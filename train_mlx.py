"""
Autoresearch pretraining script — MLX backend with Muon optimizer.
Single-device, single-file. Apple Silicon native.
Usage: uv run train_mlx.py
"""

import gc
import math
import os
import time
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map

from prepare_mlx import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, evaluate_bpb, make_dataloader

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


def norm(x):
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-5)


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def create_additive_causal_mask(seq_len, dtype=mx.float32):
    indices = mx.arange(seq_len)
    blocked = indices[None, :] > indices[:, None]
    return mx.where(blocked, mx.array(float("-inf"), dtype=dtype), mx.array(0.0, dtype=dtype))


def create_sliding_window_mask(seq_len, window_size, dtype=mx.float32):
    indices = mx.arange(seq_len)
    causal = indices[None, :] > indices[:, None]
    too_far = (indices[:, None] - indices[None, :]) >= window_size
    blocked = causal | too_far
    return mx.where(blocked, mx.array(float("-inf"), dtype=dtype), mx.array(0.0, dtype=dtype))


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = (
            nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if has_ve(layer_idx, config.n_layer)
            else None
        )
        self.rope = nn.RoPE(self.head_dim, traditional=True, base=10000)

    def __call__(self, x, ve, mask):
        B, T, _ = x.shape
        q = self.c_q(x).reshape(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).reshape(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).reshape(B, T, self.n_kv_head, self.head_dim)

        if ve is not None and self.ve_gate is not None:
            ve = ve.reshape(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * mx.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + mx.expand_dims(gate, axis=-1) * ve

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        q = norm(self.rope(q))
        k = norm(self.rope(k))

        scale = 1.0 / math.sqrt(self.head_dim)
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def __call__(self, x):
        x = self.c_fc(x)
        x = mx.maximum(x, 0) ** 2  # ReLU squared
        return self.c_proj(x)


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def __call__(self, x, ve, mask):
        x = x + self.attn(norm(x), ve, mask)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = [Block(config, i) for i in range(config.n_layer)]
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = mx.ones((config.n_layer,), dtype=mx.float32)
        self.x0_lambdas = mx.zeros((config.n_layer,), dtype=mx.float32)
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = {
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        }
        self._mask_cache = {}

    def init_weights(self):
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5

        self.wte.weight = (mx.random.normal(self.wte.weight.shape) * 1.0).astype(mx.bfloat16)
        self.lm_head.weight = (mx.random.normal(self.lm_head.weight.shape) * 0.001).astype(mx.bfloat16)

        for block in self.blocks:
            block.attn.c_q.weight = mx.random.uniform(-s, s, block.attn.c_q.weight.shape).astype(mx.bfloat16)
            block.attn.c_k.weight = mx.random.uniform(-s, s, block.attn.c_k.weight.shape).astype(mx.bfloat16)
            block.attn.c_v.weight = mx.random.uniform(-s, s, block.attn.c_v.weight.shape).astype(mx.bfloat16)
            block.attn.c_proj.weight = mx.zeros_like(block.attn.c_proj.weight).astype(mx.bfloat16)
            block.mlp.c_fc.weight = mx.random.uniform(-s, s, block.mlp.c_fc.weight.shape).astype(mx.bfloat16)
            block.mlp.c_proj.weight = mx.zeros_like(block.mlp.c_proj.weight).astype(mx.bfloat16)
            if block.attn.ve_gate is not None:
                block.attn.ve_gate.weight = mx.zeros_like(block.attn.ve_gate.weight).astype(mx.bfloat16)

        self.resid_lambdas = mx.ones((self.config.n_layer,), dtype=mx.float32)
        self.x0_lambdas = mx.full((self.config.n_layer,), 0.1, dtype=mx.float32)

        for ve in self.value_embeds.values():
            ve.weight = mx.random.uniform(-s, s, ve.weight.shape).astype(mx.bfloat16)

    def num_scaling_params(self):
        flat = dict(tree_flatten(self.parameters()))
        wte = self.wte.weight.size
        value_embeds = sum(p.size for k, p in flat.items() if "value_embeds" in k)
        lm_head = self.lm_head.weight.size
        block_params = sum(p.size for k, p in flat.items() if "blocks" in k)
        scalars = self.resid_lambdas.size + self.x0_lambdas.size
        total = wte + value_embeds + lm_head + block_params + scalars
        return {
            'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
            'transformer_matrices': block_params, 'scalars': scalars, 'total': total,
        }

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": long_window, "S": short_window}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = long_window
        return window_sizes

    def _get_masks(self, seq_len):
        unique_windows = set(self.window_sizes)
        for w in unique_windows:
            key = (seq_len, w)
            if key not in self._mask_cache:
                if w >= seq_len:
                    self._mask_cache[key] = create_additive_causal_mask(seq_len)
                else:
                    self._mask_cache[key] = create_sliding_window_mask(seq_len, w)
        return [self._mask_cache[(seq_len, w)] for w in self.window_sizes]

    def __call__(self, idx, targets=None, reduction="mean"):
        _, T = idx.shape
        masks = self._get_masks(T)

        x = self.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.blocks):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, masks[i])
        x = norm(x)

        logits = self.lm_head(x).astype(mx.float32)
        logits = 15.0 * mx.tanh(logits / 15.0)

        if targets is None:
            return logits

        valid = targets != -1
        targets_safe = mx.where(valid, targets, mx.zeros_like(targets))
        ce = nn.losses.cross_entropy(logits, targets_safe, reduction="none")
        ce = ce * valid
        if reduction == "none":
            return ce
        denom = mx.maximum(mx.sum(valid), 1)
        return mx.sum(ce) / denom

# ---------------------------------------------------------------------------
# Muon + AdamW Optimizer (MLX port)
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def _set_path_value(model, path, value):
    """Set a parameter value in an MLX model by dotted path."""
    parts = path.split(".")
    obj = model
    for part in parts[:-1]:
        if isinstance(obj, list):
            obj = obj[int(part)]
        elif isinstance(obj, dict):
            obj = obj[part]
        else:
            obj = getattr(obj, part)
    last = parts[-1]
    if isinstance(obj, dict):
        obj[last] = value
    else:
        setattr(obj, last, value)


class MuonAdamW:
    """Combined optimizer: Muon for 2D matrix params, AdamW for others (MLX)."""

    def __init__(self, model, unembedding_lr, embedding_lr, matrix_lr,
                 weight_decay, adam_betas, scalar_lr):
        self.model = model
        model_dim = model.config.n_embd
        dmodel_lr_scale = (model_dim / 768) ** -0.5

        # Classify all parameters
        self.adamw_params = {}   # path -> config
        self.muon_groups = {}    # shape -> {paths, lr, ...}
        self.adam_state = {}     # path -> {m, v, t}
        self.muon_state = {}    # shape -> {momentum_buffer, second_momentum_buffer}

        flat_params = tree_flatten(model.parameters())
        muon_paths_by_shape = {}

        for path, param in flat_params:
            if "blocks" in path and param.ndim == 2:
                # Matrix params -> Muon
                shape = param.shape
                if shape not in muon_paths_by_shape:
                    muon_paths_by_shape[shape] = []
                muon_paths_by_shape[shape].append(path)
            elif "wte" in path:
                self.adamw_params[path] = {
                    "lr": embedding_lr * dmodel_lr_scale,
                    "betas": adam_betas, "eps": 1e-10, "weight_decay": 0.0,
                }
            elif "value_embeds" in path:
                self.adamw_params[path] = {
                    "lr": embedding_lr * dmodel_lr_scale,
                    "betas": adam_betas, "eps": 1e-10, "weight_decay": 0.0,
                }
            elif "lm_head" in path:
                self.adamw_params[path] = {
                    "lr": unembedding_lr * dmodel_lr_scale,
                    "betas": adam_betas, "eps": 1e-10, "weight_decay": 0.0,
                }
            elif "resid_lambdas" in path:
                self.adamw_params[path] = {
                    "lr": scalar_lr * 0.01,
                    "betas": adam_betas, "eps": 1e-10, "weight_decay": 0.0,
                }
            elif "x0_lambdas" in path:
                self.adamw_params[path] = {
                    "lr": scalar_lr,
                    "betas": (0.96, 0.95), "eps": 1e-10, "weight_decay": 0.0,
                }
            else:
                self.adamw_params[path] = {
                    "lr": unembedding_lr * dmodel_lr_scale,
                    "betas": adam_betas, "eps": 1e-10, "weight_decay": 0.0,
                }

        # Build Muon groups keyed by shape
        for shape, paths in sorted(muon_paths_by_shape.items()):
            self.muon_groups[shape] = {
                "paths": paths,
                "lr": matrix_lr,
                "momentum": 0.95,
                "ns_steps": 5,
                "beta2": 0.95,
                "weight_decay": weight_decay,
            }

        # Store initial LRs for scheduling
        self.initial_adamw_lrs = {p: c["lr"] for p, c in self.adamw_params.items()}
        self.initial_muon_lr = matrix_lr

        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
        print(f"MuonAdamW: {len(self.adamw_params)} AdamW params, "
              f"{sum(len(g['paths']) for g in self.muon_groups.values())} Muon params "
              f"in {len(self.muon_groups)} shape groups")

    def _adamw_step(self, path, grad, param, config):
        grad_f32 = grad.astype(mx.float32)
        param_f32 = param.astype(mx.float32)
        lr = config["lr"]
        beta1, beta2 = config["betas"]
        eps = config["eps"]
        wd = config["weight_decay"]

        if path not in self.adam_state:
            self.adam_state[path] = {
                "m": mx.zeros_like(grad_f32),
                "v": mx.zeros_like(grad_f32),
                "t": 0,
            }

        state = self.adam_state[path]
        state["t"] += 1
        state["m"] = beta1 * state["m"] + (1 - beta1) * grad_f32
        state["v"] = beta2 * state["v"] + (1 - beta2) * (grad_f32 * grad_f32)

        bias1 = 1 - beta1 ** state["t"]
        bias2 = 1 - beta2 ** state["t"]
        denom = mx.sqrt(state["v"] / bias2) + eps
        step_size = lr / bias1

        param_f32 = param_f32 * (1 - lr * wd)
        param_f32 = param_f32 - step_size * (state["m"] / denom)
        return param_f32.astype(param.dtype)

    def _muon_step(self, shape, group, flat_grads, flat_params):
        paths = group["paths"]
        momentum_val = group["momentum"]
        lr = group["lr"]
        wd = group["weight_decay"]
        beta2 = group["beta2"]
        ns_steps = group["ns_steps"]
        num_params = len(paths)

        # Stack grads and params
        stacked_grads = mx.stack([flat_grads[p].astype(mx.float32) for p in paths])
        stacked_params = mx.stack([flat_params[p].astype(mx.float32) for p in paths])

        # Initialize state
        if shape not in self.muon_state:
            self.muon_state[shape] = {
                "momentum_buffer": mx.zeros_like(stacked_grads),
                "second_momentum_buffer": mx.zeros(
                    (num_params, shape[-2], 1) if shape[-2] >= shape[-1]
                    else (num_params, 1, shape[-1]),
                    dtype=mx.float32,
                ),
            }

        state = self.muon_state[shape]
        red_dim = -1 if shape[-2] >= shape[-1] else -2

        # Nesterov momentum
        state["momentum_buffer"] = momentum_val * state["momentum_buffer"] + (1 - momentum_val) * stacked_grads
        g = momentum_val * state["momentum_buffer"] + (1 - momentum_val) * stacked_grads

        # Polar express orthogonalization (bfloat16 for speed)
        X = g.astype(mx.bfloat16)
        X_norm = mx.linalg.norm(X.astype(mx.float32), axis=(-2, -1), keepdims=True).astype(mx.bfloat16)
        X = X / (X_norm * 1.02 + 1e-6)

        if g.shape[-2] > g.shape[-1]:
            for a, b, c in polar_express_coeffs[:ns_steps]:
                A = mx.swapaxes(X, -2, -1) @ X
                B = b * A + c * (A @ A)
                X = a * X + X @ B
        else:
            for a, b, c in polar_express_coeffs[:ns_steps]:
                A = X @ mx.swapaxes(X, -2, -1)
                B = b * A + c * (A @ A)
                X = a * X + B @ X

        g = X.astype(mx.float32)

        # NorMuon variance reduction
        v_mean = (g * g).mean(axis=red_dim, keepdims=True)
        red_dim_size = g.shape[red_dim]
        v_norm_sq = v_mean.sum(axis=(-2, -1), keepdims=True) * red_dim_size
        v_norm = mx.sqrt(v_norm_sq)

        state["second_momentum_buffer"] = (
            beta2 * state["second_momentum_buffer"] + (1 - beta2) * v_mean
        )
        step_size = mx.rsqrt(mx.maximum(state["second_momentum_buffer"], 1e-10))
        scaled_sq_sum = (v_mean * red_dim_size) * (step_size * step_size)
        v_norm_new = mx.sqrt(scaled_sq_sum.sum(axis=(-2, -1), keepdims=True))
        final_scale = step_size * (v_norm / mx.maximum(v_norm_new, 1e-10))
        g = g * final_scale

        # LR scaling for non-square matrices
        lr_scaled = lr * max(1.0, shape[-2] / shape[-1]) ** 0.5

        # Cautious weight decay + parameter update
        mask = (g * stacked_params) >= 0
        new_params = stacked_params - lr_scaled * g - lr_scaled * wd * stacked_params * mask

        # Unstack and write back
        for i, path in enumerate(paths):
            _set_path_value(self.model, path, new_params[i].astype(flat_params[path].dtype))

    def update(self, grads):
        flat_grads = dict(tree_flatten(grads))
        flat_params = dict(tree_flatten(self.model.parameters()))

        # AdamW step
        for path, config in self.adamw_params.items():
            if path not in flat_grads:
                continue
            new_param = self._adamw_step(path, flat_grads[path], flat_params[path], config)
            _set_path_value(self.model, path, new_param)

        # Muon step (grouped by shape)
        for shape, group in self.muon_groups.items():
            self._muon_step(shape, group, flat_grads, flat_params)

    def set_schedule(self, lr_multiplier, muon_momentum, muon_weight_decay):
        for path, config in self.adamw_params.items():
            config["lr"] = self.initial_adamw_lrs[path] * lr_multiplier
        for group in self.muon_groups.values():
            group["lr"] = self.initial_muon_lr * lr_multiplier
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay

    @property
    def state_arrays(self):
        """Return all optimizer state arrays for mx.eval()."""
        arrays = []
        for s in self.adam_state.values():
            arrays.extend([s["m"], s["v"]])
        for s in self.muon_state.values():
            arrays.extend([s["momentum_buffer"], s["second_momentum_buffer"]])
        return arrays


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

ASPECT_RATIO = 64
HEAD_DIM = 128
WINDOW_PATTERN = "L"

TOTAL_BATCH_SIZE = 2**14
EMBEDDING_LR = 1.0
UNEMBEDDING_LR = 0.008
MATRIX_LR = 0.04
SCALAR_LR = 0.5
WEIGHT_DECAY = 0.2
ADAM_BETAS = (0.8, 0.95)
WARMUP_RATIO = 0.0
WARMDOWN_RATIO = 0.5
FINAL_LR_FRAC = 0.0

DEPTH = 4
DEVICE_BATCH_SIZE = 8
FINAL_EVAL_BATCH_SIZE = 16
STARTUP_EXCLUDE_STEPS = 1

# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------

def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    if progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    cooldown = (1.0 - progress) / WARMDOWN_RATIO
    return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
mx.random.seed(42)

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
x, y, epoch = next(train_loader)
t_data = time.time()
print(f"Data/tokenizer loaded in {t_data - t_start:.1f}s")

model_dim = ((DEPTH * ASPECT_RATIO + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
config = GPTConfig(
    sequence_len=MAX_SEQ_LEN,
    vocab_size=vocab_size,
    n_layer=DEPTH,
    n_head=model_dim // HEAD_DIM,
    n_kv_head=model_dim // HEAD_DIM,
    n_embd=model_dim,
    window_pattern=WINDOW_PATTERN,
)

model = GPT(config)
model.init_weights()
mx.eval(model.parameters())
num_params = sum(p.size for _, p in tree_flatten(model.parameters()))

param_counts = model.num_scaling_params()
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

optimizer = MuonAdamW(
    model,
    unembedding_lr=UNEMBEDDING_LR,
    embedding_lr=EMBEDDING_LR,
    matrix_lr=MATRIX_LR,
    weight_decay=WEIGHT_DECAY,
    adam_betas=ADAM_BETAS,
    scalar_lr=SCALAR_LR,
)

loss_grad_fn = nn.value_and_grad(model, lambda mdl, inputs, targets: mdl(inputs, targets=targets))

print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

smooth_train_loss = 0.0
total_training_time = 0.0
step = 0
t_compiled = None

while True:
    t0 = time.time()
    accum_grads = None
    train_loss = 0.0

    for _ in range(grad_accum_steps):
        loss, grads = loss_grad_fn(model, x, y)
        mx.eval(loss, grads)
        if t_compiled is None:
            t_compiled = time.time()
            print(f"Model compiled in {t_compiled - t_data:.1f}s")
        train_loss += float(loss.item()) / grad_accum_steps
        if accum_grads is None:
            accum_grads = grads
        else:
            accum_grads = tree_map(lambda a, b: a + b, accum_grads, grads)
        x, y, epoch = next(train_loader)

    if grad_accum_steps > 1:
        accum_grads = tree_map(lambda g: g * (1.0 / grad_accum_steps), accum_grads)

    # Schedules
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    muon_wd = get_weight_decay(progress)
    optimizer.set_schedule(lrm, muon_momentum, muon_wd)

    optimizer.update(accum_grads)
    mx.eval(model.parameters(), *optimizer.state_arrays)

    train_loss_f = train_loss
    # Fast fail: abort if loss is exploding or NaN
    if not train_loss_f <= 100:
        print(f"FAIL: Loss exploded at step {step} (loss={train_loss_f:.2f})")
        raise SystemExit(1)

    dt = time.time() - t0
    if step >= STARTUP_EXCLUDE_STEPS:
        total_training_time += dt

    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt) if dt > 0 else 0
    remaining = max(0.0, TIME_BUDGET - total_training_time)

    print(
        f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | "
        f"lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | "
        f"epoch: {epoch} | remaining: {remaining:.0f}s    ",
        end="", flush=True,
    )

    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1
    if step >= STARTUP_EXCLUDE_STEPS and total_training_time >= TIME_BUDGET:
        break

print()

total_tokens = step * TOTAL_BATCH_SIZE

# Save pre-eval checkpoint weights
print("Saving pre-eval checkpoint...")
flat_params = dict(tree_flatten(model.parameters()))
import numpy as np
np.savez('pre_eval_checkpoint.npz', **{k: np.array(v.astype(mx.float32)) for k, v in flat_params.items()})

# Final eval (reduced tokens for Apple Silicon)
print("Starting final eval...")
eval_tok = 2 * 524288
val_bpb = evaluate_bpb(model, tokenizer, FINAL_EVAL_BATCH_SIZE, eval_tokens=eval_tok)

# Eval succeeded — remove safety checkpoint
os.remove('pre_eval_checkpoint.npz')

t_end = time.time()
peak_vram_mb = mx.get_peak_memory() / 1024 / 1024

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"mfu_percent:      0.00")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")
