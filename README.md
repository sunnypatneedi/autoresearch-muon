# autoresearch-muon

Muon optimizer on every platform. Fork of [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

Full credit to [@karpathy](https://github.com/karpathy) for the core idea: fixed-time
autonomous research loops controlled entirely through `program.md`. This fork
adds Apple Silicon support (MPS + MLX) and is the only repo with the
[Muon optimizer](https://kellerjordan.github.io/posts/muon/) running on non-CUDA hardware.

If you are new to neural networks, this ["Dummy's Guide"](https://x.com/hooeem/status/2030720614752039185) looks pretty good for a lot more context.

## Quick start

Requirements: Python 3.10+, [uv](https://docs.astral.sh/uv/).

### NVIDIA GPU (CUDA)
```bash
uv sync --extra torch
uv run prepare.py
uv run train.py
```

### Apple Silicon (MPS / PyTorch)
```bash
uv sync --extra torch
uv run prepare.py
uv run train.py          # auto-detects MPS
```

### Apple Silicon (MLX)
```bash
uv sync --extra mlx
uv run prepare_mlx.py    # or reuse data from prepare.py
uv run train_mlx.py
```

If the above commands all work ok, your setup is working and you can go into autonomous research mode.

## Running the agent

Simply spin up your Claude/Codex or whatever you want in this repo (and disable all permissions), then you can prompt something like:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

## How it works

Same as the original. Three files per backend:

| | PyTorch (CUDA/MPS) | MLX |
|---|---|---|
| Data/eval (read-only) | `prepare.py` | `prepare_mlx.py` |
| Model/training (agent edits) | `train.py` | `train_mlx.py` |
| Agent instructions | `program.md` | `program.md` |

The agent reads `program.md`, picks a backend, modifies the training
script, runs a 5-minute experiment, checks `val_bpb`, and commits or reverts.
Repeat overnight. Wake up to results.

## Results

Results are hardware-dependent (more steps in 5 min = lower BPB). TBD after benchmark runs.

### Comparison with other forks

| Fork | Platform | Optimizer | Hardware |
|---|---|---|---|
| upstream (karpathy) | CUDA | Muon+AdamW | H100 80GB |
| autoresearch-mlx | MLX | AdamW only | M1 Mac Studio 48GB |
| **autoresearch-muon (MPS)** | MPS | Muon+AdamW | Apple Silicon |
| **autoresearch-muon (MLX)** | MLX | Muon+AdamW | Apple Silicon |

Note: val_bpb depends on hardware throughput. Cross-hardware comparison shows
relative optimizer advantage, not absolute numbers.

## What's different from upstream

- **Apple Silicon support.** MPS (via PyTorch) and MLX (native) backends.
- **Muon optimizer everywhere.** Polar express orthogonalization + NorMuon variance
  reduction ported to both MPS (float16) and MLX (bfloat16). The autoresearch-mlx
  fork uses AdamW only.
- **Cross-platform.** Same architecture, same optimizer, three backends
  (CUDA/MPS/MLX) in one repo.
- **Clean eval API.** Configurable `eval_tokens` parameter instead of
  monkey-patching module globals.

## Design choices

- **Single file to modify.** The agent only touches the training script (`train.py` or `train_mlx.py`). This keeps the scope manageable and diffs reviewable.
- **Fixed time budget.** Training always runs for exactly 5 minutes, regardless of your specific platform. This means you can expect approx 12 experiments/hour and approx 100 experiments while you sleep.
- **Self-contained.** No external dependencies beyond PyTorch/MLX and a few small packages. No distributed training, no complex configs. One device, one file, one metric.

## Tips for smaller hardware

If you're running on a MacBook or other memory-constrained device:

1. Use a dataset with less entropy, e.g. the [TinyStories dataset](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean).
2. Experiment with decreasing `vocab_size` (8192 -> 4096, 2048, etc.).
3. Lower `MAX_SEQ_LEN` in `prepare.py` (down to 256 etc.) and increase `DEVICE_BATCH_SIZE` to compensate.
4. Decrease `EVAL_TOKENS` so validation is evaluated on less data.
5. Lower `DEPTH` (default 8 for PyTorch, 4 for MLX) -- this is the primary model complexity knob.
6. Use `WINDOW_PATTERN = "L"` -- the alternating "SSSL" pattern may be inefficient on smaller hardware.
7. Lower `TOTAL_BATCH_SIZE` (keep powers of 2, e.g. `2**14`).

## Collaborative research

autoresearch-muon follows the emerging collaborative protocol from
[upstream](https://github.com/karpathy/autoresearch/discussions/43):
agents run independently on different hardware, share findings via
GitHub Discussions/PRs, and read each other's reports for inspiration.

Branch convention: `exp/{platform}/{tag}` (e.g., `exp/MPS/mar9`,
`exp/MLX/mar9`). Branches are never merged -- each is a permanent
record of one agent's exploration. Other agents read open PRs and
Discussions before starting their own sessions.

**Cross-platform transfer** is our unique contribution: does an
H100 finding (like "init scale 0.68x") transfer to Apple Silicon?
Having CUDA, MPS, and MLX in one repo makes this testable.

## Project structure

```
prepare.py        -- PyTorch data pipeline + eval (shared constants/tokenizer)
prepare_mlx.py    -- MLX data pipeline + eval (imports shared code from prepare.py)
train.py          -- PyTorch backend: model + Muon + training loop
train_mlx.py      -- MLX backend: model + Muon + training loop
program.md        -- unified agent instructions (both backends)
pyproject.toml    -- optional dependency groups (torch / mlx)
```

## Notable forks

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (MacOS)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) (MacOS)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (Windows)

## Acknowledgments

- [Andrej Karpathy](https://github.com/karpathy) -- autoresearch and nanochat
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) -- MLX port reference
- [Apple MLX team](https://github.com/ml-explore/mlx)

## License

MIT. Original copyright preserved.
