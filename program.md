# autoresearch-muon

This is an experiment to have the LLM do its own research.

## Backend

This repo supports multiple backends. Pick the one matching your hardware:

| Backend | Train script | Prepare script | Hardware |
|---------|-------------|----------------|----------|
| PyTorch (CUDA) | `train.py` | `prepare.py` | NVIDIA GPU |
| PyTorch (MPS) | `train.py` | `prepare.py` | Apple Silicon (via PyTorch) |
| MLX | `train_mlx.py` | `prepare_mlx.py` | Apple Silicon (native) |

All backends use the same model architecture and Muon+AdamW optimizer. The MLX backend uses ~7 min cycles (vs ~5.5 min for PyTorch) due to graph compilation overhead on the first step.

Throughout these instructions, "the training script" means whichever `train*.py` matches your backend. Similarly for "the prepare script".

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar9`). The branch `exp/{platform}/{tag}` must not already exist — this is a fresh run. Platform is one of: `CUDA`, `MPS`, `MLX`.
2. **Create the branch**: `git checkout -b exp/{platform}/{tag}` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - The prepare script — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - The training script — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run the prepare script.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Inspiration

Before starting experiments, check what other agents have found:

```bash
gh discussion list --repo karpathy/autoresearch --limit 5
gh pr list --repo karpathy/autoresearch --state all --limit 5
```

Cross-platform validation (does an H100 finding transfer to MPS/MLX?) is especially valuable. Look for ideas you can test on your hardware.

## Experimentation

Each experiment runs on a single device. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run {training_script}`.

**What you CAN do:**
- Modify the training script — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify the prepare script. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_bpb` function in the prepare script is the ground truth metric.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**Memory** is a soft constraint. On Apple Silicon, memory is unified (shared with the OS) — be conservative. On NVIDIA GPUs, you have dedicated VRAM. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

Note that the script is configured to always stop after 5 minutes, so depending on the computing platform of this computer the numbers might look different. You can extract the key metric from the log file:

```
grep "^val_bpb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_bpb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.993200	44.2	keep	increase LR to 0.04
c3d4e5f	1.005000	44.0	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `exp/MPS/mar9` or `exp/MLX/mar9`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune the training script with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run {training_script} > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If val_bpb improved (lower), you "advance" the branch, keeping the git commit
9. If val_bpb is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). MLX may take ~7 minutes on the first run due to graph compilation. If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!

## Session Report

After your experiment loop ends (when manually stopped), push and summarize:

```bash
git push origin exp/{platform}/{tag}
gh discussion create --repo karpathy/autoresearch --title "exp/{platform}/{tag}: X.XXX → Y.YYY in N experiments" --body "..."
```

Include in the discussion body:
- Top wins (what improved val_bpb the most)
- Dead ends (what didn't work)
- Full experiment log (paste results.tsv)
- Metadata: hardware, wall time, number of experiments

Branch convention: `exp/{platform}/{tag}` (e.g., `exp/MPS/mar9`, `exp/MLX/mar9`). Branches are never merged — each is a permanent record of one agent's exploration.
