"""
MLX data preparation and evaluation for autoresearch experiments.
Imports shared code from prepare.py, defines only MLX-specific functions.

Usage:
    python prepare_mlx.py                  # full prep (download + tokenizer)
    python prepare_mlx.py --num-shards 8   # download only 8 shards (for testing)
"""

import argparse
import math
import os

import mlx.core as mx
import numpy as np

from prepare import (
    MAX_SEQ_LEN, TIME_BUDGET, EVAL_TOKENS, CACHE_DIR, TOKENIZER_DIR,
    DATA_DIR, VAL_FILENAME, MAX_SHARD, Tokenizer,
    _document_batches, download_data, train_tokenizer, list_parquet_files,
)

# ---------------------------------------------------------------------------
# MLX-specific utilities
# ---------------------------------------------------------------------------

def get_token_bytes():
    """Load token_bytes as mx.array. Converts from .pt if .npy doesn't exist."""
    npy_path = os.path.join(TOKENIZER_DIR, "token_bytes.npy")
    pt_path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")
    if not os.path.exists(npy_path):
        if os.path.exists(pt_path):
            import torch
            arr = torch.load(pt_path, map_location="cpu").numpy()
            np.save(npy_path, arr)
            print(f"Converted {pt_path} -> {npy_path}")
        else:
            raise FileNotFoundError(f"No token_bytes found at {npy_path} or {pt_path}. Run prepare.py first.")
    return mx.array(np.load(npy_path), dtype=mx.int32)


def make_dataloader(tokenizer, batch_size, seq_len, split, buffer_size=1000):
    """
    BOS-aligned dataloader with best-fit packing (MLX version).
    Every row starts with BOS. Documents packed using best-fit to minimize cropping.
    100% utilization (no padding). Returns mx.array batches.
    """
    assert split in ["train", "val"]
    row_capacity = seq_len + 1
    batches = _document_batches(split)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    epoch = 1

    def refill_buffer():
        nonlocal epoch
        doc_batch, epoch = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        doc_buffer.extend(token_lists)

    while True:
        all_rows = []
        for _ in range(batch_size):
            row = []
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row.extend(doc)
                    pos += len(doc)
                else:
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row.extend(doc[:remaining])
                    pos += remaining

            all_rows.append(row[:row_capacity])

        row_array = mx.array(all_rows, dtype=mx.int32)
        inputs = row_array[:, :-1]
        targets = row_array[:, 1:]
        yield inputs, targets, epoch


def evaluate_bpb(model, tokenizer, batch_size, eval_tokens=None):
    """
    Bits per byte (BPB): vocab size-independent evaluation metric.
    Uses fixed MAX_SEQ_LEN so results are comparable across configs.
    """
    token_bytes = get_token_bytes()
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    steps = (eval_tokens or EVAL_TOKENS) // (batch_size * MAX_SEQ_LEN)
    total_nats = 0.0
    total_bytes = 0

    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction="none").reshape(-1)
        y_flat = y.reshape(-1)
        nbytes = mx.take(token_bytes, y_flat, axis=0)
        mask = nbytes > 0
        total_nats += mx.sum(loss_flat * mask).item()
        total_bytes += int(mx.sum(nbytes).item())

    if total_bytes == 0:
        return float("inf")
    return total_nats / (math.log(2) * total_bytes)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data and tokenizer for autoresearch (MLX)")
    parser.add_argument("--num-shards", type=int, default=10, help="Number of training shards to download (-1 = all)")
    parser.add_argument("--download-workers", type=int, default=8, help="Number of parallel download workers")
    args = parser.parse_args()

    num_shards = MAX_SHARD if args.num_shards == -1 else args.num_shards

    print(f"Cache directory: {CACHE_DIR}")
    print()

    download_data(num_shards, download_workers=args.download_workers)
    print()

    train_tokenizer()
    print()

    # Ensure .npy token_bytes exists
    npy_path = os.path.join(TOKENIZER_DIR, "token_bytes.npy")
    if not os.path.exists(npy_path):
        get_token_bytes()  # triggers conversion from .pt

    print("Done! Ready to train with MLX.")
