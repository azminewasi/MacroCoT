"""Data-scaling experiment.

The macro vocabulary grows with the order k, so each macro token appears in a
smaller fraction of the data. This script measures how the training-set size
needed for length generalization changes with k, holding the counting range,
the model size, and the optimizer budget fixed. For each order k and each
training size, it trains one model and records exact match on the longest test
split.

The prediction is that larger k needs more data to reach the same accuracy,
because the wider alphabet spreads the supervision over more distinct tokens.
Every number is produced by a real training run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List

from ..train.config import Config
from ..train.trainer import run_config


def run_grid(
    task: str,
    ks: List[int],
    sizes: List[int],
    out_dir: str = "results/datascaling",
    steps: int = 1200,
    seed: int = 0,
    **overrides,
) -> List[Dict]:
    os.makedirs(out_dir, exist_ok=True)
    rows: List[Dict] = []
    for k in ks:
        for n in sizes:
            cfg = Config(
                task=task,
                macro_k=k,
                base_b=1,
                seed=seed,
                steps=steps,
                n_train=n,
                tag=f"ds{n}",
                eval_interval=steps,  # evaluate once at the end to save time
                **overrides,
            )
            r = run_config(cfg, out_dir, verbose=False)
            rows.append({
                "task": task,
                "macro_k": k,
                "n_train": n,
                "train_unique": r["split_stats"]["train"]["n"],
                "vocab_size": r["vocab_size"],
                "test0_em": r["final"].get("test0_em"),
                "test1_em": r["final"].get("test1_em"),
                "test2_em": r["final"].get("test2_em"),
            })
            print(
                f"task {task} k{k} n{n}: unique {rows[-1]['train_unique']}, "
                f"vocab {rows[-1]['vocab_size']}, "
                f"em0 {rows[-1]['test0_em']:.2f} em1 {rows[-1]['test1_em']:.2f} "
                f"em2 {rows[-1]['test2_em']:.2f}",
                flush=True,
            )
    _write(rows, out_dir)
    return rows


def _write(rows: List[Dict], out_dir: str) -> None:
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(rows, f, indent=2)
    tab = os.path.join("results", "tables")
    os.makedirs(tab, exist_ok=True)
    if rows:
        with open(os.path.join(tab, "datascaling_table.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/datascaling")
    args = p.parse_args()
    run_grid(
        task="comparison",
        ks=[1, 2, 3],
        sizes=[150, 400, 1000],
        out_dir=args.out,
        steps=1200,
        train_max=16,
        ood1_max=24,
        ood2_max=32,
        n_test=48,
        d_model=64,
        d_ff=128,
    )
