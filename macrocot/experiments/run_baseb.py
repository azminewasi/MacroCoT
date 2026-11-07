"""Base-b counting speedup experiment.

A single-counter pure-decrement loop of mass V produces V unit-decrement
tokens in the base machine. Rewriting the loop in base b lets each block step
remove up to b units, so the same computation emits about V / b tokens. This
script measures that reduction and checks that length generalization and
membership accuracy are preserved as b grows.

Base-b applies only to tasks whose accepting computation is a single-counter
monotone decrement burst. In this suite those are the mod-c divisibility
tasks, which count occurrences of one symbol and then test a residue. The
comparison and multiplication tasks are excluded here because their loops are
not single-counter pure-decrement bursts; macro compression handles them
instead.

Every number is produced by a real training run on synthetic traces from the
reference machine.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

from ..train.config import Config
from ..train.trainer import run_config
from .run_speedup import summarize


def run_grid(
    tasks: List[str],
    bs: List[int],
    seeds: List[int],
    out_dir: str = "results/baseb",
    steps: int = 1500,
    **overrides,
) -> List[Dict]:
    os.makedirs(out_dir, exist_ok=True)
    rows: List[Dict] = []
    for task in tasks:
        for b in bs:
            for seed in seeds:
                cfg = Config(
                    task=task,
                    macro_k=1,
                    base_b=b,
                    seed=seed,
                    steps=steps,
                    tag="baseb",
                    **overrides,
                )
                result = run_config(cfg, out_dir)
                rows.append(summarize(result))
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(rows, f, indent=2)
    return rows


def default_grid(out_dir: str = "results/baseb") -> List[Dict]:
    rows = []
    rows += run_grid(
        tasks=["parity"],
        bs=[1, 2, 4],
        seeds=[0, 1],
        out_dir=out_dir,
        steps=1200,
        train_max=14,
        ood1_max=22,
        ood2_max=30,
        n_train=1200,
        n_test=48,
        d_model=64,
        d_ff=128,
        eval_interval=300,
    )
    rows += run_grid(
        tasks=["mod3"],
        bs=[1, 2, 4],
        seeds=[0],
        out_dir=out_dir,
        steps=1500,
        train_max=16,
        ood1_max=24,
        ood2_max=32,
        n_train=1300,
        n_test=48,
        d_model=64,
        d_ff=128,
        eval_interval=300,
    )
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(rows, f, indent=2)
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/baseb")
    p.add_argument("--quick", action="store_true", help="tiny grid for a wiring check")
    args = p.parse_args()
    if args.quick:
        run_grid(["parity"], [1, 2], [0], out_dir=args.out, steps=300,
                 n_train=400, n_test=24, eval_interval=150)
    else:
        default_grid(args.out)
