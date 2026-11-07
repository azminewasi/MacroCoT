"""Aggregate experiment summaries into tables, CSV files, and figures.

Reads the per-experiment ``summary.json`` files produced by ``run_speedup``
and ``run_baseb`` and writes:

* ``results/tables/macro_table.csv`` and ``baseb_table.csv``
* a compact grouped view averaged over seeds, printed to stdout and saved as
  ``*_grouped.csv``
* optional PNG figures (trace length vs compression, steps-to-solve vs
  compression) when matplotlib is available.

The script is read-only over the training results; running it never launches a
training job.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import pandas as pd


def _load(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    with open(path) as f:
        rows = json.load(f)
    return pd.DataFrame(rows)


def _group(df: pd.DataFrame, factor: str) -> pd.DataFrame:
    if df.empty:
        return df
    keep = [
        "train_mean_trace",
        "test0_em",
        "test1_em",
        "test2_em",
        "steps_test2",
        "vocab_size",
        "num_params",
    ]
    agg = (
        df.groupby(["task", factor])[keep]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values(["task", factor])
    )
    return agg


def _relative_length(df: pd.DataFrame, factor: str) -> pd.DataFrame:
    """Add observed length reduction relative to factor == 1."""
    if df.empty:
        return df
    out = []
    for task, sub in df.groupby("task"):
        sub = sub.copy()
        base = sub.loc[sub[factor] == 1, "train_mean_trace"]
        base_val = float(base.iloc[0]) if len(base) else float("nan")
        sub["length_ratio_vs_base"] = base_val / sub["train_mean_trace"]
        out.append(sub)
    return pd.concat(out, ignore_index=True)


def make_plots(macro_g: pd.DataFrame, baseb_g: pd.DataFrame, fig_dir: str) -> List[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    os.makedirs(fig_dir, exist_ok=True)
    paths = []

    def _plot(df, factor, xlabel, fname, title):
        if df.empty:
            return
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
        for task, sub in df.groupby("task"):
            ax[0].plot(sub[factor], sub["train_mean_trace"], marker="o", label=task)
            ax[1].plot(sub[factor], sub["test2_em"], marker="o", label=task)
        ax[0].set_xlabel(xlabel)
        ax[0].set_ylabel("mean trace length (train)")
        ax[0].set_title("trace length")
        ax[1].set_xlabel(xlabel)
        ax[1].set_ylabel("exact match (longest test split)")
        ax[1].set_title("length generalization")
        ax[1].set_ylim(-0.05, 1.05)
        for a in ax:
            a.legend()
            a.grid(True, alpha=0.3)
        fig.suptitle(title)
        fig.tight_layout()
        p = os.path.join(fig_dir, fname)
        fig.savefig(p, dpi=140)
        plt.close(fig)
        paths.append(p)

    _plot(macro_g, "macro_k", "macro order k", "macro_speedup.png",
          "Macro transitions: trace length and generalization")
    _plot(baseb_g, "base_b", "base b", "baseb_speedup.png",
          "Base-b counting: trace length and generalization")
    return paths


def main(results_dir: str = "results") -> None:
    tab_dir = os.path.join(results_dir, "tables")
    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(tab_dir, exist_ok=True)

    macro = _load(os.path.join(results_dir, "macro", "summary.json"))
    baseb = _load(os.path.join(results_dir, "baseb", "summary.json"))

    if not macro.empty:
        macro.to_csv(os.path.join(tab_dir, "macro_table.csv"), index=False)
    if not baseb.empty:
        baseb.to_csv(os.path.join(tab_dir, "baseb_table.csv"), index=False)

    macro_g = _relative_length(_group(macro, "macro_k"), "macro_k")
    baseb_g = _relative_length(_group(baseb, "base_b"), "base_b")

    if not macro_g.empty:
        macro_g.to_csv(os.path.join(tab_dir, "macro_grouped.csv"), index=False)
        print("\n=== Macro transitions (averaged over seeds) ===")
        print(macro_g.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if not baseb_g.empty:
        baseb_g.to_csv(os.path.join(tab_dir, "baseb_grouped.csv"), index=False)
        print("\n=== Base-b counting (averaged over seeds) ===")
        print(baseb_g.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    figs = make_plots(macro_g, baseb_g, fig_dir)
    if figs:
        print("\nFigures written:")
        for p in figs:
            print(" ", p)


if __name__ == "__main__":
    main()
