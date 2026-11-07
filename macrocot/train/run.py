"""Run a single training configuration from a YAML file.

Usage:
    python -m macrocot.train.run --config configs/parity_baseb.yaml --out results/single

The YAML keys are any subset of the fields of ``macrocot.train.config.Config``.
Missing keys fall back to the dataclass defaults.
"""

from __future__ import annotations

import argparse
import os

import yaml

from .config import Config
from .trainer import run_config


def load_config(path: str) -> Config:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    known = set(Config().to_dict().keys())
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    return Config(**data)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out", default="results/single")
    args = p.parse_args()
    cfg = load_config(args.config)
    os.makedirs(args.out, exist_ok=True)
    result = run_config(cfg, args.out, verbose=True)
    f = result["final"]
    print(
        f"done {cfg.name()}: "
        f"test0_em {f.get('test0_em')}, test1_em {f.get('test1_em')}, "
        f"test2_em {f.get('test2_em')}, steps_to_solve {result['steps_to_solve']}"
    )


if __name__ == "__main__":
    main()
