"""Experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Config:
    # task and compression
    task: str = "parity"
    macro_k: int = 1          # order-k macro compression (1 = none)
    base_b: int = 1           # base-b counting (1 = none)

    # data sizes (kept small for CPU; scale up on GPU to match the paper)
    train_max: int = 16       # max input length in training / test0
    ood1_max: int = 24        # test1 length range is (train_max, ood1_max]
    ood2_max: int = 32        # test2 length range is (ood1_max, ood2_max]
    n_train: int = 2000
    n_test: int = 60
    length_min: int = 2

    # model
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 128
    pos: str = "nope"         # "nope" or "rel"
    max_len: int = 512

    # optimization
    steps: int = 3000
    batch_size: int = 64
    lr: float = 3e-3
    weight_decay: float = 0.01
    warmup: int = 100
    eval_interval: int = 250
    grad_clip: float = 1.0
    threshold: float = 0.99   # exact-match threshold for steps-to-solve
    seed: int = 0

    tag: str = ""             # optional label for outputs

    def name(self) -> str:
        base = f"{self.task}_k{self.macro_k}_b{self.base_b}_{self.pos}_s{self.seed}"
        return f"{base}_{self.tag}" if self.tag else base

    def to_dict(self):
        return asdict(self)
