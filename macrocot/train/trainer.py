"""Training orchestration for one configuration."""

from __future__ import annotations

import json
import math
import os
import random
import time
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..counter_machines.base import CounterMachine
from ..counter_machines.baseb import build_baseb_machine
from ..counter_machines.macro import build_macro_machine
from ..counter_machines.tasks import build_task
from ..data.dataset import TraceDataset, make_collate
from ..data.generate import make_splits, split_stats
from ..data.tokenizer import Tokenizer
from ..eval.evaluate import evaluate_all
from ..model.transformer import ModelConfig, TraceTransformer
from .config import Config


def build_config_machine(cfg: Config) -> Tuple[CounterMachine, Dict, CounterMachine]:
    base, meta = build_task(cfg.task)
    base_machine = base
    machine = base
    if cfg.base_b > 1:
        idx = meta.get("single_decrement_counter")
        if idx is None:
            raise ValueError(f"task {cfg.task} does not support base-b counting")
        machine = build_baseb_machine(machine, cfg.base_b, idx)
    if cfg.macro_k > 1:
        machine = build_macro_machine(machine, cfg.macro_k)
    return machine, meta, base_machine


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def run_config(cfg: Config, out_dir: str, verbose: bool = True) -> Dict:
    torch.set_num_threads(1)
    set_seed(cfg.seed)
    os.makedirs(out_dir, exist_ok=True)

    machine, meta, base_machine = build_config_machine(cfg)
    splits = make_splits(
        machine,
        meta,
        train_max=cfg.train_max,
        ood1_max=cfg.ood1_max,
        ood2_max=cfg.ood2_max,
        n_train=cfg.n_train,
        n_test=cfg.n_test,
        seed=cfg.seed,
        base_machine=base_machine,
        length_min=cfg.length_min,
    )
    stats = split_stats(splits)

    labels = [t.label for t in machine.transitions]
    tokenizer = Tokenizer(meta["alphabet"], labels)

    max_trace_len = max(
        (len(e.trace) for exs in splits.values() for e in exs), default=1
    )
    max_new_tokens = max_trace_len + 2
    max_len = max(
        (len(tokenizer.build_sequence(e.word, e.trace)[0]) for exs in splits.values() for e in exs),
        default=8,
    ) + 2

    mcfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        max_len=max(max_len, cfg.max_len),
        pos=cfg.pos,
    )
    model = TraceTransformer(mcfg)

    train_ds = TraceDataset(splits["train"], tokenizer)
    collate = make_collate(tokenizer.pad_id)
    eff_batch = max(1, min(cfg.batch_size, len(train_ds)))
    loader = DataLoader(
        train_ds, batch_size=eff_batch, shuffle=True, collate_fn=collate, drop_last=False
    )

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def lr_at(step):
        if step < cfg.warmup:
            return cfg.lr * (step + 1) / cfg.warmup
        prog = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
        return 0.5 * cfg.lr * (1 + math.cos(math.pi * min(1.0, prog)))

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    history = []
    steps_to_solve = {"test0": None, "test1": None, "test2": None}

    start = time.time()
    step = 0
    data_iter = iter(loader)
    model.train()
    while step < cfg.steps:
        try:
            xb, yb = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            xb, yb = next(data_iter)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        logits = model(xb)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        step += 1

        if step % cfg.eval_interval == 0 or step == cfg.steps:
            metrics = evaluate_all(model, tokenizer, splits, max_new_tokens)
            rec = {"step": step, "loss": float(loss.item())}
            for sp, m in metrics.items():
                rec[f"{sp}_em"] = m["exact_match"]
                rec[f"{sp}_acc"] = m["accept_acc"]
                if steps_to_solve[sp] is None and m["exact_match"] >= cfg.threshold:
                    steps_to_solve[sp] = step
            history.append(rec)
            model.train()
            if verbose:
                print(
                    f"[{cfg.name()}] step {step} loss {loss.item():.3f} "
                    f"em0 {rec.get('test0_em', 0):.3f} "
                    f"em1 {rec.get('test1_em', 0):.3f} "
                    f"em2 {rec.get('test2_em', 0):.3f}",
                    flush=True,
                )

    wall = time.time() - start
    final = history[-1] if history else {}
    result = {
        "config": cfg.to_dict(),
        "name": cfg.name(),
        "num_params": model.num_params(),
        "vocab_size": tokenizer.vocab_size,
        "num_transitions": len(machine.transitions),
        "split_stats": stats,
        "max_trace_len": max_trace_len,
        "history": history,
        "final": final,
        "steps_to_solve": steps_to_solve,
        "wall_time_sec": wall,
    }
    with open(os.path.join(out_dir, cfg.name() + ".json"), "w") as f:
        json.dump(result, f, indent=2)
    return result
