"""Generate word-then-trace examples from a (possibly compressed) machine.

The tasks are formal languages, so there is no external corpus to download: the
data is produced deterministically by the reference counter machine. The
``download_data.sh`` script simply invokes this module. All generated data is
therefore synthetic by construction, exactly as in the base construction whose
supervision is the transition trace of a hand-built deterministic machine.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from ..counter_machines.base import CounterMachine


@dataclass
class Example:
    word: List[int]
    trace: List[str]
    accepted: bool
    base_length: int  # length of the underlying base trace (for accounting)


def generate_examples(
    machine: CounterMachine,
    meta: Dict,
    length_min: int,
    length_max: int,
    count: int,
    rng: random.Random,
    base_machine: CounterMachine = None,
    dedup: bool = True,
    max_steps: int = 200000,
) -> List[Example]:
    """Sample ``count`` unique examples with input length in the given range."""
    sampler = meta["sample_by_length"]
    seen = set()
    out: List[Example] = []
    attempts = 0
    limit = count * 50 + 1000
    while len(out) < count and attempts < limit:
        attempts += 1
        length = rng.randint(length_min, length_max)
        word = sampler(length, rng)
        key = tuple(word)
        if dedup and key in seen:
            continue
        seen.add(key)
        run = machine.run(word, max_steps=max_steps)
        if not run.halted:
            continue
        base_len = run.length
        if base_machine is not None:
            base_len = base_machine.run(word, max_steps=max_steps).length
        out.append(Example(list(word), list(run.trace), run.accepted, base_len))
    return out


def make_splits(
    machine: CounterMachine,
    meta: Dict,
    train_max: int,
    ood1_max: int,
    ood2_max: int,
    n_train: int,
    n_test: int,
    seed: int,
    base_machine: CounterMachine = None,
    length_min: int = 2,
) -> Dict[str, List[Example]]:
    rng = random.Random(seed)
    splits = {
        "train": generate_examples(
            machine, meta, length_min, train_max, n_train, rng, base_machine
        ),
        "test0": generate_examples(
            machine, meta, length_min, train_max, n_test, rng, base_machine
        ),
        "test1": generate_examples(
            machine, meta, train_max + 1, ood1_max, n_test, rng, base_machine
        ),
        "test2": generate_examples(
            machine, meta, ood1_max + 1, ood2_max, n_test, rng, base_machine
        ),
    }
    return splits


def split_stats(splits: Dict[str, List[Example]]) -> Dict[str, Dict[str, float]]:
    stats = {}
    for name, exs in splits.items():
        if not exs:
            stats[name] = {"n": 0}
            continue
        lens = [len(e.trace) for e in exs]
        base_lens = [e.base_length for e in exs]
        acc = [e.accepted for e in exs]
        stats[name] = {
            "n": len(exs),
            "mean_trace_len": sum(lens) / len(lens),
            "max_trace_len": max(lens),
            "mean_base_len": sum(base_lens) / len(base_lens),
            "accept_rate": sum(acc) / len(acc),
        }
    return stats
