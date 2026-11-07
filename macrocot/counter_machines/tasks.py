"""Concrete deterministic counter machines used as tasks.

Every task is a formal language over a small alphabet, recognized by a
hand-built deterministic guarded-update counter machine. Supervision for a
transformer is the exact transition trace produced by the machine, and the
membership decision is the final trace token (an accept or reject transition).
This matches the protocol of the base construction, where the target is the
transition trace of a deterministic counter machine.

Tasks
-----
mod_c        Unary divisibility: input a^n, accept iff n mod c == 0.
             Single counter, pure decrement loop. Base-b applicable.
comparison   Strict comparison: input a^m b^n, accept iff m > n.
             Two counters, paired decrement loop.
unary_mult   Unary multiplication: input a^m b^n c^p, accept iff p == m*n.
             Nested loops (repeated addition with restore). Macro applicable.

Each builder returns a CounterMachine plus a metadata dict describing the
alphabet and a length-parameterized word sampler, so the data pipeline can
generate inputs of controlled size.
"""

from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

from .base import CounterMachine, Transition


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_symbols(word: Sequence[int], num_symbols: int) -> Tuple[int, ...]:
    counts = [0] * num_symbols
    for s in word:
        counts[s] += 1
    return tuple(counts)


# ---------------------------------------------------------------------------
# Task: mod-c divisibility (parity is c=2)
# ---------------------------------------------------------------------------

def build_mod(c: int = 2) -> Tuple[CounterMachine, Dict]:
    assert c >= 2
    states = [f"s{j}" for j in range(c)] + ["ACC", "REJ"]
    transitions: List[Transition] = []
    for j in range(c):
        nxt = (j + 1) % c
        # one decrement token per residue state, so the trace records the taken transition
        transitions.append(
            Transition(
                src=f"s{j}",
                guard=((0, ">", 0),),
                effect=(-1,),
                dst=f"s{nxt}",
                label=f"dec@s{j}",
            )
        )
        halt = "ACC" if j == 0 else "REJ"
        transitions.append(
            Transition(
                src=f"s{j}",
                guard=((0, "==", 0),),
                effect=(0,),
                dst=halt,
                label=f"halt@s{j}",
            )
        )

    def init_counters(word: Sequence[int]) -> Tuple[int, ...]:
        # count occurrences of symbol 'a' (index 0); symbol 'b' is a distractor
        na = sum(1 for s in word if s == 0)
        return (na,)

    m = CounterMachine(
        name=f"mod{c}",
        num_counters=1,
        states=states,
        start_state="s0",
        accept_states={"ACC"},
        reject_states={"REJ"},
        transitions=transitions,
        init_counters=init_counters,
    )

    def sample_by_length(length: int, rng: random.Random) -> List[int]:
        # random binary string over {a, b} with at least one 'a'
        length = max(1, length)
        w = [rng.randint(0, 1) for _ in range(length)]
        if 0 not in w:
            w[rng.randrange(length)] = 0
        return w

    meta = {
        "alphabet": ["a", "b"],
        "sample_by_length": sample_by_length,
        "single_decrement_counter": 0,  # marks base-b applicability
        "description": f"strings over (a,b), accept iff (#a) mod {c} == 0",
    }
    return m, meta


# ---------------------------------------------------------------------------
# Task: strict comparison  a^m b^n, accept iff m > n
# ---------------------------------------------------------------------------

def build_comparison() -> Tuple[CounterMachine, Dict]:
    states = ["pair", "ACC", "REJ"]
    transitions = [
        Transition("pair", ((0, ">", 0), (1, ">", 0)), (-1, -1), "pair", "pair_dec"),
        Transition("pair", ((0, ">", 0), (1, "==", 0)), (0, 0), "ACC", "a_left"),
        Transition("pair", ((0, "==", 0), (1, ">", 0)), (0, 0), "REJ", "b_left"),
        Transition("pair", ((0, "==", 0), (1, "==", 0)), (0, 0), "REJ", "equal"),
    ]

    def init_counters(word: Sequence[int]) -> Tuple[int, ...]:
        m, n = _count_symbols(word, 2)
        return (m, n)

    machine = CounterMachine(
        name="comparison",
        num_counters=2,
        states=states,
        start_state="pair",
        accept_states={"ACC"},
        reject_states={"REJ"},
        transitions=transitions,
        init_counters=init_counters,
    )

    def sample_by_length(length: int, rng: random.Random) -> List[int]:
        length = max(2, length)
        m = rng.randint(1, length - 1)
        n = length - m
        return [0] * m + [1] * n

    meta = {
        "alphabet": ["a", "b"],
        "sample_by_length": sample_by_length,
        "single_decrement_counter": None,  # base-b not applied (multi counter loop)
        "description": "a^m b^n, accept iff m > n",
    }
    return machine, meta


# ---------------------------------------------------------------------------
# Task: unary multiplication  a^m b^n c^p, accept iff p == m*n
# counters: 0=m 1=n 2=acc 3=p 4=tmp
# ---------------------------------------------------------------------------

def build_unary_mult() -> Tuple[CounterMachine, Dict]:
    M, N, ACC_C, P, TMP = 0, 1, 2, 3, 4
    states = ["OUTER", "ADD", "RESTORE", "CMP", "ACC", "REJ"]

    def e(**kw) -> Tuple[int, ...]:
        v = [0, 0, 0, 0, 0]
        for k, val in kw.items():
            v[{"m": M, "n": N, "acc": ACC_C, "p": P, "tmp": TMP}[k]] = val
        return tuple(v)

    transitions = [
        # outer: start an iteration if n>0 else compare
        Transition("OUTER", ((N, ">", 0),), e(), "ADD", "outer_iter"),
        Transition("OUTER", ((N, "==", 0),), e(), "CMP", "outer_done"),
        # inner add: move m into tmp and acc  (acc += m, tmp := m, m := 0)
        Transition("ADD", ((M, ">", 0),), e(m=-1, tmp=1, acc=1), "ADD", "add_step"),
        Transition("ADD", ((M, "==", 0),), e(), "RESTORE", "add_done"),
        # restore m from tmp
        Transition("RESTORE", ((TMP, ">", 0),), e(tmp=-1, m=1), "RESTORE", "restore_step"),
        Transition("RESTORE", ((TMP, "==", 0),), e(n=-1), "OUTER", "restore_done"),
        # compare acc with p
        Transition("CMP", ((ACC_C, ">", 0), (P, ">", 0)), e(acc=-1, p=-1), "CMP", "cmp_step"),
        Transition("CMP", ((ACC_C, "==", 0), (P, "==", 0)), e(), "ACC", "cmp_eq"),
        Transition("CMP", ((ACC_C, ">", 0), (P, "==", 0)), e(), "REJ", "cmp_acc_left"),
        Transition("CMP", ((ACC_C, "==", 0), (P, ">", 0)), e(), "REJ", "cmp_p_left"),
    ]

    def init_counters(word: Sequence[int]) -> Tuple[int, ...]:
        m, n, p = _count_symbols(word, 3)
        return (m, n, 0, p, 0)

    machine = CounterMachine(
        name="unary_mult",
        num_counters=5,
        states=states,
        start_state="OUTER",
        accept_states={"ACC"},
        reject_states={"REJ"},
        transitions=transitions,
        init_counters=init_counters,
    )

    def sample_by_length(length: int, rng: random.Random) -> List[int]:
        # length controls m; keep n small so trace length ~ 3*m*n stays modest
        m = max(1, min(length, 6))
        n = rng.randint(1, 4)
        true_p = m * n
        if rng.random() < 0.5:
            p = true_p
        else:
            delta = rng.choice([-2, -1, 1, 2])
            p = max(0, true_p + delta)
        return [0] * m + [1] * n + [2] * p

    meta = {
        "alphabet": ["a", "b", "c"],
        "sample_by_length": sample_by_length,
        "single_decrement_counter": None,
        "description": "a^m b^n c^p, accept iff p == m*n",
    }
    return machine, meta


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def build_task(name: str) -> Tuple[CounterMachine, Dict]:
    if name == "parity":
        return build_mod(2)
    if name == "mod3":
        return build_mod(3)
    if name == "mod5":
        return build_mod(5)
    if name == "comparison":
        return build_comparison()
    if name == "unary_mult":
        return build_unary_mult()
    raise ValueError(f"unknown task {name}")


TASKS = ["parity", "mod3", "mod5", "comparison", "unary_mult"]
