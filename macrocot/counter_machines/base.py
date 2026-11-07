"""Deterministic guarded-update counter machines.

A guarded-update counter machine (CM) operates on a finite state set and a
fixed number of non-negative integer counters. Each transition carries a guard
(a conjunction of single-counter threshold tests), an integer effect vector, a
source state, and a target state. This normal form is the one that CoT C-RASP
simulates natively: counter values are recovered as weighted prefix counts of
the emitted transition tokens, and each guard atom is a threshold test on such a
count.

The module is deliberately small and dependency free so that correctness is easy
to audit. Macro compilation (macro.py) and base-b counting (baseb.py) both
consume and produce objects defined here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# A guard atom is (counter_index, op, constant) meaning  x[i] <op> constant.
# Supported ops cover everything the constructions need after constant shifts.
Op = str  # one of "==", ">=", "<=", ">", "<"
Atom = Tuple[int, Op, int]
Guard = Tuple[Atom, ...]
Effect = Tuple[int, ...]  # integer effect vector, one entry per counter

_OPS: Dict[Op, Callable[[int, int], bool]] = {
    "==": lambda a, b: a == b,
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
}


def atom_holds(atom: Atom, x: Sequence[int]) -> bool:
    i, op, c = atom
    return _OPS[op](x[i], c)


def guard_holds(guard: Guard, x: Sequence[int]) -> bool:
    return all(atom_holds(a, x) for a in guard)


@dataclass(frozen=True)
class Transition:
    """A single guarded update.

    src: source state label
    guard: conjunction of threshold atoms on counter values
    effect: integer effect vector applied to the counters
    dst: target state label
    label: human readable name used for token vocabulary and traces
    """

    src: str
    guard: Guard
    effect: Effect
    dst: str
    label: str

    def enabled(self, state: str, x: Sequence[int]) -> bool:
        if state != self.src:
            return False
        if not guard_holds(self.guard, x):
            return False
        # No counter may become negative after the update.
        return all(x[i] + self.effect[i] >= 0 for i in range(len(x)))

    def apply(self, x: Sequence[int]) -> Tuple[int, ...]:
        return tuple(x[i] + self.effect[i] for i in range(len(x)))


@dataclass
class CounterMachine:
    """A deterministic guarded-update counter machine.

    name: identifier
    num_counters: number of counters k
    states: list of state labels
    start_state: initial state
    accept_states: halting states counted as acceptance
    reject_states: halting states counted as rejection (optional)
    transitions: list of Transition
    init_counters: maps an input word (tuple of symbol ids or a task specific
        structure) to the initial counter vector
    """

    name: str
    num_counters: int
    states: List[str]
    start_state: str
    accept_states: set
    transitions: List[Transition]
    init_counters: Callable[[Sequence[int]], Tuple[int, ...]]
    reject_states: set = field(default_factory=set)

    def halting(self, state: str) -> bool:
        return state in self.accept_states or state in self.reject_states

    def enabled_transitions(self, state: str, x: Sequence[int]) -> List[Transition]:
        return [t for t in self.transitions if t.enabled(state, x)]

    def check_deterministic_sample(self, configs: Sequence[Tuple[str, Sequence[int]]]) -> None:
        """Assert at most one transition is enabled on the given sample configs."""
        for state, x in configs:
            if self.halting(state):
                continue
            en = self.enabled_transitions(state, x)
            if len(en) > 1:
                raise ValueError(
                    f"non deterministic at state={state} x={tuple(x)}: "
                    f"{[t.label for t in en]}"
                )

    def run(self, word: Sequence[int], max_steps: int = 100000) -> "Run":
        """Simulate the machine on an input word.

        Returns a Run holding the configuration path, the emitted transition
        trace, whether it halted, and the acceptance decision.
        """
        x = self.init_counters(word)
        state = self.start_state
        path: List[Tuple[str, Tuple[int, ...]]] = [(state, tuple(x))]
        trace: List[str] = []
        steps = 0
        while not self.halting(state):
            if steps >= max_steps:
                return Run(word, path, trace, halted=False, accepted=False)
            en = self.enabled_transitions(state, x)
            if len(en) == 0:
                # Stuck: treat as rejecting halt.
                return Run(word, path, trace, halted=True, accepted=False)
            if len(en) > 1:
                raise ValueError(
                    f"machine {self.name} is non deterministic at "
                    f"state={state} x={tuple(x)}"
                )
            t = en[0]
            x = t.apply(x)
            state = t.dst
            trace.append(t.label)
            path.append((state, tuple(x)))
            steps += 1
        accepted = state in self.accept_states
        return Run(word, path, trace, halted=True, accepted=accepted)


@dataclass
class Run:
    word: Sequence[int]
    path: List[Tuple[str, Tuple[int, ...]]]
    trace: List[str]
    halted: bool
    accepted: bool

    @property
    def length(self) -> int:
        return len(self.trace)


def counters_from_trace(
    machine: CounterMachine,
    word: Sequence[int],
    trace: Sequence[str],
    label_to_effect: Optional[Dict[str, Effect]] = None,
) -> List[Tuple[int, ...]]:
    """Recover the counter path from a trace as weighted prefix counts.

    This mirrors the C-RASP recovery: C_i(t) = x0[i] + sum_{s<=t} effect(tok_s)[i].
    Used in tests to confirm that a trace alone determines the counter path.
    """
    if label_to_effect is None:
        label_to_effect = {t.label: t.effect for t in machine.transitions}
    x = list(machine.init_counters(word))
    out = [tuple(x)]
    for tok in trace:
        eff = label_to_effect[tok]
        x = [x[i] + eff[i] for i in range(len(x))]
        out.append(tuple(x))
    return out
