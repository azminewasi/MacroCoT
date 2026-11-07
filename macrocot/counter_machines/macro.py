"""Order-k macro-transition compilation.

Given a deterministic guarded-update counter machine M, build the order-<=k
macro machine M^(k): every k consecutive base steps (or fewer, when the base
run halts inside the block) become a single macro token. The macro token's
effect is the sum of the base effects, and its guard is the conjunction of each
intermediate guard evaluated at the intermediate counter values, expressed as a
threshold test on the counter values at the start of the block.

Because every guard atom is a single-counter threshold test, shifting its
argument by an integer constant keeps it a single-counter threshold test, and
the conjunction over the block is exactly an interval per counter. We compute
those intervals during enumeration, prune blocks whose interval is empty, and
emit a canonical guard. Determinism of M guarantees that at each block boundary
at most one macro is enabled.

See paper Section 4 (Macro Transitions) for the correctness and expressibility
theorems this file realizes.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .base import Atom, CounterMachine, Guard, Op, Transition

INF = 10**9


class _Interval:
    """Feasible interval [lo, hi] for one counter over non-negative integers."""

    __slots__ = ("lo", "hi")

    def __init__(self) -> None:
        self.lo = 0
        self.hi = INF

    def copy(self) -> "_Interval":
        n = _Interval()
        n.lo, n.hi = self.lo, self.hi
        return n

    def tighten(self, op: Op, c: int) -> bool:
        if op == "==":
            self.lo = max(self.lo, c)
            self.hi = min(self.hi, c)
        elif op == ">=":
            self.lo = max(self.lo, c)
        elif op == ">":
            self.lo = max(self.lo, c + 1)
        elif op == "<=":
            self.hi = min(self.hi, c)
        elif op == "<":
            self.hi = min(self.hi, c - 1)
        else:
            raise ValueError(f"bad op {op}")
        return self.lo <= self.hi

    def atoms(self, i: int) -> Tuple[Atom, ...]:
        if self.lo == self.hi:
            return ((i, "==", self.lo),)
        out: List[Atom] = []
        if self.lo > 0:
            out.append((i, ">=", self.lo))
        if self.hi < INF:
            out.append((i, "<=", self.hi))
        return tuple(out)


def _canonical_guard(intervals: List[_Interval]) -> Guard:
    atoms: List[Atom] = []
    for i, iv in enumerate(intervals):
        atoms.extend(iv.atoms(i))
    return tuple(atoms)


def _minimize(transitions: List[Transition]) -> List[Transition]:
    """Merge macro tokens that induce the same transition.

    Two length-<=k blocks that share source, destination, effect vector, and
    canonical guard are the same guarded-update transition, so they can be
    represented by a single token without changing the machine. Because the
    base machine is deterministic, distinct blocks that are not identical in
    this sense have disjoint guards, so merging only removes exact duplicates
    and never merges two behaviorally different tokens.
    """
    seen: Dict[Tuple, Transition] = {}
    for t in transitions:
        key = (t.src, t.dst, t.effect, frozenset(t.guard))
        if key not in seen:
            seen[key] = t
    return list(seen.values())


def macro_vocab_report(base: CounterMachine, k: int) -> Dict[str, int]:
    """Compare the worst-case bound to the reachable and minimized counts.

    Returns the size of the base vocabulary, the closed-form worst-case bound
    sum_{r=1}^{k} |Delta|^r, the number of reachable guard-satisfiable blocks
    that the enumeration actually emits, and the number that remain after
    canonical merging. All counts are exact for the given machine.
    """
    n = len(base.transitions)
    bound = sum(n ** r for r in range(1, k + 1))
    if k == 1:
        return {"base": n, "bound": bound, "reachable": n, "minimized": n}
    raw = _macro_transitions(base, k)
    minimized = _minimize(raw)
    return {
        "base": n,
        "bound": bound,
        "reachable": len(raw),
        "minimized": len(minimized),
    }


def _macro_transitions(base: CounterMachine, k: int, max_macros: int = 200000) -> List[Transition]:
    """Enumerate the reachable guard-satisfiable order-<=k blocks."""
    num = base.num_counters
    out_by_src: Dict[str, List[Transition]] = {}
    for t in base.transitions:
        out_by_src.setdefault(t.src, []).append(t)

    macros: List[Transition] = []

    def dfs(start_state, cur, depth, shift, intervals, labels):
        if len(macros) > max_macros:
            raise RuntimeError("macro explosion; lower k or check the machine")
        for t in out_by_src.get(cur, []):
            child_iv = [iv.copy() for iv in intervals]
            feasible = True
            for (i, op, c) in t.guard:
                if not child_iv[i].tighten(op, c - shift[i]):
                    feasible = False
                    break
            if not feasible:
                continue
            new_shift = [shift[j] + t.effect[j] for j in range(num)]
            ok = True
            for i in range(num):
                if not child_iv[i].tighten(">=", -new_shift[i]):
                    ok = False
                    break
            if not ok:
                continue
            new_labels = labels + [t.label]
            new_depth = depth + 1
            halting = base.halting(t.dst)
            if halting or new_depth == k:
                guard = _canonical_guard(child_iv)
                label = "[" + "|".join(new_labels) + "]"
                macros.append(
                    Transition(
                        src=start_state,
                        guard=guard,
                        effect=tuple(new_shift),
                        dst=t.dst,
                        label=label,
                    )
                )
                if halting:
                    continue
            else:
                dfs(start_state, t.dst, new_depth, new_shift, child_iv, new_labels)

    for q in base.states:
        if base.halting(q):
            continue
        dfs(q, q, 0, [0] * num, [_Interval() for _ in range(num)], [])
    return macros


def build_macro_machine(
    base: CounterMachine, k: int, max_macros: int = 200000, minimize: bool = False
) -> CounterMachine:
    """Compile the order-<=k macro machine of ``base``.

    When ``minimize`` is set, macro tokens that induce the same transition are
    merged (see :func:`_minimize`), which lowers the token vocabulary without
    changing the machine.
    """
    assert k >= 1
    if k == 1:
        return base

    num = base.num_counters
    macros = _macro_transitions(base, k, max_macros=max_macros)
    if minimize:
        macros = _minimize(macros)

    macro = CounterMachine(
        name=f"{base.name}_macro{k}",
        num_counters=num,
        states=list(base.states),
        start_state=base.start_state,
        accept_states=set(base.accept_states),
        reject_states=set(base.reject_states),
        transitions=macros,
        init_counters=base.init_counters,
    )
    return macro


def macro_effect_table(macro: CounterMachine) -> Dict[str, Tuple[int, ...]]:
    """Effect vector per macro label, for counter recovery from a macro trace."""
    return {t.label: t.effect for t in macro.transitions}
