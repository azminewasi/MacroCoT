"""Base-b counting transform.

Replaces unit decrement steps on a designated counter by block updates of size up
to b. A monotone decrement burst of total mass V is realized in ceil(V / b)
block steps instead of V unit steps, while every control (non-decrement)
transition is kept at order one. The result is again a deterministic
guarded-update counter machine, hence a CoT C-RASP program, so length
generalizable learnability is preserved (paper Section 5).

Scope: the transform targets pure single-counter decrement transitions, that is
transitions whose effect is -e_idx and whose guard requires x[idx] > 0. This is
exactly the counting-burst structure for which the base-b speedup proposition is
stated. Multi-counter loops (for example the paired decrement in the comparison
task) are left unchanged; use macro compilation there.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .base import CounterMachine, Transition
from .macro import _Interval, _canonical_guard


def _is_unit_dec(effect: Tuple[int, ...], idx: int) -> bool:
    return all(
        (effect[j] == -1 if j == idx else effect[j] == 0) for j in range(len(effect))
    )


def build_baseb_machine(base: CounterMachine, b: int, idx: int) -> CounterMachine:
    """Compile the base-b machine for decrement counter ``idx``."""
    assert b >= 1
    if b == 1:
        return base
    num = base.num_counters
    dec_by_src: Dict[str, Transition] = {}
    other: List[Transition] = []
    for t in base.transitions:
        if _is_unit_dec(t.effect, idx) and any(
            a == (idx, ">", 0) or a == (idx, ">=", 1) for a in t.guard
        ):
            # A pure decrement of idx. There should be at most one per source
            # state (determinism), which we rely on for the chain.
            dec_by_src[t.src] = t
        else:
            other.append(t)

    new_transitions: List[Transition] = list(other)

    def block_from(q: str) -> None:
        # Build order-1..b decrement blocks starting at state q.
        # continue branch carries interval x[idx] >= r+1; stop branch pins == r.
        shift = 0
        cur = q
        labels: List[str] = []
        for r in range(1, b + 1):
            t = dec_by_src.get(cur)
            if t is None:
                break
            labels = labels + [t.label]
            shift -= 1
            dst = t.dst
            # stop-here block of order r
            iv = [_Interval() for _ in range(num)]
            if r < b:
                iv[idx].tighten("==", r)  # exactly r remaining -> maximal short block
            else:
                iv[idx].tighten(">=", b)  # full block
            guard = _canonical_guard(iv)
            effect = tuple(-r if j == idx else 0 for j in range(num))
            label = "<" + "|".join(labels) + ">"
            new_transitions.append(
                Transition(src=q, guard=guard, effect=effect, dst=dst, label=label)
            )
            cur = dst
            # continue to r+1 only if the chain can extend
            if dst not in dec_by_src:
                break

    for q in base.states:
        if base.halting(q):
            continue
        if q in dec_by_src:
            block_from(q)

    machine = CounterMachine(
        name=f"{base.name}_base{b}",
        num_counters=num,
        states=list(base.states),
        start_state=base.start_state,
        accept_states=set(base.accept_states),
        reject_states=set(base.reject_states),
        transitions=new_transitions,
        init_counters=base.init_counters,
    )
    return machine
