"""Correctness tests: base machines, macro compression, base-b counting.

Run with:  python -m pytest tests/  (or)  python tests/test_counter_machines.py
"""

import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from macrocot.counter_machines.base import counters_from_trace
from macrocot.counter_machines.baseb import build_baseb_machine
from macrocot.counter_machines.macro import build_macro_machine, macro_effect_table
from macrocot.counter_machines.tasks import build_task


def _words(task_meta, n_min, n_max, count, seed=0):
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        length = rng.randint(n_min, n_max)
        out.append(task_meta["sample_by_length"](length, rng))
    return out


def test_base_determinism_and_recovery():
    for name in ["parity", "mod3", "mod5", "comparison", "unary_mult"]:
        m, meta = build_task(name)
        for w in _words(meta, 2, 10, 40, seed=1):
            run = m.run(w)
            assert run.halted, f"{name} did not halt on {w}"
            # counter path recovered from the trace equals the simulated path
            rec = counters_from_trace(m, w, run.trace)
            path_counters = [c for (_, c) in run.path]
            assert rec == path_counters, f"{name} recovery mismatch"
    print("ok: base determinism + counter recovery")


def test_macro_correctness():
    for name in ["parity", "mod3", "mod5", "comparison", "unary_mult"]:
        m, meta = build_task(name)
        for k in [2, 3, 4]:
            mk = build_macro_machine(m, k)
            eff = macro_effect_table(mk)
            for w in _words(meta, 2, 12, 40, seed=2):
                base_run = m.run(w)
                macro_run = mk.run(w)
                # same acceptance
                assert base_run.accepted == macro_run.accepted, (
                    f"{name} k={k} acceptance mismatch on {w}"
                )
                # trace length is exactly ceil(T/k)
                expected = math.ceil(base_run.length / k)
                assert macro_run.length == expected, (
                    f"{name} k={k} length {macro_run.length} != ceil "
                    f"{base_run.length}/{k}={expected}"
                )
                # counter recovery from macro trace matches base final counters
                rec = counters_from_trace(mk, w, macro_run.trace, eff)
                assert rec[-1] == base_run.path[-1][1], (
                    f"{name} k={k} final counter mismatch"
                )
    print("ok: macro acceptance + exact ceil(T/k) length + counter recovery")


def test_baseb_correctness():
    for name in ["parity", "mod3", "mod5"]:
        m, meta = build_task(name)
        idx = meta["single_decrement_counter"]
        for b in [2, 4]:
            mb = build_baseb_machine(m, b, idx)
            for w in _words(meta, 2, 30, 40, seed=3):
                base_run = m.run(w)
                bb_run = mb.run(w)
                assert base_run.accepted == bb_run.accepted, (
                    f"{name} b={b} acceptance mismatch on len {len(w)}"
                )
                # counting tokens shrink: base-b trace is no longer than base,
                # and for pure decrement tasks it is about ceil(n/b) + O(1).
                n = len(w)
                assert bb_run.length <= base_run.length
                assert bb_run.length <= math.ceil(n / b) + 2, (
                    f"{name} b={b} length {bb_run.length} exceeds ceil(n/b)+2 "
                    f"for n={n}"
                )
    print("ok: base-b acceptance + ceil(n/b) counting length")


def test_vocab_growth_bound():
    m, meta = build_task("mod3")
    base_delta = len(m.transitions)
    for k in [2, 3, 4]:
        mk = build_macro_machine(m, k)
        # reachable-and-feasible macro vocabulary is at most |Delta|^k
        assert len(mk.transitions) <= base_delta ** k
    print("ok: macro vocabulary <= |Delta|^k")


def test_vocab_report_and_minimization():
    from macrocot.counter_machines.macro import macro_vocab_report

    for name in ["parity", "mod3", "comparison", "unary_mult"]:
        m, meta = build_task(name)
        for k in [2, 3, 4]:
            rep = macro_vocab_report(m, k)
            # ordering the paper reports: minimized <= reachable <= bound
            assert rep["minimized"] <= rep["reachable"] <= rep["bound"]
            # minimization preserves the language: acceptance and exact length
            plain = build_macro_machine(m, k, minimize=False)
            mini = build_macro_machine(m, k, minimize=True)
            assert len(mini.transitions) == rep["minimized"]
            for w in _words(meta, 2, 12, 24, seed=7):
                rp = plain.run(w)
                rm = mini.run(w)
                assert rp.accepted == rm.accepted
                assert rp.length == rm.length
    print("ok: vocabulary report ordering + minimization soundness")


if __name__ == "__main__":
    test_base_determinism_and_recovery()
    test_macro_correctness()
    test_baseb_correctness()
    test_vocab_growth_bound()
    test_vocab_report_and_minimization()
    print("ALL COUNTER MACHINE TESTS PASSED")
