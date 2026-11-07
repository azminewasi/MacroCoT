"""Cost measurements for the positions-not-bits claim.

The compression theorems reduce the number of generated positions in a trace.
This module measures what that reduction buys, separating three quantities:

1. Serial generation steps. One autoregressive step is emitted per trace token,
   so the step count equals the trace length. This is the latency-critical term.
2. Attention cost. With a key-value cache, generating a trace of N tokens costs
   about sum_{t=1}^{N} t = N (N + 1) / 2 attention interactions per head and
   layer, hence Theta(N^2) in the trace length. Reducing N by a factor c reduces
   this by about c^2.
3. Key-value cache size. The cache holds one key and one value vector per layer
   per generated position, so its peak size is linear in N and shrinks by c.

We report both an analytic account of these quantities at the measured mean
trace length and a wall-clock measurement of greedy generation with the actual
model, using an untrained network of the target size. Generation cost depends on
the trace length and the model dimensions, not on the trained weights, so an
untrained model gives a faithful timing of the decoder at that length. The
decoder here recomputes the prefix at each step rather than caching, so its
measured latency scales with the sum of prefix lengths and upper bounds the
key-value-cached cost; we report it as a decoder latency measurement and keep
the cached attention FLOPs as the analytic figure.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Dict, List

import torch

from ..counter_machines.tasks import build_task
from ..counter_machines.macro import build_macro_machine
from ..counter_machines.baseb import build_baseb_machine
from ..data.tokenizer import Tokenizer
from ..data.dataset import encode_prompt
from ..model.transformer import ModelConfig, TraceTransformer


def analytic_costs(mean_trace_len: float, cfg: ModelConfig, bytes_per_float: int = 4) -> Dict[str, float]:
    """Analytic per-example costs at a given mean trace length.

    Uses the standard decoder accounting with a key-value cache. Attention FLOPs
    count the score and context passes, both Theta(N^2 d). Linear FLOPs count the
    projection and feed-forward layers, Theta(N d^2). Cache size counts one key
    and one value vector per layer per position.
    """
    n = float(mean_trace_len)
    d = cfg.d_model
    L = cfg.n_layers
    dff = cfg.d_ff
    # attention: per layer, sum_t t interactions, each of width d, score + context
    attn_flops = L * 2.0 * (n * (n + 1) / 2.0) * d * 2.0
    # linear: per token, qkv (3 d^2) + proj (d^2) + ffn (2 d dff), times 2 for mul-add
    linear_per_token = (4.0 * d * d + 2.0 * d * dff) * 2.0
    linear_flops = L * n * linear_per_token
    kv_floats = 2.0 * L * n * d
    return {
        "serial_steps": n,
        "attn_flops": attn_flops,
        "linear_flops": linear_flops,
        "total_flops": attn_flops + linear_flops,
        "kv_cache_floats": kv_floats,
        "kv_cache_bytes": kv_floats * bytes_per_float,
    }


@dataclass
class CostRow:
    task: str
    macro_k: int
    base_b: int
    mean_trace_len: float
    gen_latency_ms: float
    serial_steps: float
    attn_flops: float
    kv_cache_bytes: float
    vocab_size: int


def _build_machine(task: str, macro_k: int, base_b: int):
    machine, meta = build_task(task)
    if base_b > 1:
        idx = meta.get("single_decrement_counter", 0)
        machine = build_baseb_machine(machine, base_b, idx)
    if macro_k > 1:
        machine = build_macro_machine(machine, macro_k)
    return machine, meta


def benchmark(
    task: str,
    macro_k: int = 1,
    base_b: int = 1,
    n_examples: int = 40,
    length: int = 24,
    d_model: int = 64,
    n_layers: int = 2,
    n_heads: int = 4,
    d_ff: int = 128,
    seed: int = 0,
) -> CostRow:
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    machine, meta = _build_machine(task, macro_k, base_b)
    labels = [t.label for t in machine.transitions]
    tok = Tokenizer(meta["alphabet"], labels)

    rng = random.Random(seed)
    words, traces = [], []
    for _ in range(n_examples):
        w = meta["sample_by_length"](length, rng)
        run = machine.run(w)
        words.append(w)
        traces.append(run.trace)
    mean_len = sum(len(t) for t in traces) / max(1, len(traces))
    max_new = max((len(t) for t in traces), default=1) + 2

    mcfg = ModelConfig(
        vocab_size=tok.vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_len=length + max_new + 8,
        pos="nope",
    )
    model = TraceTransformer(mcfg)
    model.eval()

    prompts = [encode_prompt(tok, w) for w in words]
    steps = [len(t) for t in traces]  # force the true number of decode steps

    @torch.no_grad()
    def decode_fixed(prompt, n_steps):
        # emit exactly n_steps tokens, recomputing the prefix each step, which is
        # the work an uncached decoder does to produce a trace of that length
        seq = list(prompt)
        for _ in range(n_steps):
            x = torch.tensor([seq], dtype=torch.long)
            if x.shape[1] > mcfg.max_len:
                x = x[:, -mcfg.max_len:]
            logits = model.forward(x)[0, -1]
            seq.append(int(torch.argmax(logits).item()))
        return seq

    # warm up the dispatcher and caches before timing
    for _ in range(5):
        decode_fixed(prompts[0], steps[0])
    # time several passes and take the median to reduce CPU timing noise
    passes = []
    for _ in range(3):
        t0 = time.time()
        for p, n in zip(prompts, steps):
            decode_fixed(p, n)
        passes.append((time.time() - t0) / max(1, len(prompts)))
    passes.sort()
    latency_ms = 1000.0 * passes[len(passes) // 2]

    costs = analytic_costs(mean_len, mcfg)
    return CostRow(
        task=task,
        macro_k=macro_k,
        base_b=base_b,
        mean_trace_len=mean_len,
        gen_latency_ms=latency_ms,
        serial_steps=costs["serial_steps"],
        attn_flops=costs["attn_flops"],
        kv_cache_bytes=costs["kv_cache_bytes"],
        vocab_size=tok.vocab_size,
    )


def run_suite() -> List[CostRow]:
    # one discarded pass to absorb one-time process and dispatcher startup, so
    # the first reported row is not inflated relative to the rest
    benchmark("comparison", macro_k=1, base_b=1, n_examples=8, length=32)
    rows: List[CostRow] = []
    for k in [1, 2, 3]:
        rows.append(benchmark("comparison", macro_k=k, base_b=1, n_examples=24, length=64))
    for b in [1, 2, 4]:
        rows.append(benchmark("mod3", macro_k=1, base_b=b, n_examples=24, length=64))
    return rows


if __name__ == "__main__":
    import csv
    import os

    rows = run_suite()
    os.makedirs("results/tables", exist_ok=True)
    header = [
        "task", "macro_k", "base_b", "mean_trace_len", "gen_latency_ms",
        "serial_steps", "attn_flops", "kv_cache_bytes", "vocab_size",
    ]
    with open("results/tables/cost_table.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(header)
        for r in rows:
            wr.writerow([
                r.task, r.macro_k, r.base_b, round(r.mean_trace_len, 3),
                round(r.gen_latency_ms, 3), round(r.serial_steps, 3),
                int(r.attn_flops), int(r.kv_cache_bytes), r.vocab_size,
            ])
    for r in rows:
        print(
            f"{r.task} k{r.macro_k} b{r.base_b}: "
            f"trace {r.mean_trace_len:.2f}, latency {r.gen_latency_ms:.2f} ms, "
            f"attn_flops {r.attn_flops:.3e}, kv_bytes {int(r.kv_cache_bytes)}, "
            f"vocab {r.vocab_size}"
        )
