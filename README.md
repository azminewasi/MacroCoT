# Shorter Chains via Macro Transitions: Constant-Factor Compression for Chain-of-Thought Counter Simulation

## Repository layout

```
macrocot/
  counter_machines/
    base.py        deterministic guarded-update counter machine and its runner
    tasks.py       concrete tasks (mod-c divisibility, comparison, multiplication)
    macro.py       order-k macro-transition construction
    baseb.py       base-b rewrite of single-counter decrement loops
  data/
    tokenizer.py   symbol and transition-token vocabulary
    generate.py    trace generation and length-stratified splits
    dataset.py     torch dataset and trace-only supervision masking
  model/
    transformer.py decoder-only transformer, NoPE or learned relative bias
  train/
    config.py      experiment configuration dataclass
    trainer.py     training loop and periodic exact-match evaluation
    run.py         run a single YAML config
  eval/
    evaluate.py    greedy trace generation, exact match, membership accuracy
  experiments/
    run_speedup.py macro-transition grid
    run_baseb.py   base-b grid
    analyze.py     aggregation into tables and figures
tests/             theorem-level checks on the constructions
configs/           example single-run YAML configurations
scripts/           data materialization and full-run driver
```

## Installation

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The code runs on CPU. A GPU is only needed to reach the larger scale described
below.

## Data

All data is synthetic. Each input word is sampled by a reference counter machine
and its label is the exact transition trace that machine emits, so there is no
external corpus to download. To materialize a small sample for inspection:

```
./scripts/download_data.sh data_cache
```

## Quickstart

Run one configuration:

```
python -m macrocot.train.run --config configs/comparison_macro.yaml --out results/single
```

Reproduce the full CPU-scale demonstration and build tables and figures:

```
./scripts/run_all.sh
```

Run the construction tests (these check the theorems directly, for example that
the macro trace length is exactly ceil(T / k) and that the macro vocabulary is
bounded by the number of transitions raised to the power k):

```
python -m pytest tests/ -q
```

## What the experiments measure

Two experiments correspond to the two claims.

* `run_speedup.py` varies the macro order k on the comparison and parity tasks
  and records the mean trace length and the exact-match accuracy on three
  length-stratified test splits (in-range, and two ranges longer than anything
  seen in training).

* `run_baseb.py` varies the base b on the mod-c divisibility tasks and records
  the same quantities, plus the optimizer step at which the longest split first
  reaches the exact-match threshold.

For every configuration the trace length is expected to fall with k or b, while
the model is expected to keep generalizing to longer inputs until the compressed
vocabulary grows large enough to reduce the number of examples per token below
what the model can fit.

## Generation cost, vocabulary, and data scaling

Three further studies substantiate the cost model and probe the boundaries.

Generation cost (`python -m macrocot.eval.cost`, writes
`results/tables/cost_table.csv`). For each compression setting it measures the
wall-clock latency of forced-length greedy generation with the actual model on
one CPU core, and reports the analytic serial steps, attention FLOPs with a
key-value cache, and cache bytes at the measured mean trace length. Latency
tracks the serial step count; for comparison it falls from 16.4 ms at `k=1` to
5.4 ms at `k=3`, and the analytic attention FLOPs fall by about the square of the
compression factor.

Vocabulary growth (`macrocot.counter_machines.macro.macro_vocab_report`, table at
`results/tables/vocab_report.csv`). The reachable macro vocabulary is far below
the worst-case bound and grows about linearly in `k`, for example 15 reachable
tokens against a bound of 1554 for mod-3 at `k=4`, and 63 against 11110 for
unary-mult at `k=4`. Canonical merging removes nothing further here because
determinism already gives distinct feasible blocks disjoint guards.

Data scaling (`python -m macrocot.experiments.run_datascaling`, table at
`results/tables/datascaling_table.csv`, figure at
`results/figures/datascaling.png`). Holding the counting range, model size, and
optimizer budget fixed on parity, exact match on the longest split rises with the
training-set size for every order, and the widest-vocabulary order (`k=4`) is the
most data-starved at the smallest budget (0.44 at 60 examples), which is the
sample-complexity cost of the wider alphabet.

## Scaling to the paper protocol

The experiment scripts expose the sizes as arguments. To match the larger
protocol, raise `train_max` toward 100, raise `n_train` and `steps`, and run on
a GPU. The constructions, the tokenizer, the model, and the evaluation are
unchanged; only the scale differs.

## Tests as theorem checks

`tests/test_counter_machines.py` verifies the parts of the theory that are
exact and do not depend on training: base-machine determinism and counter
recovery, macro acceptance with trace length exactly ceil(T / k), base-b
acceptance with trace length ceil(V / b) plus a constant, and the macro
vocabulary bound. These pass independently of any learning result.

## License

MIT. See `LICENSE`.
