#!/usr/bin/env bash
# Reproduce the CPU-scale demonstration: run the macro and base-b grids, then
# aggregate tables and figures. On a GPU, raise the size fields in
# macrocot/experiments/*.py (train_max, n_train, steps) to the paper protocol.
set -euo pipefail
python3 -m macrocot.experiments.run_speedup --out results/macro
python3 -m macrocot.experiments.run_baseb  --out results/baseb
python3 -m macrocot.experiments.analyze
echo "tables in results/tables, figures in results/figures"
