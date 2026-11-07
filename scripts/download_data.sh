#!/usr/bin/env bash
# Data for this project is synthetic. Every input word is sampled by a
# reference counter machine and its supervised trace is the exact transition
# sequence that machine emits. There is no external corpus to download, so this
# script materializes a small dataset to disk so the pipeline can be inspected
# without launching a training run.
set -euo pipefail
OUT="${1:-data_cache}"
mkdir -p "$OUT"
python3 - "$OUT" <<'PY'
import sys, os, json, random
from macrocot.counter_machines.tasks import build_task
from macrocot.data.generate import make_splits
out = sys.argv[1]
for task in ["parity", "mod3", "comparison"]:
    machine, meta = build_task(task)
    splits = make_splits(
        machine, meta, n_train=200, n_test=40,
        train_max=16, ood1_max=24, ood2_max=32, length_min=2,
        seed=0,
    )
    rec = {}
    for name, exs in splits.items():
        rec[name] = [
            {"word": e.word, "trace": e.trace, "accepted": e.accepted}
            for e in exs[:5]
        ]
    with open(os.path.join(out, f"{task}_sample.json"), "w") as f:
        json.dump(rec, f, indent=2)
    print(f"wrote {task}: " + ", ".join(f"{k}={len(v)} splits" for k, v in [(k, splits[k]) for k in splits]))
print("sample data written to", out)
PY
