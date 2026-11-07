"""Exact-match evaluation.

For each example we greedily decode the trace from the word-plus-BOS prompt and
compare token-for-token with the gold trace produced by the reference machine.
Exact match is the strict metric of the base construction. We also report
acceptance accuracy, the fraction of examples whose predicted final token yields
the correct accept or reject decision.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch

from ..data.dataset import encode_prompt
from ..data.generate import Example
from ..data.tokenizer import Tokenizer


@torch.no_grad()
def evaluate_split(
    model,
    tokenizer: Tokenizer,
    examples: Sequence[Example],
    max_new_tokens: int,
) -> Dict[str, float]:
    if len(examples) == 0:
        return {"exact_match": float("nan"), "accept_acc": float("nan"), "n": 0}
    model.eval()
    exact = 0
    accept_correct = 0
    accept_labels = set()  # trace labels that indicate acceptance
    for ex in examples:
        prompt = encode_prompt(tokenizer, ex.word)
        gen_ids = model.generate(prompt, tokenizer.eos_id, max_new_tokens)
        gold_ids = tokenizer.encode_trace(ex.trace)
        if gen_ids == gold_ids:
            exact += 1
        # acceptance from predicted trace: does its final token match the gold
        # final token's acceptance status. We infer acceptance from the gold
        # example and count it correct only when the whole final token matches.
        pred_last = gen_ids[-1] if gen_ids else -1
        gold_last = gold_ids[-1] if gold_ids else -2
        if (pred_last == gold_last) == True and ex.accepted:
            accept_correct += 1
        elif (pred_last == gold_last) and not ex.accepted:
            accept_correct += 1
    n = len(examples)
    return {
        "exact_match": exact / n,
        "accept_acc": accept_correct / n,
        "n": n,
    }


def evaluate_all(model, tokenizer, splits, max_new_tokens) -> Dict[str, Dict[str, float]]:
    return {
        name: evaluate_split(model, tokenizer, exs, max_new_tokens)
        for name, exs in splits.items()
        if name.startswith("test")
    }
