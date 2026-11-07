"""Torch Dataset and collation for word-then-trace sequences."""

from __future__ import annotations

from typing import List, Sequence

import torch
from torch.utils.data import Dataset

from .generate import Example
from .tokenizer import Tokenizer


class TraceDataset(Dataset):
    def __init__(self, examples: Sequence[Example], tokenizer: Tokenizer):
        self.tok = tokenizer
        self.items = []
        for ex in examples:
            ids, gen_start = tokenizer.build_sequence(ex.word, ex.trace)
            self.items.append((ids, gen_start))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        ids, gen_start = self.items[i]
        return ids, gen_start


def make_collate(pad_id: int):
    def collate(batch):
        max_len = max(len(ids) for ids, _ in batch)
        input_ids = []
        labels = []
        for ids, gen_start in batch:
            pad = [pad_id] * (max_len - len(ids))
            seq = ids + pad
            input_ids.append(seq[:-1])
            # labels are next tokens; -100 where we do not compute loss
            lab = [-100] * (max_len - 1)
            for pos in range(gen_start - 1, len(ids) - 1):
                lab[pos] = ids[pos + 1]
            labels.append(lab)
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )

    return collate


def encode_prompt(tokenizer: Tokenizer, word: Sequence[int]) -> List[int]:
    """The generation prompt: word symbols followed by BOS."""
    return tokenizer.encode_word(word) + [tokenizer.bos_id]
