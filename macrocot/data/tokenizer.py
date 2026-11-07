"""Tokenizer for word-then-trace sequences.

A training example is laid out as

    <word symbols>  <BOS>  <trace tokens>  <EOS>

The model conditions on the word and the BOS marker and predicts the trace and
the terminating EOS. The vocabulary is built per configuration from the alphabet
symbols and the transition labels of the (possibly compressed) machine, since
macro and base-b compression change the trace token set.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

PAD, BOS, EOS = "<PAD>", "<BOS>", "<EOS>"
SPECIALS = [PAD, BOS, EOS]


class Tokenizer:
    def __init__(self, alphabet: Sequence[str], trace_labels: Sequence[str]):
        self.alphabet = list(alphabet)
        self.trace_labels = sorted(set(trace_labels))
        self.itos: List[str] = list(SPECIALS) + list(self.alphabet) + self.trace_labels
        self.stoi: Dict[str, int] = {s: i for i, s in enumerate(self.itos)}

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    @property
    def bos_id(self) -> int:
        return self.stoi[BOS]

    @property
    def eos_id(self) -> int:
        return self.stoi[EOS]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode_word(self, word: Sequence[int]) -> List[int]:
        # word entries are symbol indices into alphabet
        return [self.stoi[self.alphabet[s]] for s in word]

    def encode_trace(self, trace: Sequence[str]) -> List[int]:
        return [self.stoi[t] for t in trace]

    def build_sequence(self, word: Sequence[int], trace: Sequence[str]) -> Tuple[List[int], int]:
        """Return (token_ids, gen_start).

        gen_start is the index of the first predicted token (the first trace
        token). Loss is computed on positions gen_start-1 .. end-1 predicting
        tokens gen_start .. end.
        """
        ids = self.encode_word(word) + [self.bos_id] + self.encode_trace(trace) + [self.eos_id]
        gen_start = len(self.encode_word(word)) + 1  # first token after BOS
        return ids, gen_start
