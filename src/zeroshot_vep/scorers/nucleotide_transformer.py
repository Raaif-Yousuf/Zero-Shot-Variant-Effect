"""Masked-marginal scorer for the Nucleotide Transformer v2 (6-mer tokens).

The tokenizer prepends a ``<cls>`` token, then walks the sequence from its
start in non-overlapping 6-base blocks. A block becomes a single 6-mer token
only if it is exactly 6 bases of A/C/G/T; a short trailing block, or a block
containing N or any other character, is instead split into one single-base
token per character. This was verified by decoding token ids for sequences
with N's and lengths that are not multiples of 6 (see
``zeroshot_vep.scorers._llr.nt_token_layout``).

Because chunking starts from position 0, whether the variant lands inside a
clean 6-mer token depends on the window's frame. The engine hands us a window
centered on the variant, so when the covering block is not a clean 6-mer we
deterministically trim 0 to 5 bases off the start of the window (smallest trim
first) until the variant falls inside a full, N-free 6-mer token, then mask
that one token and read off the ref/alt log-probabilities from one forward
pass. This never changes which bases are ref or alt, only which of the six
possible reading frames is used to tokenize the window.

Model weights (InstaDeepAI/nucleotide-transformer-v2-50m-multi-species) are
downloaded at run time from the Hugging Face Hub under the CC BY-NC-SA 4.0
license; this project never redistributes them.

torch and transformers are imported lazily (inside methods) so importing this
module never requires torch to be installed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from zeroshot_vep.scorers._llr import masked_marginal_llr, nt_locate_kmer
from zeroshot_vep.scorers.base import Scorer

_K = 6


class NucleotideTransformerScorer(Scorer):
    """Masked-marginal log-likelihood-ratio scorer for Nucleotide Transformer v2.

    For each position, the 6-mer token covering the variant is masked and a
    single forward pass gives log p(6-mer) for every candidate 6-mer at that
    position; the score for each alt is log p(alt 6-mer) - log p(ref 6-mer).
    All alts at a position share this one forward pass.
    """

    def __init__(
        self,
        name: str,
        model_id: str,
        revision: str,
        max_window: int,
        num_threads: int | None = None,
        batch_size: int | None = None,
        model: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        self.name = name
        self.model_id = model_id
        self.revision = revision
        self.max_window = max_window
        self.num_threads = num_threads
        self.batch_size = batch_size  # not used for scoring; kept for CLI compatibility
        self._model = model
        self._tokenizer = tokenizer
        self._loaded = model is not None and tokenizer is not None

    def cache_key(self) -> str:
        return f"{self.model_id}@{self.revision}"

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, revision=self.revision, trust_remote_code=True
        )
        self._model = AutoModelForMaskedLM.from_pretrained(
            self.model_id, revision=self.revision, trust_remote_code=True
        )
        self._model.eval()
        self._loaded = True

    def _set_threads(self) -> None:
        if self.num_threads:
            import torch

            torch.set_num_threads(self.num_threads)

    def score_window(self, ref_seq: str, center: int, alts: Sequence[str]) -> list[float]:
        if len(ref_seq) > self.max_window:
            raise ValueError(
                f"window length {len(ref_seq)} exceeds max_window {self.max_window} "
                f"for scorer {self.name!r}"
            )
        if not 0 <= center < len(ref_seq):
            raise ValueError(f"center {center} out of range for window of length {len(ref_seq)}")

        import torch

        self._ensure_loaded()
        self._set_threads()

        tok = self._tokenizer
        trim, token_index, chunk_start, offset = nt_locate_kmer(ref_seq, center, k=_K)
        windowed = ref_seq[trim:]
        ids = tok(windowed)["input_ids"]

        ref_kmer = windowed[chunk_start : chunk_start + _K]
        if ref_kmer[offset] != ref_seq[center]:
            raise ValueError("internal alignment error: ref base mismatch after trimming")
        ref_id = tok.convert_tokens_to_ids(ref_kmer)
        if ids[token_index] != ref_id:
            raise ValueError(
                "internal alignment error: computed token index does not match the "
                "tokenizer's own output for the reference k-mer"
            )

        masked_ids = list(ids)
        masked_ids[token_index] = tok.mask_token_id
        batch = torch.tensor([masked_ids], dtype=torch.long)
        with torch.inference_mode():
            logits = self._model(input_ids=batch).logits
        log_probs = torch.log_softmax(logits[0, token_index].float(), dim=-1).numpy()

        scores = []
        for alt in alts:
            alt_kmer = ref_kmer[:offset] + alt + ref_kmer[offset + 1 :]
            alt_id = tok.convert_tokens_to_ids(alt_kmer)
            scores.append(masked_marginal_llr(log_probs, ref_id, alt_id))
        return scores
