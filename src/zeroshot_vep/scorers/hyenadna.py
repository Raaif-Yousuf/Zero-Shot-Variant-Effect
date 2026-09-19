"""Causal, character-level scorer for the HyenaDNA model family.

HyenaDNA's tokenizer maps each base to its own token (one id per character) and
appends a single trailing ``[SEP]`` token; it does not prepend a beginning-of
-sequence token. Verified for ``hyenadna-tiny-1k-seqlen-hf``: tokenizing 1000
bases yields 1001 ids, the last of which is ``[SEP]`` (id 1) and the first
1000 are the per-base ids.

torch and transformers are imported lazily (inside methods) so importing this
module never requires torch to be installed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from zeroshot_vep.scorers._llr import causal_loglik, causal_site_llr
from zeroshot_vep.scorers.base import Scorer

_VALID_MODES = ("full", "site")


class HyenaDNAScorer(Scorer):
    """Log-likelihood-ratio scorer for HyenaDNA causal language models.

    ``mode="full"`` (default) sums log p(base | prefix) over the whole
    sequence for the ref and alt windows and takes the difference. Bases
    before the variant are identical between ref and alt, so their
    contributions cancel; the sum is restricted to positions ``>= center``,
    which is mathematically equal to summing the full sequence (see
    ``tests/test_hyenadna.py`` for a brute-force check against the
    unrestricted sum). Ref and every alt are run as a single batch (same
    length, no padding needed).

    ``mode="site"`` runs only the reference window through the model once and
    compares log p(alt) vs log p(ref) at the variant position, using the
    shared prefix. This is cheaper but ignores how the variant changes
    predictions of bases downstream of it.
    """

    def __init__(
        self,
        name: str,
        model_id: str,
        revision: str,
        max_window: int,
        mode: str = "full",
        num_threads: int | None = None,
        batch_size: int | None = None,
        model: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        if mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")
        self.name = name
        self.model_id = model_id
        self.revision = revision
        self.max_window = max_window
        self.mode = mode
        self.num_threads = num_threads
        self.batch_size = batch_size  # not used for scoring; kept for CLI compatibility
        self._model = model
        self._tokenizer = tokenizer
        self._loaded = model is not None and tokenizer is not None

    def cache_key(self) -> str:
        return f"{self.model_id}@{self.revision}:mode={self.mode}"

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, revision=self.revision, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
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
        n_bases = len(ref_seq)
        ref_ids = tok(ref_seq)["input_ids"]
        if len(ref_ids) != n_bases + 1:
            raise ValueError(
                f"expected {n_bases + 1} tokens (one per base plus a trailing special "
                f"token) but tokenizer returned {len(ref_ids)}"
            )

        if self.mode == "site":
            if center == 0:
                raise ValueError("site mode needs at least one base of left context")
            batch = torch.tensor([ref_ids], dtype=torch.long)
            with torch.inference_mode():
                logits = self._model(input_ids=batch).logits
            log_probs = torch.log_softmax(logits[0, center - 1].float(), dim=-1).numpy()
            ref_base_id = ref_ids[center]
            return [
                causal_site_llr(log_probs, ref_base_id, tok.convert_tokens_to_ids(alt))
                for alt in alts
            ]

        # full mode: batch ref + every alt sequence (equal length, no padding).
        start = max(center, 1)
        sequences = [ref_ids]
        for alt in alts:
            alt_seq = ref_seq[:center] + alt + ref_seq[center + 1 :]
            sequences.append(tok(alt_seq)["input_ids"])

        batch = torch.tensor(sequences, dtype=torch.long)
        with torch.inference_mode():
            logits = self._model(input_ids=batch).logits
        log_probs = torch.log_softmax(logits.float(), dim=-1).numpy()

        ref_ll = causal_loglik(log_probs[0], sequences[0], start=start, end=n_bases)
        scores = []
        for row, ids in zip(log_probs[1:], sequences[1:], strict=True):
            alt_ll = causal_loglik(row, ids, start=start, end=n_bases)
            scores.append(alt_ll - ref_ll)
        return scores
