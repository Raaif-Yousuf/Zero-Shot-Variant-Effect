"""Pure-numpy math shared by the model scorers.

Kept free of torch and transformers so the log-likelihood arithmetic and the
Nucleotide Transformer tokenization logic can be unit-tested without a torch
install and without downloading any weights.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

ACGT = frozenset("ACGT")


def causal_loglik(log_probs: np.ndarray, token_ids: Sequence[int], start: int, end: int) -> float:
    """Sum of log p(token_ids[t] | token_ids[:t]) for t in [start, end).

    ``log_probs`` is the log-softmax of a causal LM's logits: ``log_probs[i]`` is
    the distribution used to predict ``token_ids[i + 1]`` from the prefix
    ``token_ids[: i + 1]`` (the usual next-token shift). ``start`` must be at
    least 1, since the first token in a sequence has no prefix to condition on.

    Restricting ``[start, end)`` to a suffix of the sequence gives the same
    value as summing the whole sequence and then subtracting the (identical)
    prefix contribution, as long as ``token_ids`` agrees with the full sequence
    on every position before ``start``.
    """
    if start < 1:
        raise ValueError(f"start must be >= 1 (no context for token 0), got {start}")
    if end > log_probs.shape[0] + 1:
        raise ValueError(f"end={end} exceeds available logits ({log_probs.shape[0]} rows)")
    total = 0.0
    for t in range(start, end):
        total += float(log_probs[t - 1, token_ids[t]])
    return total


def causal_site_llr(log_probs_row: np.ndarray, ref_id: int, alt_id: int) -> float:
    """log p(alt_id) - log p(ref_id) from one row of log-softmaxed logits."""
    return float(log_probs_row[alt_id] - log_probs_row[ref_id])


def masked_marginal_llr(log_probs_row: np.ndarray, ref_id: int, alt_id: int) -> float:
    """log p(alt_id) - log p(ref_id) at a masked position (masked-marginal score)."""
    return float(log_probs_row[alt_id] - log_probs_row[ref_id])


def nt_token_layout(seq: str, k: int = 6) -> list[tuple[int, int, bool]]:
    """Replicate the Nucleotide Transformer tokenizer's non-overlapping k-mer split.

    Walking from the start of ``seq``, each consecutive block of ``k`` bases
    becomes one token if it is exactly ``k`` bases long and every base is in
    A/C/G/T; otherwise (a short trailing block, or a block containing N or any
    other character) it is split into one single-character token per base.
    This matches the InstaDeepAI Nucleotide Transformer tokenizer, verified by
    decoding token ids for sequences with N's and non-multiple-of-6 lengths.

    Returns a list of ``(start, length, is_kmer)`` covering ``seq`` in order,
    excluding the leading ``<cls>`` token the tokenizer always prepends.
    """
    layout: list[tuple[int, int, bool]] = []
    n = len(seq)
    i = 0
    while i < n:
        block = seq[i : i + k]
        if len(block) == k and all(c in ACGT for c in block):
            layout.append((i, k, True))
            i += k
        else:
            for j in range(len(block)):
                layout.append((i + j, 1, False))
            i += len(block)
    return layout


def nt_locate_kmer(
    seq: str, center: int, k: int = 6, max_trim: int | None = None
) -> tuple[int, int, int, int]:
    """Find a clean k-mer token covering ``center``, trimming the start if needed.

    The Nucleotide Transformer tokenizer chunks from the start of the sequence,
    so whether ``center`` falls inside a clean k-mer token depends on the frame.
    This tries trimming 0, 1, 2, ... bases off the start of ``seq`` (in that
    order, so the smallest trim that works wins, which is deterministic) until
    ``center`` lands inside a full, N-free k-mer token.

    Returns ``(trim, token_index, chunk_start, offset)``:
      - ``trim``: bases removed from the start of ``seq``.
      - ``token_index``: index of the covering token in the tokenizer's output
        (accounting for the leading ``<cls>`` token).
      - ``chunk_start``: start of the k-mer within ``seq[trim:]``.
      - ``offset``: position of ``center`` within that k-mer (0 to k - 1).

    Raises ``ValueError`` if no trim in ``0..max_trim`` (default ``k - 1``)
    aligns the variant to a clean k-mer.
    """
    if max_trim is None:
        max_trim = k - 1
    for trim in range(0, max_trim + 1):
        if trim > center:
            break
        new_center = center - trim
        windowed = seq[trim:]
        layout = nt_token_layout(windowed, k)
        for token_index, (start, length, is_kmer) in enumerate(layout):
            if start <= new_center < start + length:
                if is_kmer:
                    return trim, token_index + 1, start, new_center - start
                break
    raise ValueError(f"cannot align position {center} in {seq!r} to a clean {k}-mer token")
