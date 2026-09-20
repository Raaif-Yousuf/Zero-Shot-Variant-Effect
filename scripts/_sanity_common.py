"""Pure logic shared by sanity_checks.py.

Kept free of torch, transformers and any file I/O so it can be unit-tested
with tiny in-memory fixtures and no model downloads.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

PURINES = frozenset("AG")
PYRIMIDINES = frozenset("CT")


def all_kmers(k: int = 6) -> list[str]:
    """Every A/C/G/T k-mer, in a fixed deterministic order (itertools.product)."""
    return ["".join(c) for c in itertools.product("ACGT", repeat=k)]


def is_transition(ref: str, alt: str) -> bool:
    """True if ref->alt is a transition (purine<->purine or pyrimidine<->pyrimidine)."""
    ref, alt = ref.upper(), alt.upper()
    return (ref in PURINES and alt in PURINES) or (ref in PYRIMIDINES and alt in PYRIMIDINES)


def classify_cpg(left: str, ref_base: str, right: str) -> bool:
    """True if the variant sits in a CpG dinucleotide context.

    A variant is "in a CpG context" if it substitutes the C of a CG
    dinucleotide (``ref_base == "C"`` and the next base is ``"G"``) or the G
    of one (``ref_base == "G"`` and the previous base is ``"C"``).
    """
    left, ref_base, right = left.upper(), ref_base.upper(), right.upper()
    return (ref_base == "C" and right == "G") or (ref_base == "G" and left == "C")


def dinucleotide_shuffle(seq: str, rng: np.random.Generator, max_attempts: int = 200) -> str:
    """Randomly reorder ``seq`` while preserving its exact dinucleotide composition.

    Implements the Altschul-Erikson (1985) algorithm: build a directed
    multigraph whose edges are the dinucleotides of ``seq`` in order, reserve
    one "last-departing" edge per base (so an Eulerian path from ``seq[0]``
    to ``seq[-1]`` is guaranteed to exist once the reserved edges are used
    last), randomly shuffle the remaining edges leaving each base, check that
    the reserved edges alone can still walk from every base to ``seq[-1]``,
    and reconstruct a new sequence by walking the shuffled graph starting at
    ``seq[0]``.

    The result has the same length, the same multiset of dinucleotides, the
    same base composition and the same first/last character as ``seq``.
    Retries with a fresh shuffle (up to ``max_attempts`` times) if the
    reserved edges do not reach ``seq[-1]``; this is rare and only possible
    to fail to satisfy for very short or degenerate sequences.
    """
    seq = seq.upper()
    if len(seq) < 2:
        raise ValueError("dinucleotide_shuffle needs at least 2 bases")
    if any(c not in "ACGT" for c in seq):
        raise ValueError("dinucleotide_shuffle requires an ACGT-only sequence")

    first, last = seq[0], seq[-1]
    edges: dict[str, list[str]] = {b: [] for b in "ACGT"}
    for a, b in zip(seq, seq[1:], strict=False):
        edges[a].append(b)

    for _ in range(max_attempts):
        remaining = {b: v[:] for b, v in edges.items()}
        fixed_last = {b: remaining[b].pop() for b in "ACGT" if remaining[b]}
        for v in remaining.values():
            rng.shuffle(v)
        if _reaches_target(fixed_last, last):
            avail = {b: v[:] for b, v in remaining.items()}
            for b, nxt in fixed_last.items():
                avail[b].append(nxt)
            out = [first]
            cur = first
            for _ in range(len(seq) - 1):
                cur = avail[cur].pop(0)
                out.append(cur)
            return "".join(out)
    raise RuntimeError(f"dinucleotide_shuffle: no valid rearrangement in {max_attempts} attempts")


def _reaches_target(fixed_last: Mapping[str, str], target: str) -> bool:
    """True if, from every base with a reserved edge, following those edges reaches ``target``."""
    for start in fixed_last:
        node = start
        seen: set[str] = set()
        while node != target:
            if node in seen or node not in fixed_last:
                return False
            seen.add(node)
            node = fixed_last[node]
    return True


def bits_per_base(mean_log_likelihood_nats: float) -> float:
    """Convert a mean natural-log likelihood per base into bits per base (-log2 P)."""
    return -mean_log_likelihood_nats / math.log(2)


def marginal_base_log_probs(
    kmer_log_probs: Sequence[float], kmers: Sequence[str], offset: int
) -> dict[str, float]:
    """Marginalize a distribution over k-mers down to a distribution over one base.

    ``kmer_log_probs[i]`` is the log-probability of ``kmers[i]``; returns, for
    each base seen at position ``offset`` across ``kmers``, the log of the
    summed probability mass of every k-mer with that base at ``offset``
    (log-sum-exp, done per base).
    """
    buckets: dict[str, list[float]] = {}
    for lp, kmer in zip(kmer_log_probs, kmers, strict=True):
        buckets.setdefault(kmer[offset], []).append(lp)
    out: dict[str, float] = {}
    for base, lps in buckets.items():
        arr = np.asarray(lps, dtype=float)
        m = arr.max()
        out[base] = float(m + np.log(np.sum(np.exp(arr - m))))
    return out


def sample_valid_positions(
    rng: np.random.Generator,
    low: int,
    high: int,
    n: int,
    is_valid: Callable[[int], bool],
    max_attempts: int | None = None,
) -> list[int]:
    """Seeded sample of ``n`` unique positions in ``[low, high)`` satisfying ``is_valid``.

    Draws candidates one at a time from ``rng`` and keeps the ones
    ``is_valid`` accepts, until ``n`` are found or ``max_attempts`` candidate
    draws are exhausted. Returned sorted ascending. Deterministic for a given
    ``rng`` state and callable.
    """
    if max_attempts is None:
        max_attempts = max(n * 200, 1000)
    chosen: list[int] = []
    seen: set[int] = set()
    attempts = 0
    while len(chosen) < n and attempts < max_attempts:
        attempts += 1
        pos = int(rng.integers(low, high))
        if pos in seen:
            continue
        seen.add(pos)
        if is_valid(pos):
            chosen.append(pos)
    if len(chosen) < n:
        raise RuntimeError(
            f"only found {len(chosen)}/{n} valid positions after {attempts} attempts"
        )
    return sorted(chosen)


def mean_llr_by_group(df: pd.DataFrame, group_col: str, value_col: str = "llr") -> pd.DataFrame:
    """Group-wise mean/median/std/n of ``value_col``, NaNs dropped, sorted by group."""
    sub = df[[group_col, value_col]].dropna()
    grouped = sub.groupby(group_col)[value_col].agg(["mean", "median", "std", "count"])
    grouped = grouped.rename(columns={"count": "n"}).reset_index()
    grouped["n"] = grouped["n"].astype(int)
    return grouped.sort_values(group_col).reset_index(drop=True)


def format_markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """A compact GitHub-flavored Markdown table from a header row and data rows."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines) + "\n"
