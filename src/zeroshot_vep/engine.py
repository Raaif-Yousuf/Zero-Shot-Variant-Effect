"""Scoring engine: extracts reference windows around variants and calls scorers."""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from collections.abc import Sequence

import pandas as pd

from zeroshot_vep.cache import ScoreCache
from zeroshot_vep.reference import ReferenceGenome
from zeroshot_vep.scorers.base import Scorer
from zeroshot_vep.variant import Variant, reverse_complement

_OUTPUT_COLUMNS = ["chrom", "pos", "ref", "alt", "id", "llr", "llr_fwd", "llr_rev", "status"]


def _score_group(
    scorer: Scorer,
    seq: str,
    center: int,
    alts: list[str],
    strand: str,
    window: int,
    cache: ScoreCache | None,
) -> list[float]:
    """Score all alts at one position/strand, going through the cache if given."""
    if cache is None:
        return list(scorer.score_window(seq, center, alts))
    scorer_key = scorer.cache_key()
    cached = cache.get_many(scorer_key, window, strand, seq, center, alts)
    missing = [a for a in alts if a not in cached]
    if missing:
        fresh = scorer.score_window(seq, center, missing)
        new_values = dict(zip(missing, fresh, strict=True))
        cache.put_many(scorer_key, window, strand, seq, center, new_values)
        cached.update(new_values)
    return [cached[a] for a in alts]


def score_variants(
    variants: Sequence[Variant],
    reference: ReferenceGenome,
    scorer: Scorer,
    window: int,
    strands: str = "both",
    cache: ScoreCache | None = None,
    progress: bool = False,
) -> pd.DataFrame:
    """Score each variant's reference window and return a results table.

    Variants sharing a position are grouped so the scorer is called once per
    position per strand, with all alts at that position batched together.
    Rows whose ``ref`` does not match the reference base get status
    ``"ref_mismatch"`` and are not scored. For ``strands="both"`` the reverse
    complement window is also scored and ``llr`` is the mean of the two
    strand scores; for ``"forward"`` ``llr`` equals ``llr_fwd``.

    Output columns: chrom, pos, ref, alt, id, llr, llr_fwd, llr_rev, status.
    """
    if strands not in ("both", "forward"):
        raise ValueError(f"strands must be 'both' or 'forward', got {strands!r}")
    if window < 1:
        raise ValueError(f"window must be positive, got {window}")
    if scorer.max_window is not None and window > scorer.max_window:
        raise ValueError(f"window {window} exceeds {scorer.name}.max_window ({scorer.max_window})")

    variants = list(variants)
    center = window // 2

    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for i, v in enumerate(variants):
        groups[(v.chrom, v.pos)].append(i)

    results: list[dict[str, object]] = [{} for _ in variants]
    total = len(groups)
    start_time = time.monotonic()

    for done, ((chrom, pos), indices) in enumerate(groups.items(), start=1):
        start0 = (pos - 1) - center
        end0 = start0 + window
        seq = reference.fetch(chrom, start0, end0)
        actual_ref = seq[center]

        valid_idx: list[int] = []
        valid_alts: list[str] = []
        for i in indices:
            v = variants[i]
            if v.ref != actual_ref:
                results[i] = {
                    "chrom": v.chrom,
                    "pos": v.pos,
                    "ref": v.ref,
                    "alt": v.alt,
                    "id": v.id,
                    "llr": float("nan"),
                    "llr_fwd": float("nan"),
                    "llr_rev": float("nan"),
                    "status": "ref_mismatch",
                }
            else:
                valid_idx.append(i)
                valid_alts.append(v.alt)

        if valid_alts:
            llr_fwd = _score_group(scorer, seq, center, valid_alts, "fwd", window, cache)
            if strands == "both":
                rev_seq = reverse_complement(seq)
                rev_center = window - 1 - center
                rev_alts = [reverse_complement(a) for a in valid_alts]
                llr_rev = _score_group(scorer, rev_seq, rev_center, rev_alts, "rev", window, cache)
            else:
                llr_rev = [float("nan")] * len(valid_alts)

            for j, i in enumerate(valid_idx):
                v = variants[i]
                fwd = llr_fwd[j]
                if strands == "both":
                    rev = llr_rev[j]
                    llr = (fwd + rev) / 2.0
                else:
                    rev = float("nan")
                    llr = fwd
                results[i] = {
                    "chrom": v.chrom,
                    "pos": v.pos,
                    "ref": v.ref,
                    "alt": v.alt,
                    "id": v.id,
                    "llr": llr,
                    "llr_fwd": fwd,
                    "llr_rev": rev,
                    "status": "ok",
                }

        if progress and (done % 200 == 0 or done == total):
            elapsed = time.monotonic() - start_time
            rate = done / elapsed if elapsed > 0 else 0.0
            print(f"[zsvep] scored {done}/{total} positions ({rate:.1f} pos/s)", file=sys.stderr)

    return pd.DataFrame(results, columns=_OUTPUT_COLUMNS)
