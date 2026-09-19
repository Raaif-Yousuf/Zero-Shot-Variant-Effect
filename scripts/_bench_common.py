"""Pure logic shared between run_benchmark.py and summarize_results.py.

Kept free of torch, the engine and any I/O so it can be unit-tested with tiny
in-memory fixtures.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

#: plan.json scorer names that read an existing dataset column instead of
#: going through the scoring engine.
COLUMN_SCORERS = {
    "phylop100way": "phylop100way",
    "phylop_mammalian": "phylop_mammalian",
    "cadd": "cadd",
}

#: which score columns are already "higher = more damaging" (no sign flip),
#: as opposed to a log-likelihood-ratio where the flip is -llr.
NATURAL_DIRECTION_SCORERS = set(COLUMN_SCORERS)

VARIANT_KEY = ["chrom", "pos", "ref", "alt"]


def run_id_for(entry: dict) -> str:
    """Deterministic, filesystem-safe run id from one plan.json entry.

    Only fields that actually distinguish two runs are included, so e.g. two
    kmer entries at different orders get different ids, but the (irrelevant
    to non-HyenaDNA scorers) ``mode`` field never appears for kmer.
    """
    parts = [entry["dataset"], entry["scorer"]]
    if entry.get("mode"):
        parts.append(entry["mode"])
    if entry.get("order") is not None:
        parts.append(f"o{entry['order']}")
    if entry.get("window") is not None:
        parts.append(f"w{entry['window']}")
    if entry.get("subset", "all") not in ("all", None):
        parts.append(entry["subset"])
    return "_".join(str(p) for p in parts)


def select_sweep_positions(positions: Iterable[int], n: int, seed: int) -> list[int]:
    """Seeded selection of up to ``n`` unique positions, returned sorted ascending.

    If there are ``n`` or fewer unique positions, all of them are returned
    (nothing to sample). Deterministic for a given ``seed``.
    """
    unique = np.sort(np.unique(np.asarray(list(positions))))
    if len(unique) <= n:
        return [int(x) for x in unique]
    rng = np.random.default_rng(seed)
    chosen = rng.choice(unique, size=n, replace=False)
    return sorted(int(x) for x in chosen)


def filter_to_positions(
    df: pd.DataFrame, positions: Iterable[int], pos_col: str = "pos"
) -> pd.DataFrame:
    """Rows of ``df`` whose ``pos_col`` is in ``positions``, order preserved."""
    wanted = set(int(p) for p in positions)
    return df[df[pos_col].isin(wanted)].reset_index(drop=True)


def join_scores_with_dataset(
    dataset: pd.DataFrame, scores: pd.DataFrame, key_cols: list[str] = VARIANT_KEY
) -> pd.DataFrame:
    """Left-join a run's scores onto the processed dataset by variant key.

    A left join keeps every dataset row even if a variant is missing from the
    scores (e.g. it hit ``ref_mismatch`` and was excluded upstream), so
    downstream coverage accounting in evaluate.py sees the true denominator.
    """
    score_cols = [c for c in scores.columns if c not in key_cols and c != "id"]
    return dataset.merge(scores[[*key_cols, *score_cols]], on=key_cols, how="left")


def scorer_label(entry: dict) -> str:
    """Human-readable scorer name for metrics tables: no dataset, window or subset."""
    parts = [entry["scorer"]]
    if entry.get("mode"):
        parts.append(entry["mode"])
    if entry.get("order") is not None:
        parts.append(f"o{entry['order']}")
    return "_".join(str(p) for p in parts)


def pick_best_model(metrics: pd.DataFrame, candidates: list[str], by: str = "spearman_rho") -> str:
    """Pick the candidate scorer with the largest ``|by|`` value in an overall metrics table."""
    subset = metrics[metrics["scorer"].isin(candidates) & metrics[by].notna()]
    if subset.empty:
        raise ValueError(f"no candidate among {candidates} has a non-null {by!r}")
    idx = subset[by].abs().idxmax()
    return str(subset.loc[idx, "scorer"])


def damaging_score(scorer: str, llr: pd.Series) -> pd.Series:
    """Convert a raw score column to the "higher = more damaging" convention.

    Baseline columns (phyloP, CADD) are already in that direction; scorer
    outputs (kmer, HyenaDNA, Nucleotide Transformer) are natural-log
    likelihood-ratios where negative means damaging, so they are negated.
    """
    if scorer in NATURAL_DIRECTION_SCORERS:
        return llr
    return -llr
