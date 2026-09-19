"""Benchmark metrics for zero-shot variant effect scores.

Every score column is assumed to follow the convention **higher = more
damaging**: callers pass ``-llr`` when a scorer's raw output is a
log-likelihood-ratio where negative means damaging (see
``zeroshot_vep.scorers.base.Scorer``). Under that convention, the positive
class (``label == 1``, e.g. loss-of-function or pathogenic) is expected to
score higher than the negative class.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

DEFAULT_N_BOOT = 1000
DEFAULT_SEED = 0
MIN_STRATUM_PER_CLASS = 10

_METRIC_COLUMNS = [
    "stratum",
    "scorer",
    "n",
    "n_scored",
    "coverage",
    "n_pos",
    "n_neg",
    "prevalence",
    "auroc",
    "auroc_lo",
    "auroc_hi",
    "auprc",
    "auprc_lo",
    "auprc_hi",
    "spearman_rho",
    "spearman_p",
    "n_spearman",
]


def _bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    """Stratified-by-label bootstrap CI: resample positives and negatives separately."""
    rng = np.random.default_rng(seed)
    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return float("nan"), float("nan")

    values = np.empty(n_boot)
    for i in range(n_boot):
        boot_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        boot_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([boot_pos, boot_neg])
        values[i] = metric_fn(y_true[idx], y_score[idx])
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def _evaluate_one(
    df: pd.DataFrame,
    score_col: str,
    label_col: str,
    continuous_col: str | None,
    n_boot: int,
    seed: int,
    stratum: str,
) -> dict[str, object]:
    n_total = len(df)
    score_notna = df[score_col].notna()
    coverage = float(score_notna.mean()) if n_total else float("nan")

    labelled = df[score_notna & df[label_col].notna()]
    y = labelled[label_col].to_numpy(dtype=float)
    s = labelled[score_col].to_numpy(dtype=float)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())

    row: dict[str, object] = {
        "stratum": stratum,
        "scorer": score_col,
        "n": n_total,
        "n_scored": len(labelled),
        "coverage": coverage,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "prevalence": n_pos / (n_pos + n_neg) if (n_pos + n_neg) else float("nan"),
    }

    if n_pos > 0 and n_neg > 0:
        row["auroc"] = float(roc_auc_score(y, s))
        row["auroc_lo"], row["auroc_hi"] = _bootstrap_ci(y, s, roc_auc_score, n_boot, seed)
        row["auprc"] = float(average_precision_score(y, s))
        row["auprc_lo"], row["auprc_hi"] = _bootstrap_ci(
            y, s, average_precision_score, n_boot, seed
        )
    else:
        row["auroc"] = row["auroc_lo"] = row["auroc_hi"] = float("nan")
        row["auprc"] = row["auprc_lo"] = row["auprc_hi"] = float("nan")

    if continuous_col is not None:
        cont_valid = df[score_notna & df[continuous_col].notna()]
        if len(cont_valid) >= 3:
            rho, p = spearmanr(
                cont_valid[score_col].to_numpy(dtype=float),
                cont_valid[continuous_col].to_numpy(dtype=float),
            )
            row["spearman_rho"] = float(rho)
            row["spearman_p"] = float(p)
        else:
            row["spearman_rho"] = float("nan")
            row["spearman_p"] = float("nan")
        row["n_spearman"] = len(cont_valid)
    else:
        row["spearman_rho"] = float("nan")
        row["spearman_p"] = float("nan")
        row["n_spearman"] = 0

    return row


def evaluate_scorers(
    df: pd.DataFrame,
    score_cols: Sequence[str],
    *,
    label_col: str = "label",
    continuous_col: str | None = None,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """One overall metrics row per score column.

    NaN scores are excluded per column and reported via ``coverage``; NaN
    labels are excluded from AUROC/AUPRC/prevalence but do not affect Spearman.
    """
    rows = [
        _evaluate_one(df, col, label_col, continuous_col, n_boot, seed, stratum="overall")
        for col in score_cols
    ]
    return pd.DataFrame(rows, columns=_METRIC_COLUMNS)


def evaluate_by_stratum(
    df: pd.DataFrame,
    score_cols: Sequence[str],
    stratum_col: str,
    *,
    label_col: str = "label",
    continuous_col: str | None = None,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    min_per_class: int = MIN_STRATUM_PER_CLASS,
) -> pd.DataFrame:
    """Metrics per (score column, stratum), skipping strata short on either class.

    A stratum needs at least ``min_per_class`` labelled rows of *both* classes
    (checked once, over rows with a non-null label, independent of score
    column) or it is left out entirely.
    """
    rows = []
    for stratum_value, group in df.groupby(stratum_col, dropna=True):
        labelled = group[group[label_col].notna()]
        n_pos = int((labelled[label_col] == 1).sum())
        n_neg = int((labelled[label_col] == 0).sum())
        if n_pos < min_per_class or n_neg < min_per_class:
            continue
        for col in score_cols:
            rows.append(
                _evaluate_one(
                    group, col, label_col, continuous_col, n_boot, seed, stratum=str(stratum_value)
                )
            )
    return pd.DataFrame(rows, columns=_METRIC_COLUMNS)


def write_metrics_tsv(metrics: pd.DataFrame, path: str) -> None:
    metrics.to_csv(path, sep="\t", index=False)


def write_metrics_markdown(metrics: pd.DataFrame, path: str) -> None:
    """Write a compact Markdown table (rounded floats) for easy pasting into reports."""
    display = metrics.copy()
    for col in display.select_dtypes(include="number").columns:
        display[col] = display[col].round(4)

    columns = list(display.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.tolist()) + " |")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
