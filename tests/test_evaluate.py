"""Metric sanity tests for evaluate.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zeroshot_vep.evaluate import (
    evaluate_by_stratum,
    evaluate_scorers,
    write_metrics_markdown,
    write_metrics_tsv,
)


def test_perfect_separation_gives_auroc_and_auprc_one():
    df = pd.DataFrame({"label": [0, 0, 0, 1, 1, 1], "score": [0.1, 0.2, 0.3, 0.8, 0.9, 1.0]})
    result = evaluate_scorers(df, ["score"], n_boot=50, seed=0)
    row = result.iloc[0]
    assert row["auroc"] == 1.0
    assert row["auprc"] == 1.0


def test_reversed_scores_give_auroc_zero():
    df = pd.DataFrame({"label": [0, 0, 0, 1, 1, 1], "score": [1.0, 0.9, 0.8, 0.3, 0.2, 0.1]})
    result = evaluate_scorers(df, ["score"], n_boot=50, seed=0)
    assert result.iloc[0]["auroc"] == 0.0


def test_spearman_perfect_monotonic_relationship():
    df = pd.DataFrame(
        {
            "label": [0, 1, 0, 1, 0, 1],
            "score": [1, 2, 3, 4, 5, 6],
            "func_score": [10, 20, 30, 40, 50, 60],
        }
    )
    result = evaluate_scorers(df, ["score"], continuous_col="func_score", n_boot=10, seed=0)
    assert result.iloc[0]["spearman_rho"] == pytest.approx(1.0)
    assert result.iloc[0]["n_spearman"] == 6


def test_bootstrap_ci_is_deterministic_given_seed():
    rng = np.random.default_rng(1)
    n = 200
    label = (rng.random(n) < 0.3).astype(float)
    score = label * 2 + rng.normal(size=n)
    df = pd.DataFrame({"label": label, "score": score})

    r1 = evaluate_scorers(df, ["score"], n_boot=200, seed=42)
    r2 = evaluate_scorers(df, ["score"], n_boot=200, seed=42)

    assert r1.iloc[0]["auroc_lo"] == r2.iloc[0]["auroc_lo"]
    assert r1.iloc[0]["auroc_hi"] == r2.iloc[0]["auroc_hi"]
    assert r1.iloc[0]["auprc_lo"] == r2.iloc[0]["auprc_lo"]


def test_nan_scores_excluded_and_coverage_reported():
    df = pd.DataFrame({"label": [0, 1, 0, 1], "score": [0.1, np.nan, 0.3, 0.9]})
    result = evaluate_scorers(df, ["score"], n_boot=10, seed=0)
    row = result.iloc[0]
    assert row["n"] == 4
    assert row["n_scored"] == 3
    assert row["coverage"] == pytest.approx(0.75)


def test_stratum_below_min_per_class_is_skipped():
    n_a = 12
    label_a = [0] * n_a + [1] * n_a
    score_a = list(np.linspace(0, 1, n_a)) + list(np.linspace(0.5, 1.5, n_a))
    label_b = [0, 0, 0, 1, 1, 1]
    score_b = [0.1, 0.2, 0.3, 0.8, 0.9, 1.0]
    df = pd.DataFrame(
        {
            "label": label_a + label_b,
            "score": score_a + score_b,
            "consequence": ["a"] * (2 * n_a) + ["b"] * len(label_b),
        }
    )
    result = evaluate_by_stratum(df, ["score"], "consequence", n_boot=10, seed=0)
    assert set(result["stratum"]) == {"a"}


def test_write_metrics_tsv_and_markdown(tmp_path):
    df = pd.DataFrame({"label": [0, 1, 0, 1], "score": [0.1, 0.9, 0.2, 0.8]})
    result = evaluate_scorers(df, ["score"], n_boot=10, seed=0)
    tsv_path = tmp_path / "metrics.tsv"
    md_path = tmp_path / "metrics.md"

    write_metrics_tsv(result, tsv_path)
    write_metrics_markdown(result, md_path)

    assert tsv_path.exists() and tsv_path.stat().st_size > 0
    assert md_path.exists() and md_path.stat().st_size > 0
