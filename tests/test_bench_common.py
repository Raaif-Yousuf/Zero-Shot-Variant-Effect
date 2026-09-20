"""Tests for the pure benchmark-orchestration logic in scripts/_bench_common.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from _bench_common import (  # noqa: E402
    damaging_score,
    filter_to_positions,
    join_scores_with_dataset,
    pick_best_model,
    run_id_for,
    scorer_label,
    select_sweep_positions,
)


def test_run_id_for_engine_scorer():
    entry = {"dataset": "brca1", "scorer": "hyenadna-small-32k", "mode": "full", "window": 1024}
    assert run_id_for(entry) == "brca1_hyenadna-small-32k_full_w1024"


def test_run_id_for_kmer_includes_order():
    entry = {"dataset": "brca1", "scorer": "kmer", "order": 6, "window": 1024}
    assert run_id_for(entry) == "brca1_kmer_o6_w1024"


def test_run_id_for_column_scorer_has_no_window_suffix_when_absent():
    entry = {"dataset": "brca1", "scorer": "cadd", "window": None}
    assert run_id_for(entry) == "brca1_cadd"


def test_run_id_for_sweep_subset_appends_suffix():
    entry = {
        "dataset": "brca1",
        "scorer": "hyenadna-small-32k",
        "mode": "site",
        "window": 4096,
        "subset": "sweep",
    }
    assert run_id_for(entry) == "brca1_hyenadna-small-32k_site_w4096_sweep"


def test_run_id_for_all_subset_has_no_suffix():
    entry = {"dataset": "clinvar", "scorer": "kmer", "order": 6, "window": 1024, "subset": "all"}
    assert run_id_for(entry) == "clinvar_kmer_o6_w1024"


def test_select_sweep_positions_returns_all_when_fewer_than_n():
    positions = [10, 20, 30]
    assert select_sweep_positions(positions, n=150, seed=0) == [10, 20, 30]


def test_select_sweep_positions_is_deterministic_and_sorted():
    positions = list(range(1000))
    first = select_sweep_positions(positions, n=50, seed=0)
    second = select_sweep_positions(positions, n=50, seed=0)
    assert first == second
    assert len(first) == 50
    assert first == sorted(first)
    assert len(set(first)) == 50


def test_select_sweep_positions_different_seed_differs():
    positions = list(range(1000))
    a = select_sweep_positions(positions, n=50, seed=0)
    b = select_sweep_positions(positions, n=50, seed=1)
    assert a != b


def test_select_sweep_positions_is_nested_across_n():
    positions = list(range(1000))
    smaller = select_sweep_positions(positions, n=30, seed=0)
    larger = select_sweep_positions(positions, n=60, seed=0)
    assert set(smaller).issubset(set(larger))
    assert len(smaller) == 30
    assert len(larger) == 60


def test_filter_to_positions():
    df = pd.DataFrame({"pos": [1, 2, 3, 4], "value": ["a", "b", "c", "d"]})
    filtered = filter_to_positions(df, [2, 4])
    assert list(filtered["pos"]) == [2, 4]
    assert list(filtered["value"]) == ["b", "d"]


def test_join_scores_with_dataset_left_join_keeps_missing_rows():
    dataset = pd.DataFrame(
        {
            "chrom": ["chr17", "chr17"],
            "pos": [100, 200],
            "ref": ["A", "C"],
            "alt": ["G", "T"],
            "label": [1.0, 0.0],
        }
    )
    scores = pd.DataFrame(
        {
            "chrom": ["chr17"],
            "pos": [100],
            "ref": ["A"],
            "alt": ["G"],
            "id": ["."],
            "llr": [-0.5],
            "llr_fwd": [-0.4],
            "llr_rev": [-0.6],
            "status": ["ok"],
        }
    )
    joined = join_scores_with_dataset(dataset, scores)
    assert len(joined) == 2
    assert joined.loc[joined["pos"] == 100, "llr"].iloc[0] == -0.5
    assert pd.isna(joined.loc[joined["pos"] == 200, "llr"].iloc[0])


def test_damaging_score_negates_llr_but_not_baseline_columns():
    llr = pd.Series([-1.0, 2.0])
    assert list(damaging_score("kmer", llr)) == [1.0, -2.0]
    assert list(damaging_score("hyenadna-small-32k", llr)) == [1.0, -2.0]
    assert list(damaging_score("phylop100way", llr)) == [-1.0, 2.0]
    assert list(damaging_score("cadd", llr)) == [-1.0, 2.0]


def test_scorer_label_variants():
    assert scorer_label({"scorer": "kmer", "order": 6}) == "kmer_o6"
    assert (
        scorer_label({"scorer": "hyenadna-small-32k", "mode": "full"}) == "hyenadna-small-32k_full"
    )
    assert scorer_label({"scorer": "nt-v2-50m"}) == "nt-v2-50m"
    assert scorer_label({"scorer": "phylop100way"}) == "phylop100way"


def test_pick_best_model_uses_largest_absolute_value():
    metrics = pd.DataFrame(
        {
            "scorer": ["kmer_o6", "hyenadna-small-32k_full", "nt-v2-50m"],
            "spearman_rho": [-0.6, 0.2, float("nan")],
        }
    )
    assert (
        pick_best_model(metrics, ["kmer_o6", "hyenadna-small-32k_full", "nt-v2-50m"]) == "kmer_o6"
    )


def test_pick_best_model_ignores_nan_candidates():
    metrics = pd.DataFrame({"scorer": ["a", "b"], "spearman_rho": [float("nan"), 0.1]})
    assert pick_best_model(metrics, ["a", "b"]) == "b"


def test_pick_best_model_raises_when_no_candidate_has_a_value():
    metrics = pd.DataFrame({"scorer": ["a"], "spearman_rho": [float("nan")]})
    with pytest.raises(ValueError):
        pick_best_model(metrics, ["a"])
