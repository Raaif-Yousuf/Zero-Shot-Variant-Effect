"""Unit tests for scripts/_sanity_common.py -- pure logic, tiny fixtures, no models."""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from _sanity_common import (  # noqa: E402
    all_kmers,
    bits_per_base,
    classify_cpg,
    dinucleotide_shuffle,
    format_markdown_table,
    is_transition,
    marginal_base_log_probs,
    mean_llr_by_group,
    sample_valid_positions,
)


def test_all_kmers_count_and_uniqueness():
    kmers = all_kmers(6)
    assert len(kmers) == 4096
    assert len(set(kmers)) == 4096
    assert all(len(k) == 6 and set(k) <= set("ACGT") for k in kmers)


def test_all_kmers_small_k():
    assert set(all_kmers(2)) == {
        "AA",
        "AC",
        "AG",
        "AT",
        "CA",
        "CC",
        "CG",
        "CT",
        "GA",
        "GC",
        "GG",
        "GT",
        "TA",
        "TC",
        "TG",
        "TT",
    }


@pytest.mark.parametrize(
    ("ref", "alt", "expected"),
    [
        ("A", "G", True),
        ("G", "A", True),
        ("C", "T", True),
        ("T", "C", True),
        ("A", "C", False),
        ("A", "T", False),
        ("G", "C", False),
        ("C", "G", False),
        ("g", "a", True),
    ],
)
def test_is_transition(ref, alt, expected):
    assert is_transition(ref, alt) is expected


@pytest.mark.parametrize(
    ("left", "ref", "right", "expected"),
    [
        ("A", "C", "G", True),
        ("C", "G", "A", True),
        ("C", "C", "G", True),  # C of a CG, left base irrelevant
        ("A", "C", "A", False),
        ("A", "G", "A", False),
        ("T", "A", "G", False),
    ],
)
def test_classify_cpg(left, ref, right, expected):
    assert classify_cpg(left, ref, right) is expected


def test_dinucleotide_shuffle_preserves_length_and_endpoints():
    rng = np.random.default_rng(0)
    seq = "ACGTACGGCTAGCTAGCATCGATCGATCGGCATCG"
    shuffled = dinucleotide_shuffle(seq, rng)
    assert len(shuffled) == len(seq)
    assert shuffled[0] == seq[0]
    assert shuffled[-1] == seq[-1]


def test_dinucleotide_shuffle_preserves_base_composition():
    rng = np.random.default_rng(1)
    seq = "ACGTACGGCTAGCTAGCATCGATCGATCGGCATCG" * 3
    shuffled = dinucleotide_shuffle(seq, rng)
    assert Counter(seq) == Counter(shuffled)


def test_dinucleotide_shuffle_preserves_dinucleotide_counts():
    rng = np.random.default_rng(2)
    seq = "ACGTACGGCTAGCTAGCATCGATCGATCGGCATCGACGTACGTGGCATTACG"
    shuffled = dinucleotide_shuffle(seq, rng)
    orig_dinucs = Counter(zip(seq, seq[1:], strict=False))
    shuf_dinucs = Counter(zip(shuffled, shuffled[1:], strict=False))
    assert orig_dinucs == shuf_dinucs


def test_dinucleotide_shuffle_is_seeded_deterministic():
    seq = "ACGTACGGCTAGCTAGCATCGATCGATCGGCATCG"
    a = dinucleotide_shuffle(seq, np.random.default_rng(42))
    b = dinucleotide_shuffle(seq, np.random.default_rng(42))
    assert a == b


def test_dinucleotide_shuffle_actually_reorders_a_longer_sequence():
    rng = np.random.default_rng(3)
    seq = ("ACGTACGGCTAGCTAGCATCGATCGATCGGCATCG" * 5) + "T"
    shuffled = dinucleotide_shuffle(seq, rng)
    assert shuffled != seq


def test_dinucleotide_shuffle_rejects_non_acgt():
    with pytest.raises(ValueError):
        dinucleotide_shuffle("ACGTN", np.random.default_rng(0))


def test_dinucleotide_shuffle_rejects_too_short():
    with pytest.raises(ValueError):
        dinucleotide_shuffle("A", np.random.default_rng(0))


def test_bits_per_base_matches_manual_conversion():
    # log(0.25) nats per base = 2 bits per base (uniform over 4 symbols)
    assert bits_per_base(np.log(0.25)) == pytest.approx(2.0)
    assert bits_per_base(0.0) == pytest.approx(0.0)


def test_marginal_base_log_probs_sums_matching_kmers():
    kmers = ["AA", "AC", "GA", "GC"]
    probs = [0.1, 0.2, 0.3, 0.4]
    log_probs = np.log(probs)
    result = marginal_base_log_probs(log_probs, kmers, offset=0)
    assert np.exp(result["A"]) == pytest.approx(0.3)  # AA + AC
    assert np.exp(result["G"]) == pytest.approx(0.7)  # GA + GC


def test_marginal_base_log_probs_offset_one():
    kmers = ["AA", "AC", "GA", "GC"]
    probs = [0.1, 0.2, 0.3, 0.4]
    log_probs = np.log(probs)
    result = marginal_base_log_probs(log_probs, kmers, offset=1)
    assert np.exp(result["A"]) == pytest.approx(0.4)  # AA + GA
    assert np.exp(result["C"]) == pytest.approx(0.6)  # AC + GC


def test_sample_valid_positions_only_returns_valid_and_is_sorted():
    rng = np.random.default_rng(0)
    positions = sample_valid_positions(rng, 0, 1000, 20, is_valid=lambda p: p % 3 == 0)
    assert len(positions) == 20
    assert positions == sorted(positions)
    assert all(p % 3 == 0 for p in positions)
    assert len(set(positions)) == 20


def test_sample_valid_positions_deterministic_for_same_seed():
    a = sample_valid_positions(np.random.default_rng(7), 0, 1000, 10, is_valid=lambda p: p % 2 == 0)
    b = sample_valid_positions(np.random.default_rng(7), 0, 1000, 10, is_valid=lambda p: p % 2 == 0)
    assert a == b


def test_sample_valid_positions_raises_when_impossible():
    with pytest.raises(RuntimeError):
        sample_valid_positions(
            np.random.default_rng(0), 0, 10, 5, is_valid=lambda p: False, max_attempts=50
        )


def test_mean_llr_by_group_aggregates_and_drops_nan():
    df = pd.DataFrame(
        {
            "group": ["a", "a", "b", "b", "a"],
            "llr": [1.0, 3.0, -2.0, -4.0, float("nan")],
        }
    )
    out = mean_llr_by_group(df, "group", "llr")
    assert list(out["group"]) == ["a", "b"]
    a_row = out[out["group"] == "a"].iloc[0]
    assert a_row["n"] == 2
    assert a_row["mean"] == pytest.approx(2.0)
    b_row = out[out["group"] == "b"].iloc[0]
    assert b_row["n"] == 2
    assert b_row["mean"] == pytest.approx(-3.0)


def test_format_markdown_table_shape():
    text = format_markdown_table(["a", "b"], [[1, 2], [3, 4]])
    lines = text.strip().splitlines()
    assert lines[0] == "| a | b |"
    assert lines[1] == "| --- | --- |"
    assert lines[2] == "| 1 | 2 |"
    assert lines[3] == "| 3 | 4 |"
