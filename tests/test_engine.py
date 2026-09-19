import math

import pytest

from zeroshot_vep.engine import score_variants
from zeroshot_vep.reference import ReferenceGenome
from zeroshot_vep.scorers.base import Scorer
from zeroshot_vep.variant import Variant


class RecordingScorer(Scorer):
    name = "recording"
    max_window = 100

    def __init__(self):
        self.calls = []

    def score_window(self, ref_seq, center, alts):
        self.calls.append((ref_seq, center, list(alts)))
        return [float(ord(a) - ord(ref_seq[center])) for a in alts]

    def cache_key(self):
        return "recording"


def _make_reference(tmp_path, seq, chrom="chr1"):
    path = tmp_path / "ref.fa"
    path.write_text(f">{chrom}\n{seq}\n")
    return ReferenceGenome(path)


# index: 0-3 A, 4-7 C, 8-11 G, 12-15 T, 16-19 A
SEQ = "AAAACCCCGGGGTTTTAAAA"


def test_window_and_reverse_complement_mirroring(tmp_path):
    ref = _make_reference(tmp_path, SEQ)
    scorer = RecordingScorer()
    variants = [Variant("chr1", 10, "G", "A")]  # 1-based pos 10 -> 0-based idx 9 = 'G'

    result = score_variants(variants, ref, scorer, window=6, strands="both")

    assert len(scorer.calls) == 2
    fwd_seq, fwd_center, fwd_alts = scorer.calls[0]
    rev_seq, rev_center, rev_alts = scorer.calls[1]
    assert fwd_seq == "CCGGGG"
    assert fwd_center == 3
    assert fwd_alts == ["A"]
    assert rev_seq == "CCCCGG"
    assert rev_center == 2
    assert rev_alts == ["T"]

    row = result.iloc[0]
    assert row["status"] == "ok"
    assert row["llr_fwd"] == pytest.approx(-6.0)
    assert row["llr_rev"] == pytest.approx(17.0)
    assert row["llr"] == pytest.approx(5.5)


def test_forward_only_strand(tmp_path):
    ref = _make_reference(tmp_path, SEQ)
    scorer = RecordingScorer()
    variants = [Variant("chr1", 10, "G", "A")]

    result = score_variants(variants, ref, scorer, window=6, strands="forward")

    assert len(scorer.calls) == 1
    row = result.iloc[0]
    assert row["llr"] == row["llr_fwd"]
    assert math.isnan(row["llr_rev"])


def test_ref_mismatch_is_not_scored(tmp_path):
    ref = _make_reference(tmp_path, SEQ)
    scorer = RecordingScorer()
    variants = [Variant("chr1", 10, "A", "G")]  # actual ref at pos 10 is 'G'

    result = score_variants(variants, ref, scorer, window=6, strands="both")

    assert scorer.calls == []
    row = result.iloc[0]
    assert row["status"] == "ref_mismatch"
    assert math.isnan(row["llr"])
    assert math.isnan(row["llr_fwd"])
    assert math.isnan(row["llr_rev"])


def test_groups_multiple_alts_into_one_call(tmp_path):
    ref = _make_reference(tmp_path, SEQ)
    scorer = RecordingScorer()
    variants = [
        Variant("chr1", 10, "G", "A"),
        Variant("chr1", 10, "G", "C"),
    ]

    result = score_variants(variants, ref, scorer, window=6, strands="both")

    assert len(scorer.calls) == 2  # one fwd + one rev call, both alts batched
    _, _, fwd_alts = scorer.calls[0]
    assert fwd_alts == ["A", "C"]
    assert len(result) == 2


def test_window_exceeding_max_window_raises(tmp_path):
    ref = _make_reference(tmp_path, SEQ)
    scorer = RecordingScorer()
    variants = [Variant("chr1", 10, "G", "A")]
    with pytest.raises(ValueError):
        score_variants(variants, ref, scorer, window=1000, strands="both")


def test_window_with_n_padding_is_allowed(tmp_path):
    ref = _make_reference(tmp_path, "ACGT")
    scorer = RecordingScorer()
    variants = [Variant("chr1", 2, "C", "G")]  # 0-based idx 1 = 'C', window 6 pads with N
    result = score_variants(variants, ref, scorer, window=6, strands="forward")
    fwd_seq, fwd_center, _ = scorer.calls[0]
    assert "N" in fwd_seq
    assert result.iloc[0]["status"] == "ok"
