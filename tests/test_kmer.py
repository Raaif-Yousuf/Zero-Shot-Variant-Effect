import math

import numpy as np
import pytest

from zeroshot_vep.scorers.kmer import KmerMarkovScorer, _encode
from zeroshot_vep.variant import reverse_complement


def _make_fasta(tmp_path, seq, chrom="chrT"):
    path = tmp_path / "ref.fa"
    path.write_text(f">{chrom}\n{seq}\n")
    return path


def test_kmer_score_matches_hand_computed_llr(tmp_path):
    # order=1 Markov model trained on "AACG" (both strands): dimer counts are
    # AA=1 AC=1 CG=2 GT=1 TT=1 (rest 0), pseudocount=1 -> Dirichlet smoothing.
    # Scoring window "AACG", center=2 ('C'), alt='G':
    #   i=2 (predicted pos, context 'A'): log p(G|A) - log p(C|A)
    #                                   = log(1/6) - log(2/6) = log(1/2)
    #   i=3 (context pos, predicted 'G'): log p(G|G) - log p(G|C)
    #                                   = log(1/5) - log(3/6) = log(2/5)
    # total = log(1/2) + log(2/5) = log(1/5)
    fasta = _make_fasta(tmp_path, "AACG")
    scorer = KmerMarkovScorer(order=1, fasta_path=fasta, chroms=["chrT"], pseudocount=1.0)

    scores = scorer.score_window("AACG", center=2, alts=["G"])

    assert scores[0] == pytest.approx(math.log(1 / 5), abs=1e-9)


def test_kmer_defaults_to_all_chromosomes(tmp_path):
    path = tmp_path / "ref.fa"
    path.write_text(">chrA\nAACG\n>chrB\nGGCC\n")
    scorer = KmerMarkovScorer(order=1, fasta_path=path, pseudocount=1.0)
    assert scorer.chroms == ["chrA", "chrB"]


def test_kmer_requires_fasta_path():
    with pytest.raises(ValueError):
        KmerMarkovScorer(order=1, fasta_path=None)


def test_kmer_cache_key_includes_order_and_chroms(tmp_path):
    fasta = _make_fasta(tmp_path, "AACG")
    a = KmerMarkovScorer(order=1, fasta_path=fasta, chroms=["chrT"], pseudocount=1.0)
    b = KmerMarkovScorer(order=2, fasta_path=fasta, chroms=["chrT"], pseudocount=1.0)
    assert a.cache_key() != b.cache_key()


def test_kmer_npz_cache_round_trip(tmp_path):
    fasta = _make_fasta(tmp_path, "AACG")
    cache_dir = tmp_path / "cache"
    scorer1 = KmerMarkovScorer(order=1, fasta_path=fasta, chroms=["chrT"], cache_dir=cache_dir)
    cached_files = list(cache_dir.glob("*.npz"))
    assert len(cached_files) == 1

    scorer2 = KmerMarkovScorer(order=1, fasta_path=fasta, chroms=["chrT"], cache_dir=cache_dir)
    np.testing.assert_array_equal(scorer1._joint_counts, scorer2._joint_counts)


def test_kmer_max_window_is_none(tmp_path):
    fasta = _make_fasta(tmp_path, "AACG")
    scorer = KmerMarkovScorer(order=1, fasta_path=fasta, chroms=["chrT"])
    assert scorer.max_window is None


def test_reverse_strand_counts_match_explicit_reverse_complement():
    # Counts derived by permuting the forward counts (the fast path used in
    # training) must equal counts obtained by literally reverse-complementing
    # the sequence and counting that directly (the naive/reference path).
    k = 2
    seq = "ACGTACGGTTAACCGGTTAGCATGACGTN"  # includes an N to exercise masking
    size = 4 ** (k + 1)
    powers = 4 ** np.arange(k, -1, -1)

    fwd_counts = KmerMarkovScorer._count_kmers(_encode(seq), k, powers, size)
    perm = KmerMarkovScorer._rc_permutation(k)
    derived_rc_counts = np.zeros(size, dtype=np.int64)
    derived_rc_counts[perm] = fwd_counts

    explicit_rc_counts = KmerMarkovScorer._count_kmers(
        _encode(reverse_complement(seq)), k, powers, size
    )

    np.testing.assert_array_equal(derived_rc_counts, explicit_rc_counts)


def test_rc_permutation_is_an_involution():
    perm = KmerMarkovScorer._rc_permutation(3)
    np.testing.assert_array_equal(perm[perm], np.arange(len(perm)))


def test_count_kmers_chunking_matches_single_pass(monkeypatch):
    # Force a tiny chunk size so a >1-chunk sequence exercises the boundary
    # handling, and confirm it gives the same counts as one big chunk.
    import zeroshot_vep.scorers.kmer as kmer_module

    k = 2
    seq = "ACGTACGGTTAACCGGTTAGCATGACGTNACGT"
    codes = _encode(seq)
    size = 4 ** (k + 1)
    powers = 4 ** np.arange(k, -1, -1)

    baseline = KmerMarkovScorer._count_kmers(codes, k, powers, size)

    monkeypatch.setattr(kmer_module, "_COUNT_CHUNK", 5)
    chunked = KmerMarkovScorer._count_kmers(codes, k, powers, size)

    np.testing.assert_array_equal(baseline, chunked)


def test_kmer_skips_n_context(tmp_path):
    fasta = _make_fasta(tmp_path, "AACG")
    scorer = KmerMarkovScorer(order=1, fasta_path=fasta, chroms=["chrT"], pseudocount=1.0)
    # window has an N right after the variant base; the i=center+1 term
    # should be skipped (contributes 0), leaving only the i=center term.
    scores = scorer.score_window("ACN", center=1, alts=["G"])
    # p(G|A) = (joint[AG]=0 + 1) / (context_total[A]=2 + 4)
    # p(C|A) = (joint[AC]=1 + 1) / (context_total[A]=2 + 4)
    expected = math.log((0 + 1) / (2 + 4)) - math.log((1 + 1) / (2 + 4))
    assert scores[0] == pytest.approx(expected, abs=1e-9)
