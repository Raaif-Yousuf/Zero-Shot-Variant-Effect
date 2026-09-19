import numpy as np
import pytest

from zeroshot_vep.scorers._llr import (
    causal_loglik,
    causal_site_llr,
    masked_marginal_llr,
    nt_locate_kmer,
    nt_token_layout,
)


def _uniform_log_probs(n_positions: int, vocab: int) -> np.ndarray:
    return np.full((n_positions, vocab), -np.log(vocab))


def test_causal_loglik_matches_manual_sum():
    # 4 tokens, vocab of 3; log_probs[i] predicts token i + 1.
    log_probs = np.log(
        np.array(
            [
                [0.1, 0.2, 0.7],
                [0.5, 0.4, 0.1],
                [0.2, 0.2, 0.6],
            ]
        )
    )
    token_ids = [0, 2, 0, 1]
    expected = log_probs[0, 2] + log_probs[1, 0] + log_probs[2, 1]
    assert causal_loglik(log_probs, token_ids, start=1, end=4) == pytest.approx(expected)


def test_causal_loglik_restricted_range_is_a_partial_sum():
    log_probs = np.log(
        np.array(
            [
                [0.1, 0.2, 0.7],
                [0.5, 0.4, 0.1],
                [0.2, 0.2, 0.6],
            ]
        )
    )
    token_ids = [0, 2, 0, 1]
    full = causal_loglik(log_probs, token_ids, start=1, end=4)
    prefix = causal_loglik(log_probs, token_ids, start=1, end=2)
    suffix = causal_loglik(log_probs, token_ids, start=2, end=4)
    assert full == pytest.approx(prefix + suffix)


def test_causal_loglik_rejects_start_zero():
    log_probs = _uniform_log_probs(3, 4)
    with pytest.raises(ValueError):
        causal_loglik(log_probs, [0, 1, 2, 3], start=0, end=4)


def test_causal_loglik_rejects_end_past_available_logits():
    log_probs = _uniform_log_probs(3, 4)
    with pytest.raises(ValueError):
        causal_loglik(log_probs, [0, 1, 2, 3], start=1, end=5)


def test_causal_site_llr():
    row = np.log(np.array([0.1, 0.6, 0.3]))
    assert causal_site_llr(row, ref_id=0, alt_id=1) == pytest.approx(row[1] - row[0])


def test_masked_marginal_llr_same_formula_as_site():
    row = np.log(np.array([0.25, 0.25, 0.5]))
    assert masked_marginal_llr(row, ref_id=2, alt_id=0) == pytest.approx(row[0] - row[2])


def test_nt_token_layout_clean_multiple_of_six():
    layout = nt_token_layout("ACGTAC" * 4)
    assert layout == [(0, 6, True), (6, 6, True), (12, 6, True), (18, 6, True)]


def test_nt_token_layout_trailing_leftover_splits_to_chars():
    layout = nt_token_layout("A" * 23)
    assert layout[:3] == [(0, 6, True), (6, 6, True), (12, 6, True)]
    assert layout[3:] == [
        (18, 1, False),
        (19, 1, False),
        (20, 1, False),
        (21, 1, False),
        (22, 1, False),
    ]


def test_nt_token_layout_n_forces_char_split_for_that_block():
    seq = "ACGTAN" + "ACGTAC" * 3
    layout = nt_token_layout(seq)
    assert layout[:6] == [(i, 1, False) for i in range(6)]
    assert layout[6:] == [(6, 6, True), (12, 6, True), (18, 6, True)]


def test_nt_token_layout_matches_real_tokenizer_decode_examples():
    # Matches decoded tokens observed from the real tokenizer for these inputs:
    # "ACGTACGTAC" (10 bases) -> <cls>, ACGTAC, G, T, A, C
    layout = nt_token_layout("ACGTACGTAC")
    assert layout == [(0, 6, True), (6, 1, False), (7, 1, False), (8, 1, False), (9, 1, False)]


def test_nt_locate_kmer_no_trim_needed():
    seq = "ACGTAC" * 4
    trim, token_index, chunk_start, offset = nt_locate_kmer(seq, center=7)
    assert trim == 0
    assert token_index == 2  # cls + chunk 0
    assert chunk_start == 6
    assert offset == 1
    assert seq[chunk_start + offset] == seq[7]


def test_nt_locate_kmer_trims_to_dodge_leading_n():
    # An N at position 0 breaks the first block; shifting the frame by one
    # base lets the variant at position 4 land in a clean downstream 6-mer.
    seq = "N" + "ACGTAC" * 4
    trim, token_index, chunk_start, offset = nt_locate_kmer(seq, center=4)
    windowed = seq[trim:]
    assert windowed[chunk_start : chunk_start + 6].isalpha()
    assert all(c in "ACGT" for c in windowed[chunk_start : chunk_start + 6])
    assert windowed[chunk_start + offset] == seq[4]


def test_nt_locate_kmer_accounts_for_earlier_dirty_chunks_in_token_index():
    # First block has an N (splits into 6 single-char tokens), so the clean
    # k-mer covering the variant is NOT simply "1 + chunk_start // 6".
    seq = "ACGTAN" + "ACGTAC" * 3
    center = 6 + 3  # inside the second clean 6-mer block
    trim, token_index, chunk_start, offset = nt_locate_kmer(seq, center)
    assert trim == 0
    # cls (1) + 6 single-char tokens for "ACGTAN" + this is the first kmer token
    assert token_index == 1 + 6
    assert chunk_start == 6
    assert offset == 3


def test_nt_locate_kmer_raises_when_unalignable():
    with pytest.raises(ValueError):
        nt_locate_kmer("N", center=0)
