"""Tests for HyenaDNAScorer using a tiny fake causal model (no real weights)."""

import os
import random
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from zeroshot_vep.scorers.hyenadna import HyenaDNAScorer  # noqa: E402
from zeroshot_vep.scorers.registry import get_scorer  # noqa: E402

# Downloading and running the real weights needs HF_HOME pointed at a short
# path (long paths break trust_remote_code on Windows); set before running
# `pytest -m models`, e.g. `export HF_HOME=C:/Users/<you>/.cache/vep-hf`.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

VOCAB = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4, "[SEP]": 5}
VOCAB_SIZE = len(VOCAB)


class FakeHyenaTokenizer:
    def __call__(self, seq, **kwargs):
        return {"input_ids": [VOCAB[c] for c in seq] + [VOCAB["[SEP]"]]}

    def convert_tokens_to_ids(self, token):
        return VOCAB[token]


class FakeCausalModel:
    """Deterministic causal model: an exponentially decaying prefix scan.

    logits[:, t, :] depends on tokens[0..t] only (never on later positions),
    so it behaves like a real causal LM for the purposes of these tests.
    """

    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=5, seed=0, decay=0.85):
        g = torch.Generator().manual_seed(seed)
        self.embed = torch.randn(vocab_size, embed_dim, generator=g)
        self.head = torch.randn(embed_dim, vocab_size, generator=g)
        self.decay = decay
        self.calls = 0

    def eval(self):
        return self

    def __call__(self, input_ids):
        self.calls += 1
        b, t = input_ids.shape
        state = torch.zeros(b, self.embed.shape[1])
        logits = torch.zeros(b, t, self.head.shape[1])
        for i in range(t):
            state = state * self.decay + self.embed[input_ids[:, i]]
            logits[:, i, :] = state @ self.head
        return SimpleNamespace(logits=logits)


def make_scorer(mode="full", **kwargs):
    return HyenaDNAScorer(
        name="fake-hyena",
        model_id="fake/hyena",
        revision="fake",
        max_window=64,
        mode=mode,
        model=FakeCausalModel(),
        tokenizer=FakeHyenaTokenizer(),
        **kwargs,
    )


def _manual_full_ll(model, tokenizer, seq, start, end):
    ids = tokenizer(seq)["input_ids"]
    batch = torch.tensor([ids], dtype=torch.long)
    with torch.inference_mode():
        logits = model(batch).logits
    log_probs = torch.log_softmax(logits[0].float(), dim=-1).numpy()
    total = 0.0
    for t in range(start, end):
        total += float(log_probs[t - 1, ids[t]])
    return total


def test_full_mode_matches_brute_force_full_sequence_sum():
    ref_seq = "ACGTACGTAC"
    center = 4
    alts = ["G", "T"]
    scorer = make_scorer(mode="full")
    scores = scorer.score_window(ref_seq, center, alts)

    model, tok = scorer._model, scorer._tokenizer
    ref_ll_full = _manual_full_ll(model, tok, ref_seq, start=1, end=len(ref_seq))
    for alt, score in zip(alts, scores, strict=True):
        alt_seq = ref_seq[:center] + alt + ref_seq[center + 1 :]
        alt_ll_full = _manual_full_ll(model, tok, alt_seq, start=1, end=len(ref_seq))
        assert score == pytest.approx(alt_ll_full - ref_ll_full, abs=1e-5)


def test_full_mode_restricted_sum_equals_unrestricted_for_prefix_positions():
    # Direct check that positions before the variant contribute zero to the
    # difference, i.e. restricting the sum to t >= center is not an approximation.
    ref_seq = "ACGTACGTACGT"
    center = 6
    scorer = make_scorer(mode="full")
    model, tok = scorer._model, scorer._tokenizer

    alt_seq = ref_seq[:center] + "G" + ref_seq[center + 1 :]
    ref_ids = tok(ref_seq)["input_ids"]
    alt_ids = tok(alt_seq)["input_ids"]
    with torch.inference_mode():
        ref_logits = model(torch.tensor([ref_ids])).logits[0]
        alt_logits = model(torch.tensor([alt_ids])).logits[0]
    ref_lp = torch.log_softmax(ref_logits.float(), dim=-1).numpy()
    alt_lp = torch.log_softmax(alt_logits.float(), dim=-1).numpy()

    for t in range(1, center):
        assert ref_lp[t - 1, ref_ids[t]] == pytest.approx(alt_lp[t - 1, alt_ids[t]], abs=1e-6)


def test_site_mode_uses_single_ref_pass_and_shared_prefix():
    ref_seq = "ACGTACGTAC"
    center = 5
    alts = ["G", "T"]
    scorer = make_scorer(mode="site")
    scores = scorer.score_window(ref_seq, center, alts)
    assert scorer._model.calls == 1

    model, tok = scorer._model, scorer._tokenizer
    ref_ids = tok(ref_seq)["input_ids"]
    with torch.inference_mode():
        logits = model(torch.tensor([ref_ids])).logits
    row = torch.log_softmax(logits[0, center - 1].float(), dim=-1).numpy()
    ref_base_id = VOCAB[ref_seq[center]]
    for alt, score in zip(alts, scores, strict=True):
        expected = row[VOCAB[alt]] - row[ref_base_id]
        assert score == pytest.approx(expected, abs=1e-6)


def test_full_mode_runs_ref_and_alts_as_one_batch():
    scorer = make_scorer(mode="full")
    scorer.score_window("ACGTACGTAC", 4, ["G", "T", "N"])
    assert scorer._model.calls == 1


def test_batched_alts_equal_one_by_one_scoring():
    ref_seq = "ACGTACGTACGT"
    center = 7
    alts = ["G", "T"]

    batched_scorer = make_scorer(mode="full")
    batched_scores = batched_scorer.score_window(ref_seq, center, alts)

    one_by_one = []
    for alt in alts:
        scorer = HyenaDNAScorer(
            name="fake-hyena",
            model_id="fake/hyena",
            revision="fake",
            max_window=64,
            mode="full",
            model=FakeCausalModel(),
            tokenizer=FakeHyenaTokenizer(),
        )
        one_by_one.append(scorer.score_window(ref_seq, center, [alt])[0])

    for batched, single in zip(batched_scores, one_by_one, strict=True):
        assert batched == pytest.approx(single, abs=1e-5)


def test_site_mode_batched_alts_equal_one_by_one():
    ref_seq = "ACGTACGTACGT"
    center = 7
    alts = ["G", "T"]
    scorer = make_scorer(mode="site")
    batched_scores = scorer.score_window(ref_seq, center, alts)

    singles = []
    for alt in alts:
        s = make_scorer(mode="site")
        singles.append(s.score_window(ref_seq, center, [alt])[0])

    for batched, single in zip(batched_scores, singles, strict=True):
        assert batched == pytest.approx(single, abs=1e-6)


def test_special_token_prediction_is_never_scored():
    # The row predicting the trailing [SEP] token (and the row after it) must
    # never enter the sum. Corrupt those two rows with fresh randomness on
    # every forward pass: if they leaked into the score, repeated calls with
    # the same inputs would give different results.
    class RandomTailModel(FakeCausalModel):
        def __call__(self, input_ids):
            out = super().__call__(input_ids)
            out.logits[:, -2:, :] = torch.randn_like(out.logits[:, -2:, :]) * 100
            return out

    ref_seq = "ACGTACGTAC"
    center = 4
    scores_a = HyenaDNAScorer(
        name="fake-hyena",
        model_id="fake/hyena",
        revision="fake",
        max_window=64,
        mode="full",
        model=RandomTailModel(),
        tokenizer=FakeHyenaTokenizer(),
    ).score_window(ref_seq, center, ["G"])
    scores_b = HyenaDNAScorer(
        name="fake-hyena",
        model_id="fake/hyena",
        revision="fake",
        max_window=64,
        mode="full",
        model=RandomTailModel(),
        tokenizer=FakeHyenaTokenizer(),
    ).score_window(ref_seq, center, ["G"])
    assert all(np.isfinite(scores_a))
    assert scores_a == pytest.approx(scores_b, abs=1e-6)


def test_rejects_window_longer_than_max_window():
    scorer = make_scorer(mode="full")
    with pytest.raises(ValueError):
        scorer.score_window("A" * 100, 10, ["C"])


def test_rejects_unknown_mode():
    with pytest.raises(ValueError):
        make_scorer(mode="bogus")


def test_cache_key_includes_model_revision_and_mode():
    full_scorer = HyenaDNAScorer(
        name="fake-hyena",
        model_id="my/model",
        revision="rev1",
        max_window=64,
        mode="full",
        model=FakeCausalModel(),
        tokenizer=FakeHyenaTokenizer(),
    )
    site_scorer = HyenaDNAScorer(
        name="fake-hyena",
        model_id="my/model",
        revision="rev1",
        max_window=64,
        mode="site",
        model=FakeCausalModel(),
        tokenizer=FakeHyenaTokenizer(),
    )
    assert "my/model" in full_scorer.cache_key()
    assert "rev1" in full_scorer.cache_key()
    assert full_scorer.cache_key() != site_scorer.cache_key()


@pytest.mark.models
@pytest.mark.parametrize("mode", ["full", "site"])
def test_real_model_gives_finite_scores(mode):
    random.seed(0)
    ref_seq = "".join(random.choice("ACGT") for _ in range(200))
    center = 100
    other_bases = [b for b in "ACGT" if b != ref_seq[center]]
    scorer = get_scorer("hyenadna-tiny-1k", mode=mode)
    scores = scorer.score_window(ref_seq, center, other_bases)
    assert len(scores) == len(other_bases)
    assert all(np.isfinite(s) for s in scores)


@pytest.mark.models
@pytest.mark.parametrize("mode", ["full", "site"])
def test_real_model_ref_in_alt_position_is_zero(mode):
    random.seed(1)
    ref_seq = "".join(random.choice("ACGT") for _ in range(200))
    center = 100
    scorer = get_scorer("hyenadna-tiny-1k", mode=mode)
    scores = scorer.score_window(ref_seq, center, [ref_seq[center]])
    assert scores[0] == pytest.approx(0.0, abs=1e-4)


@pytest.mark.models
@pytest.mark.parametrize("mode", ["full", "site"])
def test_real_model_is_deterministic_across_calls(mode):
    random.seed(2)
    ref_seq = "".join(random.choice("ACGT") for _ in range(200))
    center = 100
    other_bases = [b for b in "ACGT" if b != ref_seq[center]]
    scorer = get_scorer("hyenadna-tiny-1k", mode=mode)
    first = scorer.score_window(ref_seq, center, other_bases)
    second = scorer.score_window(ref_seq, center, other_bases)
    assert first == pytest.approx(second, abs=1e-6)
