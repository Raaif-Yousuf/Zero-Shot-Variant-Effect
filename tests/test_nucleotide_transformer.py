"""Tests for NucleotideTransformerScorer using a tiny fake masked model."""

import itertools
import os
import random
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from zeroshot_vep.scorers._llr import nt_token_layout  # noqa: E402
from zeroshot_vep.scorers.nucleotide_transformer import NucleotideTransformerScorer  # noqa: E402
from zeroshot_vep.scorers.registry import get_scorer  # noqa: E402

# Downloading and running the real weights needs HF_HOME pointed at a short
# path (long paths break trust_remote_code on Windows); set before running
# `pytest -m models`, e.g. `export HF_HOME=C:/Users/<you>/.cache/vep-hf`.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

_SPECIALS = {"<unk>": 0, "<pad>": 1, "<mask>": 2, "<cls>": 3, "<eos>": 4, "<bos>": 5}


def _build_vocab():
    vocab = dict(_SPECIALS)
    next_id = len(vocab)
    for c in "ACGTN":
        vocab[c] = next_id
        next_id += 1
    for combo in itertools.product("ACGT", repeat=6):
        vocab["".join(combo)] = next_id
        next_id += 1
    return vocab


VOCAB = _build_vocab()
VOCAB_SIZE = len(VOCAB)


class FakeNTTokenizer:
    """Reimplements the 6-mer, non-overlapping chunking independently of
    ``nt_token_layout`` (production code) so the scorer test exercises real
    integration, not a tautology; ``nt_token_layout`` is used only to
    cross-check, not to build the fake's tokenization.
    """

    mask_token_id = VOCAB["<mask>"]

    def __call__(self, seq, **kwargs):
        ids = [VOCAB["<cls>"]]
        i = 0
        n = len(seq)
        while i < n:
            block = seq[i : i + 6]
            if len(block) == 6 and all(c in "ACGT" for c in block):
                ids.append(VOCAB[block])
                i += 6
            else:
                for c in block:
                    ids.append(VOCAB[c])
                i += len(block)
        return {"input_ids": ids}

    def convert_tokens_to_ids(self, token):
        return VOCAB[token]


class FakeMaskedModel:
    """Deterministic bidirectional model: each position's logits depend on its
    own token plus a bag-of-tokens summary of the whole sequence, so masking
    one position changes only that position's context contribution.
    """

    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=6, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.embed = torch.randn(vocab_size, embed_dim, generator=g)
        self.head = torch.randn(embed_dim, vocab_size, generator=g)
        self.calls = 0

    def eval(self):
        return self

    def __call__(self, input_ids):
        self.calls += 1
        emb = self.embed[input_ids]
        context = emb.sum(dim=1, keepdim=True)
        combined = emb + context
        logits = combined @ self.head
        return SimpleNamespace(logits=logits)


def make_scorer(**kwargs):
    return NucleotideTransformerScorer(
        name="fake-nt",
        model_id="fake/nt",
        revision="fake",
        max_window=256,
        model=FakeMaskedModel(),
        tokenizer=FakeNTTokenizer(),
        **kwargs,
    )


def test_matches_manual_masked_marginal_computation():
    ref_seq = "ACGTAC" * 4  # clean 6-mer blocks, no trim needed
    center = 7  # second block, offset 1
    alts = ["G", "T"]
    scorer = make_scorer()
    scores = scorer.score_window(ref_seq, center, alts)

    tok, model = scorer._tokenizer, scorer._model
    ids = tok(ref_seq)["input_ids"]
    token_index = 2  # <cls> + block0 + this is block1
    masked = list(ids)
    masked[token_index] = tok.mask_token_id
    with torch.inference_mode():
        logits = model(torch.tensor([masked])).logits
    row = torch.log_softmax(logits[0, token_index].float(), dim=-1).numpy()
    ref_kmer = ref_seq[6:12]
    ref_id = VOCAB[ref_kmer]
    for alt, score in zip(alts, scores, strict=True):
        alt_kmer = ref_kmer[:1] + alt + ref_kmer[2:]
        expected = row[VOCAB[alt_kmer]] - row[ref_id]
        assert score == pytest.approx(expected, abs=1e-6)


def test_all_alts_share_one_forward_pass():
    scorer = make_scorer()
    scorer.score_window("ACGTAC" * 4, 7, ["G", "T"])
    assert scorer._model.calls == 1


def test_trims_window_to_dodge_a_dirty_leading_block():
    seq = "N" + "ACGTAC" * 4
    center = 4
    scorer = make_scorer()
    scores = scorer.score_window(seq, center, ["G"])
    assert all(np.isfinite(scores))

    # Cross-check the frame our production locator picked matches what the
    # fake tokenizer actually produces for the trimmed window.
    from zeroshot_vep.scorers._llr import nt_locate_kmer

    trim, token_index, chunk_start, offset = nt_locate_kmer(seq, center)
    windowed = seq[trim:]
    layout = nt_token_layout(windowed)
    start, length, is_kmer = layout[token_index - 1]
    assert is_kmer
    assert start == chunk_start


def test_ref_in_alt_position_gives_zero_llr():
    ref_seq = "ACGTAC" * 4
    center = 7
    ref_base = ref_seq[center]
    scores = make_scorer().score_window(ref_seq, center, [ref_base])
    assert scores[0] == pytest.approx(0.0, abs=1e-9)


def test_rejects_window_longer_than_max_window():
    scorer = make_scorer()
    with pytest.raises(ValueError):
        scorer.score_window("A" * 1000, 10, ["C"])


def test_cache_key_includes_model_id_and_revision():
    scorer = NucleotideTransformerScorer(
        name="fake-nt",
        model_id="my/model",
        revision="rev1",
        max_window=256,
        model=FakeMaskedModel(),
        tokenizer=FakeNTTokenizer(),
    )
    assert "my/model" in scorer.cache_key()
    assert "rev1" in scorer.cache_key()


@pytest.mark.models
def test_real_model_gives_finite_scores():
    random.seed(0)
    ref_seq = "".join(random.choice("ACGT") for _ in range(200))
    center = 100
    other_bases = [b for b in "ACGT" if b != ref_seq[center]]
    scorer = get_scorer("nt-v2-50m")
    scores = scorer.score_window(ref_seq, center, other_bases)
    assert len(scores) == len(other_bases)
    assert all(np.isfinite(s) for s in scores)


@pytest.mark.models
def test_real_model_ref_in_alt_position_is_zero():
    random.seed(1)
    ref_seq = "".join(random.choice("ACGT") for _ in range(200))
    center = 100
    scorer = get_scorer("nt-v2-50m")
    scores = scorer.score_window(ref_seq, center, [ref_seq[center]])
    assert scores[0] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.models
def test_real_model_is_deterministic_across_calls():
    random.seed(2)
    ref_seq = "".join(random.choice("ACGT") for _ in range(200))
    center = 100
    other_bases = [b for b in "ACGT" if b != ref_seq[center]]
    scorer = get_scorer("nt-v2-50m")
    first = scorer.score_window(ref_seq, center, other_bases)
    second = scorer.score_window(ref_seq, center, other_bases)
    assert first == pytest.approx(second, abs=1e-6)
