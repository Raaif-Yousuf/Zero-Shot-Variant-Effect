from zeroshot_vep.cache import ScoreCache
from zeroshot_vep.engine import score_variants
from zeroshot_vep.reference import ReferenceGenome
from zeroshot_vep.scorers.base import Scorer
from zeroshot_vep.variant import Variant


def test_cache_round_trip(tmp_path):
    cache = ScoreCache(cache_dir=tmp_path)
    assert cache.get_many("scorerA", 10, "fwd", "ACGTAC", 3, ["G", "T"]) == {}

    cache.put_many("scorerA", 10, "fwd", "ACGTAC", 3, {"G": -1.5, "T": 0.25})
    cache.close()

    cache2 = ScoreCache(cache_dir=tmp_path)
    hits = cache2.get_many("scorerA", 10, "fwd", "ACGTAC", 3, ["G", "T", "C"])
    assert hits == {"G": -1.5, "T": 0.25}
    cache2.close()


def test_cache_distinguishes_sequence_and_strand(tmp_path):
    cache = ScoreCache(cache_dir=tmp_path)
    cache.put_many("scorerA", 10, "fwd", "ACGTAC", 3, {"G": -1.5})
    assert cache.get_many("scorerA", 10, "rev", "ACGTAC", 3, ["G"]) == {}
    assert cache.get_many("scorerA", 10, "fwd", "TTTTTT", 3, ["G"]) == {}
    cache.close()


class CountingScorer(Scorer):
    name = "counting"
    max_window = 100
    calls = 0

    def score_window(self, ref_seq, center, alts):
        CountingScorer.calls += 1
        return [1.0 for _ in alts]

    def cache_key(self):
        return "counting"


def test_engine_uses_cache_to_avoid_rescoring(tmp_path):
    ref_path = tmp_path / "ref.fa"
    ref_path.write_text(">chr1\nAAAACCCCGGGGTTTTAAAA\n")
    reference = ReferenceGenome(ref_path)
    variants = [Variant("chr1", 10, "G", "A")]

    CountingScorer.calls = 0
    cache = ScoreCache(cache_dir=tmp_path / "cache")
    score_variants(variants, reference, CountingScorer(), window=6, strands="both", cache=cache)
    first_calls = CountingScorer.calls
    assert first_calls == 2  # one fwd call + one rev call

    score_variants(variants, reference, CountingScorer(), window=6, strands="both", cache=cache)
    assert CountingScorer.calls == first_calls  # cache hit, scorer not called again
    cache.close()
