import pytest

from zeroshot_vep.scorers import SCORERS
from zeroshot_vep.variant import Variant, reverse_complement


def test_reverse_complement():
    assert reverse_complement("AACGTN") == "NACGTT"


def test_variant_validation():
    Variant("chr1", 10, "A", "G")
    with pytest.raises(ValueError):
        Variant("chr1", 0, "A", "G")
    with pytest.raises(ValueError):
        Variant("chr1", 5, "A", "A")
    with pytest.raises(ValueError):
        Variant("chr1", 5, "AT", "A")


def test_registry_names():
    assert {"hyenadna-small-32k", "nt-v2-50m", "kmer"} <= set(SCORERS)
