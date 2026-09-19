"""Core variant record shared by readers, the scoring engine and scorers."""

from __future__ import annotations

from dataclasses import dataclass

COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def reverse_complement(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]


@dataclass(frozen=True)
class Variant:
    """A single-nucleotide variant on the forward strand of a reference.

    ``pos`` is 1-based, as in VCF. ``ref`` and ``alt`` are single uppercase bases.
    """

    chrom: str
    pos: int
    ref: str
    alt: str
    id: str = "."

    def __post_init__(self) -> None:
        if self.pos < 1:
            raise ValueError(f"pos must be 1-based and positive, got {self.pos}")
        for name, base in (("ref", self.ref), ("alt", self.alt)):
            if len(base) != 1 or base not in "ACGT":
                raise ValueError(f"{name} must be one of A/C/G/T, got {base!r}")
        if self.ref == self.alt:
            raise ValueError(f"ref and alt are identical at {self.chrom}:{self.pos}")
