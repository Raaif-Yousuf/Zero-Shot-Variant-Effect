"""The contract every scorer implements.

The scoring engine extracts a reference window around each variant position,
groups the alternate alleles observed at that position, and calls
:meth:`Scorer.score_window` once per strand. Scorers never see coordinates, only
sequence, so the same code handles the forward strand and the reverse complement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


class Scorer(ABC):
    #: Short registry name, for example ``"hyenadna-small-32k"``.
    name: str = "base"

    #: Largest window (in bases) the scorer accepts, or ``None`` for no limit.
    max_window: int | None = None

    @abstractmethod
    def score_window(self, ref_seq: str, center: int, alts: Sequence[str]) -> list[float]:
        """Score alternate alleles at one position.

        Args:
            ref_seq: Uppercase reference window (A/C/G/T/N).
            center: 0-based index of the variant base inside ``ref_seq``.
            alts: Alternate bases (each differs from ``ref_seq[center]``).

        Returns:
            One float per alt: log P(alt sequence) - log P(ref sequence) under the
            model (natural log). Negative means the alt is less likely than the
            reference, which is the expected direction for damaging variants.
        """

    def cache_key(self) -> str:
        """Identifies everything that changes the output (model, revision, mode, ...).

        The engine combines this with the window size and strand setting to key the
        on-disk score cache, so two scorers with equal keys must give equal scores.
        """
        return self.name
