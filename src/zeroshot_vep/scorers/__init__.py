"""Variant scorers. Use :func:`get_scorer` to build one by name."""

from zeroshot_vep.scorers.base import Scorer
from zeroshot_vep.scorers.registry import SCORERS, get_scorer

__all__ = ["SCORERS", "Scorer", "get_scorer"]
