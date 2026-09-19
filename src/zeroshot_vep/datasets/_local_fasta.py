"""Internal, minimal single-chromosome FASTA reader used only inside ``datasets``.

This exists solely so the dataset builders can verify published ref alleles
against hg19 chr17 before a shared reference-genome reader lands on ``main``.
Do not import this outside ``zeroshot_vep.datasets``: it is intentionally
private (leading underscore) and should be swapped for the shared reader once
that contract exists.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd


class LocalChromFasta:
    """Loads one gzip-compressed, single-record FASTA fully into memory."""

    def __init__(self, path: Path | str) -> None:
        self._seq = _read_fasta_gz(Path(path))

    def base_at(self, pos: int) -> str:
        """Return the uppercase base at a 1-based position."""
        if pos < 1 or pos > len(self._seq):
            raise IndexError(f"position {pos} outside sequence of length {len(self._seq)}")
        return self._seq[pos - 1]

    def __len__(self) -> int:
        return len(self._seq)


def _read_fasta_gz(path: Path) -> str:
    parts: list[str] = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                continue
            parts.append(line.strip())
    return "".join(parts).upper()


def verify_ref_alleles(
    df: pd.DataFrame, fasta: LocalChromFasta, *, pos_col: str = "pos", ref_col: str = "ref"
) -> tuple[pd.DataFrame, int]:
    """Drop rows whose published ref base disagrees with the reference FASTA.

    Returns the cleaned DataFrame and the number of rows dropped.
    """
    actual = df[pos_col].map(fasta.base_at)
    mismatch = actual != df[ref_col]
    n_mismatch = int(mismatch.sum())
    return df.loc[~mismatch].reset_index(drop=True), n_mismatch
