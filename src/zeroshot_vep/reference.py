"""Reference genome FASTA loader with lazy per-chromosome caching."""

from __future__ import annotations

import gzip
from pathlib import Path


class ReferenceGenome:
    """Reads chromosome sequences from a plain or gzip FASTA file.

    The file is parsed once, on first use, into per-chromosome lists of
    sequence lines. Joining those lines into an uppercase string only
    happens for chromosomes that are actually requested, and the result is
    cached so each chromosome is joined at most once.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"reference FASTA not found: {self.path}")
        self._raw: dict[str, list[str]] | None = None
        self._names: list[str] = []
        self._seqs: dict[str, str] = {}

    def _open(self):
        if self.path.name.endswith(".gz"):
            return gzip.open(self.path, "rt")
        return open(self.path)

    def _parse(self) -> None:
        records: dict[str, list[str]] = {}
        names: list[str] = []
        name: str | None = None
        with self._open() as fh:
            for line in fh:
                if line.startswith(">"):
                    name = line[1:].split()[0].strip()
                    records[name] = []
                    names.append(name)
                elif name is not None:
                    stripped = line.rstrip("\r\n")
                    if stripped:
                        records[name].append(stripped)
        if not names:
            raise ValueError(f"no sequences found in FASTA {self.path}")
        self._raw = records
        self._names = names

    def chromosomes(self) -> list[str]:
        """Names of the chromosomes/contigs present in the FASTA, in file order."""
        if self._raw is None:
            self._parse()
        return list(self._names)

    def _resolve_name(self, chrom: str) -> str:
        if chrom in self._names:
            return chrom
        alt = chrom[3:] if chrom.startswith("chr") else f"chr{chrom}"
        if alt in self._names:
            return alt
        raise KeyError(f"unknown chromosome {chrom!r}; available: {sorted(self._names)}")

    def _load(self, chrom: str) -> str:
        if self._raw is None:
            self._parse()
        name = self._resolve_name(chrom)
        seq = self._seqs.get(name)
        if seq is None:
            assert self._raw is not None
            seq = "".join(self._raw.pop(name)).upper()
            self._seqs[name] = seq
        return seq

    def length(self, chrom: str) -> int:
        """Length in bases of ``chrom``."""
        return len(self._load(chrom))

    def fetch(self, chrom: str, start0: int, end0: int) -> str:
        """Return the uppercase sequence on ``chrom`` over ``[start0, end0)``.

        Coordinates are 0-based and half-open. Positions outside the
        chromosome are padded with ``"N"``.
        """
        if end0 <= start0:
            raise ValueError(f"end0 ({end0}) must be greater than start0 ({start0})")
        seq = self._load(chrom)
        n = len(seq)
        left_pad = max(0, -start0)
        right_pad = max(0, end0 - n)
        clip_start = max(0, start0)
        clip_end = min(n, end0)
        body = seq[clip_start:clip_end] if clip_start < clip_end else ""
        return ("N" * left_pad) + body + ("N" * right_pad)
