"""Order-k Markov model of the reference genome: a genomic language model baseline."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from zeroshot_vep.reference import ReferenceGenome
from zeroshot_vep.scorers.base import Scorer
from zeroshot_vep.variant import reverse_complement

_BASE_CODE = {"A": 0, "C": 1, "G": 2, "T": 3}

_LUT = np.full(256, -1, dtype=np.int8)
for _base, _code in _BASE_CODE.items():
    _LUT[ord(_base)] = _code


def _encode(seq: str) -> np.ndarray:
    """Map an uppercase A/C/G/T/N string to 2-bit codes, -1 for anything else."""
    raw = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    return _LUT[raw]


class KmerMarkovScorer(Scorer):
    """Order-k Markov chain over A/C/G/T, trained on both strands of a reference.

    Scores a variant by the change in log-likelihood of the local
    ``order + 1``-length contexts that span the substituted base, under a
    Dirichlet-smoothed count model. This is a genomic language model
    without deep learning, kept deliberately simple.
    """

    max_window: int | None = None

    def __init__(
        self,
        name: str = "kmer",
        order: int = 6,
        fasta_path: str | Path | None = None,
        chroms: Sequence[str] | None = None,
        pseudocount: float = 1.0,
        cache_dir: str | Path | None = None,
    ) -> None:
        if fasta_path is None:
            raise ValueError("KmerMarkovScorer requires fasta_path to train on")
        self.name = name
        self.order = order
        self.pseudocount = pseudocount
        self.fasta_path = Path(fasta_path)
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None

        reference = ReferenceGenome(self.fasta_path)
        self.chroms = sorted(chroms) if chroms is not None else reference.chromosomes()

        self._context_powers = 4 ** np.arange(order - 1, -1, -1)
        self._train_powers = 4 ** np.arange(order, -1, -1)

        joint = self._load_or_train(reference)
        self._joint_counts = joint
        self._context_totals = joint.reshape(-1, 4).sum(axis=1)

    def cache_key(self) -> str:
        stat = self.fasta_path.stat()
        return (
            f"kmer:order={self.order}:pseudocount={self.pseudocount}:"
            f"fasta={self.fasta_path.resolve()}:{stat.st_size}:{int(stat.st_mtime)}:"
            f"chroms={','.join(self.chroms)}"
        )

    # -- training / caching --------------------------------------------------

    def _training_cache_id(self) -> str:
        stat = self.fasta_path.stat()
        payload = (
            f"{self.fasta_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|"
            f"{self.order}|{','.join(self.chroms)}"
        )
        return hashlib.sha1(payload.encode()).hexdigest()[:16]

    def _load_or_train(self, reference: ReferenceGenome) -> np.ndarray:
        size = 4 ** (self.order + 1)
        cache_path = None
        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = self._cache_dir / f"kmer_order{self.order}_{self._training_cache_id()}.npz"
            if cache_path.exists():
                data = np.load(cache_path)
                joint = data["joint_counts"]
                if joint.shape == (size,):
                    return joint
        joint = self._train(reference)
        if cache_path is not None:
            np.savez(cache_path, joint_counts=joint)
        return joint

    def _train(self, reference: ReferenceGenome) -> np.ndarray:
        k = self.order
        size = 4 ** (k + 1)
        joint = np.zeros(size, dtype=np.int64)
        for chrom in self.chroms:
            seq = reference.fetch(chrom, 0, reference.length(chrom))
            joint += self._count_kmers(_encode(seq), k, self._train_powers, size)
            joint += self._count_kmers(
                _encode(reverse_complement(seq)), k, self._train_powers, size
            )
        return joint

    @staticmethod
    def _count_kmers(codes: np.ndarray, k: int, powers: np.ndarray, size: int) -> np.ndarray:
        if len(codes) < k + 1:
            return np.zeros(size, dtype=np.int64)
        windows = sliding_window_view(codes, k + 1)
        valid = np.all(windows >= 0, axis=1)
        if not np.any(valid):
            return np.zeros(size, dtype=np.int64)
        idx = windows[valid].astype(np.int64) @ powers
        return np.bincount(idx, minlength=size).astype(np.int64)

    # -- scoring --------------------------------------------------------------

    def _logp(self, window_codes: np.ndarray) -> float:
        k = self.order
        context_idx = int(window_codes[:k].astype(np.int64) @ self._context_powers)
        predicted = int(window_codes[k])
        joint_idx = context_idx * 4 + predicted
        numer = float(self._joint_counts[joint_idx]) + self.pseudocount
        denom = float(self._context_totals[context_idx]) + 4 * self.pseudocount
        return math.log(numer / denom)

    def score_window(self, ref_seq: str, center: int, alts: Sequence[str]) -> list[float]:
        k = self.order
        codes = _encode(ref_seq)
        n = len(ref_seq)
        scores: list[float] = []
        for alt in alts:
            alt_code = _BASE_CODE[alt]
            total = 0.0
            for i in range(center, center + k + 1):
                ctx_start = i - k
                if ctx_start < 0 or i >= n:
                    continue
                window = codes[ctx_start : i + 1]
                if np.any(window < 0):
                    continue
                local_center = center - ctx_start
                alt_window = window.copy()
                alt_window[local_center] = alt_code
                total += self._logp(alt_window) - self._logp(window)
            scores.append(total)
        return scores
