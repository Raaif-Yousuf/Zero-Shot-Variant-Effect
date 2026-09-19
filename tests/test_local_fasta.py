"""Tests for the private hg19 chr17 FASTA reader and ref-allele verification."""

from __future__ import annotations

import gzip

import pandas as pd
import pytest

from zeroshot_vep.datasets._local_fasta import LocalChromFasta, verify_ref_alleles


def _write_fasta(tmp_path, seq: str):
    path = tmp_path / "chr.fa.gz"
    with gzip.open(path, "wt") as fh:
        fh.write(">chr17 test fixture\n")
        fh.write(seq[:4] + "\n")
        fh.write(seq[4:] + "\n")
    return path


def test_base_at(tmp_path):
    fasta = LocalChromFasta(_write_fasta(tmp_path, "acgtACGT"))
    assert len(fasta) == 8
    assert fasta.base_at(1) == "A"
    assert fasta.base_at(5) == "A"
    assert fasta.base_at(8) == "T"


def test_base_at_out_of_range_raises(tmp_path):
    fasta = LocalChromFasta(_write_fasta(tmp_path, "ACGT"))
    with pytest.raises(IndexError):
        fasta.base_at(0)
    with pytest.raises(IndexError):
        fasta.base_at(5)


def test_verify_ref_alleles_drops_mismatches(tmp_path):
    fasta = LocalChromFasta(_write_fasta(tmp_path, "ACGTACGT"))
    df = pd.DataFrame({"pos": [1, 2, 3], "ref": ["A", "X", "G"]})
    clean, n_mismatch = verify_ref_alleles(df, fasta)
    assert n_mismatch == 1
    assert list(clean["pos"]) == [1, 3]
