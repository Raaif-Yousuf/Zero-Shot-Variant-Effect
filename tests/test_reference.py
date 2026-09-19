import gzip
from pathlib import Path

import pytest

from zeroshot_vep.reference import ReferenceGenome


def _write_fasta(
    path: Path, records: dict[str, str], gz: bool = False, line_width: int = 10
) -> Path:
    lines = []
    for name, seq in records.items():
        lines.append(f">{name}")
        for i in range(0, len(seq), line_width):
            lines.append(seq[i : i + line_width])
    text = "\n".join(lines) + "\n"
    if gz:
        path.write_bytes(gzip.compress(text.encode()))
    else:
        path.write_text(text)
    return path


def test_fetch_plain_fasta(tmp_path):
    fasta = _write_fasta(tmp_path / "ref.fa", {"chr17": "ACGTACGTAC"})
    ref = ReferenceGenome(fasta)
    assert ref.fetch("chr17", 0, 4) == "ACGT"
    assert ref.fetch("chr17", 4, 10) == "ACGTAC"


def test_fetch_gzip_fasta(tmp_path):
    fasta = _write_fasta(tmp_path / "ref.fa.gz", {"chr17": "ACGTACGTAC"}, gz=True)
    ref = ReferenceGenome(fasta)
    assert ref.fetch("chr17", 0, 4) == "ACGT"


def test_fetch_uppercases(tmp_path):
    fasta = _write_fasta(tmp_path / "ref.fa", {"chr1": "acgtACGT"})
    ref = ReferenceGenome(fasta)
    assert ref.fetch("chr1", 0, 8) == "ACGTACGT"


def test_fetch_pads_with_n_at_edges(tmp_path):
    fasta = _write_fasta(tmp_path / "ref.fa", {"chr1": "ACGT"})
    ref = ReferenceGenome(fasta)
    assert ref.fetch("chr1", -2, 6) == "NNACGTNN"


def test_chr_prefix_mapping(tmp_path):
    fasta = _write_fasta(tmp_path / "ref.fa", {"chr17": "ACGT"})
    ref = ReferenceGenome(fasta)
    assert ref.fetch("17", 0, 4) == "ACGT"

    fasta2 = _write_fasta(tmp_path / "ref2.fa", {"17": "ACGT"})
    ref2 = ReferenceGenome(fasta2)
    assert ref2.fetch("chr17", 0, 4) == "ACGT"


def test_unknown_chromosome_raises(tmp_path):
    fasta = _write_fasta(tmp_path / "ref.fa", {"chr1": "ACGT"})
    ref = ReferenceGenome(fasta)
    with pytest.raises(KeyError):
        ref.fetch("chr2", 0, 4)


def test_caches_loaded_chromosome(tmp_path):
    fasta = _write_fasta(tmp_path / "ref.fa", {"chr1": "ACGTACGT"})
    ref = ReferenceGenome(fasta)
    ref.fetch("chr1", 0, 4)
    assert "chr1" not in (ref._raw or {})
    assert ref._seqs["chr1"] == "ACGTACGT"


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ReferenceGenome(tmp_path / "missing.fa")


def test_length(tmp_path):
    fasta = _write_fasta(tmp_path / "ref.fa", {"chr1": "ACGTACGT"})
    ref = ReferenceGenome(fasta)
    assert ref.length("chr1") == 8
