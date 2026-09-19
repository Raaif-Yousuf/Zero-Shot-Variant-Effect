"""Tests for the streaming phyloP wigFix lookup, against a tiny synthetic track."""

from __future__ import annotations

import gzip

from zeroshot_vep.datasets.phylop import lookup_phylop


def _write_wigfix(path):
    content = (
        "fixedStep chrom=chr17 start=100 step=1\n"
        "0.5\n0.6\n0.7\n"
        "fixedStep chrom=chr17 start=200 step=1\n"
        "1.1\n1.2\n"
    )
    with gzip.open(path, "wt") as fh:
        fh.write(content)


def test_lookup_phylop_matches_covered_positions(tmp_path):
    path = tmp_path / "test.wigFix.gz"
    _write_wigfix(path)

    result = lookup_phylop(path, [101, 200, 150, 999, 100])

    assert result == {100: 0.5, 101: 0.6, 200: 1.1}
    assert 150 not in result  # gap between the two fixedStep blocks
    assert 999 not in result  # past the end of the track


def test_lookup_phylop_empty_positions_returns_empty(tmp_path):
    path = tmp_path / "test.wigFix.gz"
    _write_wigfix(path)
    assert lookup_phylop(path, []) == {}
