"""Tests for the Findlay BRCA1 SGE table loader, against a tiny synthetic workbook."""

from __future__ import annotations

import pandas as pd
import pytest

openpyxl = pytest.importorskip("openpyxl")

from zeroshot_vep.datasets.findlay import COLUMN_ORDER, load_findlay_table  # noqa: E402

_HEADER = [
    "gene",
    "chromosome",
    "position (hg19)",
    "reference",
    "alt",
    "consequence",
    "function.score.mean",
    "func.class",
    "CADD.score",
    "phyloP (mammalian)",
]


def _make_xlsx(path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["title row 1"])
    ws.append(["title row 2"])
    ws.append(_HEADER)
    ws.append(["BRCA1", 17, 100, "A", "G", "Missense", -1.2, "LOF", 30.0, 5.0])
    ws.append(["BRCA1", 17, 200, "C", "T", "Synonymous", 0.1, "FUNC", 2.0, 1.0])
    ws.append(["BRCA1", 17, 300, "G", "A", "Missense", -0.2, "INT", 15.0, 3.0])
    wb.save(path)


def test_load_findlay_table(tmp_path):
    path = tmp_path / "findlay.xlsx"
    _make_xlsx(path)

    df = load_findlay_table(path)

    assert list(df.columns) == COLUMN_ORDER
    assert len(df) == 3
    assert df.loc[df["pos"] == 100, "label"].iloc[0] == 1.0
    assert df.loc[df["pos"] == 200, "label"].iloc[0] == 0.0
    assert pd.isna(df.loc[df["pos"] == 300, "label"].iloc[0])
    assert df.loc[df["pos"] == 100, "chrom"].iloc[0] == "chr17"
    assert df.loc[df["pos"] == 100, "id"].iloc[0] == "chr17:100A>G"


def test_load_findlay_table_missing_column_raises(tmp_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["title row 1"])
    ws.append(["title row 2"])
    ws.append(["gene", "chromosome"])  # missing most required columns
    path = tmp_path / "bad.xlsx"
    wb.save(path)

    with pytest.raises(ValueError):
        load_findlay_table(path)
