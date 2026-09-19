"""Load Findlay et al. 2018 BRCA1 saturation genome editing supplementary table.

The workbook has two merged title rows above the real header, so the header is
on the third sheet row. Positions are hg19; ``reference``/``alt`` are genomic
(forward-strand) alleles, not transcript alleles, so no strand flip is needed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

#: sheet column name -> tidy column name
_COLUMNS = {
    "gene": "gene",
    "chromosome": "chromosome",
    "position (hg19)": "pos",
    "reference": "ref",
    "alt": "alt",
    "consequence": "consequence",
    "function.score.mean": "function_score",
    "func.class": "func_class",
    "CADD.score": "cadd",
    "phyloP (mammalian)": "phylop_mammalian",
}

#: func.class -> binary label; INT (intermediate) is neither functional nor LOF
_LABEL = {"LOF": 1.0, "FUNC": 0.0, "INT": float("nan")}

_HEADER_ROW = 2  # 0-based; the real header is the 3rd row of the sheet

COLUMN_ORDER = [
    "chrom",
    "pos",
    "ref",
    "alt",
    "id",
    "gene",
    "consequence",
    "function_score",
    "func_class",
    "label",
    "phylop_mammalian",
    "cadd",
]


def load_findlay_table(xlsx_path: Path | str) -> pd.DataFrame:
    """Parse Supplementary Table 1 into one tidy row per SNV.

    Raises:
        ValueError: an expected column is missing (the workbook layout changed).
    """
    raw = pd.read_excel(xlsx_path, sheet_name=0, header=_HEADER_ROW, engine="openpyxl")
    missing = set(_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"Findlay table missing expected columns: {sorted(missing)}")

    df = raw[list(_COLUMNS)].rename(columns=_COLUMNS).copy()
    df["chrom"] = "chr" + df["chromosome"].astype(int).astype(str)
    df["pos"] = df["pos"].astype(int)
    df["ref"] = df["ref"].astype(str).str.upper()
    df["alt"] = df["alt"].astype(str).str.upper()
    df["id"] = df["chrom"] + ":" + df["pos"].astype(str) + df["ref"] + ">" + df["alt"]
    df["label"] = df["func_class"].map(_LABEL)

    return df[COLUMN_ORDER].reset_index(drop=True)
