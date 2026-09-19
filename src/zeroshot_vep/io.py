"""Variant readers (VCF/VCF.gz, TSV) and the scored-TSV writer."""

from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd

from zeroshot_vep.variant import Variant

_BASES = set("ACGT")
_REQUIRED_TSV_COLUMNS = {"chrom", "pos", "ref", "alt"}


def _open_text(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path)


def _is_snv_allele(ref: str, alt: str) -> bool:
    return len(ref) == 1 and len(alt) == 1 and ref in _BASES and alt in _BASES


def read_vcf(path: str | Path) -> tuple[list[Variant], int]:
    """Read SNVs from a VCF or VCF.gz file.

    Multi-allelic ALT fields are split into one :class:`Variant` per allele.
    Indels, symbolic alleles and records whose REF is not a single A/C/G/T
    base are skipped. Returns the variants and the number of skipped ALT
    alleles.
    """
    path = Path(path)
    variants: list[Variant] = []
    skipped = 0
    with _open_text(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            chrom, pos_s, vid, ref, alt_field = (
                fields[0],
                fields[1],
                fields[2],
                fields[3],
                fields[4],
            )
            ref = ref.upper()
            pos = int(pos_s)
            vid = vid if vid not in (".", "") else "."
            for alt in alt_field.split(","):
                alt = alt.upper()
                if _is_snv_allele(ref, alt):
                    variants.append(Variant(chrom, pos, ref, alt, vid))
                else:
                    skipped += 1
    return variants, skipped


def read_tsv(path: str | Path) -> tuple[list[Variant], pd.DataFrame]:
    """Read variants from a TSV with a header including chrom, pos, ref, alt.

    An optional ``id`` column is used for :attr:`Variant.id`. Any other
    columns are returned unchanged, in a DataFrame aligned by row with the
    variants list, so callers can carry benchmark labels through to output.
    """
    path = Path(path)
    df = pd.read_csv(path, sep="\t", dtype=str)
    lower_to_actual = {c.lower(): c for c in df.columns}
    missing = _REQUIRED_TSV_COLUMNS - set(lower_to_actual)
    if missing:
        raise ValueError(f"TSV {path} is missing required column(s): {sorted(missing)}")

    chrom_col = lower_to_actual["chrom"]
    pos_col = lower_to_actual["pos"]
    ref_col = lower_to_actual["ref"]
    alt_col = lower_to_actual["alt"]
    id_col = lower_to_actual.get("id")

    used_cols = {chrom_col, pos_col, ref_col, alt_col} | ({id_col} if id_col else set())
    extra_cols = [c for c in df.columns if c not in used_cols]

    records = df.to_dict("records")
    variants = [
        Variant(
            row[chrom_col],
            int(row[pos_col]),
            row[ref_col].upper(),
            row[alt_col].upper(),
            row[id_col] if id_col else ".",
        )
        for row in records
    ]
    extra = df[extra_cols].reset_index(drop=True) if extra_cols else pd.DataFrame()
    return variants, extra


def read_variants(path: str | Path) -> tuple[list[Variant], pd.DataFrame, int]:
    """Auto-detect VCF vs TSV by extension and read variants plus extra columns.

    The third element is the count of ALT alleles skipped while reading a
    VCF (always 0 for TSV input).
    """
    path = Path(path)
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".vcf") or suffixes.endswith(".vcf.gz"):
        variants, skipped = read_vcf(path)
        return variants, pd.DataFrame(), skipped
    variants, extra = read_tsv(path)
    return variants, extra, 0


def write_scores(df: pd.DataFrame, path: str | Path) -> None:
    """Write a scored variant table as tab-separated values."""
    df.to_csv(path, sep="\t", index=False)
