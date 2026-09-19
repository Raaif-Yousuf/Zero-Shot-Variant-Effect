"""Stream the ClinVar GRCh37 VCF into a tidy, labelled chr17 SNV table.

The VCF is read line by line and never loaded whole: files are hundreds of
megabytes compressed and several gigabytes uncompressed.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

#: CLNSIG values kept as positive (pathogenic) / negative (benign) labels.
PATHOGENIC_CLNSIG = {"Pathogenic", "Likely_pathogenic", "Pathogenic/Likely_pathogenic"}
BENIGN_CLNSIG = {"Benign", "Likely_benign", "Benign/Likely_benign"}

#: CLNREVSTAT values with at least one star (some review evidence behind the call).
STARRED_CLNREVSTAT = {
    "criteria_provided,_single_submitter",
    "criteria_provided,_multiple_submitters,_no_conflicts",
    "reviewed_by_expert_panel",
    "practice_guideline",
}

#: MC Sequence Ontology term -> coarse consequence bucket, in priority order when
#: a record lists more than one term (most to least severe).
_CONSEQUENCE_PRIORITY = (
    ("splice_acceptor_variant", "splice"),
    ("splice_donor_variant", "splice"),
    ("nonsense", "nonsense"),
    ("missense_variant", "missense"),
    ("synonymous_variant", "synonymous"),
    ("5_prime_UTR_variant", "UTR"),
    ("3_prime_UTR_variant", "UTR"),
    ("intron_variant", "intronic"),
)

COLUMN_ORDER = ["chrom", "pos", "ref", "alt", "id", "gene", "consequence", "clinvar_id", "label"]


def _parse_info(info: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for item in info.split(";"):
        key, sep, value = item.partition("=")
        if sep:
            fields[key] = value
    return fields


def _coarse_consequence(mc: str | None) -> str:
    if not mc:
        return "other"
    terms = {part.split("|", 1)[1] for part in mc.split(",") if "|" in part}
    for so_term, coarse in _CONSEQUENCE_PRIORITY:
        if so_term in terms:
            return coarse
    return "other"


def _gene_from_geneinfo(geneinfo: str) -> str:
    if not geneinfo:
        return ""
    return geneinfo.split("|", 1)[0].split(":", 1)[0]


def _label(clnsig: str) -> float:
    if clnsig in PATHOGENIC_CLNSIG:
        return 1.0
    if clnsig in BENIGN_CLNSIG:
        return 0.0
    return float("nan")


def iter_vcf_records(
    vcf_gz_path: Path | str, chrom: str
) -> Iterator[tuple[int, str, str, str, dict[str, str]]]:
    """Yield ``(pos, id, ref, alt, info_fields)`` for one chromosome, streaming the gzip VCF."""
    with gzip.open(vcf_gz_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t", 8)
            if fields[0] != chrom:
                continue
            pos, vid, ref, alt, info = fields[1], fields[2], fields[3], fields[4], fields[7]
            yield int(pos), vid, ref, alt, _parse_info(info)


def build_clinvar_table(
    vcf_gz_path: Path | str, chrom: str = "17"
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Filter one chromosome's SNVs down to labelled, adequately-reviewed variants.

    Returns the tidy DataFrame plus an ordered dict of row counts remaining after
    each filter step, for reporting what was dropped and why.
    """
    counts = {
        "chr17_records": 0,
        "biallelic_snv": 0,
        "clnsig_pathogenic_or_benign": 0,
        "starred_clnrevstat": 0,
    }
    rows: list[dict[str, object]] = []
    for pos, vid, ref, alt, info in iter_vcf_records(vcf_gz_path, chrom):
        counts["chr17_records"] += 1
        if len(ref) != 1 or len(alt) != 1 or ref not in "ACGT" or alt not in "ACGT":
            continue
        counts["biallelic_snv"] += 1

        label = _label(info.get("CLNSIG", ""))
        if np.isnan(label):
            continue
        counts["clnsig_pathogenic_or_benign"] += 1

        if info.get("CLNREVSTAT", "") not in STARRED_CLNREVSTAT:
            continue
        counts["starred_clnrevstat"] += 1

        rows.append(
            {
                "chrom": f"chr{chrom}",
                "pos": pos,
                "ref": ref,
                "alt": alt,
                "id": vid,
                "gene": _gene_from_geneinfo(info.get("GENEINFO", "")),
                "consequence": _coarse_consequence(info.get("MC")),
                "clinvar_id": vid,
                "label": label,
            }
        )
    df = pd.DataFrame(rows, columns=COLUMN_ORDER)
    return df, counts


def subsample_by_label(df: pd.DataFrame, n_per_label: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Take up to ``n_per_label`` rows per label value, sampled without replacement."""
    rng = np.random.default_rng(seed)
    parts = []
    for _label_value, group in df.groupby("label", sort=True):
        idx = group.index.to_numpy()
        if len(idx) > n_per_label:
            idx = np.sort(rng.choice(idx, size=n_per_label, replace=False))
        parts.append(df.loc[idx])
    if not parts:
        return df.iloc[0:0]
    return pd.concat(parts).sort_values(["chrom", "pos"]).reset_index(drop=True)


def write_filter_counts(counts: dict[str, int], path: Path | str) -> None:
    Path(path).write_text(json.dumps(counts, indent=2) + "\n")
