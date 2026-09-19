"""Build tidy benchmark TSVs from the downloaded raw data files.

Usage:
    uv run scripts/prepare_datasets.py --data-dir path/to/data
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from zeroshot_vep.datasets._local_fasta import LocalChromFasta, verify_ref_alleles
from zeroshot_vep.datasets.clinvar import (
    build_clinvar_table,
    subsample_by_label,
    write_filter_counts,
)
from zeroshot_vep.datasets.findlay import load_findlay_table
from zeroshot_vep.datasets.phylop import lookup_phylop

logger = logging.getLogger(__name__)


def _add_phylop100way(df: pd.DataFrame, wigfix_path: Path) -> pd.DataFrame:
    scores = lookup_phylop(wigfix_path, df["pos"].tolist())
    df = df.copy()
    df["phylop100way"] = df["pos"].map(scores)
    return df


def prepare_findlay(data_dir: Path) -> None:
    raw = data_dir / "raw"
    processed = data_dir / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    df = load_findlay_table(raw / "findlay2018_supp_table1.xlsx")
    n_loaded = len(df)

    fasta = LocalChromFasta(raw / "chr17.fa.gz")
    df, n_mismatch = verify_ref_alleles(df, fasta)
    logger.info("findlay: loaded %d rows, %d ref mismatches dropped", n_loaded, n_mismatch)

    df = _add_phylop100way(df, raw / "chr17.phyloP100way.wigFix.gz")
    n_phylop = int(df["phylop100way"].notna().sum())

    out_path = processed / "brca1_sge.tsv"
    df.to_csv(out_path, sep="\t", index=False)
    print(
        f"brca1_sge: loaded {n_loaded}, {n_mismatch} ref mismatches dropped, "
        f"{len(df)} rows written, phyloP100way coverage {n_phylop}/{len(df)} -> {out_path}"
    )


def prepare_clinvar(data_dir: Path, subsample_n: int, seed: int) -> None:
    raw = data_dir / "raw"
    processed = data_dir / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    df, counts = build_clinvar_table(raw / "clinvar_20260905.vcf.gz", chrom="17")
    counts["parsed_labelled_starred"] = len(df)

    fasta = LocalChromFasta(raw / "chr17.fa.gz")
    df, n_mismatch = verify_ref_alleles(df, fasta)
    counts["ref_verified"] = len(df)
    logger.info("clinvar: %d ref mismatches dropped", n_mismatch)

    df = _add_phylop100way(df, raw / "chr17.phyloP100way.wigFix.gz")
    n_phylop = int(df["phylop100way"].notna().sum())

    out_path = processed / "clinvar_chr17.tsv"
    df.to_csv(out_path, sep="\t", index=False)

    sub = subsample_by_label(df, n_per_label=subsample_n, seed=seed)
    sub_path = processed / "clinvar_chr17_subsample.tsv"
    sub.to_csv(sub_path, sep="\t", index=False)
    counts["subsample"] = len(sub)

    counts_path = processed / "clinvar_chr17_filter_counts.json"
    write_filter_counts(counts, counts_path)

    label_counts = df["label"].value_counts().to_dict()
    print(
        f"clinvar_chr17: {len(df)} rows ({n_mismatch} ref mismatches dropped), "
        f"phyloP100way coverage {n_phylop}/{len(df)}, labels {label_counts} -> {out_path}"
    )
    print(f"clinvar_chr17_subsample: {len(sub)} rows -> {sub_path}")
    print(f"filter counts -> {counts_path}: {json.dumps(counts)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"), help="base data directory (default: ./data)"
    )
    parser.add_argument(
        "--only",
        choices=["findlay", "clinvar"],
        action="append",
        default=None,
        help="restrict to one dataset (repeatable); default builds both",
    )
    parser.add_argument(
        "--subsample-n",
        type=int,
        default=1000,
        help="max ClinVar rows per label in the subsample (default: 1000)",
    )
    parser.add_argument("--seed", type=int, default=0, help="subsample random seed (default: 0)")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s"
    )

    targets = args.only or ["findlay", "clinvar"]
    if "findlay" in targets:
        prepare_findlay(args.data_dir)
    if "clinvar" in targets:
        prepare_clinvar(args.data_dir, args.subsample_n, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
