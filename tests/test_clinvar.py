"""Tests for streaming ClinVar VCF parsing, against a tiny synthetic VCF."""

from __future__ import annotations

import gzip
import json

import pandas as pd

from zeroshot_vep.datasets.clinvar import (
    build_clinvar_table,
    subsample_by_label,
    write_filter_counts,
)

_HEADER = "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"

_LINES = [
    # 1. pathogenic, single_submitter -> kept, label 1, missense
    "17\t100\t1\tA\tG\t.\t.\tCLNSIG=Pathogenic;CLNREVSTAT=criteria_provided,_single_submitter;"
    "GENEINFO=BRCA1:672;MC=SO:0001583|missense_variant\n",
    # 2. benign, multiple_submitters_no_conflicts -> kept, label 0, synonymous
    "17\t200\t2\tC\tT\t.\t.\tCLNSIG=Benign;"
    "CLNREVSTAT=criteria_provided,_multiple_submitters,_no_conflicts;GENEINFO=BRCA1:672;"
    "MC=SO:0001819|synonymous_variant\n",
    # 3. uncertain significance -> dropped (unlabelled)
    "17\t300\t3\tG\tA\t.\t.\tCLNSIG=Uncertain_significance;"
    "CLNREVSTAT=criteria_provided,_single_submitter;GENEINFO=BRCA1:672;MC=SO:0001583|missense_variant\n",
    # 4. pathogenic but not starred -> dropped
    "17\t400\t4\tT\tC\t.\t.\tCLNSIG=Pathogenic;CLNREVSTAT=no_assertion_criteria_provided;"
    "GENEINFO=BRCA1:672\n",
    # 5. indel -> dropped (not a SNV)
    "17\t500\t5\tA\tATT\t.\t.\tCLNSIG=Pathogenic;CLNREVSTAT=reviewed_by_expert_panel;"
    "GENEINFO=BRCA1:672\n",
    # 6. wrong chromosome -> dropped by the chrom filter
    "1\t600\t6\tA\tG\t.\t.\tCLNSIG=Pathogenic;CLNREVSTAT=reviewed_by_expert_panel;"
    "GENEINFO=OR4F5:79501\n",
    # 7. multi-allelic ALT -> dropped (not single base)
    "17\t700\t7\tA\tG,T\t.\t.\tCLNSIG=Pathogenic;CLNREVSTAT=reviewed_by_expert_panel;"
    "GENEINFO=BRCA1:672\n",
    # 8. pathogenic, expert panel, two MC terms -> splice wins consequence priority
    "17\t800\t8\tG\tC\t.\t.\tCLNSIG=Pathogenic;CLNREVSTAT=reviewed_by_expert_panel;"
    "GENEINFO=BRCA1:672;MC=SO:0001583|missense_variant,SO:0001574|splice_acceptor_variant\n",
    # 9. likely pathogenic, no MC field -> coarse consequence "other"
    "17\t900\t9\tT\tA\t.\t.\tCLNSIG=Likely_pathogenic;CLNREVSTAT=criteria_provided,_single_submitter;"
    "GENEINFO=BRCA1:672\n",
]


def _write_vcf(path):
    with gzip.open(path, "wt") as fh:
        fh.write(_HEADER)
        fh.writelines(_LINES)


def test_build_clinvar_table_filters_and_labels(tmp_path):
    path = tmp_path / "test.vcf.gz"
    _write_vcf(path)

    df, counts = build_clinvar_table(path, chrom="17")

    assert counts["chr17_records"] == 8  # every line except the chr1 one
    assert counts["biallelic_snv"] == 6  # excludes the indel and the multi-allelic record
    assert counts["clnsig_pathogenic_or_benign"] == 5  # excludes Uncertain_significance
    assert counts["starred_clnrevstat"] == 4  # excludes no_assertion_criteria_provided

    assert set(df["pos"]) == {100, 200, 800, 900}
    assert df.loc[df["pos"] == 100, "label"].iloc[0] == 1.0
    assert df.loc[df["pos"] == 200, "label"].iloc[0] == 0.0
    assert df.loc[df["pos"] == 100, "consequence"].iloc[0] == "missense"
    assert df.loc[df["pos"] == 200, "consequence"].iloc[0] == "synonymous"
    assert df.loc[df["pos"] == 800, "consequence"].iloc[0] == "splice"
    assert df.loc[df["pos"] == 900, "consequence"].iloc[0] == "other"
    assert df.loc[df["pos"] == 100, "gene"].iloc[0] == "BRCA1"
    assert (df["chrom"] == "chr17").all()


def test_subsample_by_label_is_seeded_and_deterministic(tmp_path):
    path = tmp_path / "test.vcf.gz"
    _write_vcf(path)
    df, _ = build_clinvar_table(path, chrom="17")

    sub1 = subsample_by_label(df, n_per_label=1, seed=0)
    sub2 = subsample_by_label(df, n_per_label=1, seed=0)

    assert len(sub1) == 2  # one row per label, both labels have >= 1 candidate
    pd.testing.assert_frame_equal(sub1.reset_index(drop=True), sub2.reset_index(drop=True))


def test_write_filter_counts(tmp_path):
    counts = {"a": 1, "b": 2}
    out = tmp_path / "counts.json"
    write_filter_counts(counts, out)
    assert json.loads(out.read_text()) == counts
