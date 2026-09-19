import gzip

import pandas as pd
import pytest

from zeroshot_vep.io import read_tsv, read_variants, read_vcf, write_scores
from zeroshot_vep.variant import Variant

VCF_HEADER = "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"


def test_read_vcf_splits_multiallelic_and_skips_indels(tmp_path):
    body = (
        "chr17\t100\trs1\tA\tG,T\t.\tPASS\t.\n"
        "chr17\t200\t.\tAT\tA\t.\tPASS\t.\n"
        "chr17\t300\t.\tC\t<DEL>\t.\tPASS\t.\n"
        "chr17\t400\trs2\tG\tC\t.\tPASS\t.\n"
    )
    path = tmp_path / "variants.vcf"
    path.write_text(VCF_HEADER + body)

    variants, skipped = read_vcf(path)

    assert [v.pos for v in variants] == [100, 100, 400]
    assert variants[0] == Variant("chr17", 100, "A", "G", "rs1")
    assert variants[1] == Variant("chr17", 100, "A", "T", "rs1")
    assert variants[2] == Variant("chr17", 400, "G", "C", "rs2")
    assert skipped == 2


def test_read_vcf_gz(tmp_path):
    body = VCF_HEADER + "chr17\t100\t.\tA\tG\t.\tPASS\t.\n"
    path = tmp_path / "variants.vcf.gz"
    path.write_bytes(gzip.compress(body.encode()))

    variants, skipped = read_vcf(path)
    assert len(variants) == 1
    assert skipped == 0


def test_read_tsv_carries_extra_columns(tmp_path):
    path = tmp_path / "variants.tsv"
    path.write_text(
        "chrom\tpos\tref\talt\tid\tlabel\n"
        "chr17\t100\tA\tG\trs1\tpathogenic\n"
        "chr17\t200\tC\tT\t.\tbenign\n"
    )
    variants, extra = read_tsv(path)
    assert variants[0] == Variant("chr17", 100, "A", "G", "rs1")
    assert variants[1] == Variant("chr17", 200, "C", "T", ".")
    assert list(extra["label"]) == ["pathogenic", "benign"]
    assert "chrom" not in extra.columns


def test_read_tsv_missing_column_raises(tmp_path):
    path = tmp_path / "bad.tsv"
    path.write_text("chrom\tpos\tref\nchr17\t1\tA\n")
    with pytest.raises(ValueError):
        read_tsv(path)


def test_read_variants_autodetects_by_extension(tmp_path):
    vcf_path = tmp_path / "v.vcf"
    vcf_path.write_text(VCF_HEADER + "chr17\t100\t.\tA\tG\t.\tPASS\t.\n")
    variants, extra, skipped = read_variants(vcf_path)
    assert len(variants) == 1
    assert extra.empty
    assert skipped == 0

    tsv_path = tmp_path / "v.tsv"
    tsv_path.write_text("chrom\tpos\tref\talt\nchr17\t100\tA\tG\n")
    variants, extra, skipped = read_variants(tsv_path)
    assert len(variants) == 1
    assert skipped == 0


def test_write_scores(tmp_path):
    df = pd.DataFrame({"chrom": ["chr17"], "pos": [100], "llr": [0.5]})
    out = tmp_path / "out.tsv"
    write_scores(df, out)
    read_back = pd.read_csv(out, sep="\t")
    assert read_back["chrom"].iloc[0] == "chr17"
    assert read_back["llr"].iloc[0] == 0.5
