import pandas as pd

from zeroshot_vep.cli import main


def _write_fasta(tmp_path, seq, chrom="chr1"):
    path = tmp_path / "ref.fa"
    path.write_text(f">{chrom}\n{seq}\n")
    return path


def _write_tsv(tmp_path, rows):
    path = tmp_path / "variants.tsv"
    header = "chrom\tpos\tref\talt\tid\n"
    body = "".join(f"{c}\t{p}\t{r}\t{a}\t{i}\n" for c, p, r, a, i in rows)
    path.write_text(header + body)
    return path


def test_models_command_lists_kmer(capsys):
    rc = main(["models"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "kmer" in out


def test_score_end_to_end_with_kmer(tmp_path):
    seq = "ACGTACGTACGTACGTACGTACGTACGTACGT" * 4  # 132 bases
    fasta = _write_fasta(tmp_path, seq)
    ref_base = seq[19]  # 1-based pos 20 -> 0-based idx 19
    alt_base = "G" if ref_base != "G" else "A"
    variants_path = _write_tsv(tmp_path, [("chr1", 20, ref_base, alt_base, "v1")])
    out_path = tmp_path / "scores.tsv"

    rc = main(
        [
            "score",
            "--variants",
            str(variants_path),
            "--fasta",
            str(fasta),
            "--model",
            "kmer",
            "--window",
            "20",
            "--strands",
            "both",
            "--no-cache",
            "--out",
            str(out_path),
        ]
    )

    assert rc == 0
    result = pd.read_csv(out_path, sep="\t")
    assert len(result) == 1
    assert result.loc[0, "status"] == "ok"
    assert "llr" in result.columns
    assert "llr_fwd" in result.columns
    assert "llr_rev" in result.columns


def test_score_defaults_kmer_cache_to_default_cache_dir(tmp_path, monkeypatch):
    # Without --cache-dir or --no-cache, the kmer scorer's own npz training
    # cache should still land under the default .cache/zsvep (cwd-relative),
    # the same place the score cache would use.
    monkeypatch.chdir(tmp_path)
    seq = "ACGTACGTACGTACGTACGTACGTACGTACGT" * 4
    fasta = _write_fasta(tmp_path, seq)
    ref_base = seq[19]
    alt_base = "G" if ref_base != "G" else "A"
    variants_path = _write_tsv(tmp_path, [("chr1", 20, ref_base, alt_base, "v1")])
    out_path = tmp_path / "scores.tsv"

    rc = main(
        [
            "score",
            "--variants",
            str(variants_path),
            "--fasta",
            str(fasta),
            "--model",
            "kmer",
            "--window",
            "20",
            "--out",
            str(out_path),
        ]
    )

    assert rc == 0
    npz_files = list((tmp_path / ".cache" / "zsvep").glob("*.npz"))
    assert len(npz_files) == 1
    assert (tmp_path / ".cache" / "zsvep" / "scores.sqlite").exists()


def test_score_reports_error_for_unknown_model(tmp_path):
    seq = "ACGT" * 10
    fasta = _write_fasta(tmp_path, seq)
    variants_path = _write_tsv(tmp_path, [("chr1", 5, "A", "G", "v1")])
    out_path = tmp_path / "scores.tsv"

    rc = main(
        [
            "score",
            "--variants",
            str(variants_path),
            "--fasta",
            str(fasta),
            "--model",
            "nope",
            "--no-cache",
            "--out",
            str(out_path),
        ]
    )
    assert rc != 0


def test_score_carries_extra_tsv_columns(tmp_path):
    seq = "ACGTACGTACGTACGTACGTACGTACGTACGT" * 4
    fasta = _write_fasta(tmp_path, seq)
    ref_base = seq[19]
    alt_base = "G" if ref_base != "G" else "A"
    path = tmp_path / "variants.tsv"
    path.write_text(
        f"chrom\tpos\tref\talt\tid\tlabel\nchr1\t20\t{ref_base}\t{alt_base}\tv1\tpathogenic\n"
    )
    out_path = tmp_path / "scores.tsv"

    rc = main(
        [
            "score",
            "--variants",
            str(path),
            "--fasta",
            str(fasta),
            "--model",
            "kmer",
            "--window",
            "20",
            "--no-cache",
            "--out",
            str(out_path),
        ]
    )
    assert rc == 0
    result = pd.read_csv(out_path, sep="\t")
    assert result.loc[0, "label"] == "pathogenic"
