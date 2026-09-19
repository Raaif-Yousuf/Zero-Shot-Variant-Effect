# Data

All raw inputs are pinned in `src/zeroshot_vep/data/manifest.py` and fetched
with `scripts/download_data.py`, which verifies each file's sha256 before
using it and refuses to proceed on a mismatch. `scripts/prepare_datasets.py`
turns the raw downloads into the tidy TSVs used for benchmarking.

## Raw files

| name | source | version / date | sha256 | size |
| --- | --- | --- | --- | --- |
| `findlay2018_supp_table1.xlsx` | [Findlay et al. 2018 Nature, Supplementary Table 1](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-018-0461-z/MediaObjects/41586_2018_461_MOESM3_ESM.xlsx) | as published, 2018 | `e9aa4186b8a5de91d61059d03f8dac1e5573d1e2802f50a6581c3053d22b923a` | 2,306,341 bytes |
| `chr17.fa.gz` | [UCSC hg19 (GRCh37) chromosome 17, plain gzip FASTA](https://hgdownload.soe.ucsc.edu/goldenPath/hg19/chromosomes/chr17.fa.gz) | hg19 assembly | `27b909064ce4470d2655a16b3e30c9987be39aa5ed8e97f6b43e66b8e01bf6b3` | 25,139,792 bytes |
| `clinvar_20260905.vcf.gz` | [NCBI ClinVar, GRCh37 weekly archive](https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh37/archive_2.0/2026/clinvar_20260905.vcf.gz) | archived 2026-09-05 | `68a479604a683df6c2431a68f1d3150922f872d8c33885fa2ad2d1fe333692f9` | 193,451,062 bytes |
| `chr17.phyloP100way.wigFix.gz` | [UCSC hg19 100-way phyloP, chromosome 17](https://hgdownload.soe.ucsc.edu/goldenPath/hg19/phyloP100way/hg19.100way.phyloP100way/chr17.phyloP100way.wigFix.gz) | 100-way track | `1fe9dad164fe271837a4f9c8c6970a914f10e4dcdac65387d94d603c24691234` | 150,943,228 bytes |

Each sha256 was cross-checked once, when it was first pinned, against the
provider's own checksum (UCSC `md5sum.txt`, NCBI's `.md5` file); routine
downloads only ever check the pinned sha256 (see `manifest.py` for the exact
md5 each one was checked against).

## BRCA1 saturation genome editing (Findlay et al. 2018)

`prepare_findlay` (`scripts/prepare_datasets.py`) reads Supplementary Table 1,
keeps one row per SNV with its genomic (forward-strand, hg19) `ref`/`alt`, and
labels it from `func.class`:

- `LOF` (loss-of-function) -> `label = 1` (823 SNVs)
- `FUNC` (functional) -> `label = 0` (2,821 SNVs)
- `INT` (intermediate) -> `label = NaN`, excluded from the LOF-vs-FUNC
  evaluation (249 SNVs)

3,893 SNVs total. Every `ref` allele was checked against hg19 chr17
(`chr17.fa.gz`); 0 mismatches. phyloP100way coverage (looked up from the
100-way track, independent of the paper's own `phyloP (mammalian)` column)
is 3,893/3,893. Output: `data/processed/brca1_sge.tsv`.

## ClinVar chr17 (GRCh37, 2026-09-05 archive)

`prepare_clinvar` streams the VCF (it is never loaded whole) and applies,
in order:

1. chromosome 17 records: 270,197
2. biallelic SNV (single-base REF and ALT, both A/C/G/T): 245,899
3. `CLNSIG` is one of Pathogenic / Likely_pathogenic /
   Pathogenic_or_Likely_pathogenic (label 1) or Benign / Likely_benign /
   Benign_or_Likely_benign (label 0): 87,972
4. `CLNREVSTAT` has at least one star (criteria provided by a single or
   multiple non-conflicting submitters, reviewed by an expert panel, or a
   practice guideline): 84,827

Every remaining `ref` allele was checked against hg19 chr17; 0 mismatches, so
84,827 rows are written to `data/processed/clinvar_chr17.tsv` (72,837 benign,
11,990 pathogenic). phyloP100way coverage is 84,825/84,827 (two positions
fall in a gap between fixedStep blocks in the track). A seeded
(`seed=0`), stratified subsample of up to 1,000 rows per label
(`clinvar_chr17_subsample.tsv`) gives 2,000 rows (1,000 benign, 1,000
pathogenic; ClinVar has far more than 1,000 of either label on chr17, so both
are capped). Per-step counts are also written to
`clinvar_chr17_filter_counts.json`.

## Licenses and redistribution

- **Findlay et al. 2018**: the publisher's supplementary table is downloaded
  directly (see the manifest above); cite the paper. The same scores are
  also archived on [MaveDB](https://www.mavedb.org/) as
  `urn:mavedb:00000097-0-2` under a CC0 (public domain) license (Esposito et
  al. 2019).
- **ClinVar**: produced by NCBI, a U.S. government agency; ClinVar data is in
  the public domain.
- **UCSC Genome Browser downloads** (hg19 chr17 FASTA, phyloP100way track):
  free for academic, nonprofit and personal use under UCSC's conditions of
  use; commercial use needs a separate license, and UCSC asks that the
  browser or its data be cited when used in a publication.

This project does not redistribute any of these raw files; `manifest.py`
only records where to fetch them and how to verify them.
