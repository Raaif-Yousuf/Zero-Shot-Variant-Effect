# Usage

## Install

```
uv sync                          # core: engine, kmer baseline, CLI
uv sync --extra models           # + HyenaDNA / Nucleotide Transformer (CPU torch, transformers<5)
uv sync --extra bench            # + benchmark data prep and plotting (matplotlib, openpyxl)
```

Extras can be combined, e.g. `uv sync --extra models --extra bench`. Run
commands with `uv run ...` so they use the project's virtual environment.

Before loading any pretrained model, set `HF_HOME` to a short path (the
default Hugging Face cache path is long enough to break `trust_remote_code`
on Windows) and silence the symlink warning:

```
export HF_HOME=C:/Users/<you>/.cache/vep-hf
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
```

## Listing available scorers

```
uv run zsvep models
```

Prints each registry name and its `max_window` in bases (`kmer`'s is
unbounded).

## Scoring variants

```
# VCF/VCF.gz input, the order-k Markov baseline
uv run zsvep score \
  --variants clinvar_chr17.vcf.gz \
  --fasta chr17.fa.gz \
  --model kmer \
  --window 1024 \
  --out scores.tsv

# TSV input, a pretrained model, forward strand only, a size cap while testing
uv run zsvep score \
  --variants data/processed/clinvar_chr17_subsample.tsv \
  --fasta chr17.fa.gz \
  --model hyenadna-tiny-1k \
  --window 1024 \
  --strands forward \
  --limit 50 \
  --out scores.tsv
```

`--model` is a name from `zsvep models`. `--window` defaults to 1024 and must
not exceed the scorer's `max_window`. `--strands` is `both` (default, forward
and reverse complement, `llr` is their mean) or `forward` (`llr` equals
`llr_fwd`, `llr_rev` is left `NaN`). `--limit N` scores only the first `N`
variants, for a quick smoke test before a full run.

`--mode {full,site}` and `--threads N` only apply to the HyenaDNA scorers
(`full`/`site` mode, see `docs/methods.md`; `--threads` calls
`torch.set_num_threads`). `--batch-size` is accepted and forwarded to model
scorers but is not currently used for scoring (the engine already batches
every alt at a position, and ref plus every alt, into one forward pass per
position per strand). The CLI only forwards a flag to a scorer if that
scorer's constructor actually declares the matching parameter, so passing
`--mode` while scoring with `kmer` or `nt-v2-50m` is harmless: it is simply
not passed through.

### Input formats (auto-detected by extension)

- **VCF / VCF.gz**: standard `CHROM POS ID REF ALT ...` columns. Only SNVs are
  scored: multi-allelic `ALT` fields are split into one row per allele,
  indels and symbolic alleles (and any record whose `REF` is not a single
  A/C/G/T base) are skipped and counted, and the count of skipped alleles is
  printed to stderr. `ID` is kept.
- **TSV**: a header row with at least `chrom`, `pos`, `ref`, `alt` (matched
  case-insensitively); an optional `id` column feeds the output `id` column.
  Any other columns (labels, scores from other tools, etc.) are carried
  through unchanged into the output, aligned by row.

### Output columns

`chrom, pos, ref, alt, id, llr, llr_fwd, llr_rev, status`, followed by
whatever extra columns the input TSV had (VCF input has none).
`status` is `"ok"` for a scored variant or `"ref_mismatch"` when the input's
`ref` did not match the reference FASTA at that position (in which case
`llr`/`llr_fwd`/`llr_rev` are all empty).

## Caching

Two independent caches make repeated runs fast:

- **Score cache**: every `(scorer, window, strand, exact window sequence,
  center, alt)` combination is looked up in a SQLite database before calling
  the scorer, and written back after. Re-running the same `zsvep score`
  command (even after an interruption) reuses every score already computed
  and only computes what is missing.
- **`kmer` training cache**: `KmerMarkovScorer` also caches its trained
  (k+1)-mer counts as a `.npz` file, keyed on the reference FASTA's path,
  size and modification time plus the Markov order and the chromosomes
  trained on, so it does not retrain from scratch on every run.

Both default to `.cache/zsvep/` relative to the current working directory
(already gitignored). `--cache-dir DIR` moves both caches to `DIR` instead.
`--no-cache` disables both: nothing is read from or written to either cache,
and `kmer` retrains from scratch every run.

## Reproducing the benchmark

1. `scripts/download_data.py` downloads and checksum-verifies the raw
   Findlay, hg19 chr17, ClinVar and phyloP100way files (see `docs/data.md`).
2. `scripts/prepare_datasets.py` turns those into the tidy `brca1_sge.tsv`
   and `clinvar_chr17.tsv` (plus a seeded subsample) benchmark tables,
   verifying every `ref` allele against hg19 chr17 along the way.
3. `scripts/run_benchmark.py` scores those tables with each registered
   scorer.
4. `scripts/summarize_results.py` turns the scored tables into the metrics
   tables and figures (`evaluate.py`, `plots.py`).
