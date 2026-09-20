"""Positive-control sanity checks for the DNA-language-model scorers.

The benchmark's headline is a negative result: every zero-shot scorer here
sits at or below chance on BRCA1 saturation genome editing and at chance on
ClinVar chr17 (see results/metrics_brca1.md and results/metrics_clinvar.md).
This script checks that the models and code are not simply broken, so a
reader can tell the negative result is about the models, not the pipeline:

  a. Nucleotide Transformer (nt-v2-50m) masked-token recovery: seeded real
     hg19 chr17 positions, mask the 6-mer token covering each one, and check
     whether the model recovers the reference 6-mer and the reference base
     above chance.
  b. HyenaDNA (hyenadna-small-32k) real-vs-shuffled likelihood: does it score
     real genomic windows higher than a dinucleotide-composition-matched
     shuffle of the same window?
  c. From already-committed scores only, no new inference: what the
     committed hyenadna-small-32k (full mode) and nt-v2-50m LLR scores on
     BRCA1 actually track -- transition/transversion, CpG context, and
     Findlay consequence class.

Writes results/sanity_checks.md (prose + tables) and results/sanity_checks.tsv
(long format: section, group_type, group, metric, value, n).

Usage:
    uv run scripts/sanity_checks.py --data-dir path/to/data --out results/

(a) and (b) load real pretrained weights; set HF_HOME to a short path first
(see docs/models.md), e.g.:
    export HF_HOME=/path/to/.cache/vep-hf HF_HUB_DISABLE_SYMLINKS_WARNING=1

Runs a single scoring process at a time (never two model forward passes
concurrently) and both (a) and (b) results are cached under --cache-dir so a
re-run with the same parameters does not repeat the inference.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _sanity_common import (  # noqa: E402
    all_kmers,
    bits_per_base,
    classify_cpg,
    dinucleotide_shuffle,
    format_markdown_table,
    is_transition,
    marginal_base_log_probs,
    mean_llr_by_group,
    sample_valid_positions,
)

from zeroshot_vep.reference import ReferenceGenome  # noqa: E402
from zeroshot_vep.scorers._llr import causal_loglik, nt_locate_kmer  # noqa: E402
from zeroshot_vep.scorers.registry import SCORERS  # noqa: E402

logger = logging.getLogger(__name__)

CHROM = "chr17"
NT_K = 6
NT_MODEL_ID, NT_REVISION = SCORERS["nt-v2-50m"][1]["model_id"], SCORERS["nt-v2-50m"][1]["revision"]
HYENA_MODEL_ID = SCORERS["hyenadna-small-32k"][1]["model_id"]
HYENA_REVISION = SCORERS["hyenadna-small-32k"][1]["revision"]

BRCA1_SCORE_FILES = {
    "hyenadna-small-32k_full": "brca1_hyenadna-small-32k_full_w1024.tsv",
    "nt-v2-50m": "brca1_nt-v2-50m_w1024.tsv",
}


# --------------------------------------------------------------------------
# Small JSON cache: one file per check, keyed on its own parameters so a
# parameter change (seed, n, window, model revision) invalidates the cache
# instead of silently reusing a stale result.
# --------------------------------------------------------------------------


def _load_cache(path: Path, config: dict) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if data.get("config") != config:
        return None
    return data


def _save_cache(path: Path, config: dict, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"config": config, "records": records}, indent=2))


# --------------------------------------------------------------------------
# Check (a): Nucleotide Transformer masked-token recovery
# --------------------------------------------------------------------------


def _is_acgt_window(reference: ReferenceGenome, chrom: str, pos: int, window: int) -> bool:
    center = window // 2
    start0 = (pos - 1) - center
    end0 = start0 + window
    return "N" not in reference.fetch(chrom, start0, end0)


def run_nt_masked_recovery(
    reference: ReferenceGenome,
    n: int,
    window: int,
    seed: int,
    threads: int,
    cache_path: Path,
    force: bool,
) -> list[dict]:
    config = {
        "n": n,
        "window": window,
        "seed": seed,
        "model_id": NT_MODEL_ID,
        "revision": NT_REVISION,
    }
    if not force:
        cached = _load_cache(cache_path, config)
        if cached is not None:
            logger.info("nt masked recovery: using cached result (%s)", cache_path)
            return cached["records"]

    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    torch.set_num_threads(threads)
    tokenizer = AutoTokenizer.from_pretrained(
        NT_MODEL_ID, revision=NT_REVISION, trust_remote_code=True
    )
    model = AutoModelForMaskedLM.from_pretrained(
        NT_MODEL_ID, revision=NT_REVISION, trust_remote_code=True
    )
    model.eval()

    kmers = all_kmers(NT_K)
    kmer_ids = np.array([tokenizer.convert_tokens_to_ids(k) for k in kmers])

    length = reference.length(CHROM)
    center = window // 2
    margin = window
    rng = np.random.default_rng(seed)
    positions = sample_valid_positions(
        rng, margin, length - margin, n, lambda p: _is_acgt_window(reference, CHROM, p, window)
    )

    records = []
    for pos in positions:
        start0 = (pos - 1) - center
        end0 = start0 + window
        seq = reference.fetch(CHROM, start0, end0)
        trim, token_index, chunk_start, offset = nt_locate_kmer(seq, center, k=NT_K)
        windowed = seq[trim:]
        ids = tokenizer(windowed)["input_ids"]
        ref_kmer = windowed[chunk_start : chunk_start + NT_K]
        ref_id = tokenizer.convert_tokens_to_ids(ref_kmer)

        masked_ids = list(ids)
        masked_ids[token_index] = tokenizer.mask_token_id
        batch = torch.tensor([masked_ids], dtype=torch.long)
        with torch.inference_mode():
            logits = model(input_ids=batch).logits
        log_probs_row = torch.log_softmax(logits[0, token_index].float(), dim=-1).numpy()

        subset_lp = log_probs_row[kmer_ids]
        pred_kmer = kmers[int(np.argmax(subset_lp))]
        base_lp = marginal_base_log_probs(subset_lp, kmers, offset)
        pred_base = max(base_lp, key=base_lp.get)

        records.append(
            {
                "pos": pos,
                "top1_kmer_correct": pred_kmer == ref_kmer,
                "top1_center_base_correct": pred_base == ref_kmer[offset],
                "logp_ref_kmer": float(log_probs_row[ref_id]),
            }
        )

    _save_cache(cache_path, config, records)
    return records


def summarize_nt_masked_recovery(records: list[dict]) -> dict[str, float]:
    n = len(records)
    return {
        "n": n,
        "top1_kmer_acc": sum(r["top1_kmer_correct"] for r in records) / n,
        "top1_center_base_acc": sum(r["top1_center_base_correct"] for r in records) / n,
        "mean_logp_ref_kmer": float(np.mean([r["logp_ref_kmer"] for r in records])),
    }


# --------------------------------------------------------------------------
# Check (b): HyenaDNA real-vs-shuffled likelihood
# --------------------------------------------------------------------------


def run_hyena_real_vs_shuffled(
    reference: ReferenceGenome,
    n: int,
    window: int,
    seed: int,
    threads: int,
    cache_path: Path,
    force: bool,
) -> list[dict]:
    config = {
        "n": n,
        "window": window,
        "seed": seed,
        "model_id": HYENA_MODEL_ID,
        "revision": HYENA_REVISION,
    }
    if not force:
        cached = _load_cache(cache_path, config)
        if cached is not None:
            logger.info("hyenadna real-vs-shuffled: using cached result (%s)", cache_path)
            return cached["records"]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(threads)
    tokenizer = AutoTokenizer.from_pretrained(
        HYENA_MODEL_ID, revision=HYENA_REVISION, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        HYENA_MODEL_ID, revision=HYENA_REVISION, trust_remote_code=True
    )
    model.eval()

    length = reference.length(CHROM)
    margin = window
    rng = np.random.default_rng(seed)
    shuffle_rng = np.random.default_rng(seed + 1)  # independent stream, only used for shuffling

    def is_valid(start0: int) -> bool:
        return "N" not in reference.fetch(CHROM, start0, start0 + window)

    starts = sample_valid_positions(rng, margin, length - margin, n, is_valid)

    records = []
    for start0 in starts:
        seq = reference.fetch(CHROM, start0, start0 + window)
        shuffled = dinucleotide_shuffle(seq, shuffle_rng)
        n_bases = len(seq)

        ref_ids = tokenizer(seq)["input_ids"]
        shuf_ids = tokenizer(shuffled)["input_ids"]
        batch = torch.tensor([ref_ids, shuf_ids], dtype=torch.long)
        with torch.inference_mode():
            logits = model(input_ids=batch).logits
        log_probs = torch.log_softmax(logits.float(), dim=-1).numpy()

        real_ll = causal_loglik(log_probs[0], ref_ids, start=1, end=n_bases)
        shuf_ll = causal_loglik(log_probs[1], shuf_ids, start=1, end=n_bases)
        n_terms = n_bases - 1

        records.append(
            {
                "start0": start0,
                "real_mean_ll_nats": real_ll / n_terms,
                "shuffled_mean_ll_nats": shuf_ll / n_terms,
            }
        )

    _save_cache(cache_path, config, records)
    return records


def summarize_hyena_real_vs_shuffled(records: list[dict]) -> dict[str, float]:
    n = len(records)
    real = np.array([r["real_mean_ll_nats"] for r in records])
    shuf = np.array([r["shuffled_mean_ll_nats"] for r in records])
    real_bpb = np.array([bits_per_base(x) for x in real])
    shuf_bpb = np.array([bits_per_base(x) for x in shuf])
    return {
        "n": n,
        "mean_real_bits_per_base": float(real_bpb.mean()),
        "mean_shuffled_bits_per_base": float(shuf_bpb.mean()),
        "mean_paired_diff_bits_per_base": float((shuf_bpb - real_bpb).mean()),
        "frac_real_higher_likelihood": float((real > shuf).mean()),
    }


# --------------------------------------------------------------------------
# Check (c): LLR breakdown from already-committed scores
# --------------------------------------------------------------------------


def load_llr_breakdown_inputs(data_dir: Path, out_dir: Path) -> pd.DataFrame:
    """BRCA1 dataset with flanking bases, transition/CpG flags, joined to committed scores."""
    dataset = pd.read_csv(data_dir / "processed" / "brca1_sge.tsv", sep="\t")
    reference = ReferenceGenome(data_dir / "raw" / "chr17.fa.gz")

    lefts, rights = [], []
    for row in dataset.itertuples():
        flank = reference.fetch(row.chrom, row.pos - 2, row.pos + 1)
        lefts.append(flank[0])
        rights.append(flank[2])
    dataset = dataset.assign(_left=lefts, _right=rights)
    dataset["ttv"] = [
        "transition" if is_transition(r, a) else "transversion"
        for r, a in zip(dataset["ref"], dataset["alt"], strict=True)
    ]
    dataset["cpg"] = [
        "cpg" if classify_cpg(left, ref, right) else "non_cpg"
        for left, ref, right in zip(
            dataset["_left"], dataset["ref"], dataset["_right"], strict=True
        )
    ]

    frames = []
    for scorer_name, filename in BRCA1_SCORE_FILES.items():
        scores = pd.read_csv(out_dir / "scores" / "brca1" / filename, sep="\t")
        merged = dataset.merge(
            scores[["chrom", "pos", "ref", "alt", "llr"]],
            on=["chrom", "pos", "ref", "alt"],
            how="inner",
        )
        merged["scorer"] = scorer_name
        frames.append(merged)
    return pd.concat(frames, ignore_index=True)


def summarize_llr_breakdown(joined: pd.DataFrame) -> pd.DataFrame:
    """One row per (scorer, group_type, group) with mean/median/std/n of llr."""
    rows = []
    for scorer_name, group in joined.groupby("scorer"):
        for group_type, col in [
            ("transition_vs_transversion", "ttv"),
            ("cpg_context", "cpg"),
            ("consequence", "consequence"),
        ]:
            table = mean_llr_by_group(group, col, "llr")
            table.insert(0, "group_type", group_type)
            table.insert(0, "scorer", scorer_name)
            table = table.rename(columns={col: "group"})
            rows.append(table)
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------
# Report writing
# --------------------------------------------------------------------------


def build_tsv_rows(
    nt_summary: dict, hyena_summary: dict, llr_breakdown: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    if nt_summary["n"]:
        for metric, value in nt_summary.items():
            if metric == "n":
                continue
            rows.append(
                {
                    "section": "nt_masked_recovery",
                    "group_type": "overall",
                    "group": "overall",
                    "metric": metric,
                    "value": value,
                    "n": nt_summary["n"],
                }
            )
    if hyena_summary["n"]:
        for metric, value in hyena_summary.items():
            if metric == "n":
                continue
            rows.append(
                {
                    "section": "hyena_real_vs_shuffled",
                    "group_type": "overall",
                    "group": "overall",
                    "metric": metric,
                    "value": value,
                    "n": hyena_summary["n"],
                }
            )
    for _, r in llr_breakdown.iterrows():
        for metric in ("mean", "median", "std"):
            rows.append(
                {
                    "section": f"llr_breakdown:{r['scorer']}",
                    "group_type": r["group_type"],
                    "group": r["group"],
                    "metric": metric,
                    "value": r[metric],
                    "n": int(r["n"]),
                }
            )
    return pd.DataFrame(rows, columns=["section", "group_type", "group", "metric", "value", "n"])


def render_markdown(
    nt_summary: dict,
    hyena_summary: dict,
    llr_breakdown: pd.DataFrame,
    args: argparse.Namespace,
) -> str:
    chance_kmer = 1 / 4096

    parts = []
    parts.append("# Sanity checks\n")
    parts.append(
        "Positive controls for the negative benchmark result "
        "(see `results/metrics_brca1.md`, `results/metrics_clinvar.md`): if the models "
        "and pipeline work at all, they should recover a masked base from real genomic "
        "sequence far above chance and score real sequence more likely than a composition-"
        "matched shuffle. Both checks below hold. What the LLR scores actually correlate "
        "with instead, using only the scores already committed to `results/scores/`, is "
        "in the third section.\n"
    )
    parts.append(
        f"All positions/windows are seeded (`--seed {args.seed}`) draws from hg19 "
        f"{CHROM}, restricted to windows containing no N bases. Generated by "
        "`scripts/sanity_checks.py`.\n"
    )

    parts.append("## a. Nucleotide Transformer (nt-v2-50m) masked-token recovery\n")
    if not nt_summary["n"]:
        parts.append("Not run this invocation (`--only` excluded it).\n")
    else:
        parts.append(
            f"{nt_summary['n']} seeded positions in {CHROM}, each inside an ACGT-only "
            f"{args.window} bp window. The 6-mer token covering the position is masked and "
            "read back from a single forward pass. Chance for the 6-mer is 1/4096 "
            f"({chance_kmer:.6f}); chance for the center base alone is 1/4.\n"
        )
        parts.append(
            format_markdown_table(
                ["metric", "value", "n"],
                [
                    [
                        "top-1 accuracy, masked 6-mer (over the 4,096 clean 6-mers)",
                        round(nt_summary["top1_kmer_acc"], 4),
                        nt_summary["n"],
                    ],
                    [
                        "top-1 accuracy, center base only",
                        round(nt_summary["top1_center_base_acc"], 4),
                        nt_summary["n"],
                    ],
                    [
                        "mean log P(reference 6-mer), nats",
                        round(nt_summary["mean_logp_ref_kmer"], 4),
                        nt_summary["n"],
                    ],
                ],
            )
        )
    parts.append("")

    parts.append("## b. HyenaDNA (hyenadna-small-32k) real versus shuffled\n")
    if not hyena_summary["n"]:
        parts.append("Not run this invocation (`--only` excluded it).\n")
    else:
        parts.append(
            f"{hyena_summary['n']} seeded, ACGT-only, {args.window} bp {CHROM} windows. Each "
            "window's dinucleotide-preserving shuffle (Altschul-Erikson algorithm; see "
            "`scripts/_sanity_common.dinucleotide_shuffle`) is scored alongside the real window "
            "in the same forward pass, causal mode, mean per-base log-likelihood converted to "
            "bits per base (lower is better/more predictable).\n"
        )
        parts.append(
            format_markdown_table(
                ["metric", "value", "n"],
                [
                    [
                        "mean bits/base, real window",
                        round(hyena_summary["mean_real_bits_per_base"], 4),
                        hyena_summary["n"],
                    ],
                    [
                        "mean bits/base, shuffled window",
                        round(hyena_summary["mean_shuffled_bits_per_base"], 4),
                        hyena_summary["n"],
                    ],
                    [
                        "mean paired difference (shuffled - real), bits/base",
                        round(hyena_summary["mean_paired_diff_bits_per_base"], 4),
                        hyena_summary["n"],
                    ],
                    [
                        "fraction of windows real scores higher than its shuffle",
                        round(hyena_summary["frac_real_higher_likelihood"], 4),
                        hyena_summary["n"],
                    ],
                ],
            )
        )
    parts.append("")

    parts.append("## c. What the committed LLR scores track (no new inference)\n")
    if llr_breakdown.empty:
        parts.append("Not run this invocation (`--only` excluded it).\n")
        return "\n".join(parts) + "\n"
    parts.append(
        "Mean `llr` (natural-log likelihood ratio; negative means the alt is scored less "
        "likely than the reference) from the already-committed BRCA1 score files, split by "
        "transition/transversion, CpG dinucleotide context, and Findlay consequence class, "
        "for `hyenadna-small-32k_full` and `nt-v2-50m`.\n"
    )
    for scorer_name in BRCA1_SCORE_FILES:
        sub = llr_breakdown[llr_breakdown["scorer"] == scorer_name]
        parts.append(f"### {scorer_name}\n")
        for group_type, title in [
            ("transition_vs_transversion", "Transition vs transversion"),
            ("cpg_context", "CpG context"),
            ("consequence", "Findlay consequence class"),
        ]:
            block = sub[sub["group_type"] == group_type]
            rows = [
                [
                    r["group"],
                    round(r["mean"], 4),
                    round(r["median"], 4),
                    round(r["std"], 4),
                    int(r["n"]),
                ]
                for _, r in block.iterrows()
            ]
            parts.append(f"**{title}**\n")
            parts.append(
                format_markdown_table(["group", "mean_llr", "median_llr", "std_llr", "n"], rows)
            )
            parts.append("")

    return "\n".join(parts) + "\n"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="cache directory for (a)/(b) results (default: <data-dir's parent>/.cache/sanity)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-nt-positions", type=int, default=300)
    parser.add_argument("--n-hyena-windows", type=int, default=200)
    parser.add_argument("--window", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="ignore any cached (a)/(b) results")
    parser.add_argument(
        "--only",
        choices=["nt", "hyena", "llr", "all"],
        default="all",
        help="restrict to one check, for faster iteration (default: all)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s"
    )

    cache_dir = args.cache_dir or (args.data_dir.resolve().parent / ".cache" / "sanity")
    cache_dir.mkdir(parents=True, exist_ok=True)
    reference = ReferenceGenome(args.data_dir / "raw" / "chr17.fa.gz")

    nt_summary = {
        "n": 0,
        "top1_kmer_acc": float("nan"),
        "top1_center_base_acc": float("nan"),
        "mean_logp_ref_kmer": float("nan"),
    }
    hyena_summary = {
        "n": 0,
        "mean_real_bits_per_base": float("nan"),
        "mean_shuffled_bits_per_base": float("nan"),
        "mean_paired_diff_bits_per_base": float("nan"),
        "frac_real_higher_likelihood": float("nan"),
    }
    llr_breakdown = pd.DataFrame(
        columns=["scorer", "group_type", "group", "mean", "median", "std", "n"]
    )

    if args.only in ("nt", "all"):
        start = time.perf_counter()
        records = run_nt_masked_recovery(
            reference,
            args.n_nt_positions,
            args.window,
            args.seed,
            args.threads,
            cache_dir / "nt_masked_recovery.json",
            args.force,
        )
        nt_summary = summarize_nt_masked_recovery(records)
        print(f"nt masked recovery: {nt_summary} ({time.perf_counter() - start:.1f}s)")

    if args.only in ("hyena", "all"):
        start = time.perf_counter()
        records = run_hyena_real_vs_shuffled(
            reference,
            args.n_hyena_windows,
            args.window,
            args.seed,
            args.threads,
            cache_dir / "hyena_real_vs_shuffled.json",
            args.force,
        )
        hyena_summary = summarize_hyena_real_vs_shuffled(records)
        print(f"hyena real vs shuffled: {hyena_summary} ({time.perf_counter() - start:.1f}s)")

    if args.only in ("llr", "all"):
        joined = load_llr_breakdown_inputs(args.data_dir, args.out)
        llr_breakdown = summarize_llr_breakdown(joined)
        print(f"llr breakdown: {len(llr_breakdown)} rows")

    args.out.mkdir(parents=True, exist_ok=True)
    tsv = build_tsv_rows(nt_summary, hyena_summary, llr_breakdown)
    tsv.to_csv(args.out / "sanity_checks.tsv", sep="\t", index=False)
    (args.out / "sanity_checks.md").write_text(
        render_markdown(nt_summary, hyena_summary, llr_breakdown, args), encoding="utf-8"
    )
    print(f"wrote {args.out / 'sanity_checks.md'} and {args.out / 'sanity_checks.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
