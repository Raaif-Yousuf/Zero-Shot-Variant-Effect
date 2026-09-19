"""Join benchmark scores with the processed datasets, compute metrics and figures.

Usage:
    uv run scripts/summarize_results.py --data-dir path/to/data \\
        --plan benchmarks/plan.json --results results/ --figures docs/figures/

Reads whatever run outputs already exist under ``<results>/scores/`` (it does
not require the full plan to be complete) and writes:
  - results/metrics_brca1.tsv/.md          (overall, BRCA1 SGE vs label/function_score)
  - results/metrics_brca1_by_consequence.tsv/.md
  - results/metrics_clinvar.tsv/.md         (overall + by consequence)
  - results/context_sweep.tsv/.md           (AUROC vs context window)
  - docs/figures/*.png                      (6 figures, see FIGURE_FILES)

Damaging-score convention: kmer and the pretrained-model scorers report a
natural-log likelihood ratio where negative means damaging, so their damaging
score is ``-llr``. phyloP and CADD are already "higher = more damaging" and
are used as-is (CADD is a supervised reference, not a zero-shot score).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bench_common import (  # noqa: E402
    damaging_score,
    join_scores_with_dataset,
    pick_best_model,
    run_id_for,
    scorer_label,
    select_sweep_positions,
)

from zeroshot_vep.evaluate import (  # noqa: E402
    evaluate_by_stratum,
    evaluate_scorers,
    write_metrics_markdown,
    write_metrics_tsv,
)
from zeroshot_vep.plots import (  # noqa: E402
    plot_auroc_vs_window,
    plot_pr_overlay,
    plot_roc_overlay,
    plot_score_distributions,
    plot_score_vs_continuous,
)

DATASET_FILES = {
    "brca1": "brca1_sge.tsv",
    "clinvar": "clinvar_chr17_subsample.tsv",
}
SWEEP_N_POSITIONS = 150
SWEEP_SEED = 0

# Curated so ROC/PR overlays and score-distribution panels stay within the
# eight-color categorical palette (see docs on the fixed hue order in plots.py).
BRCA1_FIGURE_SCORERS = [
    "kmer_o6",
    "hyenadna-tiny-1k_full",
    "hyenadna-small-32k_full",
    "hyenadna-small-32k_site",
    "nt-v2-50m",
    "phylop100way",
    "phylop_mammalian",
    "cadd",
]
CLINVAR_FIGURE_SCORERS = [
    "kmer_o6",
    "hyenadna-tiny-1k_full",
    "hyenadna-small-32k_full",
    "nt-v2-50m",
    "phylop100way",
]
BRCA1_MODEL_SCORERS = [
    "kmer_o6",
    "hyenadna-tiny-1k_full",
    "hyenadna-small-32k_full",
    "hyenadna-small-32k_site",
    "nt-v2-50m",
]

# (label, scorer, mode) for dedicated sweep runs; kmer is handled separately as a
# flat reference line, and hyenadna-tiny-1k is handled separately as a reused point.
SWEEP_MODEL_SCORERS = {"hyenadna-small-32k", "hyenadna-medium-160k", "nt-v2-50m"}
REUSED_SINGLE_WINDOW = [("hyenadna-tiny-1k_full", "hyenadna-tiny-1k", "full", 1024)]
KMER_REFERENCE_ENTRY = {
    "dataset": "brca1",
    "scorer": "kmer",
    "order": 6,
    "window": 1024,
    "subset": "sweep",
}
FIGURE_FILES = {
    "brca1_roc": "brca1_roc_overlay.png",
    "brca1_pr": "brca1_pr_overlay.png",
    "brca1_dist": "brca1_score_distributions.png",
    "context_sweep": "context_sweep_auroc.png",
    "clinvar_roc": "clinvar_roc_overlay.png",
    "best_model_scatter": "best_model_vs_function_score.png",
}


def load_plan(plan_path: Path) -> list[dict]:
    return json.loads(plan_path.read_text())


def load_dataset(data_dir: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(data_dir / "processed" / DATASET_FILES[name], sep="\t")


def build_wide_table(
    dataset_df: pd.DataFrame,
    plan: list[dict],
    dataset_name: str,
    results_dir: Path,
    subset: str = "all",
) -> tuple[pd.DataFrame, list[str]]:
    """Left-join every completed run's damaging score onto ``dataset_df`` as its own column."""
    wide = dataset_df.copy()
    score_cols: list[str] = []
    for entry in plan:
        if entry["dataset"] != dataset_name or entry.get("subset", "all") != subset:
            continue
        path = results_dir / "scores" / dataset_name / f"{run_id_for(entry)}.tsv"
        if not path.exists():
            continue
        scores = pd.read_csv(path, sep="\t")
        label = scorer_label(entry)
        joined = join_scores_with_dataset(wide[["chrom", "pos", "ref", "alt"]], scores)
        wide[label] = damaging_score(entry["scorer"], joined["llr"])
        score_cols.append(label)
    return wide, score_cols


def to_long(wide: pd.DataFrame, score_cols: list[str], label_col: str = "label") -> pd.DataFrame:
    frames = [
        pd.DataFrame({"scorer": col, "label": wide[label_col], "score": wide[col]})
        for col in score_cols
    ]
    return (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["scorer", "label", "score"])
    )


def compute_brca1_metrics(
    data_dir: Path, results_dir: Path, plan: list[dict]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    df = load_dataset(data_dir, "brca1")
    wide, score_cols = build_wide_table(df, plan, "brca1", results_dir, subset="all")
    overall = evaluate_scorers(wide, score_cols, label_col="label", continuous_col="function_score")
    by_conseq = evaluate_by_stratum(
        wide, score_cols, "consequence", label_col="label", continuous_col="function_score"
    )
    return wide, overall, by_conseq, score_cols


def compute_clinvar_metrics(
    data_dir: Path, results_dir: Path, plan: list[dict]
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    df = load_dataset(data_dir, "clinvar")
    wide, score_cols = build_wide_table(df, plan, "clinvar", results_dir, subset="all")
    overall = evaluate_scorers(wide, score_cols, label_col="label")
    by_conseq = evaluate_by_stratum(wide, score_cols, "consequence", label_col="label")
    combined = pd.concat([overall, by_conseq], ignore_index=True)
    return wide, combined, score_cols


def _sweep_auroc_row(
    sweep_df: pd.DataFrame, entry: dict, results_dir: Path, model_label: str
) -> dict | None:
    path = results_dir / "scores" / "brca1" / f"{run_id_for(entry)}.tsv"
    if not path.exists():
        return None
    scores = pd.read_csv(path, sep="\t")
    joined = join_scores_with_dataset(sweep_df[["chrom", "pos", "ref", "alt", "label"]], scores)
    joined["score"] = damaging_score(entry["scorer"], joined["llr"])
    metrics = evaluate_scorers(joined, ["score"], label_col="label")
    row = metrics.iloc[0]
    return {
        "model": model_label,
        "window": entry["window"],
        "n": int(row["n"]),
        "n_scored": int(row["n_scored"]),
        "n_pos": int(row["n_pos"]),
        "n_neg": int(row["n_neg"]),
        "auroc": row["auroc"],
        "auroc_lo": row["auroc_lo"],
        "auroc_hi": row["auroc_hi"],
    }


def build_context_sweep(
    data_dir: Path, results_dir: Path, plan: list[dict]
) -> tuple[pd.DataFrame, int]:
    df = load_dataset(data_dir, "brca1")
    sweep_positions = select_sweep_positions(df["pos"], SWEEP_N_POSITIONS, SWEEP_SEED)
    sweep_df = df[df["pos"].isin(sweep_positions)].reset_index(drop=True)

    rows: list[dict] = []
    for entry in plan:
        if (
            entry["dataset"] != "brca1"
            or entry.get("subset") != "sweep"
            or entry["scorer"] not in SWEEP_MODEL_SCORERS
        ):
            continue
        row = _sweep_auroc_row(sweep_df, entry, results_dir, scorer_label(entry))
        if row is not None:
            rows.append(row)

    for label, scorer, mode, window in REUSED_SINGLE_WINDOW:
        all_entry = {
            "dataset": "brca1",
            "scorer": scorer,
            "mode": mode,
            "window": window,
            "subset": "all",
        }
        row = _sweep_auroc_row(sweep_df, all_entry, results_dir, label)
        if row is not None:
            rows.append(row)

    kmer_row = _sweep_auroc_row(sweep_df, KMER_REFERENCE_ENTRY, results_dir, "kmer_o6_reference")
    if kmer_row is not None and rows:
        windows = sorted({r["window"] for r in rows})
        for w in windows:
            replicated = dict(kmer_row)
            replicated["window"] = w
            rows.append(replicated)

    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values(["model", "window"]).reset_index(drop=True)
    return table, len(sweep_positions)


def make_figures(
    figures_dir: Path,
    brca1_wide: pd.DataFrame,
    brca1_overall: pd.DataFrame,
    clinvar_wide: pd.DataFrame,
    clinvar_score_cols: list[str],
    sweep_table: pd.DataFrame,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    brca1_cols = [c for c in BRCA1_FIGURE_SCORERS if c in brca1_wide.columns]
    brca1_long = to_long(brca1_wide, brca1_cols)
    if not brca1_long.empty:
        plot_roc_overlay(brca1_long, figures_dir / FIGURE_FILES["brca1_roc"])
        plot_pr_overlay(brca1_long, figures_dir / FIGURE_FILES["brca1_pr"])
        plot_score_distributions(
            brca1_long, figures_dir / FIGURE_FILES["brca1_dist"], pos_name="LOF", neg_name="FUNC"
        )

    if not sweep_table.empty:
        plot_auroc_vs_window(sweep_table, figures_dir / FIGURE_FILES["context_sweep"])

    clinvar_cols = [c for c in CLINVAR_FIGURE_SCORERS if c in clinvar_wide.columns]
    clinvar_long = to_long(clinvar_wide, clinvar_cols)
    if not clinvar_long.empty:
        plot_roc_overlay(clinvar_long, figures_dir / FIGURE_FILES["clinvar_roc"])

    candidates = [c for c in BRCA1_MODEL_SCORERS if c in brca1_overall["scorer"].to_numpy()]
    if candidates:
        best = pick_best_model(brca1_overall, candidates, by="spearman_rho")
        plot_score_vs_continuous(
            brca1_wide,
            figures_dir / FIGURE_FILES["best_model_scatter"],
            score_col=best,
            continuous_col="function_score",
            score_label=f"-llr ({best})",
            continuous_label="Findlay function score",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--plan", type=Path, default=Path("benchmarks/plan.json"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--figures", type=Path, default=Path("docs/figures"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = load_plan(args.plan)
    args.results.mkdir(parents=True, exist_ok=True)

    brca1_wide, brca1_overall, brca1_by_conseq, _ = compute_brca1_metrics(
        args.data_dir, args.results, plan
    )
    write_metrics_tsv(brca1_overall, args.results / "metrics_brca1.tsv")
    write_metrics_markdown(brca1_overall, args.results / "metrics_brca1.md")
    write_metrics_tsv(brca1_by_conseq, args.results / "metrics_brca1_by_consequence.tsv")
    write_metrics_markdown(brca1_by_conseq, args.results / "metrics_brca1_by_consequence.md")

    clinvar_wide, clinvar_combined, clinvar_cols = compute_clinvar_metrics(
        args.data_dir, args.results, plan
    )
    write_metrics_tsv(clinvar_combined, args.results / "metrics_clinvar.tsv")
    write_metrics_markdown(clinvar_combined, args.results / "metrics_clinvar.md")

    sweep_table, sweep_n_positions = build_context_sweep(args.data_dir, args.results, plan)
    write_metrics_tsv(sweep_table, args.results / "context_sweep.tsv")
    write_metrics_markdown(sweep_table, args.results / "context_sweep.md")

    make_figures(args.figures, brca1_wide, brca1_overall, clinvar_wide, clinvar_cols, sweep_table)

    print(f"BRCA1 scorers summarized: {sorted(brca1_overall['scorer'].unique())}")
    print(f"ClinVar scorers summarized: {sorted(clinvar_cols)}")
    print(f"context sweep: {len(sweep_table)} rows, subsample size {sweep_n_positions} positions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
