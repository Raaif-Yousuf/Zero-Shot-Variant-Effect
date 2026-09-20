"""Run benchmark scorers over the processed datasets, per a plan.json.

Usage:
    uv run scripts/run_benchmark.py --plan benchmarks/plan.json \\
        --data-dir path/to/data --out results/ [--only RUN_ID ...] [--threads 4]

Each plan entry is scored independently and idempotently: a run whose output
TSV already exists is skipped, so a killed or partial batch can simply be
re-invoked. Baseline "scorers" that are really an existing dataset column
(phyloP, CADD) are copied through rather than run via the scoring engine.

Concurrency note: pass ``--cache-dir`` (or rely on the default, one directory
per scorer name under ``<data-dir's repo>/.cache/bench/<scorer>``) and never
launch two concurrent processes whose plan entries share the same ``scorer``
name -- they would share one sqlite score-cache file.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bench_common import (  # noqa: E402
    COLUMN_SCORERS,
    SWEEP_N_POSITIONS,
    SWEEP_SEED,
    run_id_for,
    select_sweep_positions,
)

from zeroshot_vep.cache import ScoreCache  # noqa: E402
from zeroshot_vep.engine import score_variants  # noqa: E402
from zeroshot_vep.reference import ReferenceGenome  # noqa: E402
from zeroshot_vep.scorers import get_scorer  # noqa: E402
from zeroshot_vep.variant import Variant  # noqa: E402

logger = logging.getLogger(__name__)

DATASET_FILES = {
    "brca1": "brca1_sge.tsv",
    "clinvar": "clinvar_chr17_subsample.tsv",
}
FASTA_NAME = "chr17.fa.gz"

_RUN_LOG_COLUMNS = [
    "run_id",
    "dataset",
    "scorer",
    "mode",
    "order",
    "window",
    "subset",
    "n_variants",
    "n_positions",
    "threads",
    "wall_seconds",
    "reused_from",
]


def load_dataset(data_dir: Path, dataset: str) -> pd.DataFrame:
    path = data_dir / "processed" / DATASET_FILES[dataset]
    return pd.read_csv(path, sep="\t")


def sweep_positions_for(data_dir: Path, dataset: str) -> list[int]:
    df = load_dataset(data_dir, dataset)
    return select_sweep_positions(df["pos"], SWEEP_N_POSITIONS, SWEEP_SEED)


def dataset_for_entry(data_dir: Path, entry: dict) -> pd.DataFrame:
    df = load_dataset(data_dir, entry["dataset"])
    if entry.get("subset") == "sweep":
        positions = sweep_positions_for(data_dir, entry["dataset"])
        df = df[df["pos"].isin(positions)].reset_index(drop=True)
    return df


def to_variants(df: pd.DataFrame) -> list[Variant]:
    return [
        Variant(row.chrom, int(row.pos), row.ref, row.alt, str(row.id)) for row in df.itertuples()
    ]


def run_column_scorer(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Pass an existing baseline column through unchanged, in the engine's schema."""
    out = df[["chrom", "pos", "ref", "alt", "id"]].copy()
    values = df[column]
    out["llr"] = values
    out["llr_fwd"] = values
    out["llr_rev"] = float("nan")
    out["status"] = ["ok" if pd.notna(v) else "missing" for v in values]
    return out


def run_engine_scorer(
    df: pd.DataFrame, entry: dict, fasta_path: Path, cache_dir: Path, threads: int
) -> tuple[pd.DataFrame, int]:
    variants = to_variants(df)
    reference = ReferenceGenome(fasta_path)
    if entry["scorer"] == "kmer":
        kwargs: dict[str, object] = dict(
            fasta_path=fasta_path,
            chroms=sorted({v.chrom for v in variants}),
            order=entry.get("order", 6),
            cache_dir=cache_dir,
        )
    else:
        kwargs = {"num_threads": threads}
        if entry.get("mode"):
            kwargs["mode"] = entry["mode"]
    scorer = get_scorer(entry["scorer"], **kwargs)
    n_positions = df["pos"].nunique() if not df.empty else 0
    with ScoreCache(cache_dir=cache_dir) as cache:
        result = score_variants(
            variants,
            reference,
            scorer,
            window=entry["window"],
            strands=entry.get("strands", "both"),
            cache=cache,
            progress=True,
        )
    return result, n_positions


def _matching_all_run(entry: dict, out_dir: Path) -> Path | None:
    """An already-computed "all"-subset run this sweep entry can reuse.

    Scoring is per-position independent, so filtering a wider run's output
    down to the sweep positions gives byte-identical scores to re-running the
    engine on just those positions -- this only saves compute.
    """
    if entry.get("subset") != "sweep":
        return None
    all_entry = {**entry, "subset": "all"}
    candidate = out_dir / "scores" / entry["dataset"] / f"{run_id_for(all_entry)}.tsv"
    return candidate if candidate.exists() else None


def run_one(entry: dict, data_dir: Path, out_dir: Path, cache_root: Path, threads: int) -> dict:
    run_id = run_id_for(entry)
    dataset_dir = out_dir / "scores" / entry["dataset"]
    dataset_dir.mkdir(parents=True, exist_ok=True)
    out_path = dataset_dir / f"{run_id}.tsv"
    if out_path.exists():
        logger.info("%s: output already exists, skipping", run_id)
        return {}

    df = dataset_for_entry(data_dir, entry)
    n_variants = len(df)
    fasta_path = data_dir / "raw" / FASTA_NAME

    reused_from = ""
    start = time.perf_counter()
    if entry["scorer"] in COLUMN_SCORERS:
        result = run_column_scorer(df, COLUMN_SCORERS[entry["scorer"]])
        n_positions = df["pos"].nunique() if not df.empty else 0
    else:
        reuse_path = _matching_all_run(entry, out_dir)
        if reuse_path is not None:
            all_scores = pd.read_csv(reuse_path, sep="\t")
            wanted = set(zip(df["chrom"], df["pos"], df["ref"], df["alt"], strict=False))
            mask = [
                (c, p, r, a) in wanted
                for c, p, r, a in zip(
                    all_scores["chrom"],
                    all_scores["pos"],
                    all_scores["ref"],
                    all_scores["alt"],
                    strict=False,
                )
            ]
            result = all_scores.loc[mask].reset_index(drop=True)
            n_positions = df["pos"].nunique() if not df.empty else 0
            reused_from = reuse_path.name
        else:
            cache_dir = cache_root / entry["scorer"]
            result, n_positions = run_engine_scorer(df, entry, fasta_path, cache_dir, threads)
    elapsed = time.perf_counter() - start

    result.to_csv(out_path, sep="\t", index=False)

    return {
        "run_id": run_id,
        "dataset": entry["dataset"],
        "scorer": entry["scorer"],
        "mode": entry.get("mode") or "",
        "order": entry.get("order") if entry.get("order") is not None else "",
        "window": entry.get("window") if entry.get("window") is not None else "",
        "subset": entry.get("subset", "all"),
        "n_variants": n_variants,
        "n_positions": n_positions,
        "threads": threads,
        "wall_seconds": round(elapsed, 3),
        "reused_from": reused_from,
    }


def append_run_log(row: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "runs.tsv"
    header = not log_path.exists()
    pd.DataFrame([row], columns=_RUN_LOG_COLUMNS).to_csv(
        log_path, sep="\t", index=False, mode="a", header=header
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("benchmarks/plan.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="base cache directory; each scorer gets its own subdirectory "
        "(default: <parent of --data-dir>/.cache/bench)",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        metavar="RUN_ID",
        help="restrict to one run id (repeatable)",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s"
    )

    plan = json.loads(args.plan.read_text())
    cache_root = args.cache_dir or (args.data_dir.resolve().parent / ".cache" / "bench")

    wanted = set(args.only) if args.only else None
    for entry in plan:
        run_id = run_id_for(entry)
        if wanted is not None and run_id not in wanted:
            continue
        row = run_one(entry, args.data_dir, args.out, cache_root, args.threads)
        if row:
            append_run_log(row, args.out)
            note = f" (reused {row['reused_from']})" if row["reused_from"] else ""
            print(
                f"{row['run_id']}: {row['n_variants']} variants, {row['n_positions']} positions, "
                f"{row['wall_seconds']:.1f}s, {row['threads']} threads{note}"
            )
        else:
            print(f"{run_id}: skipped (output already exists)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
