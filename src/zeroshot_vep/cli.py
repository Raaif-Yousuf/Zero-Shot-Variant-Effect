"""Command-line entry point (``zsvep``)."""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

import pandas as pd

from zeroshot_vep import __version__
from zeroshot_vep.cache import ScoreCache
from zeroshot_vep.engine import score_variants
from zeroshot_vep.io import read_variants, write_scores
from zeroshot_vep.reference import ReferenceGenome
from zeroshot_vep.scorers import SCORERS, get_scorer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="zsvep", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command")

    models_p = sub.add_parser("models", help="list available scorers")
    models_p.set_defaults(func=_cmd_models)

    score_p = sub.add_parser("score", help="score variants with a scorer")
    score_p.add_argument("--variants", required=True, type=Path)
    score_p.add_argument("--fasta", required=True, type=Path)
    score_p.add_argument("--model", required=True, dest="model_name")
    score_p.add_argument("--window", type=int, default=1024)
    score_p.add_argument("--strands", choices=["both", "forward"], default="both")
    score_p.add_argument("--mode", choices=["full", "site"], default="full")
    score_p.add_argument("--batch-size", type=int, default=8)
    score_p.add_argument("--threads", type=int, default=None)
    cache_group = score_p.add_mutually_exclusive_group()
    cache_group.add_argument("--cache-dir", type=Path, default=None)
    cache_group.add_argument("--no-cache", action="store_true")
    score_p.add_argument("--limit", type=int, default=None)
    score_p.add_argument("--out", required=True, type=Path)
    score_p.set_defaults(func=_cmd_score)

    return parser


def _cmd_models(args: argparse.Namespace) -> int:
    for name, (_, defaults) in sorted(SCORERS.items()):
        max_window = defaults.get("max_window", "unbounded")
        print(f"{name}\tmax_window={max_window}")
    return 0


def _build_scorer_kwargs(name: str, args: argparse.Namespace, chroms: list[str]) -> dict:
    """Build the kwargs a scorer's constructor actually accepts."""
    candidates = {
        "mode": args.mode,
        "batch_size": args.batch_size,
        "num_threads": args.threads,
        "fasta_path": args.fasta,
        "chroms": chroms,
        "cache_dir": args.cache_dir,
    }
    target, _ = SCORERS[name]
    module_name, class_name = target.split(":")
    try:
        module = __import__(module_name, fromlist=[class_name])
    except ImportError as exc:
        raise RuntimeError(
            f"cannot load scorer {name!r}: {exc}. "
            "If this is a pretrained-model scorer, install its extra with "
            "`uv sync --extra models`."
        ) from exc
    cls = getattr(module, class_name)
    accepted = set(inspect.signature(cls.__init__).parameters)
    return {k: v for k, v in candidates.items() if k in accepted and v is not None}


def _cmd_score(args: argparse.Namespace) -> int:
    if args.model_name not in SCORERS:
        print(
            f"error: unknown model {args.model_name!r}; choose from {sorted(SCORERS)}",
            file=sys.stderr,
        )
        return 1

    try:
        variants, extra, skipped = read_variants(args.variants)
    except Exception as exc:
        print(f"error reading variants: {exc}", file=sys.stderr)
        return 1
    if skipped:
        print(f"[zsvep] skipped {skipped} non-SNV VCF alleles", file=sys.stderr)

    if args.limit is not None:
        variants = variants[: args.limit]
        extra = extra.iloc[: args.limit].reset_index(drop=True)

    chroms = sorted({v.chrom for v in variants})

    try:
        kwargs = _build_scorer_kwargs(args.model_name, args, chroms)
        scorer = get_scorer(args.model_name, **kwargs)
    except Exception as exc:
        print(f"error building scorer {args.model_name!r}: {exc}", file=sys.stderr)
        return 1

    try:
        reference = ReferenceGenome(args.fasta)
    except Exception as exc:
        print(f"error loading reference: {exc}", file=sys.stderr)
        return 1

    cache = None if args.no_cache else ScoreCache(cache_dir=args.cache_dir)

    try:
        result = score_variants(
            variants,
            reference,
            scorer,
            window=args.window,
            strands=args.strands,
            cache=cache,
            progress=True,
        )
    except Exception as exc:
        print(f"error scoring variants: {exc}", file=sys.stderr)
        return 1
    finally:
        if cache is not None:
            cache.close()

    if not extra.empty:
        result = pd.concat([result, extra], axis=1)

    write_scores(result, args.out)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
