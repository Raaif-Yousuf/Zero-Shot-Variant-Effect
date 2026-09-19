"""Download and checksum-verify the benchmark input files.

Usage:
    uv run scripts/download_data.py --data-dir path/to/data
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from zeroshot_vep.data.manifest import MANIFEST, ChecksumError, download


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="base data directory; files land in <data-dir>/raw (default: ./data)",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        metavar="NAME",
        help="restrict to one manifest entry by name (repeatable)",
    )
    parser.add_argument(
        "--force", action="store_true", help="re-download even if a valid file is already present"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s"
    )

    specs = MANIFEST
    if args.only:
        wanted = set(args.only)
        specs = tuple(s for s in MANIFEST if s.name in wanted)
        missing = wanted - {s.name for s in specs}
        if missing:
            print(f"unknown manifest entries: {sorted(missing)}", file=sys.stderr)
            return 2

    ok = True
    for spec in specs:
        try:
            path = download(spec, args.data_dir, force=args.force)
            size = path.stat().st_size
            print(f"{spec.name}: {path} ({size:,} bytes, sha256 {spec.sha256}) OK")
        except ChecksumError as exc:
            ok = False
            print(f"{spec.name}: CHECKSUM MISMATCH: {exc}", file=sys.stderr)
        except OSError as exc:
            ok = False
            print(f"{spec.name}: download failed: {exc}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
