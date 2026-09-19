"""Stream-parse a phyloP wigFix track and look up scores at chosen positions.

UCSC distributes phyloP as ``fixedStep`` blocks of one value per line, in
increasing position order. Rather than loading the whole (multi-hundred-MB
uncompressed) track, this makes one streaming pass and reads off only the
requested positions, keeping memory to the size of the position list.
"""

from __future__ import annotations

import gzip
from pathlib import Path


def lookup_phylop(wigfix_gz_path: Path | str, positions: list[int]) -> dict[int, float]:
    """Return ``{1-based position: phyloP score}`` for the requested positions.

    Positions without coverage in the track (gaps between fixedStep blocks) are
    simply absent from the result; callers should treat a missing key as NaN.
    Only positions that fall inside a ``step=1`` block are supported, which is
    what the 100-way phyloP tracks use.
    """
    wanted = sorted(set(positions))
    if not wanted:
        return {}

    result: dict[int, float] = {}
    n = len(wanted)
    wi = 0
    cursor: int | None = None

    with gzip.open(wigfix_gz_path, "rt") as fh:
        for line in fh:
            if line.startswith("fixedStep"):
                header = dict(part.split("=") for part in line.split()[1:])
                step = int(header.get("step", "1"))
                if step != 1:
                    raise ValueError(f"unsupported wigFix step={step}, expected 1")
                cursor = int(header["start"])
                continue
            if cursor is None:
                continue

            # skip positions the track will never reach because a new block
            # already jumped past them
            while wi < n and wanted[wi] < cursor:
                wi += 1
            if wi >= n:
                break
            if wanted[wi] == cursor:
                result[wanted[wi]] = float(line)
                wi += 1
            cursor += 1

    return result
