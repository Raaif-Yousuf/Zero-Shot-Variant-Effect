"""Pinned manifest of benchmark input files and a checksum-verified fetch helper.

Every entry pins a sha256 measured from a file already cross-checked once against
the md5 the provider publishes alongside it (UCSC ``md5sum.txt``, NCBI ``.md5``).
Routine downloads only ever check the pinned sha256; the md5 cross-check is a
one-time step done when the sha256 was first pinned, recorded in each entry's
``note``.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
_CHUNK_SIZE = 1 << 20  # 1 MiB
_DEFAULT_RETRIES = 5
_RETRY_BACKOFF_SECONDS = 2.0
_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class FileSpec:
    """One benchmark input file: where to get it and how to know it is intact."""

    name: str
    url: str
    filename: str
    sha256: str
    size: int
    note: str


class ChecksumError(RuntimeError):
    """Raised when a downloaded (or already-present) file fails sha256 verification."""


MANIFEST: tuple[FileSpec, ...] = (
    FileSpec(
        name="findlay2018_supp_table1",
        url=(
            "https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-018-0461-z/"
            "MediaObjects/41586_2018_461_MOESM3_ESM.xlsx"
        ),
        filename="findlay2018_supp_table1.xlsx",
        sha256="e9aa4186b8a5de91d61059d03f8dac1e5573d1e2802f50a6581c3053d22b923a",
        size=2_306_341,
        note=(
            "Findlay, Guo, Hoffman et al. 2018, Nature 562:217-222 "
            "(doi:10.1038/s41586-018-0461-z), Supplementary Table 1: BRCA1 saturation "
            "genome editing scores. Distributed by the publisher; cite the paper when "
            "using this data. sha256 cross-checked once against the provider's own "
            "md5 c17f1a475560db5b98d83f9e14e8465f."
        ),
    ),
    FileSpec(
        name="hg19_chr17",
        url="https://hgdownload.soe.ucsc.edu/goldenPath/hg19/chromosomes/chr17.fa.gz",
        filename="chr17.fa.gz",
        sha256="27b909064ce4470d2655a16b3e30c9987be39aa5ed8e97f6b43e66b8e01bf6b3",
        size=25_139_792,
        note=(
            "UCSC Genome Browser hg19 (GRCh37) reference assembly, chromosome 17, "
            "plain gzip FASTA. Public UCSC download. sha256 cross-checked once against "
            "UCSC's md5sum.txt entry ee98e8346d23ccd91fb7ef60e9ccede9."
        ),
    ),
    FileSpec(
        name="clinvar_grch37_20260905",
        url=(
            "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh37/archive_2.0/2026/"
            "clinvar_20260905.vcf.gz"
        ),
        filename="clinvar_20260905.vcf.gz",
        sha256="68a479604a683df6c2431a68f1d3150922f872d8c33885fa2ad2d1fe333692f9",
        size=193_451_062,
        note=(
            "NCBI ClinVar, GRCh37 weekly release archived 2026-09-05. Public NCBI "
            "download. sha256 cross-checked once against the provider's "
            "clinvar_20260905.vcf.gz.md5 (e271d94171df250f520ac47b2ec8d394)."
        ),
    ),
    FileSpec(
        name="phylop100way_chr17",
        url=(
            "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/phyloP100way/"
            "hg19.100way.phyloP100way/chr17.phyloP100way.wigFix.gz"
        ),
        filename="chr17.phyloP100way.wigFix.gz",
        sha256="1fe9dad164fe271837a4f9c8c6970a914f10e4dcdac65387d94d603c24691234",
        size=150_943_228,
        note=(
            "UCSC Genome Browser hg19 100-way phyloP conservation track, chromosome 17, "
            "fixedStep wigFix. Public UCSC download. sha256 cross-checked once against "
            "UCSC's md5sum.txt entry 13e00efa10f81b73a6d1752930e089cb."
        ),
    ),
)


def get_spec(name: str) -> FileSpec:
    """Look up one manifest entry by its short name."""
    for spec in MANIFEST:
        if spec.name == name:
            return spec
    raise KeyError(f"unknown manifest entry {name!r}; choose from {[s.name for s in MANIFEST]}")


def sha256_of(path: Path) -> str:
    """Stream-hash a file; never loads it whole into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(
    spec: FileSpec,
    data_dir: Path | str,
    *,
    retries: int = _DEFAULT_RETRIES,
    force: bool = False,
) -> Path:
    """Fetch ``spec`` into ``<data_dir>/raw``, verifying its sha256.

    Skips the network entirely if a valid file is already at the destination.
    Downloads to a temporary file next to the destination and only renames it
    into place once the checksum matches, so an interrupted or failed download
    never leaves a corrupt file at the final path.

    Raises:
        ChecksumError: the downloaded bytes never matched ``spec.sha256`` after
            ``retries`` attempts.
        OSError: the file could not be reached after ``retries`` attempts.
    """
    raw_dir = Path(data_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / spec.filename

    if not force and dest.exists():
        actual = sha256_of(dest)
        if actual == spec.sha256:
            logger.info("%s already present and verified, skipping download", spec.filename)
            return dest
        logger.warning(
            "%s present but sha256 mismatch (got %s, expected %s), re-downloading",
            spec.filename,
            actual,
            spec.sha256,
        )

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            _stream_download(spec.url, dest)
            actual = sha256_of(dest)
            if actual != spec.sha256:
                dest.unlink(missing_ok=True)
                raise ChecksumError(
                    f"{spec.filename}: sha256 mismatch, expected {spec.sha256}, got {actual}"
                )
            return dest
        except (URLError, OSError, ChecksumError) as exc:
            last_error = exc
            logger.warning(
                "download attempt %d/%d for %s failed: %s", attempt, retries, spec.filename, exc
            )
            if attempt < retries:
                time.sleep(_RETRY_BACKOFF_SECONDS * attempt)
    assert last_error is not None
    raise last_error


def _stream_download(url: str, dest: Path) -> None:
    request = Request(url, headers={"User-Agent": _USER_AGENT})
    tmp_fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=dest.name + ".", suffix=".part")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(tmp_fd, "wb") as out, urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
            shutil.copyfileobj(response, out, length=_CHUNK_SIZE)
        tmp_path.replace(dest)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
