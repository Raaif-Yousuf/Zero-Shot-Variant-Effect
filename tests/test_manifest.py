"""Manifest download/verification tests. No network: uses file:// URLs and local files."""

from __future__ import annotations

import hashlib

import pytest

from zeroshot_vep.data.manifest import ChecksumError, FileSpec, download, get_spec, sha256_of


def test_sha256_of(tmp_path):
    path = tmp_path / "f.txt"
    path.write_bytes(b"hello")
    assert sha256_of(path) == hashlib.sha256(b"hello").hexdigest()


def test_get_spec_unknown_raises():
    with pytest.raises(KeyError):
        get_spec("not-a-real-entry")


def test_download_checksum_mismatch_raises(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"some content")
    spec = FileSpec(
        name="test",
        url=source.resolve().as_uri(),
        filename="dest.bin",
        sha256="0" * 64,  # deliberately wrong
        size=source.stat().st_size,
        note="test fixture",
    )
    with pytest.raises(ChecksumError):
        download(spec, tmp_path / "data", retries=1)
    # the bad file must not be left behind at the destination
    assert not (tmp_path / "data" / "raw" / "dest.bin").exists()


def test_download_succeeds_and_verifies(tmp_path):
    source = tmp_path / "source.bin"
    content = b"benchmark payload"
    source.write_bytes(content)
    spec = FileSpec(
        name="test",
        url=source.resolve().as_uri(),
        filename="dest.bin",
        sha256=hashlib.sha256(content).hexdigest(),
        size=len(content),
        note="test fixture",
    )
    dest = download(spec, tmp_path / "data")
    assert dest.read_bytes() == content


def test_download_skips_when_already_valid(tmp_path):
    data_dir = tmp_path / "data"
    raw = data_dir / "raw"
    raw.mkdir(parents=True)
    dest = raw / "dest.bin"
    content = b"cached content"
    dest.write_bytes(content)
    spec = FileSpec(
        name="test",
        url="file:///this/path/does/not/exist/and/must/not/be/read",
        filename="dest.bin",
        sha256=hashlib.sha256(content).hexdigest(),
        size=len(content),
        note="test fixture",
    )
    result = download(spec, data_dir)
    assert result == dest
    assert dest.read_bytes() == content
