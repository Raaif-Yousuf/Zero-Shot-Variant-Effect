"""SQLite score cache, keyed so a different window or reference can never collide."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

DEFAULT_CACHE_DIR = Path(".cache") / "zsvep"
DEFAULT_DB_NAME = "scores.sqlite"


class ScoreCache:
    """Persists (scorer, window, strand, window sequence, center, alt) -> llr.

    The sequence hash is part of the key, so scores computed against one
    reference or window can never be returned for a different one.
    """

    def __init__(self, cache_dir: str | Path | None = None, commit_every: int = 200) -> None:
        directory = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / DEFAULT_DB_NAME
        self._conn = sqlite3.connect(self.path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scores (
                scorer_key TEXT NOT NULL,
                window INTEGER NOT NULL,
                strand TEXT NOT NULL,
                seq_sha1 TEXT NOT NULL,
                center INTEGER NOT NULL,
                alt TEXT NOT NULL,
                score REAL NOT NULL,
                PRIMARY KEY (scorer_key, window, strand, seq_sha1, center, alt)
            )
            """
        )
        self._conn.commit()
        self._commit_every = commit_every
        self._pending = 0

    @staticmethod
    def _seq_hash(seq: str) -> str:
        return hashlib.sha1(seq.encode("ascii")).hexdigest()

    def get_many(
        self,
        scorer_key: str,
        window: int,
        strand: str,
        seq: str,
        center: int,
        alts: list[str],
    ) -> dict[str, float]:
        """Return cached scores for whichever of ``alts`` are already known."""
        if not alts:
            return {}
        seq_sha1 = self._seq_hash(seq)
        placeholders = ",".join("?" for _ in alts)
        rows = self._conn.execute(
            f"""
            SELECT alt, score FROM scores
            WHERE scorer_key = ? AND window = ? AND strand = ?
              AND seq_sha1 = ? AND center = ? AND alt IN ({placeholders})
            """,
            (scorer_key, window, strand, seq_sha1, center, *alts),
        ).fetchall()
        return dict(rows)

    def put_many(
        self,
        scorer_key: str,
        window: int,
        strand: str,
        seq: str,
        center: int,
        values: dict[str, float],
    ) -> None:
        """Store newly computed scores. Safe to call repeatedly (re-run safe)."""
        if not values:
            return
        seq_sha1 = self._seq_hash(seq)
        self._conn.executemany(
            """
            INSERT OR REPLACE INTO scores
                (scorer_key, window, strand, seq_sha1, center, alt, score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (scorer_key, window, strand, seq_sha1, center, alt, score)
                for alt, score in values.items()
            ],
        )
        self._pending += len(values)
        if self._pending >= self._commit_every:
            self._conn.commit()
            self._pending = 0

    def close(self) -> None:
        self._conn.commit()
        self._conn.close()

    def __enter__(self) -> ScoreCache:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
