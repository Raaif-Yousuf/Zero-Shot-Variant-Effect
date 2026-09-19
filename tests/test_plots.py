"""Plot functions write non-empty PNG files. Uses the non-interactive Agg backend."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from zeroshot_vep.plots import (  # noqa: E402
    plot_auroc_vs_window,
    plot_pr_overlay,
    plot_roc_overlay,
    plot_score_distributions,
)


def _score_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 40
    label = (rng.random(n) < 0.4).astype(float)
    frames = []
    for scorer in ["kmer", "phylop"]:
        score = label * 2 + rng.normal(size=n)
        frames.append(pd.DataFrame({"scorer": scorer, "label": label, "score": score}))
    return pd.concat(frames, ignore_index=True)


def test_plot_roc_overlay_writes_nonempty_file(tmp_path):
    out = plot_roc_overlay(_score_df(), tmp_path / "roc.png")
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_pr_overlay_writes_nonempty_file(tmp_path):
    out = plot_pr_overlay(_score_df(), tmp_path / "pr.png")
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_score_distributions_writes_nonempty_file(tmp_path):
    out = plot_score_distributions(_score_df(), tmp_path / "dist.png")
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_auroc_vs_window_writes_nonempty_file(tmp_path):
    df = pd.DataFrame(
        {
            "model": ["a", "a", "a", "b", "b", "b"],
            "window": [128, 512, 2048, 128, 512, 2048],
            "auroc": [0.6, 0.7, 0.75, 0.55, 0.6, 0.65],
            "auroc_lo": [0.5, 0.6, 0.65, 0.45, 0.5, 0.55],
            "auroc_hi": [0.7, 0.8, 0.85, 0.65, 0.7, 0.75],
        }
    )
    out = plot_auroc_vs_window(df, tmp_path / "window.png")
    assert out.exists()
    assert out.stat().st_size > 0
