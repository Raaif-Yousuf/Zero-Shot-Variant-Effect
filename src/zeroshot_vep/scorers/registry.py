"""Name -> scorer factory. Imports are lazy so torch is only needed for model scorers."""

from __future__ import annotations

import importlib
from typing import Any

from zeroshot_vep.scorers.base import Scorer

# Hugging Face revisions are pinned so remote code and weights cannot change underneath us.
HYENADNA = "zeroshot_vep.scorers.hyenadna:HyenaDNAScorer"
NT = "zeroshot_vep.scorers.nucleotide_transformer:NucleotideTransformerScorer"

SCORERS: dict[str, tuple[str, dict[str, Any]]] = {
    "hyenadna-tiny-1k": (
        HYENADNA,
        {
            "model_id": "LongSafari/hyenadna-tiny-1k-seqlen-hf",
            "revision": "e8c1effa8673814e257e627d2e1eda9ea5a373f6",
            "max_window": 1024,
        },
    ),
    "hyenadna-small-32k": (
        HYENADNA,
        {
            "model_id": "LongSafari/hyenadna-small-32k-seqlen-hf",
            "revision": "8fe770c78eb13fe33bf81501612faeddf4d6f331",
            "max_window": 32768,
        },
    ),
    "hyenadna-medium-160k": (
        HYENADNA,
        {
            "model_id": "LongSafari/hyenadna-medium-160k-seqlen-hf",
            "revision": "7ebf71773d22c0ede2cc55cb2be15ee8c289e1ce",
            "max_window": 160000,
        },
    ),
    "nt-v2-50m": (
        NT,
        {
            "model_id": "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
            "revision": "81b29e5786726d891dbf929404ef20adca5b36f1",
            "max_window": 12282,
        },
    ),
    "kmer": ("zeroshot_vep.scorers.kmer:KmerMarkovScorer", {}),
}


def get_scorer(name: str, **kwargs: Any) -> Scorer:
    """Build a scorer by registry name. ``kwargs`` override or extend the defaults."""
    if name not in SCORERS:
        raise KeyError(f"unknown scorer {name!r}; choose from {sorted(SCORERS)}")
    target, defaults = SCORERS[name]
    module_name, class_name = target.split(":")
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls(name=name, **{**defaults, **kwargs})
