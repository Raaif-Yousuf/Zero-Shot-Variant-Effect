# Models

All model scorers are registered in `src/zeroshot_vep/scorers/registry.py`,
which pins a Hugging Face model id and commit revision for each one so
weights and remote code cannot change underneath the project. Weights are
downloaded at run time into `HF_HOME` and are never committed to or
redistributed by this repository.

## Registry

| registry name | Hugging Face id | pinned revision | parameters | license | `max_window` | pretraining context |
| --- | --- | --- | --- | --- | --- | --- |
| `hyenadna-tiny-1k` | `LongSafari/hyenadna-tiny-1k-seqlen-hf` | `e8c1effa8673814e257e627d2e1eda9ea5a373f6` | 438,144 | BSD-3-Clause | 1,024 bp | human genome (hg38), 1,000 bp |
| `hyenadna-small-32k` | `LongSafari/hyenadna-small-32k-seqlen-hf` | `8fe770c78eb13fe33bf81501612faeddf4d6f331` | 3,281,664 | BSD-3-Clause | 32,768 bp | human genome (hg38), 32,000 bp |
| `hyenadna-medium-160k` | `LongSafari/hyenadna-medium-160k-seqlen-hf` | `7ebf71773d22c0ede2cc55cb2be15ee8c289e1ce` | 6,554,624 | BSD-3-Clause | 160,000 bp | human genome (hg38), 160,000 bp |
| `nt-v2-50m` | `InstaDeepAI/nucleotide-transformer-v2-50m-multi-species` | `81b29e5786726d891dbf929404ef20adca5b36f1` | 55,904,972 | CC BY-NC-SA 4.0 | 12,282 bp | 850 NCBI genomes, 1,000 tokens (~6,000 bp) |
| `kmer` | none (trained locally from a reference FASTA) | n/a | order-dependent (4^(order+1) counts; ~16,384 at the default order 6) | n/a (this project's code) | unbounded | trained on whichever chromosomes the input variants cover |

Parameter counts were measured with `sum(p.numel() for p in model.parameters())`
on the loaded Hugging Face model (`nt-v2-50m`'s count includes its masked-LM
head). HyenaDNA's pretraining genome and context length, and Nucleotide
Transformer's pretraining corpus and context length, are from each model's
Hugging Face model card.

Note that `nt-v2-50m`'s `max_window` (12,282 bp, driven by its
`max_position_embeddings = 2050`, verified from `config.json`) is larger than
the roughly 6,000 bp (1,000 tokens) it was actually pretrained on; scoring
near the top of that window relies on the model's rotary position embeddings
extrapolating past their training length, not on the model having seen
windows that long during pretraining.

Licenses were read from each model's Hugging Face model card metadata
(`license` field). HyenaDNA is BSD-3-Clause. Nucleotide Transformer v2 is
CC BY-NC-SA 4.0, which is non-commercial; this project only downloads the
weights to run them locally for research and never redistributes them.

## Measured CPU timings

One `score_window` call, 3 alts, a random ACGT window, on this project's
development machine (Intel Core Ultra 7 255H, a shared laptop, so these are
indicative, not a controlled benchmark):

| model | window | mode | time |
| --- | --- | --- | --- |
| `hyenadna-small-32k` | 1,024 | full | ~1.0 s |
| `hyenadna-small-32k` | 32,768 | full | ~10.8 s |
| `hyenadna-small-32k` | 1,024 | site | ~0.9 s |
| `hyenadna-small-32k` | 32,768 | site | ~4.5 s |
| `hyenadna-medium-160k` | 1,024 | full | ~2.7 s |
| `hyenadna-medium-160k` | 32,768 | full | ~20.2 s |
| `nt-v2-50m` | 1,024 | - | ~0.2 s |
| `nt-v2-50m` | 6,000 | - | ~5.5 s |

Thread count matters more than these headline numbers suggest: for
`hyenadna-small-32k` at a 4,096 bp window, `num_threads=4` (~1.8 s) beat both
fewer and more threads (~2.2-3.3 s at `num_threads=16`), so `--threads` is
worth tuning per model and window size rather than always using every core.

## Models considered but not used

Two more DNA language models were evaluated for inclusion and dropped after a
short, time-boxed check in a throwaway virtual environment (CPU-only,
`torch` + `transformers<5`, `trust_remote_code=True`), not integrated into
the registry:

- **Caduceus** (`kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16`):
  the tokenizer loads, but the model's remote modeling code unconditionally
  imports `mamba_ssm`, and `mamba_ssm` failed to build from source on this
  machine: its `setup.py` requires `nvcc` (the CUDA toolkit) to determine a
  version string, and since this is a CPU-only Windows machine with no CUDA
  toolkit installed, the build failed with
  `NameError: name 'bare_metal_version' is not defined` before it could even
  attempt a CPU code path. There is no CPU fallback in the model's remote
  code; Caduceus was not pursued further.
- **DNABERT-2** (`zhihan1996/DNABERT-2-117M`): the tokenizer loads directly.
  The model's remote modeling code additionally needs `einops` (already a
  dependency of this project's `models` extra, so trivial to add) and then
  `triton`. `triton` has no published wheel for Windows for any Python
  version compatible with this project (PyPI only ships `manylinux`/Linux
  wheels for current `triton` releases, and the last wheels old enough to
  target Windows do not cover this project's Python version), so the model
  could not be loaded at all on this platform. DNABERT-2 was not pursued
  further; it may work on Linux, which was not tested here.
