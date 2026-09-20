# Benchmark run handoff (feat/benchmark-runs, checkpoint)

Checkpointed mid-run because of a context/time limit on the driving session.
Code (plan.json, run_benchmark.py, summarize_results.py, _bench_common.py,
tests) is committed and working. The actual scoring runs are partially done;
this is the state and the exact commands to finish them.

## Machine-contention finding (read this before restarting anything)

Running 3 concurrent `run_benchmark.py` processes (as originally planned, one
per lane, `--threads 4` each = 12 requested threads on a 16-thread machine)
caused an approximately **10-17x slowdown** versus a single isolated process,
not the mild oversubscription penalty expected. Measured examples:
- `hyenadna-tiny-1k` full @ w1024: isolated pilot 0.119 s/position vs 1.399
  s/position with 3 concurrent lanes.
- `hyenadna-small-32k` site @ w1024: isolated pilot 0.288 s/position vs 3.33
  s/position with 3 concurrent lanes.

After dropping to **at most 2 concurrent lanes**, throughput recovered close
to isolated-pilot expectations (e.g. `hyenadna-small-32k` full @ w16384 on the
30-position sweep subsample: 358.5s for 30 positions = 12.0 s/position, in
line with the pilot). **Do not run all 3 lanes at once.** Run at most 2 at a
time; prefer pairing one HyenaDNA-small/medium lane with the lighter lane
(kmer/baselines/tiny/nt), and only start lane2 once lane1 or lane3 has freed
a slot.

The context-sweep subsample size was reduced from the originally planned 150
positions to **30** for the same reason (see the docstring on
`SWEEP_N_POSITIONS` in `scripts/_bench_common.py`): at 150 positions the
full sweep matrix was projected (from pre-restart, contended pilot numbers)
to take many hours. Real measured numbers after the concurrency fix suggest
30 was conservative and a rerun could likely afford more, but 30 is what
every currently-committed run used, so keep it fixed for consistency unless
you also delete and rerun everything that already used it.

## Environment for every command below

```
cd C:/Users/raaif/GitHub_Repo/Zero-Shot-Variant-Effect-wt-3
export HF_HOME=C:/Users/raaif/.cache/vep-hf
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
MAIN_DATA="C:/Users/raaif/GitHub_Repo/Zero-Shot-Variant-Effect/data"
MAIN_REPO="C:/Users/raaif/GitHub_Repo/Zero-Shot-Variant-Effect"
```

## Status as of this checkpoint (see results/runs.tsv for the authoritative, growing log)

Done (20 of 33 plan entries; wall_seconds from results/runs.tsv):
- All 4 kmer orders on BRCA1 (18-41s each), phyloP100way/phyloP_mammalian/CADD
  baselines on BRCA1 (instant), kmer + phyloP100way on ClinVar.
- `hyenadna-tiny-1k` full, BRCA1 all (1854s), `hyenadna-small-32k` full,
  BRCA1 all (2003s).
- `hyenadna-small-32k` full sweep at w256 (7.3s), w1024 (reused, 0.03s),
  w4096 (76.9s), w16384 (358.5s).
- `nt-v2-50m`, BRCA1 all (2198s); sweep at w1024 (reused), w6144 (92.1s),
  w12282 (247.3s).

Still running in the background on this machine when the checkpoint was
taken (they are real OS processes and may have finished by the time you
read this -- check `results/runs.tsv` and `results/scores/` first):
- **lane1** (task/process launched from this session, PowerShell/bash id
  `bux84dpxo`): finishing `brca1_hyenadna-small-32k_full_w32768_sweep`, then
  `clinvar_hyenadna-small-32k_full_w1024`.
- **lane3** (id `bdkqijl0p`): finishing `clinvar_hyenadna-tiny-1k_full_w1024`,
  then `clinvar_nt-v2-50m_w1024`.

Never started:
- **lane2**: `hyenadna-small-32k` site mode (BRCA1 all + 5 sweep windows) and
  `hyenadna-medium-160k` (site @ 1024/32768, full @ 1024 on the sweep
  subsample). 9 plan entries, 0 done.

## Exact resume commands

All three are idempotent: `run_benchmark.py` skips any run_id whose output
TSV already exists, so re-running a lane's full command is always safe.

Lane 1 (hyenadna-small-32k full mode; resumes at w32768 sweep if not already
past it):
```
uv run python scripts/run_benchmark.py --plan benchmarks/plan.json \
  --data-dir "$MAIN_DATA" --out results --cache-dir "$MAIN_REPO/.cache/bench_lane1" \
  --threads 4 -v \
  --only brca1_hyenadna-small-32k_full_w1024 \
  --only brca1_hyenadna-small-32k_full_w256_sweep \
  --only brca1_hyenadna-small-32k_full_w1024_sweep \
  --only brca1_hyenadna-small-32k_full_w4096_sweep \
  --only brca1_hyenadna-small-32k_full_w16384_sweep \
  --only brca1_hyenadna-small-32k_full_w32768_sweep \
  --only clinvar_hyenadna-small-32k_full_w1024
```

Lane 2 (never started -- run this once lane1 or lane3 has finished, so at
most 2 lanes are ever concurrent):
```
uv run python scripts/run_benchmark.py --plan benchmarks/plan.json \
  --data-dir "$MAIN_DATA" --out results --cache-dir "$MAIN_REPO/.cache/bench_lane2" \
  --threads 4 -v \
  --only brca1_hyenadna-small-32k_site_w1024 \
  --only brca1_hyenadna-small-32k_site_w256_sweep \
  --only brca1_hyenadna-small-32k_site_w1024_sweep \
  --only brca1_hyenadna-small-32k_site_w4096_sweep \
  --only brca1_hyenadna-small-32k_site_w16384_sweep \
  --only brca1_hyenadna-small-32k_site_w32768_sweep \
  --only brca1_hyenadna-medium-160k_site_w1024_sweep \
  --only brca1_hyenadna-medium-160k_site_w32768_sweep \
  --only brca1_hyenadna-medium-160k_full_w1024_sweep
```

Lane 3 (nt-v2-50m + hyenadna-tiny-1k + kmer + baselines; resumes at the
ClinVar tail if not already past it):
```
uv run python scripts/run_benchmark.py --plan benchmarks/plan.json \
  --data-dir "$MAIN_DATA" --out results --cache-dir "$MAIN_REPO/.cache/bench_lane3" \
  --threads 4 -v \
  --only brca1_kmer_o2_w1024 --only brca1_kmer_o4_w1024 --only brca1_kmer_o6_w1024 \
  --only brca1_kmer_o8_w1024 --only brca1_phylop100way --only brca1_phylop_mammalian \
  --only brca1_cadd --only brca1_hyenadna-tiny-1k_full_w1024 --only brca1_nt-v2-50m_w1024 \
  --only brca1_kmer_o6_w1024_sweep --only brca1_nt-v2-50m_w1024_sweep \
  --only brca1_nt-v2-50m_w6144_sweep --only brca1_nt-v2-50m_w12282_sweep \
  --only clinvar_kmer_o6_w1024 --only clinvar_phylop100way \
  --only clinvar_hyenadna-tiny-1k_full_w1024 --only clinvar_nt-v2-50m_w1024
```

Or just check `benchmarks/plan.json` and drop `--only` entirely once you are
down to a single lane -- every run_id not yet in `results/scores/**` will run.

## After all 33 plan entries have a row in results/runs.tsv

```
uv run python scripts/summarize_results.py --data-dir "$MAIN_DATA" \
  --plan benchmarks/plan.json --results results --figures docs/figures
```

This writes `results/metrics_brca1.tsv/.md`, `results/metrics_brca1_by_consequence.tsv/.md`,
`results/metrics_clinvar.tsv/.md`, `results/context_sweep.tsv/.md`, and the 6
figures in `docs/figures/`. It only needs whatever run outputs already exist,
so it is safe to run again after each lane finishes to see partial results.

## Before the final commit

- `uv run ruff check .`, `uv run ruff format .`, `uv run pytest -q` clean.
- Check `du -sh results docs/figures` stays under ~5 MB before committing
  (per the task's size cap); `results/scores/**` TSVs are small (tens of KB
  each) so this should not be close.
- Nothing from `data/` or `.cache/` gets committed (already gitignored).
