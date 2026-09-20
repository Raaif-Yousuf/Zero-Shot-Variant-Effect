## Notes

Subsample size is 60 positions (174 variants), up from an initial 30 used for a first pass; see the
`SWEEP_N_POSITIONS` docstring in `scripts/_bench_common.py`. The selection is the first N of one
seeded `rng.permutation` over all BRCA1 positions, sorted afterward, so it is nested: the 30-position
set from the first pass is a subset of this 60-position set, and any future increase would keep
these 60 as a subset too. All AUROC values here (0.38-0.55, wide bootstrap CIs given the small
n_pos=32/n_neg=128) are consistent with the near-chance BRCA1-wide results in `metrics_brca1.md`;
more context window does not recover signal for any model. Measured on the same shared laptop CPU,
see `metrics_brca1.md` for the machine spec and timing caveat.

| model | window | n | n_scored | n_pos | n_neg | auroc | auroc_lo | auroc_hi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hyenadna-medium-160k_full | 1024 | 174 | 160 | 32 | 128 | 0.436 | 0.3379 | 0.5349 |
| hyenadna-medium-160k_site | 1024 | 174 | 160 | 32 | 128 | 0.4265 | 0.332 | 0.5294 |
| hyenadna-medium-160k_site | 32768 | 174 | 160 | 32 | 128 | 0.4246 | 0.3242 | 0.5247 |
| hyenadna-small-32k_full | 256 | 174 | 160 | 32 | 128 | 0.407 | 0.311 | 0.5046 |
| hyenadna-small-32k_full | 1024 | 174 | 160 | 32 | 128 | 0.394 | 0.2954 | 0.4902 |
| hyenadna-small-32k_full | 4096 | 174 | 160 | 32 | 128 | 0.3928 | 0.2925 | 0.4941 |
| hyenadna-small-32k_full | 16384 | 174 | 160 | 32 | 128 | 0.3948 | 0.2971 | 0.4915 |
| hyenadna-small-32k_full | 32768 | 174 | 160 | 32 | 128 | 0.3909 | 0.2905 | 0.4866 |
| hyenadna-small-32k_site | 256 | 174 | 160 | 32 | 128 | 0.4199 | 0.3188 | 0.5159 |
| hyenadna-small-32k_site | 1024 | 174 | 160 | 32 | 128 | 0.4172 | 0.3147 | 0.5164 |
| hyenadna-small-32k_site | 4096 | 174 | 160 | 32 | 128 | 0.4158 | 0.3142 | 0.5144 |
| hyenadna-small-32k_site | 16384 | 174 | 160 | 32 | 128 | 0.4202 | 0.322 | 0.5193 |
| hyenadna-small-32k_site | 32768 | 174 | 160 | 32 | 128 | 0.4216 | 0.3259 | 0.5205 |
| hyenadna-tiny-1k_full | 1024 | 174 | 160 | 32 | 128 | 0.4417 | 0.3393 | 0.5425 |
| kmer_o6_reference | 256 | 174 | 160 | 32 | 128 | 0.4937 | 0.3645 | 0.6013 |
| kmer_o6_reference | 1024 | 174 | 160 | 32 | 128 | 0.4937 | 0.3645 | 0.6013 |
| kmer_o6_reference | 4096 | 174 | 160 | 32 | 128 | 0.4937 | 0.3645 | 0.6013 |
| kmer_o6_reference | 6144 | 174 | 160 | 32 | 128 | 0.4937 | 0.3645 | 0.6013 |
| kmer_o6_reference | 12282 | 174 | 160 | 32 | 128 | 0.4937 | 0.3645 | 0.6013 |
| kmer_o6_reference | 16384 | 174 | 160 | 32 | 128 | 0.4937 | 0.3645 | 0.6013 |
| kmer_o6_reference | 32768 | 174 | 160 | 32 | 128 | 0.4937 | 0.3645 | 0.6013 |
| nt-v2-50m | 1024 | 174 | 160 | 32 | 128 | 0.3794 | 0.2773 | 0.4849 |
| nt-v2-50m | 6144 | 174 | 160 | 32 | 128 | 0.4502 | 0.3369 | 0.5518 |
| nt-v2-50m | 12282 | 174 | 160 | 32 | 128 | 0.4243 | 0.3144 | 0.5332 |
