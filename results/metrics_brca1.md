## Notes

Measured on a shared Windows 11 laptop (Intel Core Ultra 7 255H, 16 threads, 32 GB RAM); other
unrelated processes on the same machine (a second ML training job, editor tooling) ran during
parts of this benchmark, so individual wall times in `results/runs.tsv` are upper bounds rather
than clean isolated measurements except where a run was the only benchmark lane active. Summed
`wall_seconds` across all 33 plan entries is about 21833s (6.1h), but at most 2 scoring lanes ran
concurrently, so true wall-clock elapsed time was well under that.

Every zero-shot scorer here (kmer at orders 2/4/6/8, hyenadna-tiny-1k, hyenadna-small-32k in both
"full" and "site" mode, nt-v2-50m) gets an AUROC significantly *below* 0.5 on BRCA1 -- the
bootstrap 95% CI is entirely below 0.5 in every case -- while phyloP100way, phyloP_mammalian and
CADD score properly above 0.79. This was checked for a labeling or sign bug and does not look like
one: `label` vs `function_score` in the raw dataset correlates at -0.92 (LOF variants have far more
negative function_score, the expected direction for Findlay et al.'s convention), and the baseline
scorers recover the expected AUROC using the same label column and evaluation code. Instead,
`-llr` for every zero-shot scorer is essentially uncorrelated with phyloP100way (Spearman rho
-0.007 to -0.03 for hyenadna-small-32k_full and nt-v2-50m, not significant) -- these small models
do not appear to carry measurable evolutionary-conservation signal at this locus, rather than
being anti-correlated with damage. On ClinVar the same scorers sit at or near chance
(AUROC 0.4965-0.5288, see `metrics_clinvar.md`) while phyloP100way reaches 0.928.

Both-strand scores (`llr`, the mean of forward and reverse complement) versus forward-only scores
(`llr_fwd`) are highly correlated for kmer (r about 1.0) and hyenadna-small-32k full mode
(r about 0.98), but only moderately correlated for nt-v2-50m (r about 0.49) and hyenadna-small-32k
site mode (r about 0.23) -- for those two the forward and reverse strand scores disagree
substantially, adding noise to the both-strand average on top of an already weak signal.

Coverage is 1.0 (no missing scores) for every scorer on both datasets in every table below; the
only rows excluded from `n_scored` are Findlay's intermediate ("INT") function class, which has no
binary label by design (249 of 3893 BRCA1 rows).

| stratum | scorer | n | n_scored | coverage | n_pos | n_neg | prevalence | auroc | auroc_lo | auroc_hi | auprc | auprc_lo | auprc_hi | spearman_rho | spearman_p | n_spearman |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | kmer_o2 | 3893 | 3644 | 1.0 | 823 | 2821 | 0.2259 | 0.4667 | 0.445 | 0.4906 | 0.2101 | 0.2007 | 0.2242 | 0.0551 | 0.0006 | 3893 |
| overall | kmer_o4 | 3893 | 3644 | 1.0 | 823 | 2821 | 0.2259 | 0.4613 | 0.4404 | 0.4844 | 0.2106 | 0.2003 | 0.2257 | 0.0819 | 0.0 | 3893 |
| overall | kmer_o6 | 3893 | 3644 | 1.0 | 823 | 2821 | 0.2259 | 0.4648 | 0.442 | 0.4878 | 0.2055 | 0.1965 | 0.2183 | 0.0763 | 0.0 | 3893 |
| overall | kmer_o8 | 3893 | 3644 | 1.0 | 823 | 2821 | 0.2259 | 0.469 | 0.4469 | 0.4927 | 0.2122 | 0.2016 | 0.2275 | 0.0767 | 0.0 | 3893 |
| overall | phylop100way | 3893 | 3644 | 1.0 | 823 | 2821 | 0.2259 | 0.8256 | 0.8087 | 0.8407 | 0.5494 | 0.5189 | 0.5833 | -0.4249 | 0.0 | 3893 |
| overall | phylop_mammalian | 3893 | 3644 | 1.0 | 823 | 2821 | 0.2259 | 0.7927 | 0.7744 | 0.809 | 0.471 | 0.4428 | 0.5012 | -0.397 | 0.0 | 3893 |
| overall | cadd | 3893 | 3644 | 1.0 | 823 | 2821 | 0.2259 | 0.8168 | 0.8005 | 0.8313 | 0.6057 | 0.578 | 0.6312 | -0.425 | 0.0 | 3893 |
| overall | hyenadna-tiny-1k_full | 3893 | 3644 | 1.0 | 823 | 2821 | 0.2259 | 0.4528 | 0.4299 | 0.4751 | 0.2036 | 0.1941 | 0.216 | 0.0956 | 0.0 | 3893 |
| overall | hyenadna-small-32k_full | 3893 | 3644 | 1.0 | 823 | 2821 | 0.2259 | 0.4523 | 0.4296 | 0.4741 | 0.2023 | 0.1933 | 0.2144 | 0.1058 | 0.0 | 3893 |
| overall | hyenadna-small-32k_site | 3893 | 3644 | 1.0 | 823 | 2821 | 0.2259 | 0.4353 | 0.4138 | 0.4574 | 0.195 | 0.1874 | 0.206 | 0.1462 | 0.0 | 3893 |
| overall | nt-v2-50m | 3893 | 3644 | 1.0 | 823 | 2821 | 0.2259 | 0.4456 | 0.4257 | 0.4664 | 0.1989 | 0.191 | 0.2107 | 0.0966 | 0.0 | 3893 |
