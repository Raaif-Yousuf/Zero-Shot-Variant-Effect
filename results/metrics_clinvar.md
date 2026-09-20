## Notes

Measured on the same shared laptop CPU as `metrics_brca1.md`; see that file for the machine spec
and timing caveat. Every zero-shot scorer (kmer, hyenadna-tiny-1k, hyenadna-small-32k, nt-v2-50m)
sits at or near chance (AUROC 0.4965-0.5288) while phyloP100way reaches 0.928, consistent with the
BRCA1 result -- see `metrics_brca1.md` for the investigation into whether this is a labeling bug
(it is not) and what the LLR scores correlate with instead. `spearman_rho`/`spearman_p` are `nan`
here because ClinVar has no continuous target column (only the binary pathogenic/benign label).

| stratum | scorer | n | n_scored | coverage | n_pos | n_neg | prevalence | auroc | auroc_lo | auroc_hi | auprc | auprc_lo | auprc_hi | spearman_rho | spearman_p | n_spearman |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | kmer_o6 | 2000 | 2000 | 1.0 | 1000 | 1000 | 0.5 | 0.5288 | 0.5041 | 0.5543 | 0.5133 | 0.4947 | 0.5373 | nan | nan | 0 |
| overall | phylop100way | 2000 | 2000 | 1.0 | 1000 | 1000 | 0.5 | 0.928 | 0.9154 | 0.9389 | 0.9369 | 0.9242 | 0.9478 | nan | nan | 0 |
| overall | hyenadna-tiny-1k_full | 2000 | 2000 | 1.0 | 1000 | 1000 | 0.5 | 0.5026 | 0.4777 | 0.5284 | 0.4979 | 0.4791 | 0.52 | nan | nan | 0 |
| overall | hyenadna-small-32k_full | 2000 | 2000 | 1.0 | 1000 | 1000 | 0.5 | 0.4965 | 0.4712 | 0.5211 | 0.4897 | 0.4706 | 0.5115 | nan | nan | 0 |
| overall | nt-v2-50m | 2000 | 2000 | 1.0 | 1000 | 1000 | 0.5 | 0.5228 | 0.499 | 0.5488 | 0.5135 | 0.4947 | 0.5375 | nan | nan | 0 |
| intronic | kmer_o6 | 338 | 338 | 1.0 | 34 | 304 | 0.1006 | 0.5217 | 0.4236 | 0.6134 | 0.1094 | 0.088 | 0.1752 | nan | nan | 0 |
| intronic | phylop100way | 338 | 338 | 1.0 | 34 | 304 | 0.1006 | 0.8619 | 0.7815 | 0.9348 | 0.6985 | 0.5594 | 0.8275 | nan | nan | 0 |
| intronic | hyenadna-tiny-1k_full | 338 | 338 | 1.0 | 34 | 304 | 0.1006 | 0.4886 | 0.3915 | 0.588 | 0.1026 | 0.0834 | 0.1582 | nan | nan | 0 |
| intronic | hyenadna-small-32k_full | 338 | 338 | 1.0 | 34 | 304 | 0.1006 | 0.4718 | 0.3775 | 0.5712 | 0.0959 | 0.0802 | 0.1408 | nan | nan | 0 |
| intronic | nt-v2-50m | 338 | 338 | 1.0 | 34 | 304 | 0.1006 | 0.5179 | 0.4255 | 0.606 | 0.1028 | 0.0866 | 0.1442 | nan | nan | 0 |
| missense | kmer_o6 | 456 | 456 | 1.0 | 355 | 101 | 0.7785 | 0.6105 | 0.553 | 0.6706 | 0.8466 | 0.8164 | 0.8778 | nan | nan | 0 |
| missense | phylop100way | 456 | 456 | 1.0 | 355 | 101 | 0.7785 | 0.9227 | 0.8867 | 0.9575 | 0.9648 | 0.9406 | 0.986 | nan | nan | 0 |
| missense | hyenadna-tiny-1k_full | 456 | 456 | 1.0 | 355 | 101 | 0.7785 | 0.579 | 0.5179 | 0.6415 | 0.8304 | 0.7987 | 0.8646 | nan | nan | 0 |
| missense | hyenadna-small-32k_full | 456 | 456 | 1.0 | 355 | 101 | 0.7785 | 0.59 | 0.535 | 0.65 | 0.8387 | 0.8099 | 0.8705 | nan | nan | 0 |
| missense | nt-v2-50m | 456 | 456 | 1.0 | 355 | 101 | 0.7785 | 0.5953 | 0.5367 | 0.6556 | 0.8355 | 0.8036 | 0.8669 | nan | nan | 0 |
