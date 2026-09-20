# Results

## What was measured

Every scorer in `src/zeroshot_vep/scorers/registry.py` (order-2/4/6/8 Markov
baselines, phyloP100way, phyloP_mammalian, CADD, HyenaDNA tiny-1k, HyenaDNA
small-32k in full and site mode, and Nucleotide Transformer v2 50M) was run
against two labelled variant sets, and each pretrained model's context
window was separately swept from 256 bp to 32,768 bp on a subsample. All
runs are logged in `results/runs.tsv`; the full per-variant scores are under
`results/scores/`.

- **BRCA1 saturation genome editing** (Findlay et al. 2018): 3,893 SNVs in
  and around *BRCA1*, hg19 chr17, each labelled loss-of-function (LOF, 823),
  functional (FUNC, 2,821) or intermediate (INT, 249, excluded from the
  binary evaluation). A continuous functional score is also available per
  variant.
- **ClinVar chr17** (2026-09-05 GRCh37 archive): a seeded, stratified
  subsample of 2,000 SNVs (1,000 benign, 1,000 pathogenic) out of 84,827
  chr17 SNVs with at least one review star, drawn from the pathogenic and
  benign classes only.

phyloP100way and CADD are not zero-shot: phyloP is a fixed, published
conservation track and CADD is a supervised classifier trained on labelled
variants, so both are baselines/reference points, not entries in the same
comparison as the DNA language models.

## BRCA1 results

Overall (`results/metrics_brca1.md`), AUROC with its bootstrap 95% CI and
Spearman rho against the continuous Findlay function score:

| scorer | auroc | auroc_lo | auroc_hi | spearman_rho | spearman_p |
| --- | --- | --- | --- | --- | --- |
| kmer_o2 | 0.4667 | 0.445 | 0.4906 | 0.0551 | 0.0006 |
| kmer_o4 | 0.4613 | 0.4404 | 0.4844 | 0.0819 | 0.0 |
| kmer_o6 | 0.4648 | 0.442 | 0.4878 | 0.0763 | 0.0 |
| kmer_o8 | 0.469 | 0.4469 | 0.4927 | 0.0767 | 0.0 |
| hyenadna-tiny-1k_full | 0.4528 | 0.4299 | 0.4751 | 0.0956 | 0.0 |
| hyenadna-small-32k_full | 0.4523 | 0.4296 | 0.4741 | 0.1058 | 0.0 |
| hyenadna-small-32k_site | 0.4353 | 0.4138 | 0.4574 | 0.1462 | 0.0 |
| nt-v2-50m | 0.4456 | 0.4257 | 0.4664 | 0.0966 | 0.0 |
| phylop_mammalian | 0.7927 | 0.7744 | 0.809 | -0.397 | 0.0 |
| phylop100way | 0.8256 | 0.8087 | 0.8407 | -0.4249 | 0.0 |
| cadd (supervised, reference only) | 0.8168 | 0.8005 | 0.8313 | -0.425 | 0.0 |

Every zero-shot scorer's 95% CI sits entirely below 0.5. phyloP100way and
CADD both clear 0.8. See `docs/figures/brca1_roc_overlay.png`,
`docs/figures/brca1_pr_overlay.png` and
`docs/figures/brca1_score_distributions.png`.

By consequence (`results/metrics_brca1_by_consequence.md`; only Missense and
Splice region have at least 10 labelled variants in both classes):

| consequence | scorer | n | auroc | auroc_lo | auroc_hi |
| --- | --- | --- | --- | --- | --- |
| Missense | kmer_o6 | 2086 | 0.4661 | 0.4343 | 0.4971 |
| Missense | hyenadna-small-32k_full | 2086 | 0.4694 | 0.4375 | 0.5007 |
| Missense | nt-v2-50m | 2086 | 0.4421 | 0.4105 | 0.4712 |
| Missense | phylop100way | 2086 | 0.7977 | 0.7737 | 0.8206 |
| Missense | cadd | 2086 | 0.756 | 0.7319 | 0.7778 |
| Splice region | kmer_o6 | 446 | 0.5079 | 0.4402 | 0.5743 |
| Splice region | hyenadna-small-32k_full | 446 | 0.5158 | 0.4494 | 0.5787 |
| Splice region | nt-v2-50m | 446 | 0.5006 | 0.4339 | 0.5683 |
| Splice region | phylop100way | 446 | 0.8124 | 0.7668 | 0.857 |
| Splice region | cadd | 446 | 0.7259 | 0.666 | 0.7759 |

The zero-shot scorers are at chance in both consequence classes; phyloP and
CADD are not.

## ClinVar results

Overall and by consequence (`results/metrics_clinvar.md`); only overall,
intronic and missense strata have at least 10 labelled variants per class:

| stratum | scorer | n | auroc | auroc_lo | auroc_hi |
| --- | --- | --- | --- | --- | --- |
| overall | kmer_o6 | 2000 | 0.5288 | 0.5041 | 0.5543 |
| overall | hyenadna-tiny-1k_full | 2000 | 0.5026 | 0.4777 | 0.5284 |
| overall | hyenadna-small-32k_full | 2000 | 0.4965 | 0.4712 | 0.5211 |
| overall | nt-v2-50m | 2000 | 0.5228 | 0.499 | 0.5488 |
| overall | phylop100way | 2000 | 0.928 | 0.9154 | 0.9389 |
| intronic | kmer_o6 | 338 | 0.5217 | 0.4236 | 0.6134 |
| intronic | phylop100way | 338 | 0.8619 | 0.7815 | 0.9348 |
| missense | kmer_o6 | 456 | 0.6105 | 0.553 | 0.6706 |
| missense | phylop100way | 456 | 0.9227 | 0.8867 | 0.9575 |

The missense stratum is the one place a zero-shot scorer clears 0.6
(kmer_o6, 0.6105), still far below phyloP100way's 0.9227 on the same rows.
See `docs/figures/clinvar_roc_overlay.png`.

## Context sweep

`results/context_sweep.md`: AUROC on a fixed 174-variant (60-position)
BRCA1 subsample, window swept from 256 bp to 32,768 bp:

| model | window | auroc | auroc_lo | auroc_hi |
| --- | --- | --- | --- | --- |
| hyenadna-small-32k_full | 256 | 0.407 | 0.311 | 0.5046 |
| hyenadna-small-32k_full | 1024 | 0.394 | 0.2954 | 0.4902 |
| hyenadna-small-32k_full | 4096 | 0.3928 | 0.2925 | 0.4941 |
| hyenadna-small-32k_full | 16384 | 0.3948 | 0.2971 | 0.4915 |
| hyenadna-small-32k_full | 32768 | 0.3909 | 0.2905 | 0.4866 |
| hyenadna-small-32k_site | 256 | 0.4199 | 0.3188 | 0.5159 |
| hyenadna-small-32k_site | 32768 | 0.4216 | 0.3259 | 0.5205 |
| hyenadna-medium-160k_full | 1024 | 0.436 | 0.3379 | 0.5349 |
| hyenadna-medium-160k_site | 1024 | 0.4265 | 0.332 | 0.5294 |
| hyenadna-medium-160k_site | 32768 | 0.4246 | 0.3242 | 0.5247 |
| nt-v2-50m | 1024 | 0.3794 | 0.2773 | 0.4849 |
| nt-v2-50m | 6144 | 0.4502 | 0.3369 | 0.5518 |
| nt-v2-50m | 12282 | 0.4243 | 0.3144 | 0.5332 |
| kmer_o6_reference | any | 0.4937 | 0.3645 | 0.6013 |

Widening the window from 256 bp up to the largest window each model
supports does not recover any signal; every AUROC stays inside roughly
0.38-0.55 with wide, overlapping confidence intervals. See
`docs/figures/context_sweep_auroc.png`.

## Cost

Measured wall-clock time on the project's development machine (Intel Core
Ultra 7 255H, 4 threads requested), scoring the same 3,893 BRCA1 variants
(`results/runs.tsv`; other unrelated processes shared the machine during
parts of the run, so these are upper bounds):

| scorer | wall time |
| --- | --- |
| phyloP100way | 0.004 s |
| order-6 k-mer | 36.457 s |
| HyenaDNA-small (full mode) | 2,003.062 s |
| Nucleotide Transformer v2 50M | 2,197.621 s |

phyloP is a table lookup; the k-mer baseline is a few dozen seconds; the two
pretrained models each take on the order of half an hour for under 4,000
variants on this laptop's CPU, roughly 500-560 ms per variant.

## Interpretation

**The negative result is about the models, not the code.** Two positive
controls (`results/sanity_checks.md`, `scripts/sanity_checks.py`) check that
the same model-loading and scoring code actually works on real sequence:

- Nucleotide Transformer (nt-v2-50m), masked-token recovery on 300 seeded
  ACGT-only hg19 chr17 windows: top-1 accuracy recovering the full masked
  6-mer is 0.1267 against a chance rate of 1/4096 (0.000244), and top-1
  accuracy for the center base alone is 0.4333 against a chance rate of
  0.25.
- HyenaDNA-small-32k, real sequence versus a dinucleotide-composition-matched
  shuffle (Altschul-Erikson shuffle) of the same 200 seeded 1,024 bp
  windows: real windows score a mean 1.5896 bits/base versus 1.9406 for
  their shuffles, and every one of the 200 windows (200/200) scores the real
  sequence more likely than its own shuffle.

Both models clearly do what a language model over real DNA should do. Their
failure on BRCA1 and ClinVar is not an artifact of broken weights, a wrong
tokenizer alignment or a sign error in this project's scoring code.

**What the LLR tracks instead.** `results/metrics_brca1.md` already records
that `-llr` from hyenadna-small-32k_full and nt-v2-50m is essentially
uncorrelated with phyloP100way (Spearman rho -0.007 to -0.03, not
significant) at this locus: these models do not appear to carry measurable
evolutionary-conservation signal here, rather than being anti-correlated
with damage. `results/sanity_checks.md` breaks the same committed BRCA1
scores down further. Both models score CpG-context substitutions with a
markedly less negative (less "damaging") mean LLR than non-CpG
substitutions: hyenadna-small-32k_full averages +2.2726 at CpG sites versus
-0.457 elsewhere, and nt-v2-50m averages +1.6685 versus -0.4592. Transitions
also score somewhat less negative than transversions for both models
(hyenadna-small-32k_full: -0.5042 vs -0.3882; nt-v2-50m: -0.4963 vs
-0.4054). Findlay consequence class makes little difference: mean LLR is
similar across Missense, Splice region, Synonymous, Nonsense and Intronic
for both models. Read together, this points at the LLR mostly tracking
local sequence and mutation-type statistics that these small models learned
from unlabelled genomic sequence (CpG sites mutate at an elevated rate and
these models apparently expect that), rather than tracking whether a
substitution actually breaks BRCA1 function. `docs/figures/best_model_vs_function_score.png`
shows the scatter of the model with the strongest (still weak) correlation
to the continuous Findlay function score, hyenadna-small-32k_site (Spearman
rho 0.1462), against that function score directly.

**Limitations.**

- The models benchmarked here range from 438,144 to 55,904,972 parameters,
  chosen specifically because they run on a laptop CPU (see
  `docs/models.md`). Published zero-shot genomic variant-effect work
  generally uses much larger models on GPUs; these results do not speak to
  whether that class of model would do better here.
- Alignment-based methods use evolutionary conservation explicitly by
  construction; phyloP100way, the strongest baseline in this benchmark, is
  exactly a measure of cross-species conservation, not a competing
  zero-shot sequence model.
- CADD is supervised (trained on labelled variant data) and is included as
  a reference point for how strong a well-established variant-effect score
  can get on this data, not as a zero-shot competitor.
- The context-sweep subsample is 174 variants at 60 positions (32
  loss-of-function, 128 functional), so its confidence intervals are wide
  (typically 0.15-0.2 AUROC wide) and it cannot rule out a modest true
  effect of window size.
- ClinVar's benign and pathogenic sets were not matched for consequence
  composition: pathogenic variants are disproportionately missense
  (355/456 missense variants in the subsample are pathogenic, versus
  34/338 intronic variants), which the by-consequence breakdown in
  `results/metrics_clinvar.md` should be read alongside the overall figure.
