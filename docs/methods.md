# Methods

## Score definition and sign convention

Every scorer implements `score_window(ref_seq, center, alts) -> list[float]`, one
natural-log likelihood ratio per alt:

```
llr = log P(alt sequence) - log P(ref sequence)
```

under whatever model the scorer wraps. Negative means the alt is less likely
than the reference under the model, which is the expected direction for a
damaging variant. Benchmark metrics (AUROC, AUPRC, Spearman) expect the
opposite convention, higher score means more damaging, so `evaluate.py`
callers pass `-llr` as the damaging score, never raw `llr`.

## Window semantics

The engine (`score_variants` in `engine.py`) extracts a window of length
`window` around each variant. For a 1-based VCF position `pos` and
`center = window // 2` (0-based):

```
start0 = (pos - 1) - center
end0   = start0 + window
ref_seq = reference.fetch(chrom, start0, end0)
```

so the variant base sits at `ref_seq[center]`. If the window runs off either
end of the chromosome, `reference.fetch` pads with `"N"`; windows containing N
elsewhere are allowed and passed through to the scorer unchanged.

Before scoring, the engine checks `ref_seq[center]` against the variant's
declared `ref` allele. A mismatch does not raise; the row is written with
`status = "ref_mismatch"` and `llr`/`llr_fwd`/`llr_rev` all `NaN`. This is the
only case where a variant is silently skipped rather than scored.

Variants at the same `(chrom, pos)` are grouped so the scorer is called once
per position per strand, with every alt at that position batched into one
call.

Each scorer also declares `max_window`; the engine rejects `window` values
above it with a clear error before doing any work (`kmer`'s `max_window` is
`None`, so it accepts any window).

## Strand averaging

For `strands="both"` (the default), the engine scores the forward window and
then the reverse complement of the same window, and reports both:

```
rev_seq    = reverse_complement(ref_seq)
rev_center = window - 1 - center
rev_alts   = [reverse_complement(a) for a in alts]
llr_fwd = scorer.score_window(ref_seq, center, alts)
llr_rev = scorer.score_window(rev_seq, rev_center, rev_alts)
llr     = (llr_fwd + llr_rev) / 2
```

The center mirrors because reversing a window of length `window` maps index
`i` to `window - 1 - i`; the alt base is complemented because the reverse
complement operation also complements every base. For `strands="forward"`,
only the forward pass runs and `llr = llr_fwd` (`llr_rev` is `NaN`).

## HyenaDNA (causal, full and site mode)

HyenaDNA is a causal, character-level language model: its tokenizer maps each
base to its own token id and appends one trailing `[SEP]` token (verified for
`hyenadna-tiny-1k-seqlen-hf`: tokenizing 1000 bases yields 1001 ids, the last
of which is `[SEP]`). A causal model's logits at position `t` predict token
`t + 1` from the prefix `tokens[:t+1]` only.

**`mode="full"`** (the default) builds the alt sequence by substituting the
alt base at `center`, tokenizes both the ref and every alt sequence (they are
the same length, so no padding is needed), and runs them through the model as
one batch. The score is:

```
sum_{t=center}^{n_bases-1} log p(alt_tokens[t] | alt_tokens[:t])
  - sum_{t=center}^{n_bases-1} log p(ref_tokens[t] | ref_tokens[:t])
```

The sum only needs to start at `t = center`, not `t = 0`: every token before
`center` is identical between the ref and alt sequences, and because the
model is causal, `log p(token[t] | prefix)` for `t < center` only depends on
that identical prefix, so those terms are identical between the two sums and
cancel exactly. Restricting the sum to `t >= center` is therefore not an
approximation; `tests/test_hyenadna.py` checks this directly against an
unrestricted brute-force sum over the whole sequence. The sum also stops
before the trailing `[SEP]` token: the row that would predict `[SEP]` (and
anything after it) is excluded, so a model's behavior at the sequence
boundary never enters the score.

**`mode="site"`** is cheaper: it runs only the reference window through the
model once, reads the logits row that predicts the base at `center` (row
`center - 1`), and returns `log p(alt) - log p(ref)` from that one row. This
needs at least one base of left context (`center >= 1`) and ignores how the
substitution would change the model's predictions for bases after `center`;
`full` mode captures that, `site` does not.

## Nucleotide Transformer v2 (masked marginal, 6-mer tokens)

The Nucleotide Transformer tokenizer prepends a `<cls>` token, then walks the
sequence from position 0 in non-overlapping 6-base blocks: a block becomes one
6-mer token only if it is exactly 6 bases of A/C/G/T, otherwise it is split
into one single-character token per base (this was verified by decoding
tokenizer output for sequences containing N and lengths that are not
multiples of 6; see `zeroshot_vep.scorers._llr.nt_token_layout`).

Because chunking starts from position 0, whether the variant lands inside a
clean 6-mer token depends on the window's frame, which the engine does not
control (it always centers the window on the variant). The scorer
deterministically trims 0 to 5 bases off the start of the window, smallest
trim first, until the variant falls inside a full, N-free 6-mer token
(`nt_locate_kmer`). This never changes which bases are ref or alt, only which
of the six possible reading frames tokenizes the window.

Once a clean frame is found, the covering 6-mer token is masked and the
window runs through the model in a single forward pass. The score is the
masked-marginal log-likelihood ratio at that position:

```
llr = log p(alt 6-mer | masked context) - log p(ref 6-mer | masked context)
```

All alts at a position share the one forward pass, since only the alt 6-mer
being read off differs.

## Order-k Markov baseline (`kmer`)

`KmerMarkovScorer` is a genomic language model without deep learning: an
order-k Markov chain over A/C/G/T, trained by counting (k+1)-mers in the
reference. Both strands are counted, by permuting each forward-strand
(k+1)-mer's count to the index of its reverse complement rather than
re-encoding and re-scanning the reverse-complemented chromosome. Any
(k+1)-mer touching an N is excluded from training.

Conditional probabilities use Dirichlet (add-`pseudocount`) smoothing over
the 4 possible next bases:

```
p(base | context) = (count(context, base) + pseudocount) / (count(context) + 4 * pseudocount)
```

`score_window(ref_seq, center, alts)` computes, for each alt, the exact
telescoping sum over the `order + 1` positions whose (k+1)-mer context
includes the substituted base (`i` from `center` to `center + order`):

```
sum_i [ log p(x_i | context_i, with alt at center)
      - log p(x_i | context_i, with ref at center) ]
```

Positions outside `ref_seq` or whose (k+1)-mer touches an N are skipped
(contribute 0). All other positions cancel exactly, the same telescoping
argument as HyenaDNA's full-mode sum, so this is an exact score under the
Markov model, not an approximation.

## Caching

The score cache (`ScoreCache` in `cache.py`) keys each stored score on
`(scorer.cache_key(), window, strand, sha1(window sequence), center, alt)`.
The SHA-1 is computed on the exact string passed to `score_window`, so a
different reference FASTA, a different window size, or a different center
can never collide with, or return a stale value for, a different one; the key
includes both strands separately (`"fwd"`/`"rev"`), never an already-averaged
`llr`. Each scorer's `cache_key()` must include everything that changes its
output (model id and pinned revision and mode for the model scorers; order,
pseudocount and the trained chromosomes/fasta identity for `kmer`), which is
enforced by scorer-level tests, not by the cache itself.

## Known limitations

- HyenaDNA is purely causal: a single forward pass only ever conditions on
  one side of the variant (upstream on the strand it is run on).
  Strand-averaging scores both directions once each, but neither individual
  pass sees both flanks at once the way a bidirectional model would.
- The Nucleotide Transformer's 6-mer tokenizer falls back to one token per
  base whenever a block is not a clean ACGT 6-mer (e.g. it contains an N). A
  window with enough Ns can therefore produce more tokens than its base
  length divided by 6, and in principle more tokens than the model was
  trained on, even though the window itself is within `max_window` bases.
- Zero-shot LLR scores are relative rankings under a language model, not
  calibrated probabilities of pathogenicity. A more negative score means
  "less likely under the model", not a probability of disease causation, and
  scores are not comparable in absolute terms across different scorers or
  window sizes.
