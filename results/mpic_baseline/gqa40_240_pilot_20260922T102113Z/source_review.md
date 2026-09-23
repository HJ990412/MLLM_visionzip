# MPIC source review

Primary reference: the complete local paper `papers/mpic.md` (SHA-256
`7253687b8a076fbea6e49fc8d9bffc856c3be33b1b7a372cba5fd5d00eaa503b`),
cross-checked against arXiv v2 and the authors' MPIC project/publication
pages.  No author-linked public source repository, commit, or software
license was found, so this work does not claim to run an official MPIC
implementation.

The reproduced paper mechanism is selective attention: all current text rows
and the first *k* canonical expanded image rows are recomputed through every
decoder layer, while cached K/V for the other image rows remains in the full
attention context.  MPIC-*k* names an image-token count; here *k*=32, not 32%,
32 chunks, a score-selected Top-32, or Ours25's 25% retention budget.  The
paper describes dummy cache slots that are replaced before attention and a
single selective prefill rather than a second full-prefix pass.

Paper-unspecified details are implementation choices: the exact Turn-1 source
prompt, Transformers indexed cache assembly, AnyRes structural-row handling,
and post-RoPE phase relocation for shifted positions.  The paper's mixed
cache-hit/cache-miss load/compute overlap has no opportunity in this
single-image all-hit pilot; layer-wise prefetch was not added.  The correct
label is **MPIC-style selective recomputation implemented in our SSD-resident
serving harness**, reported as **MPIC-32 (SSD adaptation)**.

The paper used vLLM 0.9.0, LLaVA-1.6 7B checkpoints, H800 hardware, and
MMDU/SparklesEval-style workloads.  This repository uses Transformers,
LLaVA-1.6 Vicuna 7B in 4-bit NF4/BF16 compute, one RTX 4090, and independent
single-image fixed-prefix GQA questions.  Consequently this is a mechanism
adaptation and local same-run comparison, not an end-to-end reproduction of
the paper's system or performance claims.

Reviewed web references:

- https://arxiv.org/abs/2502.01960
- https://arxiv.org/html/2502.01960v2
- https://shijuzhao.github.io/pic
