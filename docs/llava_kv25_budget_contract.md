# LLaVA-NeXT visual-KV 25% migration contract

Frozen before the GPU correctness run on 2026-09-29 UTC. This contract applies only to
the image-only VisionZip repacked store for `llava-hf/llava-v1.6-vicuna-7b-hf`.
The pre-existing Ours25 experiment is **chunk budget** and remains a separate
method (`ours_chunk25_legacy`); the new method is `ours_kv25` with
`budget_unit=visual_kv`. Existing Qwen and other LLaVA baselines retain their
original semantics and artifacts.

## Frozen inputs and environment

- GQA index: `data/index.json`, SHA256
  `514d1203d248b6f450f5e3bdacda7b931038f9c11df270b415a2e98e5c77e75a`.
  The frozen workload is the first 40 images and `questions[4:10]` for each:
  40 images, 240 questions, workload SHA256
  `97afe02f924a49cadf0c357175b50185e8f16db12b2dd4402595e2bb99d20f66`.
  A count/hash mismatch blocks the pilot; no substitute questions are allowed.
- MT-GQA reconstructed: `data/mt_gqa/dialogues.json`, SHA256
  `2c47cfad2a7ccbb673042b400304d7f3ca03d6fbe59d04fa83db50708c924224`,
  4,061 dialogues on 398 images. A seed-1234 sample of 3–5 distinct-image
  three-turn dialogues is a smoke test, not the main MT experiment.
- GPU correctness pairs, fixed before execution: `n355567/201751701`,
  `n9181/20929611`, `n390187/201861403`, `n133585/202108008`, and
  `n272098/201535625` (each image's `questions[4]`).
- Model/processor, input resolution, prompt and decoding follow the existing
  LLaVA runner. The existing environment is `conda run -n mllm_ft`, torch
  2.5.1+cu121, transformers 4.57.6, RTX 4090, 4-bit NF4, BF16 compute,
  eager attention, FP16 SSD KV payload, 64-token chunks, one separator sidecar.
- The repository's `.git` directory has no HEAD, so Git status/diff cannot
  describe the initial workspace. The pre-edit SHA256 manifest at
  `runs/llava_kv25_migration_20260929T030414Z/protected_before.jsonl`
  covers 28,889 relevant source, input, store, result and Qwen-run files,
  totalling 66,946,647,104 bytes. Its SHA256 is
  `29aaf13b5dbee707d03ffa5ae5cc14755d5abb88c659b4d497f8926b56a5c041`.

## Selection and serving

For an actual expanded image span of `V` tokens, let `S` be the count of
LLaVA structural newline/separator positions and `N=V-S` the real visual
content count. `N=0` is invalid for serving. The explicit KV mode uses
`r=0.25`, `k=ceil(r*N)=(N+3)//4`, `m=ceil(k/64)`, and normal chunk IDs
`[0,...,m-1]`. The generic helper has defined `N=0`, `r=0`, and `r=1`
behavior; bad arguments fail. `k/N` can slightly exceed 25% when `N` is not
divisible by four, with excess less than `1/N`.

The store must have one global stable descending image-only VisionZip
permutation: real rows `[0,N)` and then structural rows in original relative
order. The selected stored IDs are `[0,k)`; selected original IDs come from
the validated stored-to-original permutation. All decoder layers and KV heads
use the same positions for both K and V. Question, answer and history do not
alter selection. No online score, probe read, calibration or diversity step is
permitted.

The normal K and V files are read as whole 64-row chunks, coalesced to one
contiguous `pread` per layer and kind. The original short final chunk ends at
EOF, and no artificial padding bytes are counted. The last bought chunk can
contain real rows beyond `k` or structural tail rows. Separators are also
read through the existing sidecar, including a duplicate read if a boundary
chunk overlaps them. The full original visual KV remains on SSD.

The cache retains the existing full-size GPU buffer, physical row order,
original prefix positions/RoPE, dtype conversion, and eager attention hook.
Only selected real rows `[0,k)` and every structural image row are visible to
prefill and every decode step. Prefix/system rows and causal text suffix remain
visible as before; unselected real rows, padding and future text are hidden.
Reading or scattering an extra row must never make it visible. The mode makes
no GPU memory or H2D reduction claim unless separately measured.

## Correctness gates and fixed thresholds

CPU fixtures: `N=1,63,64,65,127,128,129,256,349,2200`, a short final
chunk, separator-tail overlap, and padding/nonexistent-tail cases. Assert
selection and original IDs independently from stable scores, exact per-layer
and per-head attended content count, minimum chunk count, actual `os.pread`
range/return bytes/calls (independent of `IOCounter`), no probe or score file
read on a hit, and correct structural/system/causal mask. Finite large K/V
sentinels in unused rows must not change prefill or decode outputs; NaN/Inf
are not valid masking sentinels. Interleaved fresh requests must not leak cache
state. Changing only the question must leave selected IDs unchanged and hit
vision/scoring counters at zero.

For five fixed GPU cases, independently rank canonical captured scores and
gather original K/V, converting through the SSD FP16 payload dtype. Assemble
the reference in the same full-size physical cache order, with independent
expected mask construction; the production KV25 selector cannot compute the
reference IDs or mask. Compare selected K/V bits, every head/layer, positions,
mask and counts exactly. Compare matched first-step FP32 logits elementwise
with **`atol=1e-4, rtol=1e-4`**, fixed here before execution; first/generated
token IDs and predictions must match exactly. A legacy-chunk/new-KV equal-set
case must have identical cache bits, mask, first token and output. The 100%
retention diagnostic must match canonical FullLoad logical content, with no
equality requirement for separator read bytes. ReComp and cached paths may
have different precision/shape; report that separately. A failed or unrun
required gate blocks both pilots.

## Conditional pilot and reporting

After correctness passes, run ReComp, FullLoad, Ours-Chunk25-Legacy and
Ours-KV25-New in the same execution. Each method's first question is normal
full-image inference. Image-only saliency/KV capture happens in that forward;
persistence is timed separately and writes only a new run-local store. The two
Ours modes may share an immutable image-only store, with shared physical
persistence and independent-deployment cost attribution reported distinctly.
Five subsequent questions per image are independent cache hits, giving 960
logical requests and 200 hits per method.

True TTFT starts before prompt/input preparation and ends after first-token
materialization and CUDA synchronization. Request completion is separate
E2E. `posix_fadvise(DONTNEED)` page-cache conditioning remains outside the
request timer and makes no NAND/controller coldness claim. Fix warmup/model,
rotate method order deterministically, and pause latency measurement if other
GPU work is active; do not terminate it. MT smoke uses only the previous
generated answer from the same method, and T2/T3 must select the same image
KV IDs.

Raw rows record method ID, budget unit, geometry, selected stored/original
IDs, all layer keep counts, planned spans, actual returned bytes/preads,
first/generated IDs, score, TTFT/E2E, counters and status. Separately report
`k/N`, `(k+S)/(N+S)`, normal payload read/full normal read, and total actual
read/full total actual read. Attribute structural and duplicate read bytes;
compute logical content bytes from true dtype/head/shape, and label physical
I/O as OS-returned bytes. Report per-image average and aggregate ratios,
all-question/hit-only/turn quality, paired old/new quality and TTFT with
image-cluster bootstrap CI, and T1/persistence/activation costs. A CI crossing
zero is not equivalence. Before a future main rerun, review QA-Chunk25 and
ReKV-Chunk25 budget semantics separately; they are unchanged here.
