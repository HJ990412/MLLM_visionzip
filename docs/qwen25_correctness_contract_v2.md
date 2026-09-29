# Qwen2.5-VL-7B-Instruct correctness contract v2

## Status and scope

This is a **prospectively frozen revised protocol** for the new
`runs/qwen25_correctness_v2_20260928T073413Z/` execution. It was written
after inspecting the v1 failures and the separate debugging investigation,
and before running the v2 GPU validation. The earlier v1 outcome remains
**13 PASS / 2 FAIL** under its original elementwise BF16 logit criterion
`atol=0.125, rtol=0.02`. The debugging rerun has the same outcome. Neither
run is relabeled or modified. V1 was a conservative initial contract; the
investigation showed that two of its strict comparisons join different
low-precision execution shapes. V2 retains those comparisons as numerical
and semantic diagnostics and tests SSD serving against independently
assembled, matched-computation memory references.

The claim available after all required gates pass is limited to storage,
selection, serving, and checked position/mask/attention semantics for this
implementation and these inputs. It does not imply numerical identity to
full-image recomputation, equal task quality, or transfer to another model.

## Frozen model and workload

| Item | Value |
|---|---|
| Model, processor, tokenizer | `Qwen/Qwen2.5-VL-7B-Instruct`, revision `cc594898137f460bfe9f0759e9844b3ce807cfb5` |
| Weight and compute | NF4 4-bit, double quantization, BF16 compute and native BF16 KV payload |
| Attention | production SDPA for every arm; eager/FP32 only isolated diagnostics |
| Input | single image, batch 1, `min_pixels=200704`, `max_pixels=802816`, unchanged system/user prompt |
| Generation | seed 1234, greedy, at most 16 new tokens, checkpoint EOS |
| KV | 28 decoder layers, 4 native KV heads, 128 head dimensions; no stored `repeat_kv` heads |
| Image ranking | last full-attention ViT block's non-CLS key-received attention; one image-only permutation per image |
| Storage | all original visual KV retained on SSD, score-ordered; structural KV separately retained |
| Hit | fixed first-k sequential read, 64 final LLM visual tokens/chunk, nominal 25% `budget_chunk_count` rounding/clamping |

No online query scoring, calibration, diversity, contextual merge, altered
resolution, smaller Qwen chunk, changed budget, or new persistent text/history
KV path is in scope. The frozen configuration and precise image/question IDs
are in `runs/qwen25_correctness_v2_20260928T073413Z/validation_manifest.json`
(canonical content SHA256
`6458545772438ef1b41cd35bbe7ec17c319be6f779d1b939fc9a94e0ada928a8`).
The first three IDs, `201751701`, `201751740`, `201751873`, are the previously
seen failure/reproduction set on image `n355567`. The additional deterministic
set is `20929611`, `201861403`, `202108008`, `201535625`, `202101069`,
`2093976`, `202144724` on the index's next seven images. These additional
examples are validation samples, not an unseen holdout. The fixed smoke
workload is a separate 20-image × 3-question manifest from index rows 20–39,
canonical digest
`878ba93ade9b5c7a179af5292bc38a716b9c4485c1372aff9281629f39d346bf`.
The GQA and MT pilot workloads must reproduce their original frozen manifests
exactly; a mismatch blocks the affected pilot.

The pre-execution freeze records SHA256 for this contract, the configuration,
both manifests, v2 validation and pilot code, production adapter, dataset
indices, and original evidence. The freeze record is written before any v2
GPU result. A changed code/config/manifest after that point requires a newly
named prospective run and is never applied retroactively.

## Correctness hierarchy and references

### Level 1: BF16 storage serialization (G1)

The reference is every captured native BF16 prefix K/V tensor before SSD
serialization. Compare against canonical SSD write/read for all 28 layers,
both K and V, structural and visual rows. Require exact dtype, shape,
valid-row count, complete BF16 raw bits and byte SHA256; also record
representative first/middle/last elements. Any difference fails G1.

### Level 2: physical repacking (G3, G7)

Starting from the same captured tensor, independently compute a stable score
order and its inverse. The repacked SSD full 100% read, restored to original
logical order, must exactly match canonical FullLoad K/V bits, prefix length,
selected/original identities, structural rows, and token-bound logical
position metadata. Padding is excluded from valid token identities and is
reported separately. Any difference fails G3 or G7.

### Level 3: matched serving (G2, G4)

FullLoad A is an in-memory `DynamicCache` built directly from the captured
native prefix; B is the SSD FullLoad cache. Ours25 A computes the expected
first-k visual IDs independently from captured image-only scores, directly
gathers their K/V from the canonical captured tensor, and sorts *all* kept
visual and structural rows by original sequence position. It does not call
the SSD loader or production compact-cache builder. Ours25 B performs the
production score-order first-k SSD read and its logical-order restoration.
The existing production path is P2; physical-order P1 is diagnostic only.

For each matched pair, hold suffix IDs, selected set/order, native K/V shape
and bits, MRoPE position IDs, `rope_deltas`, cache positions, mask, attention
backend, and model state constant. First repeat one identical input/arm twice:
if its logits or generated sequence differ, report **UNRESOLVED** and stop
claims based on an identity comparison. Then require exact BF16 KV bits,
bitwise-equal first-token logits, identical first-token ID, full generated
token sequence, and prediction. Per-layer suffix output statistics locate any
failure. A deterministic mismatch is **FAIL**. No tolerance is fit to an
observed error. Both paths share the installed stock Qwen decoder only after
their independent cache assembly.

### Level 4: independent calculation semantics (G8, G9, G10)

Derive the full logical 3-axis MRoPE positions and generation positions with
stock Qwen `get_rope_index`/generation behavior, independently of the cache
loader. Check every selected visual token's tuple
`(original_visual_index, stored_index, compact_index,
original_sequence_position, MRoPE_t, MRoPE_h, MRoPE_w)` and every structural
prefix row. Stored K is already post-MRoPE and must not be rotated again.
`cache_position` denotes a compact storage slot and must not replace logical
MRoPE coordinates. Compare an independently constructed boolean visibility
matrix, using original sequence order, against the actual production mask on
all suffix queries/layers: exactly the retained prefix and current/earlier
suffix keys are visible; future suffix and unselected visual keys are hidden.
Small deliberately corrupted mapping, position, and mask fixtures must be
rejected by these checks. Requests are interleaved across image/method and
checked for fresh mask/cache/`rope_deltas` state, including wrong-image
identity rejection.

An independent FP32 reference uses captured Q/K/V from a real GPU case,
explicit `Q_head -> KV_head` GQA mapping, `1/sqrt(head_dim)` scaling, boolean
causal masking, stable softmax, and weighted V reduction. It compares the
same selected keys in dense original layout and P2 logical-order compact
layout. It does not use production assembly or attention helpers. Its
predeclared threshold is **`atol=1e-5, rtol=1e-5` elementwise**, separately
from BF16 logits. A failed oracle or visibility check is a v2 **FAIL**, even
if a matched serving pair happens to agree. Unexplained new position,
assembly, or full-vs-split branch behavior is **UNRESOLVED** and blocks pilots.

## GPU validation gates and stop rule

All ten fixed image-question pairs are required. `validation.json` records
per-pair status/evidence for each gate and an aggregate. A gate that did not
execute is `NOT RUN`, never `PASS`.

| Gate | Required evidence / pass criterion |
|---|---|
| G1 | Native BF16 storage round-trip exact, all layer K/V, structural and visual |
| G2 | FullLoad SSD vs direct-memory split reference, deterministic and exact KV/logits/tokens |
| G3 | RepackedFull100 inverse recovery exact |
| G4 | Ours25 SSD vs independently assembled P2 memory reference, deterministic and exact KV/logits/tokens |
| G5 | source T1 vision forward exactly one; FullLoad/Ours hits zero |
| G6 | Ours hit online query-score calls zero |
| G7 | first-k chunk IDs, visual IDs, logical compact order and inverse permutation exact |
| G8 | independent stock MRoPE, structural-prefix and selected-token mapping exact; negative fixtures reject errors |
| G9 | independent causal visibility exact and FP32 GQA attention oracle within `1e-5/1e-5`; negative mask fixture rejected |
| G10 | wrong-image rejection and request/method order, cache/mask/`rope_deltas` isolation exact |
| G11 | zero unselected visual payload bytes read by hit |
| G12 | measured returned bytes, offsets, pread calls and spans match planned first-k ranges |
| G13 | 64-token policy and all frozen model/processor/backend/budget/decoding fields exact |
| G14 | existing LLaVA CPU regression runs and is reported separately from any actual LLaVA GPU regression; scope must be explicit |
| G15 | pre-v2 legacy, Qwen v1 and debugging artifact file hashes unchanged |

Required GPU system and independent semantic gates must all pass before
smoke. Any `FAIL`, `UNRESOLVED`, or `NOT RUN` in a required gate stops smoke
and both pilots. G14's declared minimum is the existing repository LLaVA
CPU test suite plus protected-file hashes; a separate actual LLaVA GPU run
may be reported only if performed, and cannot be inferred from CPU tests.
This limited G14 scope must be stated in the final verdict.

## Numerical and semantic diagnostics, outside identity gates

D1 compares normal full-image ReComp prefill with split-prefix FullLoad.
D2 compares dense original selected layout with P2 compact selected layout.
For both, retain v1's **elementwise** `torch.allclose` judgment at
`atol=0.125, rtol=0.02`, plus max/mean/p99 absolute logit difference,
number of violating elements, first-token agreement, complete generated
sequence agreement, prediction agreement, layer-zero first-divergence site,
and dataset-level quality observations when available. D3 records the prior
P1 physical-order vs P2 logical-order evidence or reruns it separately.
Neither a common token sequence nor a small maximum difference proves
numerical or quality equivalence.

The initial investigation's fixed case found FullLoad/ReComp first mismatch
at layer-0 pre-MRoPE K projection; suffix input and Q were exact. Padding
the same suffix to the full prefill matrix row count restored K/V bitwise.
The P0/P2 first mismatch was layer-0 BF16 attention output, and a one-layer
FP32 reference reduced that difference to about `3.58e-7`. These are
specific observations, not blanket excuses for future mismatches. V2
rechecks the full-vs-split projection behavior on additional fixed samples
and flags an unexplained new branch as `UNRESOLVED`.

## Conditional smoke and pilots

Only after all required v2 gates pass, run the frozen 20-image × 3-question
smoke: ReComp, FullLoad, Ours25; first request per image is normal full-image
T1, later questions are independent cache hits. Audit finite logits,
generated length, first-token/prediction, vision calls, selection, SSD bytes,
actual retention, state isolation, and failures. Equal answers or high
accuracy are not required. Structural or unexplained numerical failure
pauses further work.

Then rerun the original frozen GQA 40-image × 6-independent-question workload
for all three methods (720 logical requests; 200 hits/method), followed by
the original frozen MT-GQA-reconstructed 40-dialogue × 3-turn workload
(360 logical requests; 80 hits/method). MT T2/T3 use only the same method's
generated previous answers and no repeated image. T1 for every method is
unpruned full-image inference. A manifest mismatch marks that pilot
`BLOCKED`; earlier diagnostic raw data cannot become v2 pilot data.

## Timing, quality, and reporting

TTFT begins before request prompt/input preparation and ends after first
token materialization and CUDA synchronization. Request E2E ends after
decoding. Image file read and RGB/JPEG decode, model loading, common warmup,
page-cache conditioning, and output logging are outside request TTFT;
ReComp image preprocessing and vision/full prefill are inside. Report
conditioning success; `DONTNEED` is only an OS page-cache hint. Balance
method order by deterministic image rotation. Separately account for
activation time/bytes/resident metadata when activation is excluded from hit
TTFT. No hidden GPU/CPU visual-payload reuse between requests.

Report score, permutation, repack, write/fsync, persistence, and activation
costs; no hypothetical background overlap. A three-turn session E2E is the
sum of its three measured request E2Es and each necessary one-time
persistence/activation once. Do not add overlapping component timers to
reconstruct a total. State nominal 25%, chunk 64, image-level visual and
selected counts, mean image retention and aggregate token retention
separately, valid and padded visual bytes, structural/metadata bytes,
overall SSD read ratio, preads/spans, TTFT mean/p50/p95, request/session E2E,
actual generated-token lengths, truncation/nonfinite/failure/duplicate
counts, turn/all/hit quality, and paired differences. Use the existing
repository scorer; it is not an official external evaluator. Use a fixed
image-cluster bootstrap with 4,000 resamples and seed 1234 for paired CIs.
A confidence interval containing zero is not an equivalence claim.

## Immutable prior evidence and reproducibility

The new run's `evidence_manifest.json` holds absolute/relative paths,
SHA256s, and JSON pointers for the original 13/15 validation, unchanged
debugging rerun, report, first-divergence and padded-shape controls,
MRoPE/mask and selected-KV checks, P0/P1/P2 comparison, FP32 attention
control, and original frozen pilot manifests. The new
`protected_before.json` snapshots earlier Qwen and debug files against the
original 8,883-file legacy baseline. The v2 `protected_after.json` must
agree exactly before final claims. Earlier artifacts are never written.
