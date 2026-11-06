# Ours-ReRoPE and Pruned-T1 contract (LLaVA, GQA Stage 1)

**Status: APPROVED 2026-11-02 (plan approved with user amendments 1–8, listed
under "Approval amendments").** Frozen with a source-hash manifest (`runs/<run>/gpu_freeze.json`)
before any GPU execution. Thresholds and decision rules cannot change after any
result is seen. The pruned-T1 request had not been started before this
contract; nothing is integrated from an earlier run.

## Question

In the rev.-3 ablation (`results/recompute_kv25_ablation_20261102T061243Z`)
Ours-KV25 (full context, original positions) and Recomp-KV25-Orig (reduced
context, original positions) were similar, while Recomp-KV25-Compact (reduced
context, contiguous positions) was significantly higher than Orig. This run
asks:

1. If the Keys of the full-context Ours KV are re-rotated to the contiguous
   positions, without any extra prefill, does cache-hit quality reach the
   Compact level? (completes the 2×2 context × position design)
2. If the first turn itself runs on the important tokens only
   (VisionZip-style), how much first-turn quality is lost, and how do
   whole-dialogue designs compare?

No outcome is assumed. The workload is the development set used several times
already, so every conclusion is exploratory.

## Frozen inputs and environment

- Workload: the rev.-3 copy `runs/recompute_kv25_ablation_20261102T061243Z/gqa_manifest.json`
  (SHA256 `40e23fa7aeb22495717fc7eede1227de3a6b4811fde022d9cda71ef73c5e3895`),
  40 GQA images × `questions[4:10]` = 240 questions, same order. `data/index.json`
  SHA256 `514d1203…5e77e75a` re-verified. T1 = `questions[4]`; T2–T6 =
  `questions[5:10]`; 200 cache hits per stored arm. Stage 2 is out of scope.
- `llava-hf/llava-v1.6-vicuna-7b-hf` revision `c916e6cd…`, NF4 double-quant,
  BF16 compute, eager attention, FP16 SSD payload, 64-token chunks, separator
  sidecar, greedy, `max_new_tokens=16`, seed 1234, `mllm_ft` (torch 2.5.1,
  transformers 4.57.6), RTX 4090, no other GPU process. The serving path is the
  rev.-3 one: `request_cvpr25(budget=.25, budget_unit='visual_kv',
  mode='prefix', sep_policy='sidecar')`, `PositionServer` RoPE override for
  the text start, DONTNEED outside timers, schema-v2 TTFT.
- Reference runs for reproduction only: rev. 3 (above) and
  `runs/prefix25_persistence_20261001T050113Z` (G6 of rev. 3).

## Arms

Notation as in the rev.-3 contract: `v0` system tokens, `V` expanded image span,
`sep` row separators, `N = V − n_sep`, `k = (N+3)//4`, `order` the stable
descending image-only VisionZip permutation, `S = order[:k]`,
`keep = sorted(S ∪ sep)`, `L = v0 + k + n_sep`. Compact position of kept
original ID `i`: `p'(i) = v0 + index of i in keep`; original position
`p(i) = v0 + i`.

| Arm | KV context | Positions of stored K | Text start | T1 |
|---|---|---|---|---|
| ReComp | pixels every request | — | original | full-token |
| FullLoad | full | original (canonical visual KV) | original | full-token |
| Ours-KV25 | full T1 forward | original | `v0 + V` | full-token (own) |
| Recomp-KV25-Orig | reduced prefill | original | `v0 + V` | full-token (own) |
| Recomp-KV25-Compact | reduced prefill | contiguous | `L` | full-token (own) |
| **Ours-ReRoPE** (new) | full T1 forward (Ours capture) | contiguous, by re-rotation | `L` | Ours T1 (shared, not re-executed) |
| **Pruned-T1** (new) | none (no cache) | contiguous (pruned input) | — | pruned (own), every question |

Orig and Compact are re-run (not copied) so the 2×2 is measured in one run.

**Ours-ReRoPE persistence.** From the Ours T1 capture (the same `cache`,
`enc`, vision capture and saliency as the Ours store of the same image) a
full-size cache is built per layer: `K16 = K_bf16.to(float16)` (exact for
every row), then for the rows of `keep`
`K16[row(i)] = rotate(K_bf16[row(i)], p'(i) − p(i)).to(float16)`, where
`rotate` is the existing Llama `rotate_half` RoPE helper
(`mmimpress.recompute_kv25.rotate`, HF `inv_freq`, θ = 10000, no scaling),
applied **once** to the post-RoPE K, computed in **FP64** (exceeds the
requested FP32; angles are `(p'−p)·inv_freq`), and cast **once** to the store
dtype FP16. V and the system rows are the Ours tensors unchanged. The tensor
pair goes through the unchanged `persist_captured_visual_prefix(...,
storage_policy='prefix25')` with the same saliency capture, so S, k, the
importance storage order, file layout, sidecar and bytes are identical to
Ours. `extra_metadata`: `kv_context='full_turn1'`, `position_policy='compact'`,
`rerope=True`, `text_position_start=L`. The re-rotation is timed separately
(`rerope_ms`, CUDA-synchronized) inside the persistence wall time.

**Pruned-T1.** Every question (T1–T6) is a standalone, cache-less request:
image decode → processor → vision tower, projector and AnyRes merge (the
model's own multimodal path up to the language model, stopped by a pre-hook
that keeps `inputs_embeds` and aborts the full-token forward) → image-only
VisionZip saliency and `order` → `plan(compact)` → language-model input
`inputs_embeds[sys + keep + text]` in original relative order, with
`position_ids = cache_position = arange` → one prefill, first-token argmax
(TTFT boundary, CUDA sync) → greedy decode with the same loop semantics as
`Server._decode` (EOS stop, max 16, same token count). No contextual-token
merge, no text/question influence on S (`image_inputs` only drives saliency),
no cache read or write. TTFT is a composed measurement (full-token forward
aborted after the multimodal merge, not the schema-v2 server path); it is
reported with its breakdown (image decode, processor, vision+merge, saliency,
plan+gather, prefill-to-first-token) **for description only and is not used
for any efficiency comparison**; the report states this.

**Identity across KV25 arms.** For every image, S, k, structural handling,
storage order, stored bytes and read bytes/preads must be identical across
Ours, Orig, Compact and ReRoPE stores and hits; Pruned-T1 requests must select
the same S.

Per image: T1 is executed for ReComp, FullLoad, Ours, Orig, Compact (own
full-token T1 each, as in rev. 3) and Pruned-T1 (pruned); ReRoPE's T1 row is
the Ours T1 (`shared_t1_from='ours'`, `executed=False`). Arm order rotates by
(image, turn) over the 7 arms. No existing module is edited.

## Integrity gates (fail-closed unless marked report-only)

Rev.-3 G1–G7 stay in force for Ours/Orig/Compact (G1 identity, G2 rev. 3 incl.
structural bit-identity with the truncation control and the layer-0 S/sep
gates, G3 positions, G4, G5, G6, G7). For ReRoPE, "structural identity" means
the checks below (its sys rows come from the full forward, so the truncation
control does not apply).

**G-a Round trip** (ReRoPE vs Ours stores, every layer, S and sep rows):
`rotate(K_rerope_fp16, p − p')` in FP64 vs `K_ours_fp16`: `rel_L2 ≤ 1e-3` and
`allclose(rtol=1e-3, atol=1e-3 · max|K_ours|)` per layer/row-group. Fixed
from the rev.-3 G4 stores (CPU, read-only; evidence file
`dev/g_ab_evidence.json`, SHA256
`879706ccdd71510b0d90404012cf7a1d6b3659c796789cbaa82975d42a2d6eea`, copied
into the run directory at freeze together with its script): 5 pairs
× 32 layers, worst `rel_L2 = 2.15e-4`, max elementwise
`|Δ| / (|x| + rowmax)` = 4.1e-4, `allclose` true everywhere, i.e. FP16
rounding (≈ 2⁻¹¹) with ≥ 4× margin.

**G-b Same position** (layer 0, S and sep rows): ReRoPE K vs Compact K
`rel_L2 ≤ 1e-2` (consistent with the rev.-3 realignment tolerance) **and**
FP32-oracle ratio
`rel_L2(ReRoPE, oracle_p') ≤ 1.5 × rel_L2(Compact, oracle_p')`, where
`oracle_p'` is the rev.-3 FP32/FP64 layer-0 oracle rotated to the compact
positions. V: ReRoPE V (S, sep, sys) **bit-identical** to Ours V at every
layer; ReRoPE sys K bit-identical to Ours sys K at every layer. **Negative
control**: the un-rotated Ours K vs Compact K at layer 0 (S and sep rows) must
have `rel_L2 > 0.1`, otherwise the re-rotation is not being exercised.
Evidence: rev.-3 recorded `k_S_realigned` layer 0 over 40 images, max 0.00447
(S) / 0.00456 (sep); the FP16-cast ReRoPE value on the 5 G4 stores gives
0.0038–0.00448 / 0.00384–0.00455 (the BF16 cos/sin rounding of the Compact
prefill, stable across images); the un-rotated rev.-3 `k_S` layer-0 value is
≈ 1.17. Deeper layers of ReRoPE K vs Compact K and ReRoPE K vs Ours K
are recorded per layer (this is the context effect at identical positions),
not gated.

**G-c Text start**, every request: `PositionRecorder.verify` must hold with
`text_start = L` for ReRoPE and Compact hits, `text_start = v0 + V`
(= physical) for Ours and Orig hits; Pruned-T1 prefill/decode `position_ids`
must equal `arange` over the whole sequence (`= cache_position`); the store
`text_position_start` must equal the policy value. Recorded per request.

**G-d Identity and sentinel**: selection digest, payload sizes and T1
`inputs_embeds` SHA equal across Ours/Orig/Compact (ReRoPE shares the Ours
capture: its digest and sizes must equal Ours); Pruned-T1 `selected_original_ids`
equal to `order[:k]` of the Ours store for every request; hit read bytes and
preads equal across the five stored KV25 arms on every hit; no sentinel row in
any store (ReRoPE has none by construction) and none in unmasked GPU rows on
smoke hits; Pruned-T1 reads 0 bytes.

**G-e Reproduction vs rev. 3** (report-only): per request (image, question)
agreement of first token, generated IDs and prediction for Ours, Orig, Compact,
FullLoad and ReComp (T1 and hits) with the rev.-3 raw rows. Any mismatch is
listed with both first tokens, the new top-2 logit margin and the store
hashes, and analysed; never hidden. Rev.-3 G6 (Ours vs the 2026-10-01 run) is
also reported.

**G-f Pruned-T1 vs Compact hits** (report-only): for T2–T6, agreement of first
token and prediction between the Pruned-T1 request and the Compact hit of the
same (image, question); every mismatch is recorded with both first-token top-2
logit margins. Expected sources of mismatch: BF16 shape rounding (question
tokens inside the pruned prefill GEMM) and the FP16 store round trip. On the
G4 pairs a two-step diagnostic is recorded: pruned prefix forward
(`sys + keep`, the same call as the Compact reduced prefill) followed by the
question through `_decode`; its prefix KV must be bit-identical to the Compact
reduced prefill (gate on the G4 pairs only), and its outputs vs the one-forward
Pruned-T1 and vs the Compact hit are recorded.

**G4 extension (5 frozen pairs, before smoke):** rev.-3 (a)–(e) for
Ours/Orig/Compact, plus (f) ReRoPE: the independent in-memory reference
(`90_validate_llava_kv25._make_memory_cache`) built from the ReRoPE full-size
tensors matches the SSD path with the compact text start on both sides
(all-layer/head selected bits, mask, reads, every step's logits at
`atol=rtol=1e-4`, identical tokens); (g) G-a/G-b on the 5 pairs; (h)
Pruned-T1: a changed question leaves S unchanged, exactly one vision forward,
zero reads, positions `arange`; the two-step diagnostic above.

**G5 Protection**: SHA256 manifests of `results/`, `docs/`, `mmimpress/`,
`scripts/`, `tests/`, `README.md`, the 2026-10-01 reference run and **both**
earlier recompute runs (`…_20261102T045423Z`, `…_20261102T061243Z`) before and
after; `kvstore*/` inventory. New paths (this run's dirs, the new
source/test/doc files, the README section) are excluded; any other difference
fails. The guarded delete is limited to this run's `stores/`.

## Metrics

- Hit (T2–T6) accuracy per arm (`exact_score`), first-token and prediction
  agreement vs FullLoad, hit TTFT mean/p50/p95 and E2E, read bytes/preads.
- Paired differences, image-cluster paired bootstrap (10,000 resamples, seed
  1234, 95% percentile CI): ReRoPE − Ours, ReRoPE − Compact, plus Ours − Orig,
  Ours − Compact, Orig − Compact for the 2×2.
- 2×2 factor effects: per-image linear contrasts of hit accuracy, the three
  estimated **jointly on the same image-level bootstrap resamples** (10,000,
  seed 1234). Sign convention (positive means):

  | Effect | Contrast | Positive means |
  |---|---|---|
  | context | ((Ours + ReRoPE) − (Orig + Compact)) / 2 | full context higher than reduced, averaged over positions |
  | position | ((Compact + ReRoPE) − (Ours + Orig)) / 2 | contiguous positions higher than original, averaged over contexts |
  | interaction | (Ours − ReRoPE) − (Orig − Compact) | the original-minus-contiguous gap is larger in full context than in reduced context |
- Pruned-T1 vs full-token standalone inference (ReComp, same question, no
  cache): (a) T1, 40 questions; (b) T2–T6, 200; (c) all 240. Point estimate
  and CI (paired by request, images resampled).
- Dialogue-level expected accuracy (1 T1 + 5 hits, per image, mean over 6
  requests): [T1 full + Ours hits], [T1 full + ReRoPE hits], [all Pruned].
  "T1 full" is the ReComp T1 row (every full-token T1 is identical by G1;
  agreement reported). Means with CI and pairwise differences with CI.
- Persistence per image for FullLoad/Ours/Orig/Compact/ReRoPE: saliency,
  permutation, plan, embed gather, reduced prefill, synthetic assembly,
  **re-rotation (`rerope_ms`)**, repack, write, fsync, seal, total. ReRoPE's
  extra cost vs Ours: primary = `rerope_ms` (the analogue of the Recomp
  prefill-only Δ), full Δ = total − Ours total with CI; amortized over
  1/3/5/10 requests. Pruned-T1 cost = its per-request TTFT breakdown.
- Per-layer KV differences: ReRoPE vs Compact (context effect at identical
  positions), Orig vs Ours (context effect at original positions), ReRoPE vs
  Ours K (realignment only), Compact K raw/realigned vs Ours as in rev. 3.

## Decision rules (fixed before results)

- ReRoPE − Ours and ReRoPE − Compact: CI lower bound > 0 / upper bound < 0 →
  the higher arm is reported as higher; CI includes 0 → difference not shown
  (not equivalence).
- 2×2 effects and interaction: reported as point estimate with CI; an effect
  is called "shown" only if its CI excludes 0.
- Pruned-T1 (a)/(b)/(c) and the three dialogue designs: same CI rule.
- With 40 image clusters, no |difference| < 1 pp is interpreted as
  established, even if the CI excludes 0 (BF16 shape rounding can flip
  boundary first tokens; rev.-3 noise floor cited).
- All conclusions are exploratory: the development set has been used
  repeatedly (rev. 3, the prefix25 runs and earlier), and the ReRoPE arm was
  motivated by the rev.-3 result on the same images.
- ReRoPE needs no extra prefill; its cost is the re-rotation inside
  persistence, never cache-hit TTFT.

## Execution and storage

Order: CPU tests → freeze → protection-before → G4 (5 pairs, incl. the
extension) → smoke (3 images × 3 turns, all 7 arms; sentinel checks) →
Stage 1 (40 × 6) → protection-after → report. New outputs only in
`runs/rerope_kv25_<UTC>/` and `results/rerope_kv25_<UTC>/` (no-clobber).
Per-image stores (FullLoad 1.26 GB + 4 × 309 MB ≈ 2.5 GB) are deleted after
the image's receipts, gate record and raw rows are written, through the
guarded delete. G4 stores (5 pairs: Ours/Orig/Compact/ReRoPE ≈ 6.2 GB) are
retained; smoke stores are **not** retained this time, but every smoke
persistence receipt, gate record, raw row, log and the sealed per-file SHA256
list of each smoke store (`seal_integrity` output in the receipt) stay in the
run directory (disk: 33 GB free, floor 20 GiB checked per image; the
rev.-2/rev.-3 retained stores are untouched records). No
earlier run or results directory is modified. Expected wall time ≈ 2.5 h.
REPORT.md (2×2 table, Pruned-T1 table, dialogue table, gates, persistence
breakdown with `rerope_ms`, limits), summary.csv, REPRODUCE.md go to the
results directory; README gets one summary section. Commits only on request.

New files: `mmimpress/rerope_kv25.py`, `scripts/136_eval_rerope_kv25.py`,
`scripts/137_report_rerope_kv25.py`, `tests/test_rerope_kv25.py`, this
contract. Reused unchanged: `mmimpress/recompute_kv25.py` (plan, rotate,
`compare_stores`, `PositionServer`, `PositionRecorder`, bootstrap),
`scripts/134_eval_recompute_kv25_ablation.py` (`Ablation` base: T1, persistence,
activation, hits, G1/G2 gates, noise floor), `scripts/106_prefix25_persistence.py`.

## Approval amendments (2026-11-02)

1. G-b bound `rel_L2 ≤ 1e-2` (rev.-3 realignment tolerance); oracle ratio
   ≤ 1.5× kept; negative control un-rotated Ours K vs Compact K layer 0
   `rel_L2 > 0.1`; V and sys bit-identity unchanged.
2. Re-rotation in FP64 approved.
3. Pruned-T1 single-prefill definition and the G4 two-step diagnostic
   approved; its composed TTFT is not used for efficiency comparison (stated
   here and in REPORT).
4. 2×2 contrasts approved; sign table above; the three effects are estimated
   jointly on the same bootstrap resamples.
5. Dialogue-level "T1 full" = ReComp T1 row approved.
6. Smoke stores not retained; smoke receipts, logs and sealed store hash lists
   retained in the run directory.
7. G5 covers the rev.-2 and rev.-3 run directories.
8. `g_ab_evidence.json` copied into the run directory at freeze; SHA256
   recorded above.

## Execution note (2026-11-02, before any measurement)

The first frozen run `runs/rerope_kv25_20261102T080818Z` stopped in the G4
extension on the first pair: the script applied the rev.-3 asserting helper
(`_compare_outputs`, logits `atol=rtol=1e-4`) to the Pruned-T1 two-step
diagnostic vs the Compact hit, which this contract defines as **recorded,
not gated**. Tokens and answer were identical; the max logit delta was
0.156 (compact BF16 cache vs the FP16 store in the masked full-size buffer).
The prefix-KV bit-identity gate and the rev.-3 G4 (a)–(e) on all five pairs
had passed. Only the recording call was replaced by a non-asserting twin
(`compare_recorded`); no threshold or rule changed. The run is kept as a
record; the experiment restarts from freeze in a new run directory.
