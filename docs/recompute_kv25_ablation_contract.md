# Full-context KV25 vs. reduced-token Recompute KV25 contract (LLaVA, Stage 1)

**Status: APPROVED 2026-11-02 (user amendments 1–8; G2 revision 2 approved after dev observation; G2
revision 3 approved after the rev.-2 Stage-1 stop, before any of its quality results were opened).** Frozen with
a source-hash manifest (`runs/<run>/gpu_freeze.json`) before any GPU execution.
Thresholds and decision rules below cannot change after any result is seen.

## Question

Ours (PrefixStore-KV25 [B]) stores, for the selected set S, the decoder K/V
computed during the full-token Turn-1 forward. The alternative stores the K/V
of S computed by a separate prefill that contains only the structural tokens
and S. Under equal S, equal k, equal storage order and equal stored bytes, does
the context used to compute the KV change cache-hit quality? How much extra
prefix compute does the alternative need? A diagnostic arm measures whether
original-input text positions are needed for the Ours store.
No outcome is assumed in advance.

## Frozen inputs and environment

- Workload: `runs/prefix25_persistence_20261001T050113Z/gqa_manifest.json`,
  SHA256 `40e23fa7aeb22495717fc7eede1227de3a6b4811fde022d9cda71ef73c5e3895`.
  40 GQA images × `questions[4:10]`, same order as the reference [B] run.
  `data/index.json` SHA256 `514d1203…5e77e75a` and workload SHA256
  `97afe02f…20f66` are re-verified. T1 = `questions[4]`, hits T2–T6 =
  `questions[5:10]`, 200 hits per arm. A count/hash mismatch blocks the run.
- `llava-hf/llava-v1.6-vicuna-7b-hf` revision `c916e6cd…`, NF4 double-quant,
  BF16 compute, eager attention, FP16 SSD payload, 64-token chunks, separator
  sidecar, greedy, `max_new_tokens=16`, seed 1234. `mllm_ft` env (torch 2.5.1,
  transformers 4.57.6), RTX 4090, no other GPU compute process.
- Reference for reproducibility comparison only:
  `runs/prefix25_persistence_20261001T050113Z/llava/gqa/raw_requests.jsonl`.

## Arms

Notation: `v0` = system-token count (`v_token_start`); `V` = expanded image
span; `sep` = row-separator original IDs (`n_sep`); `N = V − n_sep`;
`k = ceil(N/4) = (N+3)//4`; `order` = the stable descending image-only
VisionZip permutation (real rows, then separators in original order);
`S = order[:k]`.

| Arm | KV context | Stored | Text position start |
|---|---|---|---|
| ReComp | pixels every request | — | original |
| FullLoad | full | full canonical visual KV | original |
| Ours-KV25 | full T1 forward | S rows of T1 KV (existing [B] writer) | `v0 + V` |
| Recomp-KV25-Orig | reduced prefill, original position IDs | S rows of reduced KV | `v0 + V` |
| Recomp-KV25-Compact | reduced prefill, contiguous position IDs | S rows of reduced KV | `v0 + k + n_sep` |
| Ours-KV25-PosShift (diagnostic) | Ours store (same files, same context) | — | `v0 + k + n_sep` |

**Reduced prefill.** The input is `[system tokens 0..v0-1] + [image tokens
with original ID in sorted(S ∪ sep)]`, in original relative order. This is
the original sequence with the unselected real tokens removed, the same order
that VisionZip's mask-gather keeps. Input embeddings are the arm's own Turn-1
`inputs_embeds` (vision tower + projector + AnyRes newline merge output),
captured by a read-only pre-hook on `model.model.language_model` during T1
prefill. Nothing is re-encoded; there is no contextual-token merge and no
text/question token. Positions:
- Orig: `position_ids = [0..v0-1] + [v0 + id for id in sorted(S ∪ sep)]`.
- Compact: `position_ids = arange(v0 + k + n_sep)`.
`cache_position = arange(len)` causal in both cases. The call is
`language_model(inputs_embeds, position_ids, cache_position, use_cache=True)`.
Additive BIAS must be empty (asserted).

**Persistence.** Reduced K/V rows are scattered into a full-size
`(1, H, v0+V, hd)` BF16 tensor at their original indices. Unselected real rows
hold the finite sentinel 4096.0. That tensor goes through the unchanged
`persist_captured_visual_prefix(..., storage_policy='prefix25')`, using the
same T1 saliency capture, so the permutation, S, k, file layout, sidecar and
writer timing path are identical to Ours. `extra_metadata` records
`kv_context` (`full_turn1` / `reduced_prefill`), `position_policy`
(`original` / `compact`) and `text_position_start`. All three stores get the
same `seal_integrity` (full checksums plus file/dir/parent fsync).

**Serving.** All KV25 arms use the existing `request_cvpr25(budget=.25,
budget_unit='visual_kv', mode='prefix', sep_policy='sidecar')`: sequential
first-⌈k/64⌉-chunk reads, sidecar, mask, `DONTNEED` outside timers, and
schema-v2 end-to-end TTFT. A `Server` subclass changes only the RoPE
`position_ids` of suffix/decode tokens to `text_position_start + i`.
`cache_position` and the full-size physical buffer stay unchanged. With
`text_position_start=None`, the subclass is the base path. The runner
fail-closes if the arm's position policy disagrees with the store metadata.
PosShift reuses the Ours context. Its T1 row is the Ours T1 (not re-executed).

Each stored arm (FullLoad, Ours, Orig, Compact) runs its own full-token T1 and
its own persistence, as in the reference run. Arm order rotates by
(image, turn). No existing module is edited.

## Integrity gates (fail-closed; any FAIL or unrun gate stops the run)

G1 **Selection and T1 identity**, per image: SHA256 over (`order`, `k`, `sep`,
`stored_row_to_original`) and over the KV payload file sizes (`layer_*/k|v.bin`,
`sep_kv.bin`, `sys_kv.pt`) must be equal for Ours, Orig and Compact. The T1
`inputs_embeds` SHA256 must also be equal across the three arms' own T1, and the
three T1 requests must have identical first token, generated token IDs and
prediction (FullLoad/ReComp T1 agreement is reported). No payload row may
equal the sentinel row.

G2 **Layer-0 sanity and structural-row identity — revision 3** (store FP16 →
FP32, all heads). Revision 2 was set after the dev samples n355567/n9181 and
before measurement; revision 3 was set after the rev.-2 Stage-1 run stopped
at its 12th image and **before any quality result of that run was opened**.
Both earlier texts and their failure logs are preserved verbatim in
"Amendment history".
- **Structural rows (gate; the only stop condition for these rows).** Rows
  whose reduced index, position and causal context equal the original ones:
  the system rows and the common-prefix rows (S∪sep IDs below the first
  unselected real ID). For each Recomp arm (Orig and Compact) a
  **same-length truncation control** is computed per image: the arm's own T1
  `inputs_embeds[0:L]` with `position_ids = cache_position = arange(L)`,
  `L` = reduced-prefill length, through the same `language_model` call. The
  stored FP16 structural rows must be **bit-identical** to the control's rows
  (BF16 → FP16 by the writer's `.to`) at **every layer (0–31), K and V**.
  On the G4 pairs the same identity is also checked in memory in BF16 before
  the FP16 store. Any bit difference stops the run.
- The control runs **after** every persistence timer, `seal` and I/O counter
  of that arm is closed and **before** store activation and every hit
  request; its wall time is recorded separately (`truncation_control`) and is
  never part of persistence or TTFT figures.
- Structural rows vs Ours (rev. 2 layer-0 K oracle / V elementwise rules and
  the layer ≥ 1 `rel_L2 ≤ 3e-2` bound for sys and common-prefix rows) are
  **report-only**: values and whether they fall within the old bound are
  recorded, not gated.
- Orig vs Ours, layer 0, **K** (S and sep rows): `rel_L2 ≤ 5e-3` **and** FP32
  oracle ratio `rel_L2(Orig, oracle) ≤ 1.5 × rel_L2(Ours, oracle)`. The oracle is
  RMSNorm → dequantized-NF4 `k_proj` → RoPE in FP32/FP64, computed from the
  arm's T1 `inputs_embeds` at original positions. No elementwise test for
  post-RoPE K (RoPE cancellation makes a relative elementwise bound meaningless).
  (Unchanged from rev. 2.)
- Orig vs Ours and Compact vs Ours, layer 0, **V** (S and sep rows):
  `rel_L2 ≤ 5e-3` and `allclose(atol=1e-2, rtol=1.6e-2)` (unchanged).
- Compact vs Ours, layer-0 K after RoPE realignment: `rel_L2 ≤ 1e-2` and
  `allclose(atol=5e-2, rtol=5e-2)` (unchanged).
- Negative controls (unchanged): un-realigned Compact K on moved rows
  `rel_L2 > 0.1`; Orig vs Ours last-layer S `rel_L2 > 5e-3`. The latter is an
  implementation check that recomputation actually happened; it is **not**
  evidence for or against the quality hypothesis.
- On mismatch: stop and report per-layer/row diagnostics. Layers ≥ 1 S/sep
  rel-L2 and cosine (Compact K raw and realigned) are recorded, not gated.
- Noise floor (recorded, not gated). For every G4 pair, four prefills are
  compared with the T1 cache: `prefix_only` (same context, question removed),
  the actual Orig reduced prefill, `truncated_same_len` and
  `truncated_half_len` (original sequence cut to `L` and `L/2`; structural
  rows keep their context, only the shape changes). The report gives the
  last-layer S K/V difference (prefix-only and Orig), the common-prefix
  ("single-row") max over layers ≥ 1 for both truncations, the ratios
  context effect / prefix-only floor and context effect / single-row drift,
  and the dev table of revision 3 below. For every Stage-1 image the control
  vs Ours difference of the structural rows (shape-only drift) is recorded.

G3 **Positions**, every request of every KV25 arm (forward pre-hook on the
model): prefill `position_ids` must equal `text_start + arange(n)` and decode
step `t` must equal `text_start + n + t`. `cache_position` must equal the
physical `v0 + V + …` and the attention mask must be all-ones over physical
length. `text_start` is `v0+V` for Ours/Orig and `v0+k+n_sep` for
Compact/PosShift. In persistence, Orig's reduced-prefill IDs must equal the
original IDs and Compact's must equal `arange`.

G4 **GPU correctness** on the 5 frozen LLaVA correctness pairs before the smoke:
(a) Ours hit output (first/generated IDs, prediction) equals the reference run's
recorded [B] output_proof. (b) The subclass with `None` gives bit-identical
logits to the base `Server`. (c) Orig: the existing independent memory
reference (`90_validate_llava_kv25._make_memory_cache`) built from the
in-memory reduced KV matches the SSD path through the same physical buffer.
It checks all-layer/head selected bits, mask, reads, and every step's logits
at `atol=rtol=1e-4` with identical tokens. (d) Compact and PosShift: the same
comparison with the position override applied to both the reference and SSD
paths. (e) Changing the question leaves the selected IDs unchanged, with zero
vision or query-scoring calls on hits.

G5 **Protection**: full SHA256 manifests of `results/`, `docs/`,
`mmimpress/`, `scripts/`, `tests/`, `README.md` and
`runs/prefix25_persistence_20261001T050113Z/` before and after; `kvstore*/`
by a (path, size, mtime_ns, inode) inventory before and after. Files this
experiment creates (new run/results dirs, the new source/test/doc files and
the final README section) are listed explicitly and excluded; any other
difference fails.

G6 **Reference reproduction (report gate A)**: each new Ours-KV25 hit is
compared per request (image, question) with the reference [B] row
(`prefix25_persistence_20261001T050113Z`, manifest `40e23fa7…`, hit 56.5%):
first token, generated IDs and prediction agreement rates are reported. Any
mismatch is analysed (row list, logit margin of the reference/new first token
where available, T1 store hashes) and reported; it is never hidden. It does
not by itself stop the run.

G7 **Sentinel non-leakage (gate B, smoke)**: for every smoke store of Ours,
Orig and Compact, no row of any `layer_*/k.bin`/`v.bin` or `sep_kv.bin`
equals the sentinel 4096.0, and on every smoke hit of all four KV25 arms the
installed GPU cache has no sentinel value in any unmasked (BIAS = 0) row of
any layer. Failure stops the run. New outputs go
only to `runs/recompute_kv25_ablation_<UTC>/` and
`results/recompute_kv25_ablation_<UTC>/`, with no-clobber creation.

## Metrics

- Hit (T2–T6) accuracy per arm (existing GQA `exact_score`). Paired
  differences vs Ours (Ours−Orig, Ours−Compact, Ours−PosShift, plus
  Orig−Compact) with an image-cluster paired bootstrap, 10,000 resamples,
  seed 1234, 95% percentile CI. A discordant-pair count is also reported.
- First-token and prediction agreement vs FullLoad. Agreement of new
  Ours/FullLoad/ReComp with the reference run is reported, not gated.
- Hit TTFT mean/p50/p95 and E2E. Actual `os.pread` bytes and calls must be
  equal across the three stored KV25 arms and PosShift; a difference fails G1.
- Persistence per image: saliency (capture), permutation, embeds capture,
  reduced prefill (CUDA-synchronized; Recomp only), synthetic scatter
  (Recomp only), repack, write, fsync/durability, seal, total wall. Extra cost
  vs Ours: the **primary** metric is prefill-only Δ (CUDA-synchronized
  reduced prefill, plus embeds gather), because the full-size synthetic
  materialization is a writer-reuse artifact. The full Δ = Recomp total −
  Ours total is also reported. Both are amortized over 1/3/5/10 requests per
  image.
- Per-layer KV difference vs Ours (from G2), averaged over images.

## Decision rules (fixed before results)

- CI lower bound of (Ours − Orig) hit accuracy > 0 → supports "reusing
  full-context KV is better".
- CI includes 0 → no quality difference shown. The paper uses only the cost
  argument (no extra prefill).
- Orig significantly higher (CI upper bound < 0) → reported as is.
- Ours − PosShift CI lower bound > 0 → evidence that original positions are
  needed. Otherwise reported as no difference shown.
- Ours − Compact is reported with the same rule, as the VisionZip-style
  alternative.
- With 40 image clusters, no |difference| < 1 pp is interpreted as
  established, even if the CI excludes 0. A CI including 0 is not equivalence.
- The report's quality interpretation states, citing the G2 noise floor, that
  BF16 shape-dependent rounding alone changes deep-layer KV of context-identical
  rows (prefix-only and truncation controls; single rows up to ~0.03 rel L2)
  and can flip boundary first tokens, so differences below 1 pp are not
  interpreted as established.
- Recomp arms' T1 equals full-token inference by construction. The extra
  prefill is persistence cost, never cache-hit TTFT.

## Execution and storage

Order: CPU tests → freeze → protection-before → G4 (5 pairs) → smoke
(3 images × 3 turns, all arms) → Stage 1 (40 × 6) → protection-after → report.
Revision 3 restarts this whole order in a new run directory. The stopped
rev.-2 run `runs/recompute_kv25_ablation_20261102T045423Z` (and its results
directory) is a record only: it is not modified, analysed or reported, and
its quality numbers are not opened before the new run completes. G5 hashes
that run directory before and after as a protected path.
Stage 2 (MT-GQA) requires separate approval after the Stage 1 report.
Payload stores are deleted per image only after G1/G2 receipts, layer-diff
statistics and raw rows are committed, as in the reference run. The delete
function resolves every target and refuses (raises, stopping the run) any
path that is not strictly inside this run's `stores/` directory, is a
symlink, or is the directory itself. G4 and smoke
Recomp stores are retained. The disk floor is 20 GiB free (checked per image).
REPORT.md, summary.csv and REPRODUCE.md go to the results directory; README gets
one summary section.

## Extension note (Qwen2.5-VL, not in scope)

Plan, embedding capture, reduced prefill and diff utilities are written against
abstract `(prefix_len, keep_ids, position_ids)` so that a Qwen port can supply:
Orig = original 3-row M-RoPE `(t,h,w)` IDs of kept tokens with the original
`rope_deltas`; Compact = the Qwen VisionZip convention (positions recomputed
for the reduced grid, or 1-D contiguous) and compact cache assembly in
`qwen25/runner`. The text-start rule then comes from the policy's max position
+ 1 instead of `v0+k+n_sep`. RoPE realignment for G2 must use M-RoPE sections.

## Amendment history

### G2 revision 1 (approved 2026-11-02, superseded before measurement)

Verbatim text of the original gate:

G2 **Layer-0 sanity** (store FP16 → FP32, all heads, rows S, sep and sys):
- Orig vs Ours, layer 0, K and V: PASS iff `rel_L2 ≤ 5e-3` and
  `allclose(atol=1e-2, rtol=1.6e-2)` (≈ 4 BF16 ULP; allows GEMM-shape
  rounding only). The bit-exact fraction is recorded.
- Sys rows, all layers, Orig and Compact vs Ours: same tolerance (system
  tokens causally precede the image).
- Common-prefix rows (S∪sep IDs below the first unselected real ID), all
  layers, Orig vs Ours: same tolerance. The count is recorded and may be 0.
- Compact vs Ours, layer 0: V same tolerance. K after RoPE realignment
  (un-rotate Ours by original position, rotate by compact position, FP32):
  `rel_L2 ≤ 1e-2` and `allclose(atol=5e-2, rtol=5e-2)`.
- Negative controls: un-realigned Compact K for rows whose position changed
  must have `rel_L2 > 0.1`. Orig vs Ours at the last layer on S must have
  `rel_L2 > 5e-3`; otherwise the reduced prefill did not change context.
  This is an implementation check that recomputation actually happened; it
  is **not** evidence for or against the quality hypothesis.
- On mismatch: stop and report the per-layer/row diagnostics.
- Layers ≥ 1: per-layer rel-L2 and cosine vs Ours (K, V; S and sep separately;
  Compact K both raw and realigned) are recorded for every image and
  are not gates.

### Revision-1 failure on dev samples (before freeze; no measurement)

Dev check in the scratchpad on correctness pairs n355567/201751701 and
n9181/20929611. G2 revision 1 failed: Orig `layer0_k` (elementwise; rel L2
2.66e-3 passed, max |Δ| 0.047–0.0625 failed) and `sys_all_layers_k/v` at
layers ≥ 1 for Orig and Compact (rel L2 up to 7.9e-3 K / 1.45e-2 V). All other
G2 parts passed (layer-0 V, Compact realigned K 4.4e-3, negative controls
1.16 and 0.21/0.42), and G4 (a)–(e) passed with G2 made non-blocking.

Controls vs the Ours T1 cache (both images identical to 2 significant digits):

| Comparison | Layer-0 S | sys rows, max over layers | Last-layer S K / V |
|---|---|---|---|
| T1 run twice | bit-exact | bit-exact | 0 / 0 |
| Full sequence through the reduced-prefill path | bit-exact | bit-exact | 0 / 0 |
| Prefix only (question removed, ~30 tokens shorter) | bit-exact | bit-exact | 0.010 / 0.021 |
| Orig reduced prefill (565–629 tokens) | rel 2.7e-3, 78% bit-exact | 7.9e-3 K / 1.4e-2 V | 0.21 / 0.42 |
| FP32 oracle vs Ours / vs Orig (layer 0) | K 3.3e-3 / 3.5e-3, V 2.4e-3 / 2.8e-3 | — | — |

Diagnosis: shape-dependent BF16 GEMM rounding (a different kernel at ~600
vs ~2,200 rows) plus RoPE cancellation, compounding over layers; the reduced
prefill path itself is exact. Raw logs: `dev_g2_revision1_failure/` in the run
and results directories.

### G2 revision 2 (approved 2026-11-02, superseded after the Stage-1 stop)

Verbatim text of the revision-2 gate:

G2 **Layer-0 sanity — revision 2** (store FP16 → FP32, all heads). These
tolerances were **revised after observing the dev samples n355567 and n9181
and before any measurement**; revision 1 and its failure log are preserved
verbatim in "Amendment history" below.
- Orig vs Ours, layer 0, **K** (S and sep rows): `rel_L2 ≤ 5e-3` **and** FP32
  oracle ratio `rel_L2(Orig, oracle) ≤ 1.5 × rel_L2(Ours, oracle)`. The oracle is
  RMSNorm → dequantized-NF4 `k_proj` → RoPE in FP32/FP64, computed from the
  arm's T1 `inputs_embeds` at original positions. No elementwise test for
  post-RoPE K (RoPE cancellation makes a relative elementwise bound meaningless).
- Orig vs Ours and Compact vs Ours, layer 0, **V**: `rel_L2 ≤ 5e-3` and
  `allclose(atol=1e-2, rtol=1.6e-2)` (unchanged).
- Sys rows and common-prefix rows (S∪sep IDs below the first unselected real
  ID), Orig and Compact vs Ours: layer 0 uses the two rules above (K oracle
  rule, V elementwise); **every layer ≥ 1 separately** must have
  `rel_L2 ≤ 3e-2` (per-layer maximum is the criterion). Applied to all five
  G4 pairs and every smoke/Stage-1 image; any exceedance stops the run and is
  reported.
- Compact vs Ours, layer-0 K after RoPE realignment: `rel_L2 ≤ 1e-2` and
  `allclose(atol=5e-2, rtol=5e-2)` (unchanged).
- Negative controls (unchanged): un-realigned Compact K on moved rows
  `rel_L2 > 0.1`; Orig vs Ours last-layer S `rel_L2 > 5e-3`. The latter is an
  implementation check that recomputation actually happened; it is **not**
  evidence for or against the quality hypothesis.
- On mismatch: stop and report per-layer/row diagnostics. Layers ≥ 1 S/sep
  rel-L2 and cosine (Compact K raw and realigned) are recorded, not gated.
- Noise floor (recorded for every G4 pair, not gated): a `prefix_only` prefill
  (same context, question removed) and the actual Orig reduced prefill are
  both compared with the T1 cache; the report gives the last-layer S K/V
  difference of each and the ratio context-effect / floor.

### Revision-2 Stage-1 stop (run `recompute_kv25_ablation_20261102T045423Z`)

G4 (5 pairs) and smoke passed. Stage 1 stopped fail-closed at the 12th image,
n51002, before any of its hits: `common_prefix_deep_v` at layer 31 had rel L2
0.0308 > 0.03 for both Orig and Compact. The common prefix there is one row
(original visual token 0, selected; context = system rows + itself). On the
same image the row's K was 0.0144, sys rows ≤ 0.0129 and the S context effect
0.21 / 0.40 (K / V). Earlier single-row images passed narrowly (n272098 0.027,
n331357 0.019). Every other G1/G2 gate passed. Protection-after: PASS.

Diagnostic (scratchpad, outside the run, no timers or stores): the
common-prefix row and the sys rows re-prefilled under four controls and
compared with the T1 cache, max over layers ≥ 1 (V unless noted):

| Image (L) | Orig reduced | Same-length truncation | Half-length truncation | Prefix only | Row K, same-length | Sys V, same-length |
|---|---|---|---|---|---|---|
| n51002 (565) | 0.0308 | 0.0308 | 0.0329 | 0 | 0.0150 | 0.0145 |
| n272098 (773) | 0.0270 | 0.0270 | 0.0297 | 0 | 0.0140 | 0.0134 |
| n331357 (539) | 0.0190 | 0.0190 | 0.0202 | 0 | 0.0105 | 0.0145 |

The same-length truncation keeps the row's context and position identical and
changes only the sequence length; it reproduces the Orig value exactly, and a
shorter sequence raises it to 0.033 with no context change. The rev.-2 bound
was calibrated on the 5-row sys aggregate (max 0.0145) and is too tight for a
single row. Single-row drift is 0.019–0.033 (V) / 0.0105–0.0165 (K), against a
last-layer S context effect of about 0.21 / 0.40 (K / V): the context effect is
12–21× the single-row V drift and 13–20× the single-row K drift. The
reduced-prefill logic is exercised exactly by a bit-identity check against the
same-length control, which replaces the bound (revision 3).

Change from revision 2 to revision 3, decided **before any quality result of
the stopped run was opened**: (1) common-prefix deep-layer bound → bit-identity
with the same-length truncation control; (2) the same bit-identity applied to
sys rows, all layers, K/V, Orig and Compact, as the single stop condition for
structural rows; (3) structural vs-Ours bounds (incl. sys-row 3e-2)
report-only; (4) control computed outside every timer; (5) noise-floor table
extended; (6) Stage 1 restarted from G4 in a new run directory. Raw
diagnostic: `dev_g2_revision2_failure/` in the new run and results
directories.
