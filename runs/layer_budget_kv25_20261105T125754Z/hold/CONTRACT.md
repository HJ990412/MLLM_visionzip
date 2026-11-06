# Layer-wise Budget Allocation for PrefixStore-KV25 — experiment contract (version 1)

Frozen 2026-11-05 UTC before any GPU measurement of Phase 1 beyond the 10-image
timing probe. Repository `/home/dblab/hj/mllm_v2` (no usable `.git`; source
protection is by SHA256 manifest). Run root `runs/layer_budget_kv25_20261105T125754Z`,
results `results/layer_budget_kv25_20261105T125754Z`. Every `[확인 필요]` item of
the draft is replaced below by the locally measured value; the measurement
provenance is recorded in the run (`data_split.json`, `phase0/n_distribution.json`,
`phase0/storage_plan.json`, `phase1/timing_probe.json`).

## A0. One-line summary

Keep the total visual-KV row budget per image equal to uniform Ours-KV25
(`B(N) = 32 · ceil(0.25·N)` rows, identical bytes) and re-distribute it across
decoder layers with a fixed per-model vector `k_l`. Selection stays
query-independent; each layer is one contiguous read; selector cost 0; 64 + 1
preads per hit. The only thing that changes is how many of the already-ranked
rows each layer sees.

## A1. Motivation (from existing results)

- `results/image_only_repack_budget_sweep_with_recomp/error_analysis.csv`
  (LLaVA, GQA 40/240): at 25 % the loss vs FullLoad is spatial −12.0 pp,
  object −5.0 pp, yes/no −3.1 pp, color/attribute no loss; FullLoad-only
  correct questions fall 19 → 6 when the budget is raised to 45 %.
- All layers currently use the same `k = ceil(0.25·N)`. Visual-token attention
  is strongly layer-dependent in MLLMs (FastV), so a re-distribution at equal
  bytes may recover accuracy.
- `keep_count_per_layer` and the per-layer `BIAS[li]` mask already exist, so
  the serving change is small.

## A2. Scope

- **Model**: LLaVA-NeXT `llava-hf/llava-v1.6-vicuna-7b-hf`
  (revision `c916e6cdcd760b4cecd1dd4907f84ac649f93b23`, 32 layers) only.
  Qwen2.5-VL is a second stage only if LLaVA is GO/PARTIAL; GLM is out of scope.
- **Changed**: per-layer stored/visible content rows `k_l` (new store policy
  `prefix25_layerwise`, new method id `ours_prefixstore_kv25_layerwise_<alloc_id>`).
- **Unchanged**: image-only VisionZip saliency and the global permutation,
  separator sidecar policy, 64-row chunks, FP16 payload, NF4/BF16/eager,
  seed 1234, greedy max 16, the Turn-1 piggyback flow, every existing baseline
  code path (`ours_prefixstore_kv25` is served by the untouched `prefix25` writer).
- **Forbidden**: using the question, history or LLM output for selection;
  image-specific `k_l` (one fixed ratio vector per model); modifying existing
  runs/results/stores; re-interpreting existing contracts.

## A3. Data split (measured; `runs/<RUN>/data_split.json`)

| Role | Use | Source | Images / questions | Overlap |
|---|---|---|---|---|
| **CAL** | layer sensitivity, allocation vectors | `data/vqav2/index.json` (sha `b83d5fa2…`), 250 images × 5 questions | 250 / 1,250 | CAL∩DEV = 0, CAL∩HOLD = 0 |
| **DEV** | choose γ / A1 vs A2 | `data/index.json` (sha `514d1203…`), 40 images, `questions[4:10]` | 40 / 240 | DEV∩HOLD = 40 |
| **HOLD** | final report | `data/mt_gqa/dialogues.json` (sha `2c47cfad…`), 398 images, 4,061 dialogues × 3 turns | 398 / 12,183 requests per arm | — |

- The draft's CAL candidate (GQA-large 395 images, `data/gqa_large/index.json`)
  is **rejected**: CAL∩DEV = 40, CAL∩HOLD = 395, 0 images remain after removing
  DEV∪HOLD (< 150 minimum). CAL is therefore the entire VQAv2 250-image workload.
  VQAv2 ids are COCO ids and GQA ids are Visual Genome ids; the image_id overlap
  is 0 by construction (pixel-level identity across datasets is not checked and
  is recorded as a limitation).
- DEV∩HOLD = 40: every DEV image is inside HOLD with different questions. This is
  the same condition as the prior KV25 contract and B main and is reported, not hidden.
- CAL gold: the most frequent normalized VQAv2 answer (ties → first occurrence).
  It is used only for the gold first-token NLL / first-token-match diagnostics;
  the primary signal (KL) needs no gold.
- Allocation vectors are computed from CAL only; DEV only picks one candidate;
  HOLD runs once.

## A4. Phase 1 — layer sensitivity (`scripts/152_profile_layer_sensitivity.py`)

Per CAL image one normal full-pixel Turn-1 inference (production
`Server.recompute` with `VisionForwardCapture`) yields the captured visual KV
and the image-only VisionZip permutation (`visionzip_repack_order`). The full
visual KV is kept in GPU memory in stored (ranked) physical order with the
same FP16→BF16 round trip as the SSD path; no SSD reads, no timing. Per
question, each variant is one suffix prefill over that cache with the
production `bias_from_keep` masks (one `keep` vector per layer) followed by
`DynamicCache.crop(prefix_len)`. An independent assertion checks, for every
layer, that the number of visible visual rows equals `k_l + S`, that the
structural rows and system rows are visible and that no unselected row is.

**S1 (primary)**. U: every layer `k_u = ceil(0.25·N)` (the Ours-KV25 mask).
R_l: layer l fully visible, all others `k_u`. D_l: layer l `k_u`, all others
fully visible. FULL: all visible. Metrics per variant: `KL(P_FULL ‖ P_variant)`
on FP32 first-token distributions, gold first-token NLL (raw and capitalized
gold, min), first-token argmax agreement with FULL and with gold.
`s_l = mean_q[KL(U) − KL(R_l)]`, clipped at 0. Reverse signal: `KL(D_l)`; the
Spearman ρ between the unclipped `s_l` and `KL(D_l)` rankings is reported.
Correctness is measured by full greedy decoding (strict EM against gold and
VQA soft score) for FULL and U only; the 64 R/D variants use the first-token
proxies (full decoding of 64 variants per question is not affordable and the
first token carries the answer for single-word VQA).

**S2 (secondary, A2 input)**. During the FULL prefill and the first decode step
the eager attention weights (text query rows → visual rows, summed over heads
and query rows) are accumulated per layer. Coverage `c_l(k)` = share of real
visual mass on the first k stored rows, saved on the 64-row grid plus the
endpoint N; separator mass is recorded separately.

Cost: (2 full decodes + 64 single prefills) per question. Timing probe on the
first 10 CAL images decides the per-image question count: if the projected
total for 5 questions/image exceeds 4 h, 3 questions/image are used
(`phase1/timing_probe.json` records the measured per-forward time and the
decision). The measured value is bound to this contract version through the
run config, not edited into this file afterwards.

Outputs: `runs/<RUN>/phase1/sensitivity.csv`, `coverage_curves.npz`,
`per_question.jsonl`, `fig_sensitivity.png`, `fig_coverage.png`.

## A5. Phase 2 — allocation vectors (`scripts/153_allocate_layer_budget.py`, `mmimpress/layer_budget.py`)

Ratio vector `r` with `Σ r_l = 32·0.25 = 8.0`, `0.10 ≤ r_l ≤ 0.60`. Per image
`k_l` is the Hamilton largest-remainder apportionment of `B(N)` with quotas
`r_l·N`: floor first, then one row at a time in descending fractional-remainder
order (ties: lower layer first), never above `min(N, ceil(0.60·N))`, repeated
until `Σ k_l = B(N)` exactly. The uniform vector reproduces `k_l = ceil(0.25·N)`
exactly at every N (tested at N ∈ {1,…,2880}). Validator bounds:
`floor(0.10·N) ≤ k_l ≤ min(N, ceil(0.60·N))`, exact rational arithmetic.

| ID | Rule | Free parameters |
|---|---|---|
| **U** | `r_l = 0.25` (= existing Ours-KV25, control) | 0 |
| **A1-γ** | `r_l ∝ s_l^γ`, γ ∈ {0.5, 1, 2}; clip to [0.10, 0.60] one bound per round (upper first), residual re-distributed proportionally to the unclipped layers | γ |
| **A2** | water-filling on the CAL-mean coverage curves (fraction grid): all layers start at 0.10; the step `64 / median N_CAL` is handed to the layer with the largest marginal coverage gain until Σ = 8.0; same clip | 0 |
| **F-front** | linear 0.40 (layer 0) → 0.10 (layer 31), normalized to Σ = 8.0 | 0 |
| **F-back** | F-front reversed (sanity) | 0 |

Vectors are saved as `runs/<RUN>/phase2/layer_budget_llava_<id>.json` with
`layer_budget_sha256` = SHA256 of the canonical JSON of the ratio list and are
never changed afterwards; the SHAs are copied into the run config.

## A6. Implementation (minimal)

1. **Store** (`mmimpress/store.py`, `mmimpress/piggyback.py`): policy
   `prefix25_layerwise`; layer l holds exactly `k_l` rows (short EOF, no padding),
   `k_l` recomputed inside the writer from the frozen vector. Meta adds
   `format=llava_fp16_importance_prefix25_layerwise_v1`, `budget_policy=layerwise_v1`,
   `layer_keep_ratios`, `layer_keep_counts`, `payload_rows_per_layer`,
   `layer_budget_id`, `layer_budget_sha256`, `layer_budget_r_min/r_max`,
   `stored_content_count_per_layer/_total`, `uniform_equivalent_content_count`.
   With the uniform vector every payload file (`layer_*/k.bin`, `v.bin`,
   `sep_kv.bin`, `sys_kv.pt`, `visionzip_layout.pt`) is byte-identical to the
   `prefix25` writer and `meta.json` agrees on every shared key except the
   policy/format labels (CPU-tested at all fixture N).
2. **Serving** (`mmimpress/serve.py`): layer l visible rows `[0, k_l)` through the
   existing `bias_from_keep` per layer; `ceil(k_l/64)` chunks coalesced into one
   contiguous span per layer and kind; 64 + 1 preads; budget must be exactly
   0.25 / `visual_kv` / `prefix`. Reader (`ChunkReader`) uses
   `payload_rows_for_layer`.
3. **Validator** (`mmimpress/prefix25.py::validate_llava_prefix_layerwise_meta`,
   `ImageContext.validate_visual_kv_layout`): file size = `k_l · row_bytes` per
   layer, `Σ k_l = B(N)`, bounds, counts recomputed from the vector, SHA match,
   mapping/permutation checks; fail closed.
4. **Raw row**: `keep_count_per_layer` kept; adds `k_per_layer`,
   `planned_normal_read_bytes_per_layer`, `k_uniform_equivalent`, `k_total`,
   `selection_rule`; actual per-file pread bytes come from the independent
   `os.pread` trace.
5. **Scripts** (local latest was 151): `152_profile_layer_sensitivity.py`,
   `153_allocate_layer_budget.py`, `154_validate_layer_budget_kv25.py`,
   `155_eval_layer_budget_kv25.py` (DEV), `156_eval_layer_budget_hold.py` (HOLD),
   `157_report_layer_budget_kv25.py`; `tests/test_layer_budget_kv25.py`.

## A7. Gates (fixed before execution; any unrun/failed gate blocks DEV/HOLD)

- **G1 regression**: with the uniform vector, for the 5 fixed GPU cases of
  `docs/llava_kv25_budget_contract.md` (`n355567/201751701`, `n9181/20929611`,
  `n390187/201861403`, `n133585/202108008`, `n272098/201535625`): payload bytes
  identical to the existing `prefix25` writer from the same capture; per-layer
  masks, first token and generated token IDs identical when served; the
  recorded fresh-[B] output of `runs/prefix25_persistence_20261001T050113Z/llava/final_gpu_regression.json`
  reproduced (tokens, prediction; logits SHA reported). All-step logits are
  reported (expected bitwise equal; equality is not a threshold).
- **G2 reference**: with F-front, an independent implementation (no production
  selector/mask helpers) builds the per-layer visible sets, memory cache and
  additive masks; production masks must match bit for bit at every layer in
  prefill and every decode step; FP32 first and all generated-step logits
  `atol = rtol = 1e-4`; first/generated token IDs identical; actual `os.pread`
  ranges equal the independent per-layer plan `(0, k_l·row_bytes)` for every
  `k.bin`/`v.bin` plus one full `sep_kv.bin` read; no probe/score/layout file read.
- **G3 budget invariance**: `Σ k_l = B(N)` exactly for every CAL/DEV/HOLD image
  (CPU, from the frozen N lists); on DEV the actual pread byte sum of every
  layerwise arm vs uniform must be within ±2 % (payloads are exact-row, so the
  expected value is 0 %; the measured value is reported).
- **G4 question invariance**: changing only the question leaves `k_per_layer`
  and `selected_original_ids` unchanged; vision forward and query-score calls are 0.
- **G5 CPU fixtures**: N ∈ {1, 63, 64, 65, 127, 128, 129, 256, 349, 2200},
  clip boundaries, largest-remainder exactness vs an independent reference,
  invalid vectors (NaN/Inf/negative/length/sum/out-of-range) fail closed,
  tampered metadata/SHA/row counts/file lengths fail closed, uniform byte identity.
- The complete existing CPU suite must still pass.

## A8. Phase 3 — DEV (`scripts/155_eval_layer_budget_kv25.py`)

Same frozen 40 × `questions[4:10]` workload and structure as `scripts/89`/`106`
(image-major; T1 = `questions[4]` normal full-pixel inference; 5 cache hits;
deterministic arm rotation per image/turn; true TTFT; DONTNEED outside timers).
Arms: ReComp (reference, additive to the draft list), FullLoad, U, A1-0.5, A1-1,
A1-2, A2, F-front, F-back. T1 is executed once per image; its capture is the
source of every store and the T1 row is recorded for every arm as shared.
Every KV25 arm is served from its **own real store** written from that capture
(`prefix25` for U, `prefix25_layerwise` otherwise), so bytes, preads, TTFT and
persistence are measured for all arms (the mask-only substrate the draft
allowed is not used). Stores are deleted after each image's receipts are sealed.

Selection: `SELECTED` = the A1/A2 candidate with the highest DEV hit accuracy
(T2–T6, 200 hits); ties → A2, A1-1, A1-0.5, A1-2 in that order. If F-front's hit
accuracy is within ±0.5 pp of SELECTED, F-front also goes to HOLD.
Reporting: per question-type Δ (regexes of `scripts/26`, spatial highlighted),
paired image-cluster bootstrap 95 % CI (10,000 resamples, seed 1234), per-layer
actual `k_l`/bytes table.

## A9. Phase 4 — HOLD (`scripts/156_eval_layer_budget_hold.py`)

LLaVA part of `docs/mt_gqa_B_main_contract.md` unchanged (398 images, 4,061
dialogues × 3 turns, generated method-local history, strict normalized EM,
12,183 requests/arm, image-atomic attempts, independent per-image audit,
atomic commit, exact-allowlist cleanup, one live image at a time). Arms:
ReComp, FullLoad, B-uniform (`ours_prefixstore_kv25`), B-SELECTED
(+ B-F-front if admitted). No number of the prior B main is reused; all arms
are re-measured in one execution. Storage plan (`phase0/storage_plan.json`):
76 GB free on 2026-11-05 (prior main required 109.5 GB with 5 arms and a 40 GiB
raw allowance); the plan for 4–5 arms is reserve 30 GiB + peak working set
(raster 1.3 GB + up to three B stores 0.31 GB each + staging) + raw/evidence
growth bound; it is re-checked before HOLD starts and the run is BLOCKED if the
free space is below the requirement (no deletion of existing artifacts).
Metrics: followup accuracy, followup TTFT mean/p95, SSD MB/followup, store
MB/image, shared-stream and standalone-equivalent sessions.

## A10. Decision rule (fixed)

Δ = HOLD followup accuracy (B-SELECTED − B-uniform), image-cluster paired
bootstrap 95 % CI:

- **GO**: Δ ≥ +1.0 pp, CI excludes 0, SSD MB/followup ≤ uniform × 1.02,
  TTFT mean ≤ uniform × 1.05.
- **PARTIAL**: Δ ≥ +0.5 pp with CI including 0, or exactly one of the byte/TTFT
  conditions violated.
- **NO-GO**: otherwise. Phase 1 sensitivity results are reported regardless.
- If F-front ≥ SELECTED − 0.5 pp (HOLD followup), the contribution is described
  as a "front-heavy layer schedule", not "profiling".

Prior expectation: similar layer-budget reports recover 1–3 pp at equal bytes;
recovering half of the 25 % loss (DEV −3.3 pp, HOLD −2.3 pp) is success. A
result near 0 is itself evidence that the layer axis has no slack.

## A11. Reporting and preservation

`results/layer_budget_kv25_<ts>/`: REPORT.md, CONTRACT.md (this file),
sensitivity.csv, allocation vectors, DEV/HOLD quality/latency/io CSVs,
paired_analysis, independent_audit, REPRODUCE.md, protection before/after
hashes. Existing runs/results/stores are never modified; large stores/raw stay
local. `[확인 필요]` resolutions: CAL = VQAv2 250/1,250 (GQA-large rejected);
script numbers 152–157; N (content tokens) CAL 1344 / 2112 / 2880,
DEV 1440 / 2208 / 2880, HOLD 1344 / 2256 / 2880 (min / median / max),
`Σ B(N)` CAL 4,414,080, DEV 698,880, HOLD 6,984,192 rows; disk 76 GB free;
Phase 1 forward time from the probe is bound through the run config.
