# Layer-wise Budget v2 — Packed PrefixStore + prior-work allocation baselines (contract, version 2)

Frozen 2026-11-06 UTC before any v2 measurement. Repository `/home/dblab/hj/mllm_v2`
(no usable `.git`; source protection by SHA256 manifest). Run root
`runs/layer_budget_kv25_v2_20261106T073502Z`, results `results/layer_budget_kv25_v2_20261106T073502Z`.
v1 run `results/layer_budget_kv25_20261105T125754Z/` (verdict PARTIAL) is read-only input.
Every `[확인 필요]`-type item of the v2 draft (`papers/layer_budget_kv25_v2_packed_prompt.md`, Part A)
is replaced below by the locally measured or source-verified value; provenance is recorded in the run
(`phase0/prior_work_sources.json`, `phase0/storage_plan.json`, `phase0/v1_reuse.json`).

## A1. What v1 established and what v2 asks

- v1 HOLD: A1-2 (layerwise) − uniform = +1.305 pp (95 % CI [+0.56, +2.05]), identical bytes and
  preads (308.05 MB / 65 preads per followup), TTFT 1.124× → **PARTIAL** (only the TTFT ≤ 1.05× condition failed).
- The TTFT increase is a **file-layout effect**, not a budget effect: cold `pread` throughput on this
  disk depends on file size (≈ 2.0 GB/s at ≈ 2 MB vs 3.6–4.0 GB/s at 5–6 MB, `read_ahead_kb` = 128;
  `runs/layer_budget_kv25_20261105T125754Z/llava/dev_io_diagnostic_pread_by_size.json`), and A1-2 shrinks
  the mid/back-layer files.
- v2 answers two questions.
  1. With the **same I/O path for uniform and layerwise** (one packed file per image, one read syscall
     per hit), does layerwise A1-2 satisfy the pre-registered TTFT condition (≤ 1.05×)? → GO/PARTIAL/NO-GO re-decision.
  2. When the layer-wise allocation criteria of **VL-Cache, SqueezeAttention and PyramidKV** are vectorised in
     the same offline frame, is output-sensitivity-based A1-2 better than them? → related-work comparison table.

## A2. Unchanged from v1 (reused, not re-measured)

Data split (CAL = VQAv2 250 images / 1,250 questions; DEV = GQA 40 × `questions[4:10]`; HOLD = MT-GQA-reconstructed
398 images / 4,061 dialogues × 3 turns), model `llava-hf/llava-v1.6-vicuna-7b-hf`
(revision `c916e6cdcd760b4cecd1dd4907f84ac649f93b23`), NF4/BF16/eager, seed 1234, greedy max 16, Turn-1 piggyback,
image-only VisionZip permutation, row budget `B(N) = 32·ceil(0.25N)`, ratio-vector definition, Hamilton
largest-remainder, clip [0.10, 0.60], the **A1-2 vector itself** (v1 frozen ratio SHA
`2c2295cfbe876adc7d9a5e2253f1e1ead13bda3718e7b30e7f34779e03bfa8c7`), the uniform vector U
(ratio SHA `d79c1e44…`, file copied byte-identical), decision rule A10 of v1, question invariance.
v1 `phase1_sensitivity.csv` (SHA `9e6100450b916c178669c51cfaade13d8f0bdb9a9ca472696fbed083835986a3`) and
`coverage_curves.npz` (SHA `80f0387c5a3987fa93f34745ab5477d2754af7f2f5810e7c16e4c30a7e26a215`) are reused,
not re-measured; the v1 `A1-2` document file SHA is `c2bb0d4e9353bc4b461b850f09ef36d25c09a912fc2385786e4a99764ef94b16`.
No new selection: SELECTED = A1-2 is fixed before any v2 measurement.

## A3. Change 1 — PackedPrefixStore (`mmimpress/packed_store.py`, `mmimpress/packed_serve.py`)

- One `kv_packed.bin` per image: in layer order `[L0 K rows 0..k_0)[L0 V rows 0..k_0)[L1 K]…[L31 V][separator K/V sidecar]`,
  FP16, token-major `(rows, heads, head_dim)`; row content and order **bit-identical** to the unpacked
  PrefixStore [B] / layerwise files and `sep_kv.bin`. `sys_kv.pt`, `visionzip_layout.pt`, `meta.json`,
  `integrity.json` are kept as separate files exactly as in v1 (read at activation, before the request timer).
- Meta: `store_format = "prefixstore_kv25_packed_v1"`, `storage_policy = "prefix25_packed"`, a per-layer table
  `packed_layout = [{layer, k_offset, v_offset, rows}]`, `sep_offset`, `packed_total_bytes`, `packed_sha256`,
  plus every key of the v1 `prefix25_layerwise` meta (`budget_policy = "layerwise_v1"`, `layer_keep_ratios`,
  `layer_keep_counts`, `payload_rows_per_layer`, `layer_budget_id/sha256`, …). The uniform vector U is a
  layerwise vector (v1 G1 proved it reproduces the `prefix25` rows byte for byte), so every packed store,
  including U, carries a frozen vector document.
- Cache-hit read: **one read syscall** — `os.preadv(fd, [pinned_buffer], 0)` reading the whole file into a
  page-locked host buffer allocated at activation (outside every timer), then per-layer zero-copy views.
  `preadv` is used instead of `pread` because `pread` returns a fresh `bytes` object that would need a second
  host copy into pinned memory; the independent trace records both `os.pread` and `os.preadv` calls, and
  G7 requires exactly one read call (a `preadv`) and zero `pread` calls per hit. H2D, scatter (`PrefixCache.write`
  with the stored-row index), separator scatter, `bias_from_keep` masks and prefill are the v1 code path.
- uniform and every layerwise vector use the same packed path; bytes per hit = `Σ_l 2·k_l·row_bytes + separator bytes`,
  vector-independent (G3).
- Persistence: the packed writer writes the packed file **directly** from the captured cache (no intermediate
  per-layer files); write/fsync time is reported separately (one-time cost per image).
- DEV-only diagnostic arm `packed+directcopy`: rows `[0, k_l)` are copied as one contiguous slice into the dense
  GPU buffer instead of the index scatter. Not used on HOLD.
- Existing unpacked paths are untouched; method ids are separated: `ours_prefixstore_kv25_<vec>_packed`,
  `ours_prefixstore_kv25_<vec>_unpacked` (`U_unpacked` = the untouched `prefix25` writer and serving path of
  v1's `ours_prefixstore_kv25`; `A1-2_unpacked` = v1's `prefix25_layerwise` path).
- Validator: per-layer slice bit identity against the unpacked writer (gates), `packed_total_bytes = Σ(2·k_l·row_bytes) + sep`,
  monotone offsets, SHA256 of the packed file, all v1 layerwise geometry checks; fail closed.

## A4. Change 2 — prior-work allocation vectors (offline, CAL only, Σ r_l = 8.0, same clip and rounding)

Source verification (Phase 0, `phase0/prior_work_sources.json`):

| Work | Source checked | Commit / SHA |
|---|---|---|
| VL-Cache (Tu et al., ICLR 2025) | **No official code found** (GitHub search; the only public repo `bhzadjnty7/Efficient-VLM-Inference` is an unofficial notebook that defers to the paper). Definitions taken from the paper PDF arXiv 2410.23317 (SHA256 `f4848462…`), Eq. 1–2 and Algorithm 1. | paper only (ASSUMPTION-free: p = 1 % is stated in the paper) |
| SqueezeAttention (Wang et al.) | `github.com/hetailang/SqueezeAttention`, `utils_hh/modify_llama_drop.py` (cosine at L314, 3-cluster KMeans and re-allocation at L527–543) | commit `a1933d18b09c668312cbabdfd7b1fa73888c0f6f` |
| PyramidKV (Cai et al.) | `github.com/Zefan-Cai/PyramidKV`, `pyramidkv/pyramidkv_utils.py` `PyramidKVCluster.update_kv` (L360–370; `beta = 20` default) | commit `68cd9551a63fedf362ddb0edb008badd04c24996` |

The draft described the VL-Cache filter as "keep entries until cumulative mass p"; the paper's
ThresholdFilter (Eq. 1) is a **relative-to-row-maximum** threshold. The paper definition is used; this
correction is recorded here before measurement.

| ID | Rule (all computed from one fresh full-prompt FullLoad prefill per CAL question; mean over 1,250 questions) |
|---|---|
| **A1-2** | v1 frozen vector, reused |
| **U** | uniform 0.25 |
| **VLC_offline** | VL-Cache Eq. 1–2 / Alg. 1 with post-vision rows: `Q' = Q[m−τ:m]` (the τ language tokens after the image span), `A' = softmax(Q'Kᵀ/√d)` over all causal keys (system, visual, text); `ThresholdFilter(A', p)_ij = A'_ij if A'_ij ≥ p·max_j A'_ij else 0`, **p = 0.01**; per-head sparsity `γ_h = #zeros / #causal entries`, `γ_l = mean_h γ_h`; `r_l ∝ (1 − γ_l)` normalised to Σ = 8.0 then clipped to [0.10, 0.60] with v1 `clip_redistribute` (VL-Cache's own clip is [0.01, 1]; the contract clip is applied to every vector). A visual-keys-only sparsity is recorded as a diagnostic column only. |
| **SA_offline** | SqueezeAttention code definition: per decoder layer, `cos_l = mean over prompt tokens of cosine(h_in, h_in + attn_out)` where `h_in` is the residual-stream input of the layer and `h_in + attn_out` the residual after the attention sub-layer (before the MLP); all prompt tokens (system, visual, text) of the prefill. Importance `1 − cos_l`; `r_l ∝ (1 − cos_l)`, normalised, clipped. |
| **SA-2g_offline** | SqueezeAttention's group rule on the same `cos_l`: 1-D 3-cluster k-means on `cos_l` (exact optimum by enumerating all contiguous splits of the sorted values; `sklearn` is not installed and random-init KMeans is not reproducible). The highest-cosine cluster ("class 3") gets budget `percent`; the other two clusters share the remainder equally: `a = (32·0.25 − |C3|·percent) / (32 − |C3|)`. **ASSUMPTION**: `percent = 0.25·(0.08/0.21) ≈ 0.0952` (the README's example ratio `KV_class3 / ini_size = 0.08 / 0.21`), which is below r_min and therefore clipped to 0.10 by the contract clip. |
| **PYR** | v1 F-front (linear 0.40 → 0.10, Σ = 8.0) re-labelled; ratio SHA identical to v1 F-front (`6858154b…` file; ratio SHA recorded in `phase2/summary.json`). |
| **PYR-beta** | PyramidKV code schedule with window 0 and β = 20: `min = 0.25/β`, `max = 2·0.25 − min`, `step = (max − min)/(L−1)`, `r_l = max − l·step` (Σ = 8.0 exactly), then the contract clip (layers whose `r_l < 0.10` are raised to 0.10; residual redistributed proportionally among the unclipped layers). |

- VLC and SA are **online, per-request** quantities in the original papers; v2 freezes their CAL mean as a
  fixed vector, hence the `_offline` suffix and the report note "allocation criterion moved offline, not a
  reproduction of the original system".
- Vectors are saved as `phase2/layer_budget_llava_<id>.json` with SHA256, never changed afterwards; all
  vectors are drawn in one figure (`phase2/fig_vectors.png`).
- G8 reproducibility: VLC/SA vectors are computed twice from the per-question records in two different CAL
  orders (forward, reversed); max |Δr_l| < 1e-6.

## A5. DEV and HOLD arms

- **DEV** (GQA 40 / 240, structure of v1 §A8: one executed T1 per image, 5 cache hits per arm, deterministic
  rotation, true TTFT, DONTNEED outside timers). Every arm is served from **its own real store** (stronger than
  the draft's allowed shared-FullStore mask; the raw rows record `store_kind`). Accuracy arms: ReComp (reference),
  FullLoad, U, A1-2, VLC_offline, SA_offline, SA-2g_offline, PYR, PYR-beta (all packed). Latency arms, each
  executed 3× per hit (cold each time; outputs must be identical across repeats; mean and SD reported):
  U_unpacked, U_packed, A1-2_unpacked, A1-2_packed, VLC_offline_packed, SA_offline_packed, PYR_packed,
  A1-2_packed_directcopy. Report: hit accuracy per vector with paired CI vs A1-2 and vs U, question-type Δ
  (spatial highlighted), packed vs unpacked TTFT / read ms / scatter ms / read-call count, packed TTFT spread
  across vectors (±2 % question), per-layer `k_l` / bytes.
- **HOLD** (MT-GQA 398 / 4,061 × 3, generated method-local history, strict normalized EM, 12,183 requests per arm),
  7 arms: ReComp, FullLoad, U_packed, A1-2_packed, VLC_offline_packed, SA_offline_packed, PYR_packed.
  Projected ≈ 10–15 h.
- **SELECTED = A1-2**, fixed. If VLC/SA/PYR beat A1-2 on DEV, SELECTED does not change; they are comparison columns only.

## A6. Storage plan (measured 2026-11-06; `phase0/storage_plan.json`)

Free space at freeze: 67 GB (62.4 GiB). Packed store per image ≈ 296 MB (N = 2112) and ≤ 400 MB (N = 2880),
so 398 images × 5 vectors ≈ 123–160 GB per vector set cannot be held at once. The v1 lifecycle
(`FRESH_IMAGE_STREAMING`: one live image at a time, stores built → all arms measured → independently
audited → committed → **run-local scratch deleted by exact SHA allowlist**) is the "sequential build → measure →
delete" policy at image granularity; deletion is restricted to `runs/<RUN>/hold/scratch/` and
`runs/<RUN>/llava/dev_stores/` files created by this run. Existing stores are never deleted.
Per-image live bundle on HOLD ≤ 1.38 GB raster + 5 × 0.40 GB packed = 3.4 GB; peak working set bound
3.4 GB + 4 GiB builder margin; raw/evidence growth bound 8 GiB (v1 4-arm main = 1.9 GB → 7 arms ≈ 3.4 GB);
minimum reserve 30 GiB. Required ≈ 45.4 GiB ≤ 62.4 GiB free → PASS. The `capacity_guard` re-checks before every
image; failure → `BLOCKED_STORAGE`, no deletion.

## A7. Gates (v1 G1–G5 re-run + new; any failed/unrun gate blocks DEV/HOLD; no threshold changes)

- **G1–G5**: v1 `scripts/154` stages cpu/g3/gpu re-run on this run (the touched v1 modules are unchanged).
- **G6 packed equivalence**: for the 5 fixed GPU cases and every DEV image's first hit (first token), the packed file's
  per-layer slices are bit-identical to the unpacked B / layerwise files, total bytes equal, the restored GPU cache,
  masks, first token and generated ids identical to the unpacked path (logits reported; bitwise expected).
- **G7 read path**: per hit exactly one read call (`os.preadv`) on `kv_packed.bin`, zero `os.pread` calls,
  returned bytes = file size; probe/score/vision calls 0; the read range is unchanged when only the question changes.
- **G8 vector reproducibility**: VLC/SA vectors from two CAL orders agree to < 1e-6.
- **G9 I/O characterisation**: `fio` is not installed and no package may be installed; GNU coreutils `dd`
  (independent of `mmimpress`) measures cold sequential read throughput at file sizes
  {256 KB, 1 MB, 4 MB, 16 MB, 64 MB, 256 MB}, 5 fresh files each, buffered after `posix_fadvise(DONTNEED)` and
  with `iflag=direct`; written to `io_microbench.csv` outside every request timer.
- Complete existing CPU suite: no regression (GLM modules recorded separately as pre-existing env errors, as in v1).

## A8. Decision

- **Primary** (v1 A10 verbatim): A1-2_packed − U_packed, HOLD followup accuracy, image-cluster paired bootstrap 95 % CI
  (10,000 resamples, seed 1234). GO: Δ ≥ +1.0 pp & CI excludes 0 & SSD MB/followup ≤ 1.02× & TTFT mean ≤ 1.05×.
  PARTIAL: Δ ≥ +0.5 pp with CI including 0, or Δ ≥ +1.0 pp with CI excluding 0 and exactly one of bytes/TTFT violated.
  NO-GO otherwise.
- **Secondary** (reported only): (a) U_packed TTFT vs v1 U_unpacked HOLD TTFT (cross-run, labelled as such),
  (b) A1-2 vs VLC/SA/PYR Δ and CI, (c) packed TTFT spread across vectors at identical bytes within ±2 %.
- Pre-registered expectations: packed uniform SSD read ≈ 137 → 80–90 ms; A1-2 − U TTFT ≈ 0; VLC/SA 0–1 pp below A1-2
  with possibly overlapping CIs; PYR significantly lower (v1 DEV −3 pp).

## A9. Scripts and artifacts (local latest script number was 157)

`scripts/158_allocate_prior_work_vectors.py` (Phase 1 vectors + G8), `scripts/159_eval_layer_budget_v2_dev.py` (DEV),
`scripts/160_report_layer_budget_v2.py` (report), and — added because the draft named no gate or HOLD script —
`scripts/161_validate_packed_store_v2.py` (G1–G7, G9, CPU suite) and `scripts/162_eval_layer_budget_v2_hold.py` (HOLD);
`mmimpress/packed_store.py`, `mmimpress/packed_serve.py`, `tests/test_packed_store.py`.
Results: `results/layer_budget_kv25_v2_<ts>/` with REPORT.md (v1 comparison table, vector figure, io_microbench figure,
question-type Δ, prior-work comparison table, decision), CONTRACT.md, vectors + SHA, dev/hold CSVs, paired_analysis,
independent_audit, REPRODUCE.md, before/after hashes, deletion logs. The report states: VLC/SA are offline-frozen
variants of online allocations and not reproductions of the original systems; PYR is a fixed-shape schedule
reproduction; all comparisons share bytes, the packed I/O path and the VisionZip token order and differ only in the
per-layer cut point.

## A10. Rules

No modification of existing stores/runs/results; no reset/clean/upgrade/commit/push; every artifact in the new run
root; a condition not in this contract is written here (with reason) before use, and post-hoc changes become a new
version; failed gates are not bypassed; storage shortage → BLOCKED and reported.
