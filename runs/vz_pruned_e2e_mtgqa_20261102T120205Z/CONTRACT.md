# VZ-Pruned-E2E vs Ours-KV25 vs ReComp on MT-GQA (LLaVA, Stage 2) — contract

**Status: APPROVED 2026-11-02 (plan approved with user amendments 1–4, listed
under "Approval amendments").** Frozen with a source-hash manifest in
`runs/<run>/gpu_freeze.json` before any GPU work; thresholds and decision rules
cannot change after any result is seen. Amendments are appended under
"Approval amendments", never edited in place. The amended text below is the
binding one; the original draft wording is quoted in the amendments.

## Question

Stage 1 (`results/rerope_kv25_20261102T093759Z`) measured a cache-less
"pruned first turn" on 40 GQA images only. This run asks, end to end on the
full reconstructed MT-GQA workload with generated history:

> Does running the **first turn itself on the 25 % important tokens**
> (VisionZip-style, "VZ-Pruned-E2E") and reusing *that* KV for turns 2–3 lose
> first-turn quality, and how do the whole dialogue (quality and latency)
> compare with "full first turn + 25 % KV reuse" (Ours-KV25) and with ReComp?

No outcome is assumed. The workload has already been used for the B main
(`runs/mt_gqa_B_llava5_qwen3_20261001T093921Z`), so the Ours−ReComp
contrast is a replication and the VZ contrasts are the new, pre-registered
exploratory comparisons.

## Frozen inputs and environment

- Workload: `data/mt_gqa/dialogues.json`, SHA256
  `2c47cfad2a7ccbb673042b400304d7f3ca03d6fbe59d04fa83db50708c924224`;
  ordered workload SHA256
  `0287e0c57813800c781633b969c5cff336b3a3c1a1bdcdbb56d63f6ddab0ca62`
  (`mmimpress.mt_gqa.workload_sha256`). 398 images, 4,061 dialogues × 3 turns
  = 12,183 requests per arm; 8,122 follow-ups per arm. Image files re-hashed
  against the B main manifest (`image_identity`). Image-major execution in the
  B main image order; each dialogue keeps its global ordinal.
- Model and serving: `llava-hf/llava-v1.6-vicuna-7b-hf` revision
  `c916e6cdcd760b4cecd1dd4907f84ac649f93b23`, slow AnyRes processor, NF4
  double-quant, BF16 compute, eager attention, FP16 SSD payload, 64-token
  chunks, separator sidecar, greedy, `max_new_tokens=16`, seed 1234, batch 1,
  CPU threads 2, `mllm_ft` (torch 2.5.1+cu121, transformers 4.57.6),
  RTX 4090, no other GPU process (`assert_gpu_exclusive` before every request;
  foreign GPU work aborts timing and is never terminated).
- Prompt, scorer, history: unchanged from 108/109 — `USER: <image>\n` +
  `Q1/A1 …` method-local generated history + `Current question Qt: …` +
  `Answer the current question with a single word or short phrase. ASSISTANT:`;
  strict normalized exact match against the first gold answer; each arm's own
  decoded answers (including empty / EOS / cap answers) become its history;
  gold is never supplied to inference.
- Reference runs (reproduction / protection only): B main
  `runs/mt_gqa_B_llava5_qwen3_20261001T093921Z` (ReComp and Ours-KV25 rows),
  `runs/prefix25_persistence_20261001T050113Z`, rev.-2/rev.-3 ablation runs
  `runs/recompute_kv25_ablation_20261102T{045423Z,061243Z}`, ReRoPE runs
  `runs/rerope_kv25_20261102T{080818Z,081118Z,093759Z}`.

## Arms

Notation: `v0` system tokens (5), `V` expanded image span (`v_token_num`),
`sep` row-separator rows, `N = V − n_sep`, `k = ⌈N/4⌉ = (N+3)//4`, `order`
the stable descending image-only VisionZip saliency permutation (penultimate
CLS→patch attention, head sum), `S = order[:k]`, `keep = sorted(S ∪ sep)`
(original relative order), `L = v0 + k + n_sep`. Compact position of kept
original ID `i`: `p'(i) = v0 + index of i in keep`.

| Arm (`method_id`) | T1 | KV reused in T2–T3 | Positions of stored K | Text RoPE start |
|---|---|---|---|---|
| ReComp (`recompute`) | full pixels, full tokens | none (pixels every turn) | — | original |
| Ours-KV25 (`ours_prefixstore_kv25`) | full pixels, full tokens (B path) | `S ∪ sep` rows of the **full-context** T1 cache, prefix25 store | original `v0+i` | `v0 + V` |
| **VZ-Pruned-E2E** (`vz_pruned_e2e`, new) | real pruned serving path: `[sys | keep | question]`, contiguous positions `0…L+q−1`, one LM prefill | `S ∪ sep` rows of the **pruned** T1 cache (`[0, L)`), prefix25 store | contiguous `p'` | `L` |
| FullLoad (`fullload`) | optional — see "Open points" | full canonical KV | original | original |

ReRoPE, Orig, PosShift, MPIC, SparseVLM-AllHead are out of scope.

### VZ-Pruned-E2E, first turn (real serving path, per dialogue)

1. TTFT timer starts before prompt/input preparation and image decode
   (same boundary as ReComp/Ours T1).
2. Processor (same `_combined_processor`), H2D.
3. **One** vision-tower call through `model.model.get_image_features(...)`
   (vision tower → feature select → projector → AnyRes `pack_image_features`
   with `image_newline`) inside `VisionForwardCapture(capture_saliency=True)`;
   the capture is finalized before step 4 (its cost is inside TTFT, because
   the method needs the saliency to build its own input).
4. `anyres_token_scores` → `visionzip_repack_order` → `reduced_plan(compact)`
   (identical functions to the B store builder and rev. 3).
5. Input assembly: `embed_tokens(input_ids)` for the text rows; the sequence
   `[sys rows | image feature rows at keep | text rows after the image span]`,
   shape `(1, L+q, D)`; no full-token `inputs_embeds` is built or run.
6. One language-model prefill with `position_ids = cache_position =
   arange(L+q)` (`runner.model(inputs_embeds=…)`, no `pixel_values`), first
   token = `int(argmax)` + `cuda.synchronize` (TTFT boundary of every arm),
   then the greedy decode loop of `Server._decode` (EOS or 16 decisions).
7. The `AbortAfterEmbeds` composition of Stage 1 is **not** used in any timed
   request; it appears only in correctness gate G4-b as the reference
   implementation.

No contextual merge, no recompute of dropped tokens, no query-aware scoring
(`online_query_score_calls = 0`). Reported TTFT of this arm is a real-path
measurement and is compared with the other arms' T1 TTFT.

### Store creation rule (amendment 2)

Stores are built **once per image**, from the T1 of the image's *source
dialogue* = the first dialogue of that image in the frozen B main ordinal
order (`manifest['dialogues']` order; the same rule as the B main's
`first_source_T1_only`). Both the Ours store and the VZ store of an image
come from that dialogue's T1; the T1 question of every other dialogue never
enters a store. The VZ prefix KV `[sys | keep]` is question-independent by
causal structure, but its BF16 values depend on the GEMM shape of the T1
prefill (`L+q`, i.e. on the source question length) — the same kind of
shape rounding that rev. 3 characterised. G2-c therefore compares the stored
structural rows with a control of the *same* length `L+q`, and G4-d / G8
record how the stored-prefix hit compares with a fresh single prefill.

**Every dialogue's T1 is executed for real**: Ours = full-pixel, full-token
T1 (`normal_request`, B path), VZ = the pruned real path above, ReComp = full
pixels every turn. No T1 answer is copied between dialogues or arms. On the
source dialogue only, Ours additionally keeps a reference to its
`inputs_embeds` (pre-hook, no copy, no sync) and VZ keeps its cache rows for
persistence; both are inside that T1's E2E exactly as in the B main and
rev. 3 ("capture exit included once"). Non-source T1 requests carry no
capture overhead beyond the saliency capture that the VZ method itself
needs.

### VZ-Pruned-E2E, persistence (source dialogue only)

From the T1 cache of the source dialogue of each image (as every other arm):
rows `[0, L)` are scattered into full-size `(1, H, v0+V, hd)` layers whose
other rows hold the finite sentinel `4096.0` (`synthetic_layers`, rev. 3), and
`persist_captured_visual_prefix(storage_policy='prefix25')` writes exactly the
`S` content rows, `sys_kv.pt` and the separator sidecar with the original
expanded `input_ids` — the same file format, sizes and chunking as the Ours
store. Metadata adds `ablation='vz_pruned_e2e'`, `kv_context='pruned_t1'`,
`position_policy='compact'`, `text_position_start=L`, `prefill_len=L+q`.
Persistence time includes the synthetic assembly and the one full completion
hash scan (`main_b_support.persist_llava`), like every other store.

### VZ-Pruned-E2E, hits (T2–T3)

`request_cvpr25(budget=.25, budget_unit='visual_kv', mode='prefix',
sep_policy='sidecar', expected_prefix_layout='visionzip_image_only')` on the
VZ store through `PositionServer(text_position_start=L)` — the Compact hit
path of rev. 3. Same conditioning (`posix_fadvise DONTNEED` outside timers),
same `legacy_hit` instrumentation (read trace, projection counts, no vision
call) as the Ours arm. Reads are identical in bytes and preads to Ours.

## Integrity gates (fail-closed unless marked report-only)

- **G1 freeze.** Dataset and image SHA256, model revision, package versions,
  source-hash manifest (`mmimpress/`, `scripts/`, `tests/`, this contract),
  CPU test suite PASS (`tests/test_recompute_kv25.py`,
  `tests/test_rerope_kv25.py`, new `tests/test_vz_pruned_e2e.py`, existing
  108/109 CPU regression). `verify_freeze` re-checks on every launch/resume.
- **G2 VZ store gate (per image, before any VZ hit; transplanted from rev. 3).**
  Inputs: Ours store, VZ store, Ours T1 `inputs_embeds` (reference-only hook
  on the source dialogue's Ours T1, as in rev. 3), VZ full original embeds
  (sys rows + all image feature rows + text rows, assembled outside timers).
  - G2-a selection identity: VZ selection digest, `k`, `full_importance_permutation`
    and payload sizes equal the Ours store's. Also every VZ T1 of the same
    image (all dialogues) reports the same `S` (**G7**).
  - G2-b prefix-embedding identity: SHA256 of the VZ `[sys | image rows]`
    BF16 tensor equals SHA256 of Ours `inputs_embeds[:, :v0+V]` (bit-identical:
    the real path reproduces HF's own merge).
  - G2-c structural bit-identity (rev.-3 rule, same-length control): the
    original sequence truncated to the VZ prefill length `L+q` is prefilled at
    `arange(L+q)` outside every timer; the stored system rows and common-prefix
    rows (`structural_rows`) of the VZ store must be **bit-identical** (FP16)
    to the control's rows at every layer, K and V. Recorded also in memory
    (BF16, before the FP16 cast) in G4.
  - G2-d layer-0 rule (amendment 1; rev.-3 Compact rule). The VZ store K is
    at contiguous positions `p'` and the Ours store K at original positions
    `p`, so layer-0 K is compared **after position realignment**: Ours
    layer-0 K of the `S` and `sep` rows rotated by `R(p'−p)` in FP64 vs the
    VZ rows — rel L2 ≤ 1e-2; FP32-oracle ratio
    `rel(VZ, oracle_p') ≤ 1.5 × rel(Ours_realigned, oracle_p')`
    (`layer0_oracle` from the Ours embeds, rotated to `p'`); **negative
    control**: un-realigned Ours K vs VZ K on the moved rows rel L2 > 0.1.
    Layer-0 V keeps the elementwise rule `LAYER0_TOL` (rel L2 5e-3,
    atol 1e-2, rtol 1.6e-2) on `S` and `sep` rows.
  - G2-e sentinel rows in the store = 0; metadata
    `text_position_start == v0 + stored_content_count + n_sep`.
  - Report-only: per-layer drift of `S` rows (K after rotation, V) vs Ours
    (the context effect, expected to be large at deep layers), system rows
    under the rev.-2 bound 3e-2.
- **G3 request semantics (per request, inside the image audit).** VZ T1:
  exactly one vision-tower call, one language-model prefill of length `L+q`
  with `position_ids == cache_position == arange` (layer-0 hook), zero SSD
  reads, projection calls `{q,k,v}=1` per layer in prefill and
  `len(tokens)−1` in decode. VZ hits: zero vision calls, `PositionRecorder`
  verifies text RoPE start `L` for every forward, reads = `⌈k/64⌉` content
  chunks per layer + separator sidecar, `selected_original_ids == order[:k]`
  (108/109 audit rules of the Ours arm). ReComp/Ours: unchanged 109 rules.
- **G4 correctness (5 fixed images, before smoke; stores retained).**
  - (a) Ours: the T1 of the B main's source dialogue is re-executed; given an
    identical T1 prediction, the T2 hit must reproduce the B main's recorded
    `first_token_id`, `generated_token_ids` and prediction.
  - (b) VZ path equivalence: the real path (above) and the Stage-1
    `AbortAfterEmbeds` + `pruned_generate` composition on the same image and
    T1 prompt must give identical token ids and first-token logits with
    `max|Δ| ≤ 1e-4` (same embeddings, same shapes — no rounding excuse).
  - (c) In-memory BF16 structural identity of the VZ prefix rows with the
    control (G2-c before the FP16 cast).
  - (d) Hit-vs-standalone: for T2 and T3 of the source dialogue, the VZ hit
    (store + history suffix) is compared with a cache-less single prefill of
    `[sys | keep | history + question]` at `arange`. Recorded (token
    agreement, first-token top-2 margins), **not gated**: the two differ only
    by GEMM shape rounding.
- **G5 protection.** Full SHA256 manifest before and after of `results/`,
  `docs/`, `mmimpress/`, `scripts/`, `tests/`, `README.md` and the reference
  runs listed above (B main, prefix25, rev.-2, rev.-3, three ReRoPE runs).
  Any change fails the run. Existing runs and stores are never modified.
- **G6 reproduction vs B main (report-only, mismatch analysis mandatory).**
  Request-level agreement of `prediction` and `generated_token_ids` for
  ReComp (12,183) and Ours (12,183) against the B main rows with the same
  `(dialog_id, turn)`. Mismatches are classified as (i) prompt-identical
  (same `prompt_sha256`, i.e. the model itself diverged) or (ii)
  history-cascade (an earlier turn already differed). For (i) the first-token
  top-2 logit margin of this run is reported; T1 cannot be class (ii).
- **G7** VZ `S` invariance across dialogues of one image (fail-closed; the
  Ours counterpart is 109's `ours_same_image_selection_invariant`).
- **G8 hit-vs-standalone sample in main (report-only).** 20 images fixed at
  freeze (seeded draw, seed 1234, listed in `gpu_freeze.json`), first
  dialogue, T2 and T3 = 40 pairs, computed after the image's timed requests
  and before cleanup; token agreement and first-token margins.
- Smoke = the B manifest's `smoke_dialogue_ids` (4 dialogues, 4 images,
  36 requests); integration = `integration_dialogue_ids` (2 dialogues of one
  image, 18 requests; store reuse across dialogues). Smoke/integration rows
  never enter main counts; their stores are deleted after audit, with sealed
  file hash lists kept in receipts.

## Metrics

Per arm: T1 / T2 / T3 / hit (T2–T3) / all strict EM; dialogue-level
"all three correct" rate (report-only). Efficiency, all MEASURED on the
shared stream: T1 TTFT and E2E (ReComp/Ours full-pixel, VZ real path), T1
LM-prefill token count (`v0+V+q` vs `L+q`), hit TTFT / E2E / SSD bytes /
preads, per-image persistence (service wall, native writer, completion hash,
OS write bytes/calls; VZ additionally synthetic assembly ms) and activation,
peak GPU memory. VZ T1 TTFT breakdown (processor+H2D, vision+projector+pack,
saliency+plan, assembly, LM prefill) is reported.

Session accounting follows 113 exactly: **shared-stream** = measured request
E2Es + build/activation attributed once to the source dialogue;
**standalone-equivalent** = requests + per-image setup reattributed to every
dialogue (DERIVED). Both are reported with Δ vs ReComp CIs.

## Decision rules (fixed before results)

Image-cluster paired bootstrap, 10,000 resamples, seed 1234, the same
resamples for every contrast (joint), pooled request-weighted means,
percentile 95 % CI, in percentage points:

| Contrast | (a) T1 EM | (b) hit EM (T2–T3) | (c) all EM |
|---|---|---|---|
| Ours − VZ | **PRIMARY** | secondary | **PRIMARY** |
| Ours − ReComp | secondary | secondary | secondary |
| VZ − ReComp | secondary | secondary | secondary |

**Primary endpoints (amendment 3):** Ours − VZ on (a) T1 EM and (c) all EM.
The other seven contrasts are secondary and are reported in a separate,
labelled table; the REPORT states the primary results first.

- A difference is "shown" only if its CI excludes 0 **and** |point| ≥ 1 pp;
  CIs containing 0 are not equivalence; differences < 1 pp are not interpreted.
- (b) compares each arm under its own generated history (method-level, as in
  the B main). Report-only supplement: the same contrasts on the subset of
  dialogues whose T1 prediction was identical between the two arms
  (history-matched); no decision is based on it.
- Latency: paired per-dialogue Δ of T1 TTFT, hit TTFT, shared-stream and
  standalone-equivalent session time vs ReComp and vs Ours, same bootstrap.
- Reproduction (G6) and G8 are described, not tested.

## Execution and storage

- New run `runs/vz_pruned_e2e_mtgqa_<UTC>/`, results
  `results/vz_pruned_e2e_mtgqa_<UTC>/`. Image-atomic lifecycle of 108:
  scratch stores per image, flush raw/events, full file hashes, independent
  per-image audit, atomic `COMMITTED.json`, close contexts, allowlisted
  cleanup of the image's stores; failures persist, partial images are re-run
  wholly into a new attempt; resume only with identical config/manifest.
- Per-dialogue order: ordinal rotation over the 3 arms (`ordinal % 3`); the
  B main rotated over 5. Every arm executes a real T1 for every dialogue;
  captures/persistence only on the first dialogue of each image.
- Disk: ≥ 30 GiB reserve + peak working set (two stores ≈ 0.7 GB per image;
  G4 stores retained ≈ 3.4 GB). Expected new artifacts ≈ 13 GB; current free
  space 85 GB.
- Time estimate (from B main per-request measurements): ≈ 5.3 s per
  dialogue × 4,061 ≈ 6.0 h of requests + ≈ 1.0 h per-image persistence /
  gates / audit / cleanup + ≈ 0.5 h freeze / G4 / smoke / integration → **≈ 7.5–9 h**
  for the three arms. FullLoad would add ≈ 2.5 h and 1.3 GB per-image scratch.
- Deliverables: `REPORT.md`, `summary.csv`, `REPRODUCE.md`, `analysis.json`,
  `kv_layer_diff.csv` (G2 per-layer), `persistence.csv`, `gates.jsonl`,
  `reproduction_vs_B.csv`, one README section. No commit unless requested.
- New sources: `mmimpress/vz_pruned_e2e.py` (real-path pruned T1, synthetic
  persistence helper, G2 wrapper), `scripts/147_eval_vz_pruned_e2e_mtgqa.py`
  (lifecycle re-using 108 helpers), `scripts/148_audit_vz_pruned_e2e_mtgqa.py`
  (109 rules + VZ rules), `scripts/149_report_vz_pruned_e2e_mtgqa.py`
  (113 definitions + decision table), `tests/test_vz_pruned_e2e.py`.

## Open points — resolved at approval (amendment 4)

1. FullLoad **excluded** (the B main already measured it on this workload;
   G6 links the runs).
2. Three-arm ordinal rotation (`ordinal % 3`) **approved**.
3. G4-b tolerance 1e-4 on first-token logits (fail-closed) **approved**.
4. G8 sample 20 images × T2,T3 (40 pairs), report-only **approved**.
5. G4 stores retained; smoke/integration/main stores deleted after commit
   with sealed hash lists **approved**.
6. History-matched subset analysis as a report-only supplement **approved**.

## Approval amendments (2026-11-02, before freeze and before any GPU work)

1. **G2-d.** Draft text: "VZ layer-0 K of `S` and `sep` rows vs Ours layer-0
   K rotated to `p'`: rel L2 ≤ 5e-3 and FP32-oracle ratio ≤ 1.5; layer-0 V
   elementwise `LAYER0_TOL`." Replaced by the rev.-3 Compact rule because the
   two stores hold K at different positions: realign (`R(p'−p)`, FP64) then
   rel L2 ≤ 1e-2, FP32-oracle ratio ≤ 1.5, negative control un-realigned
   rel L2 > 0.1 on moved rows; V elementwise rule unchanged.
2. **Store creation rule** written out (section "Store creation rule"): one
   store per image from the source dialogue's T1 (B main rule), VZ prefix KV
   question-independent by causal structure but subject to `L+q` shape
   rounding, every dialogue's T1 executed for real (Ours full, VZ pruned).
3. **Primary endpoints** pre-specified: Ours − VZ on (a) T1 EM and (c) all
   EM; the other seven contrasts are secondary and separated in the REPORT.
4. Open points 1–6 resolved as listed above (FullLoad excluded).

## Execution note (2026-11-02, development check before freeze)

Before the freeze, the implementation was exercised once on one image
(`n229548`, its first two dialogues, 18 requests) in a scratchpad directory
that is not a run: the real pruned path, both stores, the G2 gate (PASS;
structural rows bit-identical in all 32 layers, realigned layer-0 K rel L2
3.8e-3, oracle ratio 0.99–1.00, negative control 1.19–1.24), the G8 pairs
(token-identical, first-logit max |Δ| 0.125 from shape rounding), the G4-b
path equivalence (max |Δ| 0.0) and the per-image audit. ReComp and Ours
reproduced the B main's 12/12 rows. These numbers are development evidence
only — they are not run results and do not enter any table. The script and
its log are copied into `runs/<run>/dev/` and `results/<run>/dev/` at freeze;
their SHA256 are recorded in `gpu_freeze.json`. No threshold or rule was
changed after this check.

## Execution note (2026-11-02, run `vz_pruned_e2e_mtgqa_20261102T115431Z` stopped in G4)

The first frozen run stopped inside the correctness phase on its second image
(`n184551`): the per-image audit rule `first_token_margin_recorded` required
the top-2 margin's arg-max to equal the generated first token, which fails on
an exact top-2 tie (margin 0.0: the BF16 arg-max and the FP32 top-k may pick
different indices). This is an audit-rule defect, not a measurement defect;
the rule now accepts a tie. Because 148 is part of the frozen source set, the
run is preserved unmodified as a record (its two G4 images, `n14` complete
and `n184551` failed at audit, with retained stores) and a new run is frozen.
No smoke/integration/main request was executed in the stopped run and no
quality number was viewed beyond the single failing audit example.

## Execution note (2026-11-02, run `vz_pruned_e2e_mtgqa_20261102T115946Z` stopped before smoke)

The second frozen run passed G4 on all five images (`n14`, `n184551`,
`n350732`, `n481655`, `n548534`) and then stopped at the first smoke image in
the storage guard: the five retained G4 stores (≈ 3.5 GB under
`scratch/correctness/`) were counted as "incomplete payload" against the
4 GiB allowance. The guard now excludes the contractually retained
correctness stores. 147 is frozen, so the run is preserved unmodified as a
record (G4 receipts and stores retained) and a third run is frozen; G4 is
re-executed there. No smoke/integration/main request ran in this run.
