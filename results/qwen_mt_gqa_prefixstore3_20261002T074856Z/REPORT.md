# Qwen PrefixStore3 MT-GQA-reconstructed generated-history main

Run: `/home/dblab/hj/mllm_v2/runs/qwen_mt_gqa_prefixstore3_20261002T074856Z`. Qwen-only; completed LLaVA remains unchanged. MB=10^6 bytes, GiB=2^30 bytes, ms, accuracy differences in percentage points.

| Validation | Status |
|---|---|
| LEGACY G7 STATUS | FAIL (preserved and reproduced) |
| G7 ROOT-CAUSE ATTRIBUTION | PASS: FP32 operation shape/rounding in validator |
| VALIDATOR/IMPLEMENTATION CHANGE | validator + experiment adapter only; existing inference unchanged |
| FINAL SOURCE FREEZE | PASS |
| CPU REGRESSION | PASS |
| MATCHED GPU CORRECTNESS | PASS |
| INDEPENDENT SEMANTIC VALIDATION | PASS |
| GENERATED-HISTORY INTEGRATION | PASS |
| ADDITIONAL 8 DIALOGUES | PASS |
| NEGATIVE FIXTURES | PASS |
| MAIN-RUNNER SMOKE | PASS |
| FULL MAIN COVERAGE | PASS: 36,549 / 36,549 |
| HISTORY/SELECTION/I/O AUDIT | PASS |
| QUALITY RESCORING | PASS |
| SESSION ACCOUNTING | PASS |
| RAW REPRODUCIBILITY | PASS |
| ARTIFACT PROTECTION | PASS |

G7 is classified as a numerical validation problem. Actual BF16 SSD KV, positions, masks, first logits and generated IDs match independent same-shape references. Fixed-input FP32 QK, softmax and value aggregation vary with dense/compact operation shape on CPU. All-layer FP64 references (converted before operations) meet 1e-12/1e-12. FP64 is finite-precision validation, not main inference. The old 1e-5/1e-5 FAIL remains. G7_REVIEW.md and G7_stage_attribution.json contain causal controls, input hashes and stage attribution.

## Accuracy

| Method | T1 % | T2 % | T3 % | All % (correct/N) | Followup % (correct/N) |
|---|---|---|---|---|---|
| ReComp | 60.749 | 61.660 | 61.069 | 61.159 (7451/12183) | 61.364 (4984/8122) |
| FullLoad | 60.749 | 61.709 | 61.093 | 61.184 (7454/12183) | 61.401 (4987/8122) |
| Ours B | 60.749 | 57.892 | 57.843 | 58.828 (7167/12183) | 57.868 (4700/8122) |

| B minus baseline | All Δpp [95% CI] | Followup Δpp [95% CI] |
|---|---|---|
| ReComp | -2.331 [-2.840, -1.829] | -3.497 [-4.260, -2.743] |
| FullLoad | -2.356 [-2.868, -1.856] | -3.534 [-4.302, -2.784] |

These are method-level comparisons including generated-history divergence. Raw predictions/gold are independently strict-normalized rescored. Turn-specific paired CIs and first-token/full-sequence/normalized-prediction agreements and paired contingencies are in paired_analysis and prediction_agreement.

## Followup performance

| Method | TTFT mean/p50/p95 ms | E2E mean/p95 ms | Actual KV read MB | Preads | H2D tensor MB | Compact KV MB |
|---|---|---|---|---|---|---|
| ReComp | 113.279/113.182/126.619 | 133.616/157.733 | 0.000 | 0.000 | NOT_MEASURED | NOT_MEASURED |
| FullLoad | 71.777/71.044/79.072 | 92.224/115.476 | 24.407 | 57.000 | 21.765 | 21.765 |
| Ours B | 54.010/54.038/56.384 | 74.182/88.369 | 6.371 | 57.000 | 6.371 | 6.371 |

ReComp followups are full-image recomputation, not cache hits. KV reads=0 does not mean zero image-file I/O. B reads 100% of its stored k content rows, representing ceil(original N/4). All structure is retained. Page-cache DONTNEED is an OS hint, not cold NAND. H2D bytes count tensors. GPU peak includes coexisting contexts.

## Store and setup

| Method | Content % (pooled) | Valid content MB | Total/allocated MB | OS write MB | Outer persistence mean/p50/p95 ms | Activation mean/p50/p95 ms | Build/activation |
|---|---|---|---|---|---|---|---|
| FullLoad | 100.000 | 20.477 | 24.342/24.348 | 24.342 | 64.294/63.553/73.003 | 14.151/13.754/15.934 | 398 / 398 |
| Ours B | 25.127 | 5.145 | 6.400/6.486 | 6.400 | 45.969/45.621/51.920 | 5.115/5.092/5.646 | 398 / 398 |

Persistence includes durable native publication and experiment completion seal; activation includes integrity reads/metadata/FD setup. Source T1 saliency/capture-clone tail is charged exactly once in its request E2E. Component intervals can overlap and are not summed. ReComp has no KV store/build/activation.

## Three-turn session accounting

| Method | Shared-stream mean/p50/p95 ms | Derived standalone mean/p50/p95 ms |
|---|---|---|
| ReComp | 399.590/402.233/452.118 | 399.590/402.233/452.118 |
| FullLoad | 324.537/315.343/395.339 | 395.419/392.881/445.135 |
| Ours B | 285.942/276.591/336.217 | 332.083/327.759/373.246 |

| B minus baseline | Accounting | Mean delta ms [95% CI] |
|---|---|---|
| ReComp | physical_stream_session_ms | -113.648 [-115.593, -111.692] |
| ReComp | derived_standalone_session_ms | -67.507 [-69.212, -65.780] |
| FullLoad | physical_stream_session_ms | -38.595 [-39.391, -37.815] |
| FullLoad | derived_standalone_session_ms | -63.336 [-64.486, -62.229] |

These tables include measured outer writer-return time plus activation. Shared-stream charges each actual image/method setup once at its source dialogue and reconciles to 796 physical builds and 796 activations. Standalone-equivalent reattributes the same measured setup once per dialogue; it is DERIVED and omits a NOT_MEASURED hypothetical non-source capture increment. Request E2E, not TTFT sums, supplies the three request costs. The original inner durable-publication endpoint remains in final/; the reporting-only amendment and outer-endpoint CIs are in reporting_details/.

## Qwen findings and limits

B vs ReComp: followup accuracy -3.497 [-4.260, -2.743] pp; TTFT mean 52.32% reduction, p95 55.47% reduction. Actual KV read difference +6.371 MB/request.
B vs FullLoad: followup accuracy -3.534 [-4.302, -2.784] pp; TTFT mean 24.75% reduction, p95 28.69% reduction. Actual KV read difference -18.037 MB/request.

Mean and p95 are reported separately; a mean improvement alone does not establish p95 improvement or statistical equivalence. Single GPU, batch1, bounded image-major shared cache; no long-session throughput/eviction/generalization claims. LLaVA is a separate historical run and no cross-model latency pairing is implied.

Generation cap/empty/EOS counts and rates are in request_detail_summary.csv. Attempts, physical events and failed diagnostics remain separate from final main requests. Native store payloads were deleted only after audit+hash+commit within exact owned paths and require regeneration. Prior large-file protection uses labeled sampled fingerprints; new artifacts and raw use full SHA256.


## Measured I/O, residency and reporting boundary details

The root report uses the recorded outer writer return for persistence and session accounting. The original final/ report preserves the inner durable-publication endpoint; its wrapper receipt-return tail was outside that endpoint. This is a reporting-only amendment after main completion. No inference, correctness criterion, source freeze or raw measurement changed. reporting_details/paired_analysis.json supplies the matching outer-endpoint CIs.

| Method | Durable publication mean ms | Outer return mean/p50/p95 ms | Receipt-return tail mean ms |
|---|---|---|---|
| FullLoad | 64.232 | 64.294/63.553/73.003 | 0.062 |
| Ours B | 45.911 | 45.969/45.621/51.920 | 0.058 |

| Method | T1 TTFT mean/p50/p95 ms | T2 TTFT mean/p50/p95 ms | T3 TTFT mean/p50/p95 ms |
|---|---|---|---|
| ReComp | 112.050/112.675/119.910 | 112.761/112.932/120.265 | 113.797/113.327/126.943 |
| FullLoad | 112.029/112.710/119.886 | 69.404/69.304/73.174 | 74.151/72.791/84.852 |
| Ours B | 112.034/112.673/119.747 | 52.623/52.497/54.345 | 55.398/55.358/56.967 |

| Method | Original content MB | Stored valid MB | Padding MB | Structure/system/boundary MB | Metadata+seal file MB | Resident metadata MB | Payload FDs (derived) | Activation read MB / preads |
|---|---|---|---|---|---|---|---|---|
| FullLoad | 20.477 | 20.477 | 2.612 | 1.204 | 0.048 | 0.097 | 57.000 | 24.333 / 58.000 |
| Ours B | 20.477 | 5.145 | 0.000 | 1.204 | 0.050 | 0.101 | 57.000 | 6.390 / 58.000 |

Metadata residency is the native recursive Python object-size estimate of decoded meta; it excludes kernel FD memory and other process allocations. The payload FD count is derived from the verified file inventory/open path (one FD per payload), not separate kernel memory measurement. Resident metadata/FDs persist for an active image store; visual KV payloads are reread on every hit. Structure includes system and vision-boundary rows in one sidecar; those overlapping categories are not summed again. FullLoad padding is serialized and read but discarded before H2D; B has zero content padding.

| Recorded component (ms) | FullLoad mean/p50/p95 | Ours B mean/p50/p95 |
|---|---|---|
| prefix_slice_serialization_ms | 0.978/0.959/1.197 | 0.366/0.356/0.445 |
| permutation_ms | 0.633/0.631/0.727 | 0.840/0.833/0.985 |
| kv_materialize_ms | 8.480/8.335/10.086 | 7.418/7.318/8.687 |
| kv_repack_ms | 0.000/0.000/0.000 | 0.652/0.632/0.832 |
| ssd_write_ms | 8.239/8.008/10.231 | 2.944/2.895/3.683 |
| fsync_ms | 32.271/31.793/35.458 | 26.933/26.777/28.434 |
| completion_publication_ms | 0.912/0.897/0.972 | 0.961/0.876/1.003 |
| capture_score_ms | 0.000/0.000/0.000 | 1.351/1.373/1.589 |
| capture_clone_ms | 0.341/0.335/0.372 | 0.347/0.339/0.396 |
| source_post_generation_tail_ms | 0.762/0.744/0.864 | 2.198/2.210/2.529 |

These are native named intervals, some nested: do not sum them into a critical path. kv_materialize includes CPU materialization/D2H; an isolated D2H-only timer and isolated serialization-only timer are NOT_MEASURED. Prefix slicing/serialization and write/fsync retain their native boundaries. Capture score/clone/post-generation tail is already charged once in source T1 E2E and is not added to persistence. Native per-payload full hashes are reused; no second full-payload checksum pass is claimed.

| Method | Followup GPU peak allocated / reserved MB (mean) | Padding read MB (mean) | Extra valid content rows (mean) |
|---|---|---|---|
| ReComp | 6165.429 / 7933.526 | NOT_MEASURED | NOT_MEASURED |
| FullLoad | 6116.931 / 7933.526 | 2.642 | 0.000 |
| Ours B | 6101.512 / 7933.526 | 0.000 | 0.000 |

GPU peaks are request-scoped torch allocator observations in the shared process, including model/coexisting contexts; they are not incremental KV size or independently deployed memory footprints. Per-scope mean/p50/p95 and measured counts are in request_gpu_io_summary.csv. ReComp has no cached visual-KV read; unrelated image-file I/O is not zero. All 36,549 outputs ended with EOS; no empty output or generation cap occurred.

The final/ directory retains quality, latency, strict rescoring, prediction agreement, raw audits and the original durable-publication analysis. reporting_details/ contains the authoritative outer-return session_per_dialogue.csv, session_summary.csv, paired_analysis.json and the detailed storage/residency tables. The full frozen artifact paths are indexed below.

## Terminal evidence and complete validation accounting

| Scope | Phase | Committed requests | Physical executed | Failed image attempts |
|---|---|---:|---:|---:|
| final | integration | 36 | 36 | 0 |
| final | smoke | 36 | 36 | 0 |
| final | main | 36549 | 36549 | 0 |
| additional | integration | 72 | 72 | 0 |
| same_image | integration | 18 | 18 | 0 |
| diagnosis_v1 | integration | 0 | 6 | 1 |
| diagnosis_v2 | integration | 18 | 18 | 0 |

Request E2E boundary: non-source requests end when the final token-generation loop (_decode) returns; final text rendering and diagnostic serialization are outside that endpoint. Source-capture T1 ends at the outer core return and includes required saliency, capture clone and output postprocessing once. The same boundary applies across methods; the standalone-equivalent capture limitation remains.

Original ten core pairs and 24 fixed-input repeated reference controls are separate validation, not main requests. The first new diagnostic scalar-metadata assertion failed and is preserved. All final additional IDs were chosen before outcomes. No sample or tolerance was replaced after observing additional results.

Final main completed in a launcher wall of 4893.778 seconds including model load/warmup/smoke/logging/conditioning/per-image audit and cleanup. This is not request TTFT/E2E.

The authoritative terminal full-hash index is artifact_index_final.json in this report directory, produced after launcher/analysis logs closed. The nested final/ index is the earlier analysis snapshot. Existing large artifacts use the documented sampled protection fingerprints; all new artifacts in this terminal index use full-byte SHA256. LLaVA remains 398 committed images and 60,915 requests, with no rerun.
