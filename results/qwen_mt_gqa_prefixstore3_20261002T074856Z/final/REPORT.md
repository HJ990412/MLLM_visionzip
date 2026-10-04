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

| Method | Content % (pooled) | Valid content MB | Total/allocated MB | OS write MB | Persistence mean/p50/p95 ms | Activation mean/p50/p95 ms | Build/activation |
|---|---|---|---|---|---|---|---|
| FullLoad | 100.000 | 20.477 | 24.342/24.348 | 24.342 | 64.232/63.491/72.937 | 14.151/13.754/15.934 | 398 / 398 |
| Ours B | 25.127 | 5.145 | 6.400/6.486 | 6.400 | 45.911/45.563/51.864 | 5.115/5.092/5.646 | 398 / 398 |

Persistence includes durable native publication and experiment completion seal; activation includes integrity reads/metadata/FD setup. Source T1 saliency/capture-clone tail is charged exactly once in its request E2E. Component intervals can overlap and are not summed. ReComp has no KV store/build/activation.

## Three-turn session accounting

| Method | Shared-stream mean/p95 ms | Derived standalone mean/p95 ms |
|---|---|---|
| ReComp | 399.590/452.118 | 399.590/452.118 |
| FullLoad | 324.531/395.277 | 395.357/445.074 |
| Ours B | 285.937/336.155 | 332.026/373.193 |

| B − ReComp | Δms [95% CI] |
|---|---|
| Actual shared-image stream | -113.653 [-115.598, -111.698] |
| Derived standalone-equivalent | -67.564 [-69.269, -65.837] |

Shared-stream attribution charges each actual image/method setup once, at its source dialogue; sums reconcile to all physical events. Standalone-equivalent reallocates that measured setup once to each dialogue. It is DERIVED, not a fresh-store-per-dialogue measurement. Non-source T1 capture increment is NOT_MEASURED and is not filled as zero. Neither uses TTFT sums as session E2E.

## Qwen findings and limits

B vs ReComp: followup accuracy -3.497 [-4.260, -2.743] pp; TTFT mean 52.32% reduction, p95 55.47% reduction. Actual KV read difference +6.371 MB/request.
B vs FullLoad: followup accuracy -3.534 [-4.302, -2.784] pp; TTFT mean 24.75% reduction, p95 28.69% reduction. Actual KV read difference -18.037 MB/request.

Mean and p95 are reported separately; a mean improvement alone does not establish p95 improvement or statistical equivalence. Single GPU, batch1, bounded image-major shared cache; no long-session throughput/eviction/generalization claims. LLaVA is a separate historical run and no cross-model latency pairing is implied.

Generation cap/empty/EOS counts and rates are in request_detail_summary.csv. Attempts, physical events and failed diagnostics remain separate from final main requests. Native store payloads were deleted only after audit+hash+commit within exact owned paths and require regeneration. Prior large-file protection uses labeled sampled fingerprints; new artifacts and raw use full SHA256.
