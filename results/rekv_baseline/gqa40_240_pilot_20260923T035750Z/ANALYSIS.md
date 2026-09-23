# ReKV-Chunk25 (adapted): same-run GQA pilot

| Method | Acc all | Acc hit | TTFT mean | p50 | p95 | SSD MB/hit | Preads/hit |
|---|---:|---:|---:|---:|---:|---:|---:|
| ReComp | 150/240 (62.50%) | 126/200 (63.00%) | 517.70 ms | 523.57 ms | 581.71 ms | 0.000 | 0.00 |
| FullLoad | 150/240 (62.50%) | 126/200 (63.00%) | 660.77 ms | 654.92 ms | 786.35 ms | 1165.073 | 64.00 |
| MPIC-32 (SSD adaptation) | 149/240 (62.08%) | 125/200 (62.50%) | 638.17 ms | 636.43 ms | 792.47 ms | 1165.335 | 65.00 |
| QA-Chunk25 | 140/240 (58.33%) | 116/200 (58.00%) | 431.77 ms | 436.94 ms | 518.24 ms | 348.059 | 305.79 |
| ReKV-Chunk25 (adapted) | 139/240 (57.92%) | 115/200 (57.50%) | 568.51 ms | 581.53 ms | 698.61 ms | 305.236 | 168.12 |
| Ours25 | 139/240 (57.92%) | 115/200 (57.50%) | 257.32 ms | 256.33 ms | 303.46 ms | 306.079 | 65.00 |

## ReKV, QA-Chunk25, and Ours25

| Metric | ReKV-Chunk25 | QA-Chunk25 | Ours25 |
|---|---:|---:|---:|
| Acc hit | 115/200 | 116/200 | 115/200 |
| TTFT mean (ms) | 568.51 | 431.77 | 257.32 |
| Online decision cost (ms) | 10.91 | 94.51 | 0.21 |
| Retrieval-forward wall (ms) | 519.63 | N/A | N/A |
| Answer-prefill wall (ms) | 48.47 | N/A | N/A |
| SSD bytes/hit | 305236050 | 348058911 | 306079334 |
| Preads/hit | 168.12 | 305.79 | 65.00 |
| Selected-layout runs/layer | 2.61 | 4.26 | 1.00 |
| Metadata bytes/image | 9208089 | N/A | N/A |
| Actual attention key length | 595.82 | N/A | N/A |
| Question-pair chunk Jaccard | 0.865 | 0.718 | 1.000 |

The ReKV online decision cost above sums the measured Q representative, similarity, Top-k, selected-ID transfer, and I/O planning intervals. QA's entry is its separately instrumented selector decision host interval; the two implementations instrument different work. Retrieval-forward wall encloses ReKV's selected KV I/O, attention, and MLP, so its subcomponents are not added to TTFT. Selected-layout runs describe the chosen chunk IDs; they are not necessarily the same as actual pread calls.

## Paired quality and latency

Differences are ReKV minus the named reference. Accuracy confidence intervals are 10,000 image-cluster percentile bootstrap replicates, retaining each sampled image's six matched questions. A confidence interval containing zero does not establish equivalence.

| Reference | Acc all difference [95% CI] | Acc hit difference [95% CI] | ReKV TTFT minus reference | Relative reduction vs reference | Reference/ReKV speedup |
|---|---:|---:|---:|---:|---:|
| Ours25 | 0.0000 [-0.0458, 0.0458] | 0.0000 [-0.0550, 0.0550] | 311.19 ms | -120.93% | 0.453× |
| QA-Chunk25 | -0.0042 [-0.0625, 0.0542] | -0.0050 [-0.0750, 0.0650] | 136.73 ms | -31.67% | 0.759× |
| FullLoad | -0.0458 [-0.1042, 0.0083] | -0.0550 [-0.1250, 0.0100] | -92.27 ms | 13.96% | 1.162× |

The paired contingency counts are in `paired_quality.json` for all 240 and cache-hit 200 matched questions per comparison.

## Selection, I/O, and handoff

ReKV selected chunk IDs varied across questions with mean pair Jaccard 0.865 and consecutive-query Jaccard 0.871. Its per-layer selection overlap with QA-Chunk25 was 0.178. Ours25's same-image prefix selection was invariant across Q2–Q6. Identical ReKV selections are valid; selection difference alone cannot explain correctness.

Across ReKV cache hits, Stage A read 305.236 MB/request and Stage B read 0 visual payload bytes/request. Mean duplicate-read bytes were 0. The full selected K/V, separators, compact position mapping, and per-layer key lengths are recorded in raw rows; metadata remains ready on GPU before each timed request.

ReKV `ssd_read_ms` is IOCounter pread time; `ssd_read_pipeline_ms` covers broader host reading. `h2d_ms` measures host staging and asynchronous transfer submission, not isolated DMA execution. No cross-stream SSD/compute overlap is claimed. `posix_fadvise(DONTNEED)` is performed before each SSD hit and its per-file statuses are stored in raw rows; it does not guarantee a cold SSD controller cache.

## Provisioning and memory

The ReKV raw-K store used a mean 1196.985 MB/image and mean one-time persistence 1679.21 ms. Mean capture materialization, representative build, store write, and fsync are in `persistence.csv`; storage is run-local and persisted after the normal Turn-1 answer.

Active-image ReKV retrieval metadata occupied 9.208 MB GPU/image, with mean activation 2.15 ms outside TTFT. Initial/system K/V occupied an additional 2.621 MB GPU/image. The 100-image metadata figure in `memory_analysis.json` is arithmetic extrapolation, not a measured 100-image footprint. Request-local peak pinned host staging was 303.699 MB/hit on average. Linux process-wide RSS highwater sampled on those hits reached 6.267 GB maximum; it is a process lifetime highwater, not request-attributable.

## Workload and integrity

The six methods were measured in the same new run over the frozen 40-image/240-question GQA slice. The six questions per image are independent requests without answer history. Every method used normal Image+Q1 pixel inference; each Turn-1 prompt, pixel/input hash, prediction, and first output token matched across methods. Accuracy uses 240 questions per method. TTFT and I/O use Q2–Q6 (200 requests per method). Logical requests and actual root-model forward calls are reported separately in `validation.json` and raw rows; generated token IDs/counts are recorded for every request.
- ReComp: 240 logical requests; 569 observed model forwards.
- FullLoad: 240 logical requests; 571 observed model forwards.
- MPIC-32 (SSD adaptation): 240 logical requests; 370 observed model forwards.
- QA-Chunk25: 240 logical requests; 566 observed model forwards.
- ReKV-Chunk25 (adapted): 240 logical requests; 824 observed model forwards.
- Ours25: 240 logical requests; 571 observed model forwards.

MPIC's selective first-token prefill runs through its manual decoder path outside the top-level model-forward hook. If its first token is EOS, zero counted top-level forwards on that hit is valid; the raw row still records its generated token.

The run completed 1,440 unique requests with 0 durable technical failure events, 0 retries, and 0 duplicates. The before/after guard checked 67846 pre-existing runs/results/source-store entries with no missing or changed paths. Large payload files use the recorded bounded fingerprint policy.

## Implementation scope and limitations

The official source commit is `1fd9a3dbf5dbff7f27069ae2f4463674c495e830`. The paper describes cosine similarity, whereas the pinned official vector-cache path computes an unnormalized FP32 dot product. This main run uses `official_code_dot` throughout. Its video frames, video backbone, and GPU/CPU payload caches are adapted here to canonical 64-token single-image SSD chunks, the frozen LLaVA-NeXT Vicuna 7B 4-bit NF4/BF16 model, and metadata-ready/payload-cold SSD hits. The results do not reproduce original StreamingVQA table numbers. No MT-GQA, MT-VQA-v2, ConvBench, or VisDial full run is included.

## Final status

```text
IMPLEMENTATION: PASS
SOURCE RETRIEVAL PARITY: PASS
PRE-ROPE / POSITION CORRECTNESS: PASS
COMPACT KV ASSEMBLY: PASS
RETRIEVAL-TO-ANSWER HANDOFF: PASS
DUPLICATE PAYLOAD READ CHECK: PASS
ACTUAL-MODEL SMOKE: PASS
GQA PILOT: COMPLETE
ARTIFACT PROTECTION: PASS
ReKV-CHUNK25 SSD ADAPTATION VALIDATED: YES
```

`YES` validates this documented SSD image adaptation and its measured serving behavior; it does not claim a full reproduction of the original ReKV streaming system.
