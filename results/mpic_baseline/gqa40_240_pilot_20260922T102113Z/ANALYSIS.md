# MPIC-32 SSD adaptation: same-run GQA pilot

| Method | Acc all | Acc hit | TTFT mean | p50 | p95 | SSD MB/req | Preads | Recomputed image tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| ReComp | 0.6250 | 0.6300 | 518.09 ms | 524.04 ms | 584.14 ms | 0.000 | 0.00 | all |
| FullLoad | 0.6250 | 0.6300 | 651.74 ms | 650.79 ms | 776.28 ms | 1165.073 | 64.00 | 0 |
| QA-Chunk25 | 0.5833 | 0.5800 | 422.08 ms | 427.20 ms | 505.81 ms | 348.059 | 305.79 | 0 |
| Ours25 | 0.5792 | 0.5750 | 255.56 ms | 256.91 ms | 301.72 ms | 306.079 | 65.00 | 0 |
| MPIC-32 (SSD adaptation) | 0.6208 | 0.6250 | 646.72 ms | 653.46 ms | 752.99 ms | 1165.335 | 65.00 | min(32,N) |

## Scope and interpretation

All five arms were measured in this run on the frozen 40-image, 240-question GQA slice. Accuracy uses all 240 questions per method; latency and SSD statistics use Q2–Q6 (200 requests per method). The six questions per image are independent requests, not a native conversation. ReComp's table entry is zero **Visual-KV** SSD traffic; it is not a claim that ordinary model/image file I/O is zero.

MPIC and Ours do not share a 25% budget. MPIC retains the complete image context (1.0000) while recomputing a small prefix; Ours retains approximately 0.2500. Ours is a context-pruning quality–I/O trade-off; MPIC is a full-context partial-recomputation alternative.

## MPIC selective-attention evidence

Across 200 MPIC hits, image-token counts ranged from 1464 to 2928. Every layer recomputed canonical local rows 0..31 plus all current text rows (mean 31.62 text rows/request), retained every cached image key, executed one decoder selective-prefill pass, and then appended decode K/V at the exact full-cache length. The source payload hashes were unchanged. The main pilot kept source and target image positions identical, so the shifted-position path is evidenced only by the separate smoke diagnostic.

MPIC read 1165.335 MB/request versus FullLoad 1165.073 and Ours25 306.079. Its mean split was 1165.073 MB cached K/V plus 0.262 MB recomputation-input embeddings. Chunk alignment yielded exactly 65 preads/request (32 K, 32 V, one embedding); a k=32 request can therefore still read the complete visual-KV chunks in this layout.

MPIC mean timing instrumentation (overlapping intervals) was: KV pread 370.92 ms, embedding pread 5.35 ms, H2D 131.07 ms, cache assembly 16.11 ms, position processing 0.00 ms, and inclusive selective-prefill interval 642.01 ms. That interval includes reads, transfers, assembly, all decoder layers, final norm, and LM head; it is not pure GPU compute and the components must not be subtracted from TTFT.

## Paired quality and latency differences

Differences below are candidate minus FullLoad. Quality intervals use 10,000 deterministic percentile-bootstrap replicates clustered by image. An interval containing zero does not prove equivalence.

| Candidate | Acc diff all [95% CI] | Acc diff hit [95% CI] | Paired hit TTFT diff | First-token agreement |
|---|---:|---:|---:|---:|
| ReComp | 0.0000 [-0.0167, 0.0167] | 0.0000 [-0.0200, 0.0200] | -133.65 ms | 0.9833 |
| QA-Chunk25 | -0.0417 [-0.0833, -0.0042] | -0.0500 [-0.1000, -0.0050] | -229.67 ms | 0.8750 |
| Ours25 | -0.0458 [-0.0958, -0.0042] | -0.0550 [-0.1100, 0.0000] | -396.19 ms | 0.8583 |
| MPIC-32 (SSD adaptation) | -0.0042 [-0.0125, 0.0000] | -0.0050 [-0.0150, 0.0000] | -5.02 ms | 0.9958 |

## One-time persistence and storage

Persistence is measured separately from cache-hit TTFT and includes the store's durable publication. QA-Chunk25 shares FullLoad's canonical raster store, so its incremental store is zero. The MPIC provisioning total also includes visual-input D2H materialization performed when its Turn-1 capture exits; persist-call time is retained as a separate column.

| Method | Store policy | Mean bytes/image | Mean persist call | Mean post-response provisioning | Capture D2H | Added visual input |
|---|---|---:|---:|---:|---:|---:|
| ReComp | none | 0.000 MB | 0.00 ms | 0.00 ms | 0.00 ms | 0.000 MB |
| FullLoad | owned canonical raster store | 1260.565 MB | 1025.32 ms | 1025.32 ms | 0.00 ms | 0.000 MB |
| QA-Chunk25 | shares FullLoad canonical raster store | 0.000 MB | 0.00 ms | 0.00 ms | 0.00 ms | 0.000 MB |
| Ours25 | owned importance-repacked store | 1187.793 MB | 1006.97 ms | 1006.97 ms | 0.00 ms | 0.000 MB |
| MPIC-32 (SSD adaptation) | owned canonical KV + visual-input sidecar | 1183.319 MB | 922.63 ms | 925.27 ms | 2.64 ms | 18.204 MB |

## Correctness, positions, and protection

The three-image quantized smoke passed finite-logit/generation checks, k=0 FullLoad comparisons, k=N same-embedding full-prefill logit and cache comparisons under the predeclared tolerance, k=32 layer/counter checks, dummy-sentinel replacement, causal masking, request isolation, and exact decode-cache appends. Its shifted-prefix diagnostic passed mapping, causal mask, source immutability, and post-RoPE phase relocation checks. Position-change support remains LIMITED because relocation does not reconstruct hidden-state changes caused by different preceding text, and the paper does not specify this relocation detail.

The run contains 1,200 unique completed requests, 0 durable technical failure event(s), 0 retry count(s), and 0 duplicates. The prior-artifact guard verified 44459 pre-existing paths with no missing or changed protected entry. Large files use the manifest's bounded nine-window fingerprint policy to avoid warming hundreds of GiB of unrelated SSD payload; this is an accidental-change guard, not an adversarial cryptographic proof.

## Parallelism and limitations

The paper-described overlap between cache-hit image transfer and cache-miss image computation is **not applicable** to this single-image all-hit pilot. No layer-wise prefetch was implemented. Synchronous reads/transfers are reported as measured and are not claimed to overlap. `posix_fadvise(DONTNEED)` conditioning occurs outside TTFT but does not guarantee a cold SSD controller cache.

This fixed-prefix GQA workload requires little position relocation and cannot represent native multi-image MPIC scheduling or the paper's full system. No MT-GQA, MT-VQA-v2, ConvBench, or ReKV run is part of this result.

## Final status

```text
IMPLEMENTATION: PASS
SELECTIVE-ATTENTION CORRECTNESS: PASS
POSITION-CHANGE SUPPORT: LIMITED
GQA PILOT: COMPLETE
ARTIFACT PROTECTION: PASS
MPIC-32 SSD ADAPTATION VALIDATED: YES
```

`YES` validates this paper-guided SSD adaptation and its local measurements; it does not claim an official or complete reproduction of the MPIC serving system.
