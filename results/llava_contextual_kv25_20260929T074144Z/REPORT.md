# LLaVA-NeXT GQA Dominant + Contextual Representative Visual-KV25

VisionZip-inspired original-token selection variant. Contextual token merging from the paper was not reproduced: all selected rows are unchanged original decoder K/V rows.

| Method | D/C tokens (image mean) | All accuracy | Hit accuracy | Δ hit vs D25 (pp) | Hit TTFT (ms) | SSD MB/hit | Persistence ms/image |
|---|---:|---:|---:|---:|---:|---:|---:|
| ReComp | N/A | 0.625 | 0.630 | +6.50 | 524.18 | 0.000 | N/A |
| FullLoad | N/A | 0.625 | 0.630 | +6.50 | 709.61 | 1165.073 | 1122.99 |
| D25+C0 | 546.0/0.0 | 0.571 | 0.565 | +0.00 | 261.31 | 319.501 | 1114.59 |
| D22.5+C2.5 | 492.0/54.0 | 0.575 | 0.570 | +0.50 | 261.62 | 319.501 | 1111.74 |
| D20+C5 | 437.1/108.8 | 0.579 | 0.575 | +1.00 | 261.19 | 319.501 | 1129.58 |
| D17.5+C7.5 | 382.7/163.3 | 0.550 | 0.540 | -2.50 | 260.00 | 319.501 | 1110.53 |
| D15+C10 | 327.9/218.1 | 0.575 | 0.570 | +0.50 | 262.43 | 319.501 | 1125.55 |
| D20+Random5 | 437.1/108.8 | 0.562 | 0.555 | -1.00 | 263.82 | 319.501 | 1120.24 |
| D20+Uniform5 | 437.1/108.8 | 0.579 | 0.575 | +1.00 | 263.54 | 319.501 | 1126.64 |

## Preregistered comparison

D20+C5 minus D25 hit accuracy: +1.00 percentage points (95% image-paired bootstrap CI [-3.50, +5.50]; 10,000 resamples, seed 1234). The interval includes zero, so an increase or equivalence is not established.

Best observed sweep arm: D20+C5, hit accuracy 0.575; difference from D25: +1.00 pp, 95% paired CI [-3.50, +5.50] pp. This arm was chosen after seeing the sweep and is exploratory.

D20+C5 versus Random: +2.00 pp [-1.00, +5.00]. Versus Uniform: +0.00 pp [-3.00, +3.00]. Representative-specific benefit remains inconclusive.

## Budget, SSD reads, and timing

Every selective arm attended exactly ceil(0.25 × N) original visual-content tokens per layer; structural separators followed the frozen sidecar policy. All selective arms had identical measured returned SSD bytes and pread counts on each paired hit.

| Method | T1 accuracy | T2 | T3 | T4 | T5 | T6 | Hit TTFT p50/p95 (ms) | Hit E2E mean (ms) | Content / structural retention | Normal / total preads |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ReComp | 0.600 | 0.625 | 0.650 | 0.700 | 0.475 | 0.700 | 529.27 / 591.10 | 550.43 | 1.0000 / 1.0000 | 0.0 / 0.0 |
| FullLoad | 0.600 | 0.575 | 0.650 | 0.725 | 0.475 | 0.725 | 689.02 / 998.08 | 740.77 | 1.0000 / 1.0000 | 64.0 / 64.0 |
| D25+C0 | 0.600 | 0.625 | 0.600 | 0.475 | 0.425 | 0.700 | 262.02 / 319.10 | 288.77 | 0.2500 / 0.2629 | 64.0 / 65.0 |
| D22.5+C2.5 | 0.600 | 0.575 | 0.625 | 0.525 | 0.450 | 0.675 | 258.46 / 330.30 | 289.96 | 0.2500 / 0.2629 | 64.0 / 65.0 |
| D20+C5 | 0.600 | 0.575 | 0.550 | 0.575 | 0.475 | 0.700 | 259.02 / 343.15 | 289.24 | 0.2500 / 0.2629 | 64.0 / 65.0 |
| D17.5+C7.5 | 0.600 | 0.525 | 0.550 | 0.525 | 0.450 | 0.650 | 257.41 / 324.87 | 289.10 | 0.2500 / 0.2629 | 64.0 / 65.0 |
| D15+C10 | 0.600 | 0.625 | 0.550 | 0.500 | 0.450 | 0.725 | 258.95 / 324.27 | 294.29 | 0.2500 / 0.2629 | 64.0 / 65.0 |
| D20+Random5 | 0.600 | 0.600 | 0.550 | 0.525 | 0.450 | 0.650 | 259.55 / 344.02 | 292.46 | 0.2500 / 0.2629 | 64.0 / 65.0 |
| D20+Uniform5 | 0.600 | 0.575 | 0.600 | 0.575 | 0.475 | 0.650 | 258.31 / 328.44 | 292.08 | 0.2500 / 0.2629 | 64.0 / 65.0 |

D20+C5 Turn-1 TTFT was -0.17 ms versus D25; one-time persistence was +14.99 ms/image. These measured costs include the opt-in key hook and synchronous descriptor mapping, clustering, repacking, writing, and fsync. Persistence is separate from cache-hit TTFT.

Paired D20+C5 minus D25 hit TTFT: -0.12 ms [-4.37, +4.06]; ratio 1.000 [0.983, 1.015].

## Selection diagnostics

D20+C5 changed a mean of 103.6 selected IDs/image versus D25 (mean Jaccard 0.681). Mean cluster size was 16.05; 48.1% of uniform targets changed to another original representative. D25 wrong→D20+C5 right: 13 hits; D25 right→D20+C5 wrong: 11 hits. Examples are saved in prediction_flips.json. Descriptor-space coverage and VQA accuracy are separate outcomes.

## Seven requested conclusions

1. Primary D20+C5 versus D25: INCONCLUSIVE; +1.00 pp, paired 95% CI [-3.50, +5.50].
2. Best observed ratio: D20+C5; +1.00 pp, 95% paired CI [-3.50, +5.50] pp. This is exploratory.
3. Representative-specific benefit versus Random/Uniform: INCONCLUSIVE; paired intervals are above.
4. All selective arms kept exactly ceil(0.25 × N) original content rows with the frozen structural policy.
5. D20+C5 read 319.501 MB and 65.0 preads per hit; paired TTFT changed -0.12 ms (ratio 1.000), with intervals above.
6. D20+C5 Turn-1 TTFT changed -0.17 ms and persistence changed +14.99 ms/image versus D25. Mean captured key reduction 0.02 ms and key materialization 1.15 ms; component costs are in persistence.csv and summary.csv.
7. Additional validation: do not promote this uncertain pilot; revisit only with a larger independent workload and a clear expected effect.

Additional D20+C5 distributions: extra-token saliency median 0.006317, p95 0.017578; cluster-size median 14.0, p95 36.0.

## Scope and interpretation

Actual runtime packages: torch 2.5.1+cu121, transformers 4.57.6, accelerate 1.14.0, bitsandbytes 0.49.2, Pillow 12.2.0, numpy 2.2.6, psutil 7.2.2. Local checkpoint revision: c916e6cdcd760b4cecd1dd4907f84ac649f93b23.

The 40 images are the previously used GQA pilot, not an unseen holdout. ReComp uses a different numerical path from FP16 SSD KV serving. OS posix_fadvise(DONTNEED) conditions page cache outside request TTFT and does not prove a cold SSD controller or NAND. Saved read bytes are returned by os.pread. After receipt-protected cleanup, stores require rebuilding for independent SSD replay.

Independent audit: PASS, 2160/2160 requests.

Further validation: The pilot does not justify replacing the existing main method. A larger independent workload is warranted only if the unresolved effect matters operationally.

IMPLEMENTATION: PASS
CPU VALIDATION: PASS
GPU CORRECTNESS: PASS
PILOT: COMPLETE
QUALITY SIGNAL: INCONCLUSIVE
REPRESENTATIVE-SPECIFIC BENEFIT: INCONCLUSIVE
QWEN / FULL MT-GQA: NOT RUN

## Execution history and additional audit

The first two frozen attempts stopped at smoke preflight before a request; both passed their CPU/GPU correctness gates and retain their failure receipts. The third frozen attempt passed smoke, completed four pilot images and 22 requests of the fifth, then stopped with a CUDA OOM. Its partial raw rows, failed-image stores, and failure receipt remain under that separate run. This completed run released finished-request GPU buffers between timed requests; it did not alter PrefixCache allocation, attention, masks, or selection. The fixed fifth image completed all 54 requests with a measured CUDA allocation peak of 7.99 GiB. Across all 40 images, peak allocated CUDA memory was 7.99 GiB, peak process RSS was 4.50 GiB, and minimum disk free while stores existed was 110.45 GiB.

A separate read-only audit script recomputed all 2,160 GQA scores against the frozen index and checked 33,416 request, budget, I/O, receipt, and selection-artifact conditions. It found zero missing, extra, duplicate, or failing requests. Its code and script-hash receipt are saved as strict_independent_audit.py and strict_independent_audit.json. Frozen source and protected-file hashes still match the pre-GPU source freeze.

The primary and two control image-paired bootstrap statistics were also independently recomputed from raw rows with the frozen 10,000 resamples and seed; all nine recorded effect and interval fields matched (statistical_recheck.json).
