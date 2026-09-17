# ImageOnly-Repack cross-dataset generalization

This analysis merges the immutable image-at-a-time serving artifacts. No inference is run here, and no prior result is modified.

## Frozen workloads

| Dataset | Images | Questions | q/image | Index SHA256 | Workload SHA256 |
|---|---:|---:|---:|---|---|
| GQA-large | 395 | 1185 | 3 | `0d50962f0c1bac3bc6e1836978289d5fde7d60b434f55cebbc4607c4c80a797c` | `cabec1bb1035c836839b98d72ba6a04558529d246cf55f8499ca75ea2d2f290c` |
| VQAv2 | 250 | 1000 | 4 | `b83d5fa288fcb722ca073e261d3fec9086629ed0db2e568a2d5d6a24ef1589d7` | `e341b499a968c5caba4ddffc58fdf0ccafe2e0e212b6cb280ca1b11fc9f31d18` |
| TextVQA | 500 | 500 | 1 | `b1e5ff0eaba2a45c6398e7cb90631cdc7387eff25ed0997a66374f25968a2f4d` | `49fa0b15f406132162cba1b47245f28a560d28b4f3af76985c855d77a145a2a6` |

All layouts use zero calibration questions.  The original per-image question offset remains fixed only so the evaluation IDs exactly match the previous Static+Diverse generalization workload.

## Cross-dataset result

| Dataset | FullLoad score | P25 score | Δ P25 | P45 score | Δ P45 | P25 TTFT↓ vs ReComp | P45 TTFT↓ vs ReComp | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| GQA-large | 63.46% | 58.73% | -4.73 pp | 62.87% | -0.59 pp | 44.46% | 16.33% | SUPPORTED |
| VQAv2 | 81.57% | 76.33% | -5.23 pp | 80.50% | -1.07 pp | 44.50% | 15.84% | PARTIALLY SUPPORTED |
| TextVQA | 61.93% | 57.87% | -4.07 pp | 62.20% | 0.27 pp | 46.27% | 18.77% | SUPPORTED |

## Direct answers to Q1–Q10

### Q1. GQA-large

Prefix25: quality Δ -4.73 pp, TTFT reduction 44.46%, SSD reduction 73.79%. Prefix45: quality Δ -0.59 pp, TTFT reduction 16.33%, SSD reduction 52.68%.

### Q2. VQAv2

Prefix25: quality Δ -5.23 pp, TTFT reduction 44.50%, SSD reduction 73.73%. Prefix45: quality Δ -1.07 pp, TTFT reduction 15.84%, SSD reduction 52.77%.

### Q3. TextVQA

Prefix25: quality Δ -4.07 pp, TTFT reduction 46.27%, SSD reduction 73.58%. Prefix45: quality Δ 0.27 pp, TTFT reduction 18.77%, SSD reduction 52.60%.

### Q4. Prefix25 efficiency trade-off

세 dataset 모두에서 유지됩니다.

### Q5. Prefix45 quality-oriented point

세 dataset 모두에서 통계적으로 quality-oriented point라고 보기는 어렵습니다. dataset별 CI와 verdict를 확인해야 합니다.

### Q6. 가장 큰 quality degradation

VQAv2의 Prefix25: -5.23 pp vs FullLoad.

### Q7. 가장 민감한 category

동일 dataset/method에서 `other` heuristic category가 -9.52 pp로 가장 낮았습니다. 이는 official category metadata가 아닌 question-text regex 분석입니다.

### Q8. FullLoad가 ReComp보다 빠른 dataset

없습니다.

### Q9. SSD caching만으로 충분한가?

아닙니다. FullLoad가 어느 dataset에서도 ReComp보다 빠르지 않았으므로, 이 측정에서 latency 이득에는 partial loading이 필요했습니다.

### Q10. Cross-dataset generalization verdict

**PARTIALLY SUPPORTED**. Dataset별 판정을 하나의 평균으로 덮지 않았습니다.

## Statistical and measurement policy

Quality uses 10,000 paired image-cluster bootstrap resamples with seed 1234. Main latency is server-side `end_to_end_ttft_ms`; page-cache conditioning is outside that timer. The previous Static+Diverse artifacts lack the same final E2E boundary, so their latency is not directly compared. Their frozen IDs, quality, and I/O remain suitable supplementary references.

VQAv2/TextVQA scores retain the repository consensus implementation for exact old/new comparability; this implementation is not byte-for-byte the official leave-one-annotator-out evaluator. Category labels are heuristic because the frozen indexes contain no official type metadata.

## Paper-ready conclusion

Across GQA, VQAv2, and TextVQA, image-only importance-aware Visual KV repacking consistently enabled selective sequential SSD loading without query-dependent cache scoring. Prefix25 provided the most aggressive efficiency point, while Prefix45 did not satisfy the quality-oriented criterion on every dataset. No dataset made FullLoad faster than ReComp, so partial loading was necessary for the measured latency benefit. The cross-dataset verdict is reported from the measured per-dataset evidence as PARTIALLY SUPPORTED.

Automated validation: 6/6 checks passed.
