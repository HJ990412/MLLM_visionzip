# Reorder + Prefix baseline: GQA 40 images / 240 questions

Validation: **PASS**

이 디렉터리는 source run을 수정하지 않고 생성한 분석 사본이다. Primary
comparison과 판정 규칙은 다음과 같이 고정했다.

```text
GO iff observed delta Accuracy(Static+Diverse25 - Prefix25) > 0 and the 95% image-cluster paired-bootstrap CI lower bound > 0; otherwise RETHINK
```

## 결과표

| Method | Accuracy | Δ vs FullLoad | True TTFT mean / p50 / p95 | SSD MB/request | SSD/Full | Chunks/layer | Touched | Selector |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ReComp | 62.50% | +0.42 pp | 503.81 / 506.87 / 598.16 ms | 0.000 | 0.00% | - | - | - |
| FullLoad | 62.08% | +0.00 pp | 656.15 / 659.00 / 778.96 ms | 1165.073 | 100.00% | - | 100.00% | - |
| SparseVLM 25% | 60.00% | -2.08 pp | 737.90 / 746.60 / 882.80 ms | 928.098 | 79.66% | - | 74.27% | - |
| Reorder + Prefix 25% | 60.42% | -1.67 pp | 262.16 / 261.90 / 318.97 ms | 306.079 | 26.27% | 8.525 | 24.28% | 0.194 ms |
| Reorder + Static 25% | 46.67% | -15.42 pp | 296.08 / 295.69 / 348.99 ms | 306.066 | 26.27% | 8.525 | 24.28% | 3.868 ms |
| Reorder + Diverse Only 25% | 58.33% | -3.75 pp | 406.71 / 389.54 / 548.57 ms | 296.561 | 25.45% | 8.525 | 24.28% | 18.980 ms |
| Reorder + Static+Diverse 25% | 60.83% | -1.25 pp | 337.59 / 331.12 / 412.13 ms | 300.401 | 25.78% | 8.525 | 24.28% | 14.866 ms |
| Reorder + Prefix 50% | 62.50% | +0.42 pp | 426.26 / 426.69 / 505.92 ms | 603.875 | 51.83% | 17.400 | 49.60% | 0.334 ms |
| Reorder + Static 50% | 55.42% | -6.67 pp | 467.93 / 469.42 / 552.35 ms | 603.826 | 51.83% | 17.400 | 49.60% | 4.915 ms |
| Reorder + Diverse Only 50% | 62.08% | +0.00 pp | 646.19 / 644.30 / 810.83 ms | 591.489 | 50.77% | 17.400 | 49.60% | 36.004 ms |
| Reorder + Static+Diverse 50% | 62.50% | +0.42 pp | 560.92 / 560.90 / 664.24 ms | 594.209 | 51.00% | 17.400 | 49.60% | 21.865 ms |

## 핵심 질문

### Q1. Importance reorder 뒤 first 25%만 읽은 정확도

**60.42%**

### Q2. Static+Diverse25의 추가 accuracy

Static+Diverse25 - Prefix25 = **+0.42 pp**.
Primary 95% image-cluster bootstrap CI는
**[-2.08, +3.33] pp**다.

McNemar discordance는 SD-only **7**, Prefix-only
**6**, exact two-sided p=1다.

### Q3. 선택 chunk overlap

25%의 (image, layer) 1280쌍에서 평균 Jaccard는
**0.5431**, median은 **0.6000**다.
평균 intersection/SD-only/Prefix-only는 각각
5.972/2.553/
2.553 chunks다. Separator sidecar는 공통이므로
Jaccard에서 제외했다.

### Q4. SSD read와 true TTFT

- Prefix25: 306.079 MB/request,
  262.16 ms, normal/separator preads
  64.00/1.00
- Static+Diverse25: 300.401 MB/request,
  337.59 ms, normal/separator preads
  276.55/1.00
- SD - Prefix: -5.678 MB,
  +75.43 ms

### Q5. 현재 60.8% 성능의 주된 원인

**A를 주된 원인으로 재검토해야 함. 다만 RETHINK는 통계적 동등성 증명이 아니라 Static+Diverse의 양의 추가 이득이 이 실험에서 확립되지 않았다는 operational 판정이다.**

### Q6. Contribution 판정

**RETHINK**

## 통계와 범위

- Primary CI: 이미지 40개를 cluster 단위로 10,000회 paired bootstrap,
  seed=0
- Supplement: 질문 240개 paired bootstrap
- GQA score는 질문별 binary normalized exact match
- McNemar는 질문별 paired binary disagreement의 exact-binomial 결과다.
  이미지당 6개 질문의 상관 때문에 primary uncertainty는 image-cluster CI다.
- Calibration SparseVLM score magnitude는 legacy store에 보존되지 않았기 때문에
  first-4 calibration 질문으로 score를 **분석 전용으로 재계산**했다. 이 score는
  serving/selection에 전달하지 않았고, `static.pt`의 VisionZip saliency도 calibration
  importance로 대체하지 않았다.

## Calibration importance-mass coverage

Separator를 분자와 분모에서 제외한 `(image, layer)` macro mean은 다음과 같다.

| Budget | Prefix | Static | Static+Diverse |
|---:|---:|---:|---:|
| 25% | **73.50%** | 71.11% | 65.62% |
| 50% | **89.36%** | 88.09% | 83.97% |

동일 separator sidecar와의 union을 포함하면 25%에서 각각 75.38%, 73.12%,
67.98%다. 즉 reordered prefix가 Static+Diverse보다 calibration importance mass를
더 많이 보존했다. 전체 정의, rank-consistency gate 및 7,680개 image-layer-method
행은 `importance_coverage.json`/`.csv`에 있다.

## Provenance

- Source run: `/home/dblab/hj/mllm_v2/runs/reorder_prefix_baseline/calib4`
- Source results SHA256: `9c4c11ac1868419c0b4493b802e5f5ded90c430de22220e32cbd4676a88fbdec`
- Store: `/home/dblab/hj/mllm_v2/kvstore`
- Store content SHA256: `e570a6847743a203fc1e2892d736ebe8aa946647cbb0e280f212388da2c09d68`
- Index SHA256: `514d1203d248b6f450f5e3bdacda7b931038f9c11df270b415a2e98e5c77e75a`
- Workload SHA256: `97afe02f924a49cadf0c357175b50185e8f16db12b2dd4402595e2bb99d20f66`
- Separator policy: `sidecar`
- Decision rule: `GO iff observed delta Accuracy(Static+Diverse25 - Prefix25) > 0 and the 95% image-cluster paired-bootstrap CI lower bound > 0; otherwise RETHINK`

상세 검증은 `validation.json`, paired 통계는 `paired_stats.json`, 실제 선택
ID는 `selection_trace.jsonl`, overlap 행은 `chunk_overlap.csv`에 있다.
