# DIAGNOSTIC ONLY — Qwen2.5-VL 수치 불일치 조사

**진단용 결과입니다. GPU correctness validation이 FAIL이므로 유효한 benchmark 성능/품질 주장으로 사용할 수 없습니다.**

- Workload: deterministic-40-image-generated-history-subset (mt_gqa_reconstructed)
- Images: 40; requests: 360
- T1은 전 방법 정상 pixel inference입니다. GQA의 Q2 이후는 서로 독립적인 질문이며, MT의 이전 답변은 각 방법의 생성 결과만 사용합니다.
- TTFT는 첫 output token materialization 및 CUDA 동기화까지입니다. 이미지 파일 읽기와 RGB decode, page-cache conditioning은 timer 밖입니다. ReComp의 processor 전처리와 vision/full visual prefill은 timer 안입니다.
- MT-GQA는 reconstructed subset이며 공식 MetaCompress benchmark 결과가 아닙니다.

## Validation contract

- validation_status: FAIL
- benchmark_validated: false
- run_mode: diagnostic_numerical_mismatch
- FullLoad gate: FAIL
- Prefix25 gate: FAIL
- 두 수치 gate의 원본 per-question 결과와 출력 token 일치 여부는 validation_evidence.json에 그대로 보존했습니다.

## Quality 및 cache-hit TTFT

| Method | All accuracy | Hit accuracy | Hit TTFT mean ms | p50 ms | p95 ms | Hit E2E mean ms |
|---|---:|---:|---:|---:|---:|---:|
| recompute | 0.683 | 0.700 | 115.256 | 114.581 | 132.166 | 136.629 |
| fullload | 0.683 | 0.700 | 75.509 | 73.735 | 86.194 | 97.179 |
| ours25 | 0.658 | 0.662 | 61.651 | 59.877 | 69.048 | 82.265 |

## Cache-hit I/O

| Method | Visual bytes/request | Structural bytes/request | Metadata bytes/request | pread calls/request | read spans/request | kept tokens/request |
|---|---:|---:|---:|---:|---:|---:|
| recompute | 0 | 0 | 0 | 0.000 | 0.000 | NOT MEASURED |
| fullload | 23855104 | 1204224 | 0 | 57.000 | 57.000 | 370.475 |
| ours25 | 7064781 | 1204224 | 0 | 57.000 | 57.000 | 123.200 |

## Persistence 및 session

| Method | Persistence mean ms | Score mean ms | Repack mean ms | Write mean ms | Activation mean ms | 3-turn session E2E mean ms |
|---|---:|---:|---:|---:|---:|---:|
| recompute | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | 0.000 | 409.054 |
| fullload | 71.200 | 0.000 | 0.000 | 10.180 | 13.591 | 412.389 |
| ours25 | 74.065 | 1.847 | 0.846 | 10.139 | 13.960 | 388.391 |

Activation I/O는 첫 store activation에서만 발생하며 cache-hit read bytes와 분리했습니다. 세부 값은 sessions.csv에 있습니다.

## T1 capture timing

T1 TTFT 차이는 image-paired 관측치이며 method rotation에도 잡음이 큽니다. score/clone은 응답 이후 persistence에 귀속됩니다.

- fullload − ReComp T1 TTFT: -2.576 ms (95% CI [-5.486, 0.334]); score_extra 0.000 ms, capture_clone 0.337 ms
- ours25 − ReComp T1 TTFT: 0.105 ms (95% CI [-3.162, 3.057]); score_extra 1.847 ms, capture_clone 0.359 ms

## Paired image-cluster bootstrap

Ours25 − comparator 차이입니다. 95% percentile CI가 0을 포함해도 두 방법의 동등성을 뜻하지 않습니다. 작은 pilot의 추정 불확실성이 큽니다.

- vs recompute: ttft_ms=-53.605 [-56.208, -50.924], accuracy=-0.037 [-0.113, 0.025], visual_read_bytes=7064780.800 [6697779.200, 7340032.000], structural_read_bytes=1204224.000 [1204224.000, 1204224.000], metadata_read_bytes=0.000 [0.000, 0.000]
- vs fullload: ttft_ms=-13.858 [-16.232, -11.759], accuracy=-0.037 [-0.113, 0.025], visual_read_bytes=-16790323.200 [-17707827.200, -16056320.000], structural_read_bytes=0.000 [0.000, 0.000], metadata_read_bytes=0.000 [0.000, 0.000]

성능 수치는 raw JSONL에 기록된 실제 측정값에서 계산했습니다. 빈 지표는 측정되지 않은 값입니다.
