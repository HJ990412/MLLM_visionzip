# DIAGNOSTIC ONLY — Qwen2.5-VL 수치 불일치 조사

**진단용 결과입니다. GPU correctness validation이 FAIL이므로 유효한 benchmark 성능/품질 주장으로 사용할 수 없습니다.**

- Workload: frozen-gqa40-q5to10 (gqa)
- Images: 40; requests: 720
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
| recompute | 0.567 | 0.590 | 110.112 | 109.882 | 127.407 | 130.018 |
| fullload | 0.567 | 0.590 | 71.070 | 69.392 | 84.816 | 91.176 |
| ours25 | 0.575 | 0.600 | 57.513 | 55.259 | 69.805 | 77.566 |

## Cache-hit I/O

| Method | Visual bytes/request | Structural bytes/request | Metadata bytes/request | pread calls/request | read spans/request | kept tokens/request |
|---|---:|---:|---:|---:|---:|---:|
| recompute | 0 | 0 | 0 | 0.000 | 0.000 | NOT MEASURED |
| fullload | 22478848 | 1204224 | 0 | 57.000 | 57.000 | 349.000 |
| ours25 | 6422528 | 1204224 | 0 | 57.000 | 57.000 | 112.000 |

## Persistence 및 session

| Method | Persistence mean ms | Score mean ms | Repack mean ms | Write mean ms | Activation mean ms | 6 independent requests/image total mean ms |
|---|---:|---:|---:|---:|---:|---:|
| recompute | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | 0.000 | 782.813 |
| fullload | 71.970 | 0.000 | 0.000 | 9.794 | 12.888 | 669.479 |
| ours25 | 70.879 | 1.659 | 0.827 | 9.599 | 13.309 | 602.803 |

Activation I/O는 첫 store activation에서만 발생하며 cache-hit read bytes와 분리했습니다. 세부 값은 sessions.csv에 있습니다.

## T1 capture timing

T1 TTFT 차이는 image-paired 관측치이며 method rotation에도 잡음이 큽니다. score/clone은 응답 이후 persistence에 귀속됩니다.

- fullload − ReComp T1 TTFT: -4.061 ms (95% CI [-7.096, -1.066]); score_extra 0.000 ms, capture_clone 0.341 ms
- ours25 − ReComp T1 TTFT: -1.969 ms (95% CI [-5.277, 1.376]); score_extra 1.659 ms, capture_clone 0.367 ms

## Paired image-cluster bootstrap

Ours25 − comparator 차이입니다. 95% percentile CI가 0을 포함해도 두 방법의 동등성을 뜻하지 않습니다. 작은 pilot의 추정 불확실성이 큽니다.

- vs recompute: ttft_ms=-52.599 [-55.078, -50.016], accuracy=0.010 [-0.035, 0.055], visual_read_bytes=6422528.000 [5872025.600, 6881280.000], structural_read_bytes=1204224.000 [1204224.000, 1204224.000], metadata_read_bytes=0.000 [0.000, 0.000]
- vs fullload: ttft_ms=-13.557 [-15.418, -11.579], accuracy=0.010 [-0.035, 0.055], visual_read_bytes=-16056320.000 [-16790323.200, -15414067.200], structural_read_bytes=0.000 [0.000, 0.000], metadata_read_bytes=0.000 [0.000, 0.000]

성능 수치는 raw JSONL에 기록된 실제 측정값에서 계산했습니다. 빈 지표는 측정되지 않은 값입니다.
