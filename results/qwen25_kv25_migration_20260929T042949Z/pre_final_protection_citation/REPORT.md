# Qwen2.5-VL Chunk25 → Visual-KV25 전환 파일럿

동일 Qwen/Qwen2.5-VL-7B-Instruct, NF4/BF16/SDPA, 64-row chunk, r=0.25, seed=1234, 최대 16 token, frozen GQA/MT manifest에서 4-arm을 새로 실행했다. Old/New Ours hit는 동일한 보호된 v2 repacked BF16 store를 read-only로 사용했다. 각 arm의 T1은 새 full-image inference이며 진단 capture는 저장하지 않았다. Persistence와 cold-start session은 이번에 재측정하지 않았다.

GPU correctness gate: `/home/dblab/hj/mllm_v2/runs/qwen25_kv25_migration_20260929T042949Z/gpu_validation/validation.json` (`00f3135738bc3fd0831d05b4c576ca04bebc5c50a174bee2da3eab98d86b169d`), PASS. 이 구현 검증은 정확도 유지나 지연 감소의 보장을 뜻하지 않는다.

## GQA 4-arm

40 images, 960 unique physical requests, 0 실패/중복/retry. Read MB와 TTFT는 hit request당 평균이다.

| 방법 | 전체 정답률 | Hit 정답률 | T1 TTFT (ms) | Hit TTFT (ms) | 실제 content retention (macro / weighted) | Normal / total read (MB/hit) |
|---|---:|---:|---:|---:|---:|---:|
| ReComp | 56.67% | 59.00% | 115.21 | 114.81 | — / — | — / — |
| FullLoad | 56.67% | 59.00% | 113.69 | 70.72 | 100.00% / 100.00% | 22.479 / 23.683 |
| Ours Chunk25 legacy | 57.50% | 60.00% | 113.28 | 55.92 | 31.73% / 32.09% | 6.423 / 7.627 |
| Ours Visual-KV25 | 55.00% | 57.00% | 114.70 | 55.65 | 25.12% / 25.11% | 7.340 / 8.544 |

| 비교 | Hit 정답률 차이 (%p) | Hit TTFT 차이 (ms)·감소율 | SSD bytes 변화 (MB/hit) | Paired 95% CI |
|---|---:|---:|---:|---|
| KV25 − Chunk25 | -3.00 | -0.27 / +0.49% 감소 | +0.918 | quality [-6.50, +0.00] %p; TTFT [-1.28, +0.82] ms; SSD [+0.459, +1.468] MB |

선택 original-ID 집합: 동일 0 / 다름 40 images. KV25 normal chunk 수: 증가 10, 동일 30, 감소 0. 경계 chunk에서 읽고 attention에서 제외한 유효 rows 1614개 (92.553 MB/image-sum); padding rows 0개 (0.000 MB/image-sum).

Structural 보존과 실제 메모리/전송량: KV25 structural 포함 retention macro 29.46%; structural 1.204 MB/hit, H2D 6.230 MB/hit, compact KV 6.230 MB/hit, GPU peak allocated 6095.233 MB/hit. 물리적으로 읽은 유효 content KV 7.340 MB/hit와 실제 attention에 남긴 content KV 5.026 MB/hit를 구분했다. 실제 normal read는 FullLoad의 32.65%, total read는 36.08%이다.

Turn별 정답률: ReComp T1 45.00%, T2 52.50%, T3 65.00%, T4 65.00%, T5 42.50%, T6 70.00%; FullLoad T1 45.00%, T2 52.50%, T3 65.00%, T4 65.00%, T5 42.50%, T6 70.00%; Ours Chunk25 legacy T1 45.00%, T2 50.00%, T3 65.00%, T4 67.50%, T5 45.00%, T6 72.50%; Ours Visual-KV25 T1 45.00%, T2 45.00%, T3 60.00%, T4 65.00%, T5 42.50%, T6 72.50%.

95% CI에 0이 있더라도 동등성 근거로 해석하지 않는다.

## MT 4-arm

40 images, 480 unique physical requests, 0 실패/중복/retry. Read MB와 TTFT는 hit request당 평균이다.

| 방법 | 전체 정답률 | Hit 정답률 | T1 TTFT (ms) | Hit TTFT (ms) | 실제 content retention (macro / weighted) | Normal / total read (MB/hit) |
|---|---:|---:|---:|---:|---:|---:|
| ReComp | 68.33% | 70.00% | 114.07 | 114.60 | — / — | — / — |
| FullLoad | 68.33% | 70.00% | 113.76 | 75.72 | 100.00% / 100.00% | 23.855 / 25.059 |
| Ours Chunk25 legacy | 65.83% | 66.25% | 114.74 | 59.73 | 33.40% / 33.25% | 7.065 / 8.269 |
| Ours Visual-KV25 | 66.67% | 67.50% | 115.55 | 59.02 | 25.14% / 25.14% | 7.524 / 8.728 |

| 비교 | Hit 정답률 차이 (%p) | Hit TTFT 차이 (ms)·감소율 | SSD bytes 변화 (MB/hit) | Paired 95% CI |
|---|---:|---:|---:|---|
| KV25 − Chunk25 | +1.25 | -0.71 / +1.19% 감소 | +0.459 | quality [-2.50, +5.00] %p; TTFT [-1.70, +0.32] ms; SSD [+0.092, +0.826] MB |

선택 original-ID 집합: 동일 0 / 다름 40 images. KV25 normal chunk 수: 증가 5, 동일 35, 감소 0. 경계 chunk에서 읽고 attention에서 제외한 유효 rows 1523개 (87.335 MB/image-sum); padding rows 0개 (0.000 MB/image-sum).

Structural 보존과 실제 메모리/전송량: KV25 structural 포함 retention macro 29.22%; structural 1.204 MB/hit, H2D 6.544 MB/hit, compact KV 6.544 MB/hit, GPU peak allocated 6101.743 MB/hit. 물리적으로 읽은 유효 content KV 7.524 MB/hit와 실제 attention에 남긴 content KV 5.340 MB/hit를 구분했다. 실제 normal read는 FullLoad의 31.54%, total read는 34.83%이다.

Turn별 정답률: ReComp T1 65.00%, T2 65.00%, T3 75.00%; FullLoad T1 65.00%, T2 65.00%, T3 75.00%; Ours Chunk25 legacy T1 65.00%, T2 62.50%, T3 70.00%; Ours Visual-KV25 T1 65.00%, T2 62.50%, T3 72.50%.

MT 후속 turn의 차이는 method별로 생성한 history의 차이도 포함한다. 95% CI에 0이 있더라도 동등성 근거로 해석하지 않는다.

## 실행 조건과 감사

Smoke: 4 images × 3 questions × 4 arms = 48 requests, report-side 감사 PASS. 첫 smoke 시작은 기본 Python 환경에 transformers가 없어 모델 로드 전에 종료했고 physical request는 0개였다. 별도 retry1 디렉터리에서 검증된 mllm_ft 환경으로 전체 48요청을 수행했다. 첫 실패는 startup_failure.json에 보존했다. GQA/MT 원본 frozen manifest와 protected v2 store 메타데이터 해시를 각 image별로 대조했다.

Read-only 활성화 때 보호된 payload 전체 SHA를 검증하고 metadata/FD를 상주한다. PIL 이미지 파일 hash 검사는 image당 한 번 요청 timer 밖에 수행한다. RGB decode는 매 normal-pixel request 안에서 수행하여 TTFT/E2E에 더한다. Qwen processor와 vision도 pixel TTFT 안에 있다. Cache hit에는 decode가 없고 raw decode field는 null이다 (timing erratum 참조). 활성화 시간/IO 및 posix_fadvise page-cache 힌트는 hit 요청 timer 밖에 기록한다. Hit의 실제 pread bytes/calls는 timer 안에 포함된다. Controller/NAND cache를 비웠다고 주장하지 않는다. T1의 score hook은 정상 full-image forward 안에서 실행되며 capture clone은 request E2E 후 진단으로 수행된다.

통계는 image cluster 10,000 bootstrap resamples, seed 1234이다. 품질과 지연의 방향은 관측 결과와 paired CI에 한정하며, 새로운 main rerun의 보장은 아니다.

원시 요청, 고정 manifest, config, store inventory, GPU inventory, summary.csv와 report_audit.json과 별도 independent_audit.json의 해시 및 재실행 명령은 REPRODUCE.md에 있다.

## 최종 판정

- IMPLEMENTATION: PASS
- CPU TEST: 42/42 PASS 및 pilot preflight PASS
- GPU CORRECTNESS UNDER v2: PASS
- LEGACY QWEN CHUNK25 REGRESSION: GPU gate G11 PASS; pilot legacy arm VALID UNDER v2
- LLAVA CHUNK25/KV25 PROTECTION: CPU/protected-file PASS; LLaVA GPU NOT RUN in this migration
- GQA 4-ARM PILOT: VALID UNDER v2
- MT 4-ARM PILOT: VALID UNDER v2
- PERSISTENCE/SESSION: NOT_REMEASURED
- READY FOR QWEN KV25 MAIN RERUN: NO pending main-run storage/capacity plan; correctness passed and this small pilot kept its 30 GiB reserve, but main-run capacity is unverified.
