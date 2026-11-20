# 실험 B — LLaVA-1.5-7B × MT-VQA-v2-reconstructed-25k: ReComp / FullLoad / IMPRESS / Ours (IMPRESS = 공개 구현 방식)

측정값만 적는다. 측정하지 않은 값은 "미실행"이다. 계약: `docs/llava15_mtvqa_impress_code_contract.md` (사본 `CONTRACT.md`).

## 설정

- 실행: `runs/llava15_mtvqa_impress_code_20261118T053221Z` (요청 단위 원본), 이 디렉터리(`results/llava15_mtvqa_impress_code_20261118T053221Z`)는 집계와 보고서.
- 환경: Python 3.10.20, torch 2.5.1+cu121, **transformers 4.57.6** (`/home/dblab/anaconda3/envs/mllm_ft/bin/python`).
- 모델 `llava-hf/llava-1.5-7b-hf` revision `b234b804b114…`, NF4/BF16/eager, FP16 SSD KV, greedy max 16, batch 1, CPU threads 2. visual token 576개 고정, separator 없음.
- 모집단·절차: MT-VQA-v2-reconstructed-25k 25,000 images × 3턴(이미지당 dialogue 1개), method-local generated history, ten-answer leave-one-out 공식 VQA score, dialogue 단위 arm 순서 회전. 턴 1은 모든 방법이 픽셀부터 계산, 캐시 예열 없음, 저장 KV 읽기는 요청 직전에 page cache를 비운 cold 조건.
- IMPRESS(공개 구현 방식, `mmimpress/impress_official.py` — LLaVA-NeXT 실험과 같은 코드; 이 실험의 유일한 IMPRESS): probe head 0·1·2, layer마다 25% 유지(576개 중 144개), 유사도 임계값 0.3(이상이면 합의), chunk 64 토큰. 중요도 = 턴 1에서 저장한 prefix 행 기여분 + 요청 때 계산한 suffix 행 기여분(전체 행의 attention 열 합). fallback은 그 layer의 이미지 토큰 K·V 전부를 읽어 mask 없이 사용. 재배치는 턴 1 요청 하나의 통계(모든 head·모든 행 열 합)로 한 번. 다음 layer probe sidecar 선행 읽기. 공개 구현 https://github.com/ISCS-ZJU/HyperInfer 커밋 `91df7a2fc12581c6780a6485c5b50e7fd9ef9d0e`.
- FullLoad: 전체 Visual KV(raster 순서), 워커가 layer 순서로 K·V 전체를 이어 읽음.
- Ours: 34절의 PYR-b20 packed store(image-only CLIP 순서, layer별 배분 그대로), 워커가 packed 파일을 layer 순서로 이어 읽음(hit당 preadv 32회).
- 비동기 공통: 백그라운드 워커 1개, 디스크 읽기만 워커가 하고 GPU 적재는 요청 스레드, 선행 읽기는 TTFT 타이머 시작 뒤에만, 앞서 읽는 깊이 상한 없음.
- 캐시 계층 없음: 모든 방법이 cold SSD. ReComp·FullLoad의 완결된 이전 기록이 없어 네 arm 모두 이 실행 안에서 잰다.

## 검증 단계

| 단계 | 결과 | 내용 |
|---|---|---|
| 1. 공개 구현과의 동등성 (CPU) | PASS | 공개 함수 호출 테스트 11개 모두 실행·통과(skip 0), 커밋 `91df7a2` |
| 2. prefix 행 기여분 (gate 44장, GPU) | PASS | 44장 중 저장값 ≠ 재계산값인 이미지 0장 |
| 3. 정합성 (유지 100% IMPRESS vs 같은 재배치 store의 FullLoad loader) | PASS | 132건 중 생성 토큰 불일치 0건 (입력이 달라진 요청 0건). 참고(판정 아님): 재배치 store FullLoad와 raster store FullLoad의 생성 토큰이 다른 요청 1건 / 132건 (그중 입력이 같은 요청 1건) |
| 4. 비동기 정합성 (gate 44장) | PASS | FullLoad: 132건 중 불일치 0건; IMPRESS: 132건 중 불일치 0건; Ours: 132건 중 불일치 0건 |
| 5. 턴 1 일치 (gate 9 arm) | PASS | 44장 중 44장 일치 |
| 6. 기존 테스트 | PASS | transformers 4.57.6: 761개 실행, 실패 3개(모두 환경 실패), 새 실패 0개; transformers 5.13.1: 779개 실행, 실패 2개(모두 환경 실패), 새 실패 0개 |
| 7. Pilot 40장 (본조건 4 arm) | PASS | 480건 감사 PASS, 턴 1 일치 40/40장, 이미지당 5.14 s → main 예상 35.7 h, 디스크 PASS |
| 8. 본실행 진입 | PASS | 1–7 모두 PASS일 때만 실행 |

### 검증 3의 내용 (gate 44장, 132건)

- 판정: 유지 100% IMPRESS와, 같은 재배치 store를 FullLoad loader(선택·mask 없음)로 읽은 결과의 생성 토큰이 다른 요청 0건 / 132건.
- 참고(판정에 쓰지 않음): 재배치 store FullLoad와 raster store FullLoad의 생성 토큰이 다른 요청 1건 / 132건. 그중 두 요청의 입력이 같은 것은 1건이고, 나머지는 앞 턴의 답이 달라져 history가 달라진 요청이다. 두 store의 K·V 바이트는 같고 행 순서만 다르다.

참고: 재배치 vs raster FullLoad:

| image | dialogue | 턴 | 입력 동일 | `fullload_reordered` 토큰 | `fullload` 토큰 | 달라진 위치 | raster FullLoad 상위 두 후보 logit |
|---|---|---:|---|---|---|---:|---|
| 436426 | mtvqa25k_000017 | 3 | 예 | [4367, 2] | [9817, 2] | 0 | 19.625 (id 9817) / 19.5 (id 4367) |

### 참고: gate 44장의 동기·비동기 followup TTFT (88요청/arm, 판정 기준 아님)

| 방법 | 동기 평균(ms) | 비동기 평균(ms) | paired 평균 차이(ms) |
|---|---:|---:|---:|
| FullLoad | 214.2 | 117.3 | -96.8 |
| IMPRESS | 245.5 | 242.5 | -3.1 |
| Ours | 83.6 | 73.5 | -10.1 |

## 본실행 (25,000장)

요청 수: 300,000 (이미지 25,000장 × 3턴 × 4 arm).

VQA score는 ten-answer leave-one-out 공식 VQA score(%). SSD MB는 read syscall이 실제로 돌려준 바이트(1 MB = 1e6 bytes)의 요청 평균. FullLoad·ReComp 대비는 같은 실행 안의 TTFT 평균 비율이다.

| 방법 | Followup score | TTFT 평균(ms) | p50 | p95 | FullLoad 대비 | ReComp 대비 | SSD MB/followup |
|---|---:|---:|---:|---:|---:|---:|---:|
| ReComp | 48.6084 | 122.6 | 133.0 | 135.9 | 1.035× | 1.000× | 0.00 |
| FullLoad | 48.5860 | 118.4 | 117.8 | 129.0 | 1.000× | 0.966× | 301.99 |
| IMPRESS | 48.1140 | 240.6 | 237.8 | 268.4 | 2.032× | 1.962× | 263.35 |
| Ours | 47.2070 | 74.7 | 73.9 | 81.2 | 0.631× | 0.610× | 75.50 |

턴별 VQA score:

| 방법 | T1 | T2 | T3 | 전체 | Followup |
|---|---:|---:|---:|---:|---:|
| ReComp | 48.6572 | 48.6572 | 48.5596 | 48.6247 | 48.6084 |
| FullLoad | 48.6572 | 48.6548 | 48.5172 | 48.6097 | 48.5860 |
| IMPRESS | 48.6572 | 48.0956 | 48.1324 | 48.2951 | 48.1140 |
| Ours | 48.6572 | 47.2236 | 47.1904 | 47.6904 | 47.2070 |

IMPRESS의 종류별 읽기량(followup 평균): probe 14.38 MB / K 124.49 MB / V 124.49 MB (합 263.35 MB). probe에는 prefix 행 기여분 0.22 MB(32 layer × 6,912 bytes)가 포함된다.

참고(턴 1 TTFT, followup E2E, read 호출 수, 요청 스레드가 읽기를 기다린 시간):

| 방법 | T1 TTFT 평균(ms) | followup E2E 평균(ms) | read 호출/followup | 읽기 대기 평균(ms) |
|---|---:|---:|---:|---:|
| ReComp | 109.2 | 154.7 | 0.0 | — |
| FullLoad | 108.8 | 151.2 | 64.0 | 6.79 |
| IMPRESS | 113.5 | 274.3 | 99.3 | 103.20 |
| Ours | 109.6 | 108.6 | 32.0 | 0.77 |

### 같은 실행 1,000장 구간 (main manifest 순서 앞쪽 1,000장)

| 방법 | followup TTFT 평균(ms) | p50 | p95 | FullLoad 대비 | ReComp 대비 |
|---|---:|---:|---:|---:|---:|
| ReComp | 122.2 | 132.5 | 135.5 | 1.051× | 1.000× |
| FullLoad | 116.3 | 114.7 | 127.2 | 1.000× | 0.951× |
| IMPRESS | 238.5 | 235.8 | 263.0 | 2.051× | 1.951× |
| Ours | 74.4 | 73.6 | 80.8 | 0.640× | 0.609× |

### 참고: 전체 25,000장의 같은 비율

| 방법 | followup TTFT 평균(ms) | p50 | p95 | FullLoad 대비 | ReComp 대비 |
|---|---:|---:|---:|---:|---:|
| ReComp | 122.6 | 133.0 | 135.9 | 1.035× | 1.000× |
| FullLoad | 118.4 | 117.8 | 129.0 | 1.000× | 0.966× |
| IMPRESS | 240.6 | 237.8 | 268.4 | 2.032× | 1.962× |
| Ours | 74.7 | 73.9 | 81.2 | 0.631× | 0.610× |

### 이미지 단위 paired bootstrap 95% CI (10,000회, seed 1234, followup; 모두 같은 실행, 기록 재사용 없음)

| 비교 | VQA score 차이 (pp) | 해석 | TTFT 차이 (ms) |
|---|---:|---|---:|
| Ours − IMPRESS | -0.9070 [-1.0894, -0.7240] | 낮음 | -165.9 [-166.0, -165.7] |
| Ours − FullLoad | -1.3790 [-1.5512, -1.2068] | 낮음 | -43.7 [-43.7, -43.6] |
| Ours − ReComp | -1.4014 [-1.5728, -1.2292] | 낮음 | -47.9 [-47.9, -47.8] |
| IMPRESS − FullLoad | -0.4720 [-0.6184, -0.3272] | 낮음 | +122.2 [+122.0, +122.4] |

### IMPRESS 내부 통계 (followup)

- fallback 비율(layer·요청 평균): 47.96% · probe head 간 평균 Jaccard: 0.3420 (임계값 0.3, 이상이면 합의)
- fallback이 한 layer라도 있었던 요청: 100.00% · 32개 layer 모두 fallback인 요청: 0.00%
- K·V chunk 비율(읽은 chunk / layer당 9 chunk, probe sidecar 제외): K 82.45% · V 82.45% · K·V 평균 82.45% · 합의 layer만 66.26%
- 중요도 합에서 prefix 행이 차지하는 비중(probe head 3개 × 이미지 토큰 576개, layer·요청 평균): 93.94%
- 재배치하지 않았다면(계산값, 읽지 않음) K·V chunk 비율: 96.50%
- probe sidecar 선행 읽기: 요청당 31.0개 중 30.90개가 사용 시점에 완료, 대기 0.83 ms
- GPU(요청 스레드)가 읽기를 기다린 시간, followup 평균 / p95: FullLoad 6.79 / 16.35 ms · IMPRESS 103.20 / 129.02 ms · Ours 0.77 / 5.11 ms
- 읽기와 연산이 겹치므로 hook 시간과 디스크 시간의 합을 TTFT로 해석하지 않는다.

| layer | fallback 비율 | 평균 Jaccard | K chunk 비율 | V chunk 비율 | prefix 행 비중 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.00% | 0.7782 | 33.34% | 33.34% | 92.23% |
| 1 | 0.00% | 0.7960 | 34.66% | 34.66% | 95.52% |
| 2 | 0.00% | 0.6736 | 45.46% | 45.46% | 96.61% |
| 3 | 15.41% | 0.3484 | 63.74% | 63.74% | 95.45% |
| 4 | 51.40% | 0.3021 | 76.56% | 76.56% | 96.79% |
| 5 | 3.63% | 0.3755 | 66.11% | 66.11% | 97.42% |
| 6 | 0.11% | 0.4813 | 61.51% | 61.51% | 95.15% |
| 7 | 76.07% | 0.2750 | 88.27% | 88.27% | 96.87% |
| 8 | 50.04% | 0.3045 | 85.71% | 85.71% | 97.49% |
| 9 | 88.33% | 0.2544 | 97.05% | 97.05% | 96.22% |
| 10 | 67.45% | 0.2884 | 88.83% | 88.83% | 97.25% |
| 11 | 95.02% | 0.2333 | 98.10% | 98.10% | 95.59% |
| 12 | 99.94% | 0.1970 | 99.98% | 99.98% | 97.47% |
| 13 | 6.25% | 0.3594 | 81.43% | 81.43% | 95.69% |
| 14 | 75.03% | 0.2809 | 91.63% | 91.63% | 96.37% |
| 15 | 87.00% | 0.2564 | 95.50% | 95.50% | 95.12% |
| 16 | 27.07% | 0.3237 | 79.01% | 79.01% | 93.99% |
| 17 | 37.78% | 0.3202 | 83.29% | 83.29% | 91.05% |
| 18 | 88.64% | 0.2562 | 96.00% | 96.00% | 93.74% |
| 19 | 99.94% | 0.2012 | 100.00% | 100.00% | 95.29% |
| 20 | 21.00% | 0.3340 | 69.16% | 69.16% | 90.32% |
| 21 | 37.46% | 0.3164 | 89.82% | 89.82% | 83.31% |
| 22 | 38.25% | 0.3207 | 97.64% | 97.64% | 91.58% |
| 23 | 99.93% | 0.1817 | 99.97% | 99.97% | 96.79% |
| 24 | 22.13% | 0.3369 | 85.52% | 85.52% | 85.21% |
| 25 | 22.70% | 0.3318 | 86.64% | 86.64% | 83.05% |
| 26 | 4.90% | 0.3887 | 79.00% | 79.00% | 92.75% |
| 27 | 72.30% | 0.2775 | 98.63% | 98.63% | 94.54% |
| 28 | 49.62% | 0.3024 | 96.78% | 96.78% | 93.26% |
| 29 | 31.26% | 0.3251 | 79.95% | 79.95% | 91.96% |
| 30 | 78.94% | 0.2702 | 93.56% | 93.56% | 96.92% |
| 31 | 87.22% | 0.2527 | 95.39% | 95.39% | 95.01% |

### 저장·턴 1 비용 (이미지당 평균)

| 방법 | store MB | Visual KV MB | probe 사본 MB | 쓰기(store 준비까지) ms | p95 | 무결성 해시 ms | 재배치 쓰기 ms | activation ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FullLoad | 318.79 | 301.99 | 14.16 | 309.2 | 386.1 | 158.7 | — | 158.9 |
| IMPRESS | 319.16 | 301.99 | 14.16 | 308.4 | 385.7 | 155.4 | 249.6 | 156.4 |
| Ours | 78.16 | 75.50 | 0.00 | 131.5 | 169.0 | 38.8 | — | 48.4 |

IMPRESS의 턴 1 추가 비용(이미지당 평균, followup TTFT에는 들어 있지 않다):

| 항목 | 시간(ms) | 어디서 |
|---|---:|---|
| prefix 행 기여분 + 재배치 통계 캡처 (32 layer) | 1.4 | 턴 1 요청 안 (그 요청의 TTFT에 포함) |
| 캡처 결과를 host로 옮기기 | 0.2 | 요청 밖 |
| 증거 파일(.npz) 저장 | 0.5 | 요청 밖 (실험 기록용) |
| 재배치 순서 계산 (32 layer 정렬) | 22.7 | 요청 밖 |
| 재배치 쓰기 전체 (K / V / probe sidecar 읽기+치환+쓰기, meta) | 249.6 | 요청 밖 |
|   그중 읽기 / 쓰기 | 40.7 / 152.6 | |
| 요청 밖 합계 | 250.3 | |

캡처가 걸린 턴 1 요청의 TTFT는 평균 113.5 ms, 같은 이미지의 ReComp 턴 1은 평균 109.2 ms다(캡처 요청에는 KV 캡처도 함께 걸려 있다). probe key 사본 쓰기는 4.3 ms(raster writer 안)다.
"쓰기(store 준비까지)"는 raster writer(쓰기+fsync)의 시간이고 IMPRESS의 재배치 쓰기는 별도 열이다. ReComp는 store가 없다. FullLoad store에도 쓰이지 않는 probe 사본이 들어 있다(같은 writer).

### answer type별 followup VQA score (탐색적, 다중비교 보정 없음)

| answer type | 요청 수 | ReComp | FullLoad | IMPRESS | Ours |
|---|---:|---:|---:|---:|---:|
| yes/no | 18,623 | 43.5215 | 43.5182 | 43.4538 | 43.1628 |
| number | 6,391 | 53.6051 | 53.5331 | 52.9135 | 51.7227 |
| other | 24,986 | 51.1218 | 51.0978 | 50.3598 | 49.0663 |

턴 1 생성 토큰이 네 arm에서 모두 같은 이미지: 25,000 / 25,000.

## 실행 기록

- 요청 수: gate 1,188 · pilot 480 · main 300,000 / 300,000. gate·pilot 요청은 main 분모에 넣지 않았다.
- 소요 시간: gate 0.10 h (프로세스 시작 1회) · pilot 0.06 h (프로세스 시작 1회) · main 35.31 h (프로세스 시작 1회).
- 중단·재개 기록: 0건.
- 이미지당 scratch 최대 사용량: 716.1 MB. 실행 전 여유 437.1 GiB, 보고 시점 여유 422.0 GiB.
- 기존 `results/`·`runs/`·`kvstore*` 파일 목록(경로, 크기, mtime) 비교: 변화 없음 (2,454,045개 → 2,454,045개).
- 소스 freeze 대조(보고 시점): 일치.

## 실행 전 기록 (GPU 측정 전에 작성)

- **중지한 실행**: LLaVA-1.5 × MT-VQA 25k 본실행 `runs/llava15_mtvqa_impress_20261117T085444Z`(이전 IMPRESS 구현, 계약 `docs/llava15_mtvqa_impress_v2_contract.md`)을 `STOP_REQUESTED`로 이미지 경계에서 멈췄다.
  중지 시각 2026-11-17 18:15:20 UTC, 완료 5,956 / 25,000장, 71,472 / 300,000요청(마지막 commit 이미지 `314125`). 부분 결과는 삭제하지 않았고 `runs/…/STATUS_STOPPED.json`, `results/…/STATUS_STOPPED.json`으로 "중단됨" 표시를 남겼다. scratch store는 이미지마다 지워져 남은 파일이 없다. 그 실행은 재개하지 않는다.
- **디스크 여유** (freeze 시점, `prerun.json`): 437.1 GiB (469,332,004,864 bytes), 필요 여유 50.0 GiB.
- **예상 SSD 쓰기량** (`prerun.json`): 25,858.8 GB ≈ 25.9 TB (이미지당 raster store 1회 + IMPRESS 재배치 store 1회 + packed 1회, 이미지당 묶음 719.2 MB, 25,000 + 44 + 40장). 요청 원본·증거 파일은 별도다.
- 환경: transformers 4.57.6 (`mllm_ft`). 공개 구현 커밋 `91df7a2fc12581c6780a6485c5b50e7fd9ef9d0e`.
- 개발 리허설(측정 아님, 판정·표에 쓰지 않음): `runs/llava15_mtvqa_impress_code_dev_20261117T184702Z`에서 gate(44장, 1,188요청)와 검증 2·3·4·5, pilot 40장을 한 번 돌려 스크립트 오류가 없음을 확인했다. 리허설 뒤 코드를 바꾸지 않았다.
- 실험 A는 이 run 시작 전에 끝났다(`results/mt_gqa_impress_code_20261117T185832Z`). 이 run의 기존 산출물 보호 대조에는 A의 최종 결과가 이미 포함된다.

## 원본과 다른 점

"원본"은 공개 구현(커밋 91df7a2)의 동작이다. 논문이 밝히지 않은 부분은 공개 구현을 원본으로 본다. "방향"은 IMPRESS(baseline)의 TTFT·score에 유리한지, 불리한지, 중립인지다.

| # | 항목 | 원본 | 이 실험 | 방향 |
|---|---|---|---|---|
| 1 | 선택 대상 | prefix 토큰 전부 | 이미지 토큰 576개. 시스템 프롬프트는 항상 유지, 예산 밖 | score에 유리 / 읽기량에 소폭 불리 |
| 2 | prefix 행 기여분 | 정확도 경로: 매 요청 다시 계산(선택된 KV만 보는 상태) | 턴 1(mask 없음)의 값을 저장해 요청 때 더함 | 유리 (요청 때 prefix 행 연산 없음) |
| 3 | 캐시 계층 | 논문·공개 구현: GPU·DRAM·SSD 다층 캐시 | 없음. 모든 방법이 cold SSD | 불리 (모든 방법 공통 조건) |
| 4 | 다음 layer probe 선행 읽기 | 공개 구현의 IMPRESS 설정은 `--no-prefetch` | 켬 | 유리 |
| 5 | 득표 동점 | 순서 미정의 | 저장 위치가 앞선 토큰 우선 | 유리 또는 중립 |
| 6 | 재배치 통계의 추론 상태 | head별 25% mask 추론, 같은 prefix의 마지막 요청 | mask 없는 턴 1 요청 하나 | 중립 |
| 7 | 재배치 적용 | `key[reorder]` (token→rank 인덱스를 그대로 적용해 내용이 어긋남) | 위치 r = rank r 토큰, K·V·probe 같은 순서 | 중립 (접근 위치는 같다) |
| 8 | 중요도 수치 정밀도 | float16 softmax·합 | BF16 확률 + float32 합 / float32 softmax | 중립 |
| 9 | Fallback 읽기 단위 | chunk 파일별 | layer의 K 파일 1회, V 파일 1회 pread | 유리 (호출 수 적음, 바이트 같음) |
| 10 | 선택 chunk 읽기 단위 | chunk 파일별 | 인접한 chunk는 pread 하나로 합침 | 유리 |
| 11 | prefix 행 기여분 읽기 | 없음 | probe sidecar 끝 6,912 bytes/layer를 같은 read로 | 불리 (probe 읽기 +1.6%) |
| 12 | mask | 정확도 경로에만 있음(성능 경로는 0으로 채우고 답을 내지 않음) | mask (한 경로에서 score와 시간을 함께 잼) | 중립 |
| 13 | 모델·생성 | OPT(FP16) 텍스트 LLM, 1토큰 logits | LLaVA-1.5(Vicuna, RoPE, NF4/BF16), greedy 최대 16토큰, 디코딩 중 선택 고정 | 중립 |
| 14 | HyperInfer의 KV 선행 로딩 | HyperInfer 확장 | 없음 (IMPRESS의 probe 선행 읽기만) | 불리 (HyperInfer 대비; IMPRESS 정의에는 해당 없음) |

## 실행 후 기록 (본실행 종료 뒤 작성; 계약·표본·기준은 결과를 보기 전의 것 그대로)

- **실행 시각**: gate 2026-11-18 05:33–05:40 UTC, pilot 05:40–05:44, 본실행 05:44:26 UTC – 2026-11-19 17:03:25 UTC (35.31 h). 본실행 300,000 / 300,000요청, 25,000 / 25,000장. 중단·재개 없음(프로세스 시작 1회).
- **검증 단계**: 1·2·3·4·5·6·7 모두 PASS(보고서 "검증 단계" 표). 3번은 유지 100% IMPRESS와 같은 재배치 store의 FullLoad loader가 gate 132건 모두에서 같았다(불일치 0). 재배치 store FullLoad와 raster store FullLoad는 132건 중 1건 다름(이미지 436426, 턴 3; raster FullLoad의 상위 두 후보 logit 19.625 vs 19.5, BF16 근사 동점이며 참고값, 판정 아님). 모든 arm의 턴 1 생성 토큰은 25,000장 모두 같았다.
- **디스크·SSD 쓰기량**: 실행 전 여유 437.1 GiB, 보고 시점 여유 422.0 GiB, 이미지당 scratch 최대 716.1 MB. `/sys/block/nvme0n1/stat`의 기록 sector 수 차이로 잰 SSD 쓰기량은 51,061,833,640 sector = 26,143.7 GB ≈ 26.1 TB(gate+pilot+본실행 전체, 다른 프로세스의 쓰기도 포함하는 장치 전체 값). 실행 전 예상 25,858.8 GB보다 1.1% 많다.
- **기존 산출물 보호**: `results/`, `runs/`(이 run 제외), `kvstore*`의 파일 2,454,045개 대조 변화 없음. 이 개수에는 실험 A의 최종 결과가 들어 있다.
- **예상과 다른 점**
  - 공개 구현 방식의 IMPRESS는 FullLoad보다 TTFT가 2.03× 길다(240.6 vs 118.4 ms). 실험 A(1.49×)보다 격차가 크다. LLaVA-1.5에서는 FullLoad가 이미지당 302 MB뿐이라 probe 읽기 호출과 hook 오버헤드의 비중이 커진다. 요청 스레드의 읽기 대기는 IMPRESS 103.2 ms, FullLoad 6.8 ms, Ours 0.8 ms다. 이 설명은 관측값에서 읽은 해석이며 요인을 따로 떼어 재지는 않았다(미실행).
  - fallback 비율이 layer·요청 평균 47.96%이고 모든 요청에 fallback layer가 하나 이상 있다. 평균 Jaccard 0.342로 임계값 0.3 근처라, K·V chunk 비율이 82.45%에 그쳐 읽기량이 FullLoad의 87%(263 vs 302 MB)다. 합의 layer만 보면 66.3%다.
  - 정확도 방향이 실험 A와 다르다. 여기서는 Ours가 IMPRESS보다 VQA score가 낮다(Ours−IMPRESS −0.91 pp [−1.09, −0.72]). Ours−FullLoad는 −1.38 pp, IMPRESS−FullLoad는 −0.47 pp다. 실험 A는 Ours가 IMPRESS보다 높았다(+6.80 pp). 두 실험의 차이는 모델·데이터셋·지표(공식 VQA score)가 모두 달라 원인을 하나로 특정하지 않았다(미실행).
  - 중요도 합에서 prefix 행 비중은 93.94%로 A(98.7%)보다 낮고, 중간 layer(21·24·25)에서 83–85%까지 내려간다.
  - Ours의 followup score는 ReComp보다 1.40 pp, FullLoad보다 1.38 pp 낮다(CI가 0을 포함하지 않음). 유지 비율이 FullLoad의 25%인 조건의 손실이다.
- **하지 않은 것(미실행)**
  - 유지 비율·임계값·probe head 수 등 IMPRESS 옵션 변경(민감도) 실험.
  - "원본과 다른 점" 각 항목이 결과에 준 효과의 개별 측정. 표의 방향은 설계상 판단이다.
  - 공개 구현의 캐시 계층(GPU·DRAM), HyperInfer의 KV 선행 로딩.
  - 중지한 이전 구현 실행(5,956장)과의 비교, 그 실행의 재개.
  - 재배치 store와 raster store FullLoad의 생성 토큰 차이 1건의 원인 분석(BF16 근사 동점으로 보임).
  - 실행 중 디스크 최소 여유 시계열(시작·종료 시점만 기록).
- **방법 이름 표기**: 표와 본문의 방법 이름은 ReComp / FullLoad / IMPRESS / Ours만 쓴다. 내부 arm 이름은 `runs/<run>/config.json`에 있다.
