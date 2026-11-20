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
