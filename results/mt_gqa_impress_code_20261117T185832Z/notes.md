## 실행 전 기록 (GPU 측정 전에 작성)

- **중지한 실행**: LLaVA-1.5 × MT-VQA 25k 본실행 `runs/llava15_mtvqa_impress_20261117T085444Z`(이전 IMPRESS 구현, 계약 `docs/llava15_mtvqa_impress_v2_contract.md`)이 돌고 있었다.
  그 실행의 `STOP_REQUESTED` 파일로 이미지 경계에서 멈췄다. 중지 시각 2026-11-17 18:15:20 UTC, 완료 5,956 / 25,000장, 71,472 / 300,000요청(마지막 commit 이미지 `314125`).
  부분 결과는 삭제하지 않았고 `runs/…/STATUS_STOPPED.json`, `results/…/STATUS_STOPPED.json`으로 "중단됨" 표시를 남겼다. scratch store는 그 실행의 수명주기대로 이미지마다 삭제되어 남은 파일이 없다.
- **디스크 여유** (freeze 시점, `prerun.json`): 439.0 GiB (471,397,724,160 bytes), 필요 여유 47.2 GiB.
- **예상 SSD 쓰기량** (`prerun.json`, README 38절 본실행이 같은 이미지에서 잰 store 크기로 계산; 이미지당 raster store 1회 + IMPRESS store 2회 + packed 1회):
  gate 17.0 GB + DEV 163.7 GB + 본실행 1,633.8 GB = 1,814.4 GB. 요청 원본·증거 파일은 별도다.
- 환경: transformers 4.57.6 (`mllm_ft`). 공개 구현 커밋 `91df7a2fc12581c6780a6485c5b50e7fd9ef9d0e`.
- 개발 리허설(측정 아님, 판정·표에 쓰지 않음): `runs/mt_gqa_impress_code_dev_20261117T182946Z`에서 gate phase(4장, 1,323요청)와 검증 2·3·4·5를 한 번 돌려 스크립트 오류가 없음을 확인했다.
  리허설의 gate 이미지는 이 run의 gate 이미지와 같다. 리허설 뒤 코드를 바꾸지 않았다.

## 실행 후 기록 (본실행 종료 뒤 작성; 계약·표본·기준은 결과를 보기 전의 것 그대로)

- **실행 시각**: gate 2026-11-17 18:59–19:16 UTC, DEV 19:16–20:08, 본실행 20:08–2026-11-18 05:28 UTC (9 h 20 min), 본실행 48,732 / 48,732 요청, 398 / 398장. 중단·재개 없음.
- **검증 단계**: 1·2·3·4·5·6·7 모두 PASS(보고서 "검증 단계" 표). 3번은 유지 100% IMPRESS와 같은 재배치 store의 FullLoad loader가 gate 147건 모두에서 같았다(불일치 0). 재배치 store FullLoad vs raster store FullLoad는 147건 중 1건 다름(`mtgqa_000018` 턴 2, 참고값, 판정 아님). 5번: ReComp·FullLoad·Ours의 생성 토큰이 README 38절 본실행 기록과 gate 147건, DEV 1,134건, 본실행 12,183건 모두에서 같았고, 모든 arm의 턴 1 토큰도 4,061개 대화 모두에서 같았다. 본실행 진입(8번) 조건을 충족해 본실행을 진행했다.
- **디스크·SSD 쓰기량**: 실행 전 여유 439.0 GiB, 실행 중 관측된 최소 여유 434.0 GiB. `/sys/block/nvme0n1/stat`의 기록 sector 수 차이로 잰 실제 SSD 쓰기량은 3,554,085,928 sector = 1,819.7 GB(gate+DEV+본실행 전체, 다른 프로세스의 쓰기도 포함하는 장치 전체 값). 실행 전 예상 1,814.4 GB와 0.3% 차이.
- **기존 산출물 보호**: `results/`, `runs/`(이 run 제외), `kvstore*`의 파일 2,443,362개 대조 PASS.
- **예상과 다른 점**
  - 공개 구현 방식의 IMPRESS는 FullLoad보다 정답률이 낮고(−7.76 pp, CI [−8.73, −6.81]) TTFT도 더 길다(1.487×). 이전 구현(질문 토큰 행만 사용)과 비교한 수치는 이 실험의 arm이 아니므로 적지 않는다.
  - 원인으로 보이는 관측: fallback이 layer·요청 평균 53.8%이고, 모든 요청에 fallback layer가 하나 이상 있다. 평균 Jaccard가 0.338로 임계값 0.3 근처라 layer 절반 이상이 전체 읽기로 돌아간다. 그 결과 K·V chunk 비율이 84.5%라 읽기량이 FullLoad의 90%(1,052 MB vs 1,174 MB)이고, probe 55.9 MB와 read 호출 수(124 vs 64)가 더해져 GPU 읽기 대기가 381.9 ms로 FullLoad(116.8 ms)보다 길다. 이 설명은 관측값에서 읽은 해석이며 요인을 따로 떼어 재지는 않았다(미실행).
  - 중요도 합에서 prefix 행이 차지하는 비중이 98.7%라 suffix(질문) 행의 기여는 작다.
  - 2번(판정 기준): 같은 턴 1 프롬프트로 픽셀부터 다시 전체 계산한 값과 저장값이 4장 모두 모든 원소에서 같았다(prefix 행 기여분·재배치 통계, 최대 절댓값 차이 0.0, 생성 토큰도 같음). 보고서의 최대 2.39 / 평균 0.00226 / Jaccard 0.9805는 같은 이미지를 다른 질문의 요청에서 다시 캡처한 참고 진단값이며 기준이 아니다. 질문이 다르면 prefix 행 기여분이 달라지는 것(원소 단위 완전 일치 0장)은 저장값을 다른 질문에 재사용하는 설계의 근사다.
  - 선택 대상에서 separator와 system prompt를 항상 유지하는 점 때문에 IMPRESS의 chunk 읽기량이 공개 구현보다 소폭 늘 수 있다(원본과 다른 점 1번).
- **하지 않은 것(미실행)**
  - 유지 비율, 임계값, probe head 수 등 IMPRESS 옵션 변경(민감도) 실험.
  - "원본과 다른 점" 각 항목이 결과에 준 효과의 개별 측정. 표의 방향은 설계상 판단이다.
  - 공개 구현의 캐시 계층(GPU·DRAM), HyperInfer의 KV 선행 로딩.
  - 이전 구현(질문 행만 사용)과의 재비교.
  - 재배치 store와 raster store FullLoad의 생성 토큰 차이 1건의 원인 분석.
- **방법 이름 표기**: 표와 본문의 방법 이름은 ReComp / FullLoad / IMPRESS / Ours만 쓴다. 내부 arm 이름은 `runs/<run>/config.json`에 있다.
