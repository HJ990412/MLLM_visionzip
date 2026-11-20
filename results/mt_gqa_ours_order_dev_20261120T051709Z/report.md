# LLaVA-NeXT × MT-GQA: Ours에서 중요도 순 재배열을 빼도 되는가

측정값만 적는다. 측정하지 않은 값은 "미실행"이다. 계약: `docs/mt_gqa_ours_order_contract.md` (사본 `CONTRACT.md`). 두 방식의 이름은 "Ours(재배열)"(현재 구현)과 "Ours(원래 순서)"(같은 토큰 집합을 원래 이미지 토큰 순서로 저장·적재)다.

## 판정

미실행 (본실험이 끝나지 않았다).

## Gate (본실험 전)

| gate | 결과 | 내용 |
|---|---|---|
| 1. 저장 내용 일치 — gate 4장 | PASS | 4/4장 일치; 비교한 K·V 행 144,384개 중 다른 행 0개; separator 블록·payload 크기 일치 |
| 1. 저장 내용 일치 — DEV 40장 | 미실행 |  |
| 2. 질문 무관 — gate 4장 | PASS | 4/4장에서 질문 3개로 만든 store의 선택·배치가 같음 (layer별 인덱스 표, 행 수, byte table, 파일 구성, 요청 기록·payload digest 외 메타데이터), commit된 store의 인덱스 표와도 같음. KV payload 바이트는 판정하지 않고 아래에 보고 |
| 2. 질문 무관 — DEV 40장 | 미실행 |  |
| 3. Ours(원래 순서) 동기 = 비동기 — gate 4장 | PASS | 147건 중 불일치 0건 |
| 3. Ours(원래 순서) 동기 = 비동기 — DEV 40장 | 미실행 |  |
| 4. 연속성 (2026-11-17 실행과 토큰 일치) — gate 4장 | PASS | ReComp: 147건 중 불일치 0건; FullLoad: 147건 중 불일치 0건; Ours(재배열): 147건 중 불일치 0건 |
| 4. 연속성 — DEV 40장 | 미실행 |  |
| 5. 재현성 (Ours(원래 순서) 두 번 실행) — gate 4장 | PASS | 147건 중 불일치 0건 |
| 5. 재현성 — DEV 40장 | 미실행 |  |
| 6. 기존 테스트 | 미실행 |  |
| 7. DEV 40장 파일럿 실행·감사 (4 arm) | 미실행 |  |
| gate phase 실행·감사 (4장, 6 arm) | PASS | 882/882건, 4장 |
| devgate phase 실행·감사 (DEV 40장, 3 arm) | 미실행 |  |
| 본실험 진입 | 미실행 |  |

gate 2의 보고 항목(판정 아님) — 질문 3개로 만든 store의 KV payload 바이트:

| 대상 | 질문 쌍 | 프롬프트 길이가 같은 쌍: payload 일치 | 길이가 다른 쌍: payload 일치 | 길이가 다를 때 다른 FP16 원소 비율(평균) | 최대 절댓값 차 | 재배열 순서로 쓴 store도 같은 쌍에서 같고 다른가 | 첫 질문의 payload = commit된 store |
|---|---:|---:|---:|---:|---:|---:|---:|
| gate 4장 | 12 | 2/2 | 0/10 | 59.7% | 0.812 | 12/12쌍 같은 양상 | 4/4장 |

턴 1에서 캡처되는 Visual KV는 그 요청 프롬프트의 토큰 길이에 따라 값이 달라진다(질문 내용이 아니라 길이; 길이가 같으면 바이트까지 같다). 이 성질은 store 종류와 무관하며, 같은 캡처를 재배열 순서로 쓴 store의 payload도 같은 쌍에서 같고 다르다. 따라서 gate 2는 사용자 결정(2026-11-20)에 따라 선택·배치의 일치로 판정하고 payload 바이트는 보고만 한다(계약 6절, 12절).

보고 항목(gate 아님), gate 4장: 두 Ours의 생성 토큰열이 다른 요청 T2 0/49건, T3 0/49건. 모든 arm의 턴 1 토큰열이 같은 대화 49/49개.

## DEV 40장 파일럿

미실행.

## 설정

- 실행: `runs/mt_gqa_ours_order_dev_20261120T051709Z` (요청 단위 원본), 이 디렉터리(`results/mt_gqa_ours_order_dev_20261120T051709Z`)는 집계와 보고서.
- 환경: Python 3.10.20, torch 2.5.1+cu121, transformers 4.57.6 (`/home/dblab/anaconda3/envs/mllm_ft/bin/python`).
- 모델 `llava-hf/llava-v1.6-vicuna-7b-hf` revision `c916e6cdcd76…`, NF4/BF16/eager, FP16 SSD KV, greedy max 16, batch 1, CPU threads 2.
- 데이터·절차: MT-GQA-reconstructed 398 images / 4,061 dialogues / 3턴, 2026-11-17 본실험과 같은 프로토콜. 모든 arm이 매 대화 T1을 픽셀부터 계산하고 T2–T3는 자기 arm의 생성 이력을 쓴다. 저장 KV 읽기는 요청 직전에 page cache를 비운 cold 조건, 비동기 읽기, strict normalized EM.
- 두 Ours: PYR-beta-raw 벡터, 총 KV 예산 25%, packed 저장. layer별 토큰 집합과 k_l은 같고, 파일 안의 행 순서와 GPU cache에서의 자리만 다르다.
- arm 순서: 대화 단위 순환 회전에 더해, 두 Ours가 한 블록(arm 수만큼의 대화)마다 자리를 맞바꾼다(계약 4절; 이전 프로토콜에 추가한 것).

## 8. 기존 코드에서 추가하거나 바꾼 부분과 새 테스트

| 파일 | 내용 |
|---|---|
| `mmimpress/packed_store.py` | `write_packed_store(..., row_order="importance")`: `"original"`이면 layer l의 같은 토큰 집합을 원래 인덱스 오름차순으로 쓰고, separator 블록은 원래 위치에서 모으며, 메타데이터에서 중요도 순열 관련 필드를 빼고 인덱스 표 두 개를 넣는다. 기본값의 연산·바이트는 그대로다. |
| `mmimpress/packed_store.py` | 신규 `original_order_plan`(행 계획), `layer_original_indices`(메타데이터에서 layer별 목록 복원), `validate_packed_original_meta`(fail-closed 검증). `validate_packed_meta`는 store 형식에 따라 분기한다. `persist_captured_packed_prefix`에 `row_order` 인자. |
| `mmimpress/packed_serve.py` | `PackedImageContext`: 원래 순서 store면 separator 위치를 원래 위치로 두고 layer별 행 인덱스를 activation 때 만든다. 전용 layout 검증 추가. `PackedPrefixSelector._scatter_layer`: 원래 순서면 행을 원래 토큰 위치에 쓰고 그 위치만 보이게 mask를 만든다. `PackedStreamSelector`는 상속으로 같은 경로를 쓴다. 읽기(`PackedReader`), 읽기 순서, hook, 타이머 경계는 수정하지 않았다. 요청 결과에 `row_order` 필드가 하나 붙는다(기본 경로 포함). |
| `mmimpress/piggyback.py` | `persist_captured_visual_prefix(..., row_order="importance")`: `"original"`이면 순열 hash를 메타데이터에 넣지 않고 `visionzip_layout.pt`를 쓰지 않는다. staging, fsync, 원자적 publish 단계는 두 방식이 같다. |
| `scripts/202_eval_mt_gqa_ours_order.py` (신규) | 실행 harness. `scripts/196`에서 IMPRESS arm을 빼고 Ours(원래 순서) arm, 두 store의 디스크 비교, 자리 교대가 있는 arm 순서를 더했다. |
| `scripts/203_validate_mt_gqa_ours_order.py` (신규) | gate 1–7과 본실험 뒤의 사후 확인. |
| `scripts/204_report_mt_gqa_ours_order.py` (신규) | 이 보고서. 측정에 관여하지 않는다. |
| `scripts/run_ours_order.sh` (신규) | freeze → gate → 본실험 → 보고의 실행 순서. |
| `tests/test_packed_original_order.py` (신규, 9개) | 행 계획이 `order[:k_l]`의 정렬과 같음 / 입력이 순서와 개수뿐임; 두 store의 토큰별 K·V 바이트·separator·byte table·크기 일치; 메타데이터에 중요도 순열이 없음; 변조된 메타데이터 거부; 적재 위치·mask·읽기 호출이 기대와 같음; 비동기 경로가 동기 경로와 같고 읽기 계획이 재배열 store와 같음; 길이가 다른 세 캡처(질문 3개)로 만든 store가 바이트 단위로 같음; 옵션의 적용 범위 밖 사용 거부. |

2026-11-17 실행의 소스 freeze와 비교해 SHA256이 달라진 파일: `mmimpress/packed_serve.py`, `mmimpress/packed_store.py`, `mmimpress/piggyback.py`. 새로 생긴 파일: `scripts/202_eval_mt_gqa_ours_order.py`, `scripts/203_validate_mt_gqa_ours_order.py`, `tests/test_packed_original_order.py`. 기존 재배열 경로는 기본값이며, gate 4(연속성)가 ReComp·FullLoad·Ours(재배열)의 생성 토큰이 그 실행과 같음을 확인한다.

## 기존 산출물 보호

- `results/`, `runs/`(이 run 제외), `kvstore*`의 기존 파일 3,081,376개 (경로, 크기, mtime) 대조: **PASS**.
