# LLaVA-NeXT × MT-GQA: Ours에서 중요도 순 재배열을 빼도 되는가 — 실행 계약

2026-11-20 UTC, 이 run의 어떤 GPU 측정보다 먼저 고정. 저장소 `/home/dblab/hj/mllm_v2`.
Run root `runs/mt_gqa_ours_order_<UTC>`, 결과 `results/mt_gqa_ours_order_<UTC>` (같은 `<UTC>`).
이 계약의 SHA256은 `runs/<RUN>/config.json`의 `contract_sha256`에 기록한다. 결과를 본 뒤 정의·gate·판정 기준·표본·arm을 바꾸지 않는다.

## 1. 목적

Ours는 이미지 토큰을 VisionZip 점수 내림차순으로 재배열해 저장하고 GPU cache에도 그 순서로 올린다(layer l은 저장 행 [0, k_l)).
지금은 25%만 저장하므로 저장 순서가 읽기량을 바꾸지 않는다. 재배열을 뺀 방식을 같은 실행 안에서 재서, 빼도 정답률과 TTFT가 유지되는지 판정한다.
최종 결정은 사용자가 한다. 이 문서가 고정하는 것은 두 방식의 정의(3절), gate(6절), 판정 기준(7절), 보고 항목의 정의(8절)다.

보고서와 표에서 두 방식은 "Ours(재배열)", "Ours(원래 순서)"로만 부른다. 옵션 이름과 method id는 이 문서의 3–4절에만 적는다.

## 2. 환경

- 인터프리터 `/home/dblab/anaconda3/envs/mllm_ft/bin/python` — Python 3.10.20, torch 2.5.1+cu121, **transformers 4.57.6** (2026-11-17 본실험과 같다).
  실행 시점 버전과 pip freeze는 `environment_before.json`에 기록하고 각 phase 시작 때 대조한다. `.venv-glm46v`(5.13.1)는 단위 테스트 교차 확인에만 쓴다.
- 모델 `llava-hf/llava-v1.6-vicuna-7b-hf` revision `c916e6cdcd760b4cecd1dd4907f84ac649f93b23`, NF4 / BF16 / eager, FP16 SSD KV, greedy max 16,
  seed 1234, batch 1, CPU threads 2, RTX 4090 1장 단독 사용.

## 3. 두 방식의 정의

둘 다 LLaVA-NeXT-7B, PYR-beta-raw 벡터(`runs/layer_budget_kv25_pyrbeta_20261110T073604Z/vectors/layer_budget_llava_PYR-beta-raw.json`),
총 KV 예산 25%, packed 저장(`kv_packed.bin` 하나), 비동기 읽기다.

**Ours(재배열)** — 현재 구현 그대로. 저장 형식(`prefixstore_kv25_packed_v1`)과 동작을 바꾸지 않는다.
`visionzip_repack_order`가 준 순서(실제 patch를 VisionZip 점수 내림차순으로 stable 정렬, 동점은 원래 인덱스가 앞선 쪽, separator는 뒤)의
앞 k_l개를 layer l의 행 [0, k_l)로 저장하고, GPU cache의 이미지 구간 앞 k_l 자리에 올린다. separator는 저장 순서의 뒤쪽 자리에 올린다.

**Ours(원래 순서)** — 아래만 다르다 (`row_order="original"`).

- layer l이 저장하는 토큰 집합은 Ours(재배열)과 같다: `visionzip_repack_order`가 준 순서의 앞 k_l개(동점 처리 포함). k_l도 같은 벡터·같은 계산이다.
- 그 집합을 **원래 이미지 토큰 인덱스 오름차순**으로 저장한다.
- 재사용 턴에서는 각 행을 GPU cache의 **원래 토큰 위치**에 올리고, separator도 원래 위치에 올리며, 저장하지 않은 위치는 mask로 가린다.
- 파일 구성(layer별 [K|V] 구간 + separator 블록), byte table, 비동기 읽기 순서(separator 블록 → layer 0 … 31의 [K|V] 구간, 워커 1개),
  pinned buffer 준비 시점(activation, 요청 타이머 밖), 읽기 호출 수는 Ours(재배열)과 같다. 같은 reader·같은 selector 클래스·같은 hook을 쓴다.
- 메타데이터에는 중요도 순열을 두지 않는다. layer별 원래 인덱스 목록을 복원하는 표 두 개만 둔다:
  `stored_original_indices`(저장된 토큰 전체의 원래 인덱스, 오름차순, 길이 max k_l)와 `stored_min_keep_count`(각 토큰이 속하는 가장 작은 k_l).
  layer l의 목록 = `stored_min_keep_count[j] <= k_l`인 `stored_original_indices[j]` (집합이 중첩되므로 이것으로 정확히 복원된다).
  `order`, `full_importance_permutation`, `stored_row_to_original`, `newline_stored`, 순열 hash, `visionzip_layout.pt`(점수·순열 파일)는 쓰지 않는다.
- store를 만드는 입력은 턴 1에서 캡처한 Visual KV, CLIP 끝에서 둘째 layer의 CLS→patch attention(VisionZip 점수), separator 위치, 동결된 layer 예산뿐이다.
  질문 문자열, decoder attention, calibration 점수는 입력이 아니다.

두 방식의 store는 각 arm이 자기 턴 1(이미지의 첫 대화)에서 캡처한 KV로 따로 만든다(한 번의 캡처를 나눠 쓰지 않는다).

## 4. Arm과 설정값

| 표의 이름 | method id | store 종류 |
|---|---|---|
| ReComp | `recompute` | 없음 (매 턴 픽셀부터) |
| FullLoad | `fullload_async` | `raster`: 전체 Visual KV, raster 순서 |
| Ours(재배열) | `ours_prefixstore_kv25_PYR-beta-raw_packed_async` | `packed_PYR-beta-raw` (writer 기본값) |
| Ours(원래 순서) | `ours_origorder_kv25_PYR-beta-raw_packed_async` | `packedorig_PYR-beta-raw` (`row_order="original"`) |

- 본실험 arm은 위 4개, 총 12,183 × 4 = **48,732 요청**. ReComp·FullLoad·Ours(재배열)의 코드 경로와 설정은 2026-11-17 본실험
  (`runs/mt_gqa_impress_code_20261117T185832Z`, 계약 `docs/mt_gqa_impress_code_contract.md`)과 같다. FullLoad는 워커가 layer 순서로 K·V 전체를 읽는다.
- gate 전용 arm (어떤 본실험 표에도 넣지 않는다): `ours_origorder_kv25_PYR-beta-raw_packed`(Ours(원래 순서)의 동기 읽기),
  `ours_origorder_kv25_PYR-beta-raw_packed_async_repeat`(Ours(원래 순서)를 한 번 더 실행). 셋은 같은 store를 쓴다.
- **arm 순서**: 대화 단위 순서 회전(대화 ordinal o에서 `o mod n`번째 arm부터 순환)은 이전 실행과 같다. 순환 회전만으로는 각 arm의 직전 arm이 고정되어
  Ours(원래 순서)가 항상 Ours(재배열) 바로 뒤, Ours(재배열)이 항상 FullLoad 바로 뒤에 놓인다. 두 방식의 TTFT를 2% 기준으로 비교하므로,
  `(o div n)`이 홀수인 대화에서는 두 Ours의 자리를 서로 바꾼다. 이렇게 하면 두 방식이 (위치, 직전 arm) 조합을 같은 횟수로 갖는다.
  ReComp·FullLoad의 자리는 이전 실행과 같다. 이것은 이전 프로토콜에 **추가한 것**이다(12절). 요청별로 직전 arm을 기록한다.

## 5. 데이터와 절차 (2026-11-17 본실험과 동일)

- MT-GQA-reconstructed 398 images, 4,061 dialogues × 3 turns. dialogues와 image identity가 2026-11-17 본실험의 manifest와 같음을 freeze에서 assert한다.
- 수명주기는 그 실행과 같은 것(`scripts/164`를 import): method-local generated history(T2–T3는 자기 arm의 생성 이력), strict normalized EM, greedy max 16,
  image-atomic attempt, 이미지별 독립 감사 후 commit, commit 뒤 run-local scratch store만 정확한 SHA allowlist로 삭제, 중단 시 `--resume`.
  실행 스크립트 `scripts/202`는 `scripts/196`에서 IMPRESS arm·분기를 빼고 Ours(원래 순서) arm을 더한 것이다.
- **턴 1은 모든 arm이 매 대화 픽셀부터 계산한다.** 저장 KV 읽기는 요청 직전에 그 store의 모든 `.bin`을 `posix_fadvise(DONTNEED)`로 비운 cold 조건이고
  비우기는 타이머 밖이다. TTFT 경계: 요청 시작(프롬프트 조립·토큰화 직전) → 첫 토큰 argmax 직후 `cuda.synchronize`.
- store는 이미지의 첫 대화 턴 1에서, store 종류마다 그 대화의 arm 순서에서 그 종류를 쓰는 첫 arm의 턴 1 캡처로 만든다.

| phase | 이미지 | 대화 | arm | 요청 수 |
|---|---|---|---|---|
| `gate` | 2026-11-17 실행과 같은 gate 이미지 4장의 모든 대화 | 49 | 6: 본실험 4 + Ours(원래 순서) 동기 + Ours(원래 순서) 두 번째 실행 | 882 |
| `devgate` | DEV 40장의 모든 대화 | 378 | 3: Ours(원래 순서) 동기 / 비동기 / 두 번째 실행 | 3,402 |
| `dev` | DEV 40장의 모든 대화 (파일럿, 본조건) | 378 | 4 (본실험 arm) | 4,536 |
| `main` | 398장 전체 | 4,061 | 4 | 48,732 |

gate·devgate·dev 요청은 본실험 분모에 넣지 않는다. gate·DEV 이미지는 본실험에 다시 포함된다. dev는 구현 확인용이며 본실험의 arm이나 설정을 고르는 데 쓰지 않는다.

## 6. Gate (본실험 전, 4장 → DEV 40장 순서; 하나라도 실패하면 본실험 없이 실패 내용과 원인만 보고)

1. **저장 내용 일치.** 같은 이미지에서 두 store를 디스크에서 다시 읽어 비교한다(이미지의 모든 요청이 끝난 뒤, 타이머 밖).
   원래 순서 store의 layer별 원래 인덱스 집합 == 재배열 store의 `order`의 앞 k_l개; 같은 원래 토큰의 K·V 행이 모든 layer에서 비트 단위로 같음;
   separator 블록이 비트 단위로 같음; payload 총 바이트(와 byte table)가 같음. gate 4장(`gate` phase)과 DEV 40장(`dev` phase) 모두에서 전 이미지 PASS.
   이 비교는 본실험에서도 이미지별 감사 항목으로 계속 수행하며, 실패하면 그 이미지는 commit되지 않고 실행이 멈춘다.
2. **질문 무관.** gate 4장과 DEV 40장 각각에 대해, 같은 이미지에 서로 다른 질문 3개(그 이미지 대화들의 질문 중 앞에서부터 서로 다른 3개)로
   턴 1 요청을 새로 돌려 원래 순서 store를 3개 만든다. **무엇을 어디에 저장하는가**가 세 store에서 같아야 한다:
   layer별 원래 인덱스 표, 행 수, byte table, 파일 구성, 그리고 캡처 요청의 기록(요청 ID·캡처 시간·쓰기 시간)과 payload digest를 뺀 모든 메타데이터 필드.
   그 phase가 commit한 store와도 인덱스 표·행 수·byte table이 같아야 한다. 행 선택 함수의 입력이 VisionZip 순서와 layer별 개수뿐임은
   단위 테스트(`tests/test_packed_original_order.py`)로 확인한다.
   **KV payload 바이트는 판정하지 않고 보고한다**(사용자 결정, 2026-11-20; 12절 8번). 턴 1에서 캡처되는 Visual KV의 값은 그 요청 프롬프트의 토큰 길이에 따라
   달라지기 때문이다(store 종류와 무관). 보고 내용: 질문 쌍마다 payload 일치 여부와 프롬프트 길이가 같은지, 다른 FP16 원소의 비율과 최대 절댓값 차,
   같은 캡처를 재배열 순서로 쓴 store의 payload가 같은 쌍에서 같고 다른지, 첫 질문(= commit된 store를 만든 프롬프트)의 payload가 commit된 store와 같은지.
3. **동기 = 비동기.** Ours(원래 순서)의 동기 읽기 arm과 비동기 읽기 arm이 모든 요청에서 생성 토큰열과 읽기 바이트(전체, packed)가 같다.
   gate 4장(147건)과 DEV 40장(`devgate`, 1,134건).
4. **연속성.** 이번 실행의 ReComp, FullLoad, Ours(재배열) 생성 토큰열이 2026-11-17 실행의 같은 요청(같은 arm·대화·턴;
   요청 단위 원본 `runs/mt_gqa_impress_code_20261117T185832Z/main`)과 모든 요청에서 같다. gate 4장과 DEV 40장(`dev`).
5. **재현성.** Ours(원래 순서)를 같은 요청에 두 번 돌린 생성 토큰열이 모든 요청에서 같다. gate 4장과 DEV 40장(`devgate`).
6. **기존 테스트.** `python -m unittest discover -s tests`를 두 환경에서 돌린다. 이 작업 전부터 환경 문제로 실패하는 것 외에 실패가 없어야 한다.
   4.57.6(실행 환경): GLM-4.6V API가 없어 실패하는 3개(`test_glm46v_runner` setUpClass, `test_glm46v_validation…partial_rope…`, `test_glm46v_vision…sdpa_hook…`).
   5.13.1: `test_qwen25_vision…stock_forward_capture…`, `test_visdial_turn1_piggyback_core…real_tiny_clip…` 2개. 새 테스트 파일은 모두 통과해야 한다.
7. **DEV 40장 파일럿.** 4 arm 본조건으로 돌려 40장 모두 독립 감사 PASS로 commit하고 표를 만든다.

1–7이 모두 PASS면 `gates/admission.json`을 PASS로 쓰고 묻지 않고 본실험(398장 4 arm)을 이어서 실행한다. `main` stage는 admission 없이는 시작하지 않는다.

**gate가 아닌 것(보고 항목).** 두 Ours 방식의 생성 토큰이 서로 다른 것은 gate 실패가 아니다. key 순서가 바뀌면 BF16 합산 순서가 달라져 경계 사례가 뒤집힐 수 있다.
모든 arm의 턴 1 생성 토큰이 대화별로 같은지도 보고만 한다.

## 7. 판정 기준 (실행 전 고정)

본실험 398장의 followup(T2–T3, arm당 8,122건) 기준으로 다음 세 가지를 모두 만족하면 **"빼도 됨"**이다.

- **a.** 정답률 차이 (Ours(원래 순서) − Ours(재배열))의 95% CI 하한 > −0.5 pp.
  이미지 단위 paired bootstrap 10,000회, seed 1234, percentile 2.5 / 97.5 (`mmimpress.recompute_kv25.paired_bootstrap`, 요청 가중).
- **b.** followup TTFT 평균의 비율 (Ours(원래 순서) / Ours(재배열)) ≤ 1.02. 각 방식의 followup TTFT 산술 평균의 비다.
- **c.** followup당 SSD 읽기 바이트와 읽기 호출 수가 두 방식에서 같음: 모든 followup 요청 쌍(같은 대화·턴)에서 read syscall이 실제로 돌려준 바이트 합과
  read syscall 호출 수가 각각 같다(따라서 평균도 같다).

하나라도 어긋나면 **"판단 보류"**로 적고, 어긋난 항목과 수치를 그대로 보고한다. 최종 결정은 사용자가 한다.

## 8. 보고 항목의 정의 (측정하지 않은 값은 "미실행")

1. **본실험 표** (arm별, followup): 정답률(strict normalized EM), TTFT 평균·p50·p95, SSD MB/followup(read syscall이 돌려준 바이트, MB = 1e6 bytes), 읽기 호출 수.
2. **짝지은 비교** (이미지 단위 paired bootstrap 95% CI, 정답률과 TTFT): Ours(원래 순서) − Ours(재배열); 두 Ours 각각 − FullLoad, − ReComp;
   두 Ours 각각 − IMPRESS. IMPRESS는 2026-11-17 실행의 저장된 요청별 결과(`impress_official_async`)를 쓴다. 본실험 뒤 gate 4의 일치가 398장 전체에서
   성립하는지 먼저 확인하고, 성립하지 않으면 IMPRESS 비교는 내지 않는다. IMPRESS와의 TTFT 차이는 서로 다른 실행의 값임을 표에 적는다.
   CI가 0을 포함하면 "차이 미입증"으로 쓴다(동등성 아님).
3. **TTFT 비율** (followup 평균의 비): 원래 순서/재배열, 두 Ours 각각/FullLoad, /ReComp.
4. **두 Ours 사이의 차이**: 생성 토큰열이 다른 요청 수, 정오가 바뀐 요청 수(재배열만 정답 / 원래 순서만 정답)를 T2와 T3로 나눠서.
   T3는 자기 arm의 T2 답이 이력에 들어가므로, 프롬프트가 같은 요청만 센 값도 함께 적는다.
5. **T1 저장 비용** (이미지당 평균, 두 방식 각각): 중요도 계산(VisionZip 점수 캡처, 토큰 위치 매핑, 순서 계산; 원래 순서는 layer별 인덱스 표 계산 포함),
   행 골라내기(writer의 index_select), SSD 쓰기(파일 쓰기와 fsync를 따로), 메타데이터 크기(`meta.json`, 재배열은 `visionzip_layout.pt` 포함).
6. **재사용 턴 시간 분해** (followup 평균, 두 방식 각각): 읽기 대기 = 요청 스레드가 워커의 읽기를 기다린 벽시계 시간(`read_blocked_ms`),
   GPU 적재 = layer별 적재 호출(행·separator를 GPU cache에 쓰고 mask를 만드는 구간)의 벽시계 합(`scatter_ms`; 동기 H2D 복사가 앞선 GPU 연산의 완료를
   기다리는 시간이 섞인다 — 두 방식에 같은 정의). 워커의 읽기 시간, hook 시간, prefill 시간도 적는다. 읽기와 연산이 겹치므로 합을 TTFT로 해석하지 않는다.
   직전 arm별 TTFT(자리 교대의 효과)도 적는다.
7. **판정**: "빼도 됨" 또는 "판단 보류", 기준 a·b·c 각각의 수치.
8. 기존 코드에서 추가하거나 바꾼 부분 목록과 새 테스트.
9. 측정하지 않은 것과 한계.

## 9. 코드 변경 범위

- 수정 (기본값은 기존 동작 그대로, 원래 순서는 새 옵션 `row_order="original"`):
  `mmimpress/packed_store.py`(`write_packed_store`, `persist_captured_packed_prefix`, `validate_packed_meta`; 신규 `original_order_plan`, `layer_original_indices`,
  `validate_packed_original_meta`), `mmimpress/packed_serve.py`(`PackedImageContext`, `PackedPrefixSelector`; `PackedStreamSelector`는 상속으로 같은 경로),
  `mmimpress/piggyback.py`(`persist_captured_visual_prefix`에 `row_order` 인자: 원래 순서일 때 순열 hash와 `visionzip_layout.pt`를 쓰지 않음).
  `mmimpress/cvpr25.py`의 `visionzip_repack_order`는 수정하지 않는다.
- 기본 경로의 유일한 관측 가능한 차이: 요청 결과 기록에 `row_order: "importance"` 필드가 하나 더 붙는다.
- 신규: `scripts/202_eval_mt_gqa_ours_order.py`(실행), `scripts/203_validate_mt_gqa_ours_order.py`(gate), `scripts/204_report_mt_gqa_ours_order.py`(보고; 측정에 관여하지 않으므로
  소스 freeze에서 제외), `scripts/run_ours_order.sh`(실행 순서), `tests/test_packed_original_order.py`, 이 문서.
- 수정 없음: 그 밖의 `mmimpress/**`, 기존 스크립트 01–201, 기존 테스트. 기존 store, 결과 디렉터리(`results/`, `runs/`), 동결 파일(계약·벡터·config·manifest)은 건드리지 않는다.
  수정한 세 파일의 SHA256은 2026-11-17 실행의 `config.json`에 기록된 값과 달라진다(그 실행은 완료되었고 resume 대상이 아니다). freeze가 달라진 파일 목록을 기록한다.

## 10. 저장 계획·SSD 쓰기량·삭제 범위·보호

- FRESH_IMAGE_STREAMING: 이미지 하나씩 store 생성 → 측정 → 저장 내용 비교 → 감사 → commit → run-local scratch 삭제.
  한 이미지의 묶음 상한(최대 N = 2880): 전체 KV store 1개 + packed store 2개 ≈ 2.5 GB (질문 무관 확인 중에는 packed store 6개 ≈ 2.5 GB). 최소 reserve 30 GiB + 작업 묶음 + margin 4 GiB + raw·evidence 증가 8 GiB가 없으면 `BLOCKED_STORAGE`로 멈춘다.
- 예상 SSD 쓰기량은 2026-11-17 본실험이 같은 이미지에서 잰 store 크기로 freeze가 계산해 `storage_plan.json`에 기록한다
  (이미지당 raster store 1회 + packed store 2회; `devgate`는 packed 1회; 질문 무관 확인은 gate·DEV 이미지당 packed 6회 = 질문 3개 × 행 순서 2가지). 실측은 `/sys/block/nvme0n1/stat`의 쓰기 섹터 수 차이로 보고한다.
- 삭제 범위: `runs/<RUN>/scratch/**`뿐이다(감사+commit 후 정확한 allowlist; 질문 무관 확인의 임시 store는 비교 직후). 기존 `results/`, `runs/`, store는 수정·삭제하지 않는다.
  freeze 때 `results/`, `runs/`(이 run 제외), `kvstore*`의 모든 파일 (경로, 크기, mtime_ns) 목록과 그 SHA256을 남기고 보고 때 다시 비교한다.
- 소스: freeze 때 `mmimpress/**/*.py`, `scripts/*.py`(보고 스크립트 198·201·204 제외), `tests/test_*.py`의 SHA256을 `config.json`에 넣고 각 phase 시작 때 대조한다.
- 커밋과 푸시는 하지 않는다.

## 11. 하지 않는 것

- 25% 외 예산, 다른 벡터, 다른 모델·데이터셋, warm cache, 반복 실행(본실험은 1회), 본실험에서의 동기 arm.
- 저장량이 25%를 넘는 구성에서의 읽기량 비교(원래 순서는 접두 구간 읽기로 일부만 읽을 수 없다 — 이 실험의 범위 밖), O_DIRECT·io_uring, 워커 여러 개, SMART·NAND 수준 측정.
- IMPRESS의 재실행(2026-11-17의 저장된 결과만 쓴다).

## 12. 사용자 지시와의 대응, 그리고 지시에 없던 결정

- 지시대로: 두 방식의 정의(3절), arm 4개·48,732 요청, 프로토콜(5절), 통계, 결과 디렉터리 이름, 판정 기준 a·b·c와 그 임계값, gate 1–7의 내용과 순서, 보고 항목 1–9.
- 지시에 없어서 이 문서에서 정한 것:
  1. **arm 순서의 자리 교대**(4절). 이전 프로토콜의 순환 회전에 두 Ours의 자리 교대를 더했다. 이유: 순환 회전에서는 Ours(원래 순서)가 항상 Ours(재배열) 직후에 실행되어
     순서 효과와 방식 효과가 섞인다. ReComp·FullLoad·Ours(재배열)의 생성 토큰은 순서와 무관하므로 gate 4(연속성)에는 영향이 없다.
  2. **gate 3·5의 DEV 40장 확인을 별도 phase(`devgate`, 3 arm)로** 돌린다. DEV 파일럿(gate 7)을 본실험과 같은 4 arm으로 유지하기 위해서다.
  3. **gate 6의 "모두 통과"**는 이 작업 전부터 환경 문제로 실패하는 테스트(4.57.6에서 3개, 5.13.1에서 2개)를 제외한 뜻으로 읽었다. 그 외의 실패가 하나라도 있으면 FAIL이다.
  4. **gate 4의 비교 대상**은 요청 단위 원본이 있는 `runs/mt_gqa_impress_code_20261117T185832Z/main`이다(`results/…`에는 집계만 있다).
  5. **메타데이터의 인덱스 표 부호화**(3절): layer별 목록을 그대로 쓰지 않고, 그것을 정확히 복원하는 표 두 개로 둔다.
  6. **기준 c의 판정 단위**는 요청 쌍 전체의 일치(7절)다. **기준 b**는 평균의 비다(짝지은 차이의 CI는 보고 항목 2에 따로 적는다).
  7. 두 Ours의 store는 각 arm의 자기 턴 1 캡처로 따로 만든다(3절). 따라서 gate 1은 캡처의 재현성까지 포함해 확인한다.
  8. **gate 2의 판정 범위** — 이것은 리허설 결과를 본 뒤 사용자가 정했다(2026-11-20, 본실험 freeze 전).
     처음에는 "질문 3개 → store가 같다"를 payload 바이트까지 같아야 하는 것으로 구현했다. 개발 리허설(gate 이미지 4장)에서 인덱스 표·행 수·byte table·파일 구성은
     세 질문에서 모두 같았지만 payload 바이트는 프롬프트 길이가 다른 질문 사이에서 달랐다(4장 모두). 원인을 따로 확인했다
     (`runs/mt_gqa_ours_order_dev_20261120T043319Z/diag_gate2_prompt_length.json`): 같은 프롬프트를 다시 돌리면 캡처된 Visual KV가 원소 단위로 같고,
     토큰 길이가 같은 서로 다른 질문끼리도 원소 단위로 같으며, 길이가 다르면 layer 1부터 FP16 원소의 약 70%가 다르다(layer 0은 같음; 최대 절댓값 차 0.47–1.17).
     같은 캡처를 재배열 순서로 쓴 store의 payload도 정확히 같은 경우에 같고 달랐다. 즉 턴 1 캡처의 성질이며 두 방식에 공통이다.
     이 내용을 사용자에게 알리고 세 가지(선택·배치 일치로 판정 / 바이트 일치 유지 후 중단 / FAIL로 기록하고 진행) 중 선택을 받았다: **선택·배치 일치로 판정, payload 바이트는 보고 항목.**
     6절 2번이 그 정의다. 판정 기준 a·b·c와 다른 gate는 바뀌지 않았다.
- 개발 리허설(`runs/mt_gqa_ours_order_dev_<UTC>`, 측정 아님)을 freeze 전에 돌려 스크립트 오류를 확인했다. 그 값은 판정과 표에 쓰지 않는다.
  리허설에서 gate 이미지 4장의 gate 1·3·4·5는 성립했고(두 Ours의 생성 토큰은 147건 모두 같았다), gate 2는 위 8번의 경위를 거쳤다. 이 사실들은 이 계약을 고정하기 전에 알려져 있었다.
