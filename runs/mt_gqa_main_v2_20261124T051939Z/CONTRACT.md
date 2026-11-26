# LLaVA-NeXT × MT-GQA 본실험 v2 계약 (공식 토큰열; ReComp / FullLoad / IMPRESS / Ours)

작성 2026-11-24. 이 문서는 gate와 본실험을 시작하기 전에 SHA256으로 동결한다
(`runs/mt_gqa_main_v2_<UTC>/config.json`의 `contract_sha256`, 사본 `CONTRACT.md`).
동결 뒤에는 방법 정의·gate 기준·통계 방법을 바꾸지 않는다. 측정 보고이며 GO/NO-GO 판정은 두지 않는다.

## 1. 배경과 목적

ConvBench 본실험의 0단계 점검(`results/convbench_main_20261122T103401Z/report.md` 11절)에서, LLaVA-NeXT의 MT-GQA 프롬프트 경로가
원본 LLaVA 절차(`tokenizer_image_token`, 느린 legacy 토크나이저)와 다른 토큰열을 만든다는 것이 확인됐다.
`<image>` 직후의 `▁`(29871) 토큰 하나가 빠져 있다(뽑은 60개 프롬프트 모두). LLaVA-1.5 경로에는 불일치가 없다.

이 경로를 원본 절차의 토큰열과 같게 고친 뒤, 네 방법을 한 실행에서 다시 측정한다. 이 실행의 결과가 논문의 LLaVA-NeXT × MT-GQA 행이 된다.

## 2. 고치는 것

- LLaVA-NeXT의 MT-GQA 프롬프트를 토크나이저에 넘길 때 `<image>` 뒤에 공백 하나를 둔다(`<image> \n`). 그러면 HF 토크나이저가 원본 절차와 같은 토큰 ID를 낸다.
  ConvBench 기준 정렬(`docs/convbench_baseline_alignment.md`, `scripts/213`의 `OFFICIAL_RENDER["next"]`)에서 쓴 렌더링과 같은 방식이다.
  구현은 새 파일 `mmimpress/mt_gqa_tokens.py`(`tokenization="official"`)다.
- 바꾸지 않는 것: 프롬프트 문자열(기록·해시·감사가 쓰는 문자열은 기존과 같은 `USER: <image>\n…ASSISTANT:`), 질문 순서, 이력 구성(자기 방법의 생성 답),
  생성 설정(greedy, cap 16), 채점(strict normalized EM). 바뀌는 것은 토크나이저 입력 문자열과 그 결과인 토큰열뿐이다.
- `<image>` 앞의 토큰과 이미지 구간은 두 경로에서 같다. 이미지 뒤의 모든 토큰이 한 자리씩 뒤로 간다.
- 기존 경로는 옵션(`tokenization="legacy"`)으로 남긴다. 기존 결과의 재현은 gate 3이 확인한다.
- LLaVA-1.5 경로(`mmimpress/llava15/`), 방법 코드(`mmimpress/serve.py`, `packed_serve.py`, `packed_store.py`, `piggyback.py`, `impress_official.py`),
  기존 스크립트·테스트는 수정하지 않는다. 이 작업은 새 파일만 더한다(10절). 기존 store, 결과 디렉터리, 동결 파일은 수정하지 않는다.
- 요청마다, 요청이 끝난 뒤 타이머 밖에서, 그 요청이 실제로 쓴 토큰 ID(픽셀 요청은 모델에 들어간 입력 ID, 저장 KV 요청은 store의 prefix ID + suffix ID)를
  원본 절차의 토큰 ID와 비교해 기록한다. `official`에서 하나라도 다르면 그 이미지를 중단한다(따라서 그 단계도 멈춘다).

## 3. 공통 조건

- 모델 `llava-hf/llava-v1.6-vicuna-7b-hf` revision `c916e6cdcd760b4cecd1dd4907f84ac649f93b23`, NF4 / BF16 / eager attention, FP16 SSD KV,
  greedy cap 16, batch 1, seed 1234, CPU thread 2, RTX 4090 1장 단독 사용.
- 환경: `/home/dblab/anaconda3/envs/mllm_ft/bin/python`, transformers 4.57.6, torch 2.5.1+cu121, numpy 2.2.6 — 11/17·11/20 실행과 같다. freeze가 그 실행들의 기록과 대조하고 다르면 멈춘다.
- 모든 방법은 매 대화 T1을 픽셀부터 계산한다(저장 KV의 warm-up 없음). T2–T3는 자기 방법의 생성 이력을 쓴다.
- cold SSD: 저장 KV를 읽는 요청 직전에 그 store 파일의 page cache를 비운다(`posix_fadvise(DONTNEED)`, TTFT 밖).
- 저장된 KV를 읽는 세 방법은 모두 비동기 읽기다. 미리 읽기는 TTFT 타이머가 시작된 뒤에만 한다.
- TTFT 경계: 요청 시작(프롬프트 조립·토큰화 직전) → 첫 토큰 argmax 직후 `cuda.synchronize`. 기존 실행과 같다(같은 요청 함수).
- store는 이미지마다 그 이미지의 첫 대화 T1에서 한 번 만들고(각 store 종류를 쓰는 arm 중 회전 순서상 첫 arm의 T1 캡처), 그 이미지의 모든 대화가 쓴다.
  다른 대화의 T1은 픽셀부터 계산해 답만 낸다. 기존 본실험과 같은 프로토콜이다.

## 4. 방법 정의

1. **ReComp**: 매 턴 이미지부터 다시 계산한다.
2. **FullLoad**: `results/mt_gqa_impress_code_20261117T185832Z`의 구현 그대로(전체 Visual KV, raster 순서, 워커가 layer 순서로 K·V 전체를 읽음).
3. **IMPRESS**: 같은 실행의 구현과 설정 그대로(`docs/mt_gqa_impress_code_contract.md`, `mmimpress/impress_official.py`): 전체 Visual KV + probe key 사본,
   probe head 0·1·2, 전체 행 열 합 중요도, 유지 25%, 평균 Jaccard ≥ 0.3이면 투표 선택 아니면 fallback, K/V 분리, 64토큰 chunk,
   첫 대화 T1 통계로 layer별 재배치, 다음 layer probe key 미리 읽기.
4. **Ours**: `results/mt_gqa_ours_order_20261120T052725Z`의 "Ours(원래 순서)" 구현 그대로. PYR-beta-raw 벡터
   (`runs/layer_budget_kv25_pyrbeta_20261110T073604Z/vectors/layer_budget_llava_PYR-beta-raw.json`, ratio SHA256 `51f9b0db…`), 총 KV 예산 25%,
   packed 저장, `row_order="original"`(원래 토큰 인덱스 오름차순으로 저장하고 원래 cache 위치에 올림), 중요도 순 재배열 없음.

구현이 기준 실행과 같다는 것은 두 가지로 확인한다. (a) 실행기 `scripts/222`는 `scripts/209`(11/17의 `scripts/196`을 import하고 11/20의 `scripts/202`와 같은 호출을 쓰는 하니스)를
import해 요청·캡처·저장·활성화 함수를 그대로 부른다. (b) freeze가 11/17·11/20 freeze의 소스 해시와 대조해, 달라진 파일이 그 뒤의 알려진 변경
(11/20의 `row_order` 옵션: `mmimpress/packed_store.py`, `packed_serve.py`, `piggyback.py`; 11/21의 LLaVA-1.5 이식: `mmimpress/llava15/store.py`, `serve.py`)뿐이 아니면 멈춘다.

| 표의 이름 | method id | store 종류 |
|---|---|---|
| ReComp | `recompute` | 없음 |
| FullLoad | `fullload_async` | `raster` |
| IMPRESS | `impress_official_async` | `impress_official` |
| Ours | `ours_origorder_kv25_PYR-beta-raw_packed_async` | `packedorig_PYR-beta-raw` |

gate 판정용 arm(본실험 표에 넣지 않는다): IMPRESS 동기 읽기 `impress_official`, IMPRESS 유지 100% `impress_official_r100`, 같은 재배치 store를 읽는 FullLoad loader `fullload_reordered`,
Ours 동기 읽기 `ours_origorder_kv25_PYR-beta-raw_packed`, Ours 두 번째 실행 `…_packed_async_repeat`.

## 5. 실험

- 데이터: MT-GQA reconstructed 398 images / 4,061 dialogues × 3턴(INDEX_SHA `2c47cfad…`, WORKLOAD_SHA `0287e0c5…`). 대화·이미지가 11/17·11/20 실행의 manifest와 같음을 freeze에서 확인한다.
- 본실험 arm 4개 × 12,183요청 = 48,732요청. arm 순서는 대화의 데이터셋 내 순번으로 회전한다(`순번 % arm 수`). arm 목록의 순서는 ReComp, FullLoad, IMPRESS, Ours다.
- 지표: followup(T2–T3) strict normalized EM, 턴별 EM, TTFT, SSD 읽기 바이트, 읽기 호출 수. SSD 바이트·호출 수는 read syscall이 실제로 돌려준 값이다.
- 통계: 이미지 단위 paired bootstrap 10,000회, seed 1234, 95% CI(`mmimpress.recompute_kv25.paired_bootstrap`).
- 결과 디렉터리: `results/mt_gqa_main_v2_<UTC>/`(보고서·집계·status.json·driver.log·PID 파일). 요청 단위 원본과 증거 파일은 기존 실행들과 같이 `runs/mt_gqa_main_v2_<UTC>/`에 둔다.

| 단계 | 이미지 | 토큰열 | arm | 요청 수 |
|---|---|---|---|---:|
| `legacy` | gate 4장(49 dialogues) | 기존 경로 | ReComp, FullLoad, IMPRESS, Ours | 588 |
| `gate` | gate 4장 | 공식 | 본실험 4 + IMPRESS(동기), IMPRESS(유지 100%), FullLoad loader(재배치 store), Ours(동기), Ours(두 번째 실행) | 1,323 |
| `devgate` | DEV 40장(378 dialogues) | 공식 | IMPRESS(동기), IMPRESS, IMPRESS(유지 100%), FullLoad loader(재배치 store), Ours(동기), Ours, Ours(두 번째 실행) | 7,938 |
| `dev` | DEV 40장 | 공식 | ReComp, FullLoad, IMPRESS, Ours | 4,536 |
| `main` | 398장 | 공식 | ReComp, FullLoad, IMPRESS, Ours | 48,732 |

gate 4장은 11/17·11/20 실행의 gate 이미지, DEV 40장은 `results/layer_budget_kv25_v2_20261106T073502Z/data_split.json`의 DEV 이미지다.
legacy·gate·devgate·dev 요청은 본실험 분모에 넣지 않는다. 그 이미지들은 본실험에 다시 포함된다.

## 6. Gate (본실험 전, 4장 → DEV 40장 순서)

1. **토큰열.** 본실험의 프롬프트 12,183개(대화 4,061 × 3턴) 전부에서, 고친 경로의 토큰열이 원본 LLaVA 절차의 토큰열과 일치한다.
   T2·T3 프롬프트는 생성 이력에 따라 정해지므로 본실험 전에는 기존 본실험이 실제로 보낸 프롬프트로 판정한다:
   11/20 실행의 ReComp·FullLoad·Ours와 11/17 실행의 IMPRESS, 방법마다 12,183개, 합 48,732개가 모두 일치해야 한다.
   같은 프롬프트에서 기존 경로의 토큰열이 그 실행들이 기록한 suffix 토큰열과 같아야 한다(비교 대상이 실제로 쓰인 경로임을 확인).
   본실험에서는 요청마다 다시 비교하며 불일치가 나오면 멈춘다(2절). 본실험 요청 전체의 일치 건수는 보고 항목이다.
2. **LLaVA-1.5 경로의 토큰열이 고치기 전과 같다(프롬프트 60개).** LLaVA-1.5 × MT-GQA 본실험(`runs/llava15_mt_gqa_main_20261120T165001Z`)에서 고르게 뽑은 이미지 20장의
   첫 대화 T1·T2·T3(ReComp가 보낸 프롬프트)에 대해, 지금의 코드로 다시 만든 프롬프트와 그 토큰 ID(이미지 뒤 ID, 이미지 앞 ID 수)가 그 실행의 기록과 같고,
   그 실행이 동결한 소스 파일이 모두 지금도 같다.
3. **기존 경로 옵션으로 돌리면 11/20 실행의 ReComp 생성 토큰이 재현된다(gate 4장).** `legacy` 단계의 ReComp 147건이 11/20 실행의 같은 요청(같은 대화, 같은 턴)과 모두 같다.
   같은 단계의 FullLoad·Ours(11/20 대비)와 IMPRESS(11/17 대비)는 참고로 보고한다(판정 아님).
4. **Ours.** 동기 읽기와 비동기 읽기의 생성 토큰·읽기 바이트(전체, packed)가 모든 요청에서 일치하고, 같은 요청을 두 번 돌려 생성 토큰이 모든 요청에서 같다.
5. **IMPRESS.** 유지 비율 100%일 때의 생성 토큰이, 같은 재배치 store를 읽는 FullLoad loader의 생성 토큰과 모든 요청에서 일치한다.
   비동기와 동기의 생성 토큰·읽기 바이트(전체와 종류별)가 모든 요청에서 일치한다.
6. **T1 생성 토큰이 네 방법에서 모두 같다**(대화마다 ReComp, FullLoad, IMPRESS, Ours의 T1 생성 토큰열이 같다).
7. **기존 테스트에서 새 실패가 0개다**(이 작업 전부터 실패하던 테스트는 목록만 적는다). 실행 환경과 교차 확인용 5.13.1 환경 두 곳에서 돌린다. 새 테스트 파일은 통과해야 한다.
   이 작업 전부터 실패하던 테스트: transformers 4.57.6 환경의 GLM-4.6V 테스트 3개(`test_glm46v_runner` setUpClass, `test_glm46v_validation…partial_rope…`, `test_glm46v_vision…sdpa_hook…`);
   5.13.1 환경의 vision-capture 테스트 2개(`test_qwen25_vision…stock_forward_capture…`, `test_visdial_turn1_piggyback_core…real_tiny_clip…`)와
   11/22에 만든 `tests/test_convbench_main.py`의 2개(`FrozenSettings.test_official_token_check_on_both_tokenizers`, `ReferenceHarness.test_reference_checks_pass_and_source_is_unchanged`:
   실행 환경이 4.57.6 환경임을 assert하거나 4.x의 느린 legacy 토크나이저 동작에 기대는 테스트). 뒤의 2개는 2026-11-24에 이 작업의 새 파일을 모두 치운 상태에서 같은 실패를 확인했다.
8. **DEV 40장 파일럿**을 4 arm으로 돌려 표를 만든다(40장 모두 이미지 감사 PASS로 commit).

판정 범위: gate 1·2·7은 공통(CPU). gate 3은 4장. gate 4·5는 4장(`gate` 단계)과 DEV 40장(`devgate` 단계). gate 6은 4장(`gate` 단계의 본실험 arm 4개)과 DEV 40장(`dev` 단계).

- 기존 실행과 생성 토큰이 다른 것은 gate 실패가 아니다(토큰열이 바뀌었다). 보고 항목이다.
- gate 1–7이 통과하고 DEV 파일럿이 실행되면 드라이버가 본실험까지 이어서 진행한다.
- gate가 실패하면 드라이버는 본실험을 시작하지 않고, 실패 내용과 원인을 `status.json`과 `report.md`에 적은 뒤 종료한다.
  4장 단계에서 실패하면 DEV 단계도 시작하지 않는다. 기준을 임의로 바꾸지 않는다. 판정이 기록된 gate는 다시 판정하지 않는다.
- gate 검사 프로그램이 결과를 내지 못하고 끝난 경우(판정이 아닌 실행 오류)와 요청 단계가 끝나지 못한 경우는 FAIL로 확정하지 않고 오류를 기록한 뒤 멈추며,
  같은 명령으로 그 지점부터 다시 실행된다. 이 경우에도 본실험은 gate가 모두 PASS로 기록되기 전에는 시작하지 않는다.
- 모든 단계에서 이미지별 독립 감사가 실패하면 그 이미지는 commit되지 않고 실행이 멈춘다(기존 실행의 감사 항목 + 2절의 토큰열 항목).

## 7. 실행 방식

- gate → DEV → 본실험 → 보고서 생성을 드라이버 하나(`scripts/run_mt_gqa_main_v2.sh runs/mt_gqa_main_v2_<UTC>`)가 터미널·세션과 분리해 실행한다(setsid + nohup).
- 드라이버에는 사람의 입력을 기다리는 지점이 없다. 같은 드라이버가 두 개 뜨지 않게 잠금 파일(`driver.lock`, flock)을 둔다.
- 상태 기록(`results/<run>/`): `status.json`(현재 단계, 완료 요청 수 / 전체 요청 수, 예상 종료 시각, 누적 SSD 쓰기량, 마지막 오류), `driver.log`, `driver.pid`.
- 이어서 실행: 같은 명령을 다시 실행하면 이어서 돈다.
  - commit된 이미지와 완료된 대화는 건너뛴다. 끊긴 대화는 그 대화의 T1부터 다시 한다. 끊긴 대화의 불완전한 기록은 결과에서 버린다(증거로는 남긴다).
  - store는 이미지의 첫 대화 T1에서 만들므로, 끊긴 대화가 그 이미지의 첫 대화이거나 디스크의 store가 만들 때 기록한 SHA256과 맞지 않으면
    그 이미지의 store와 기록을 버리고 이미지의 첫 대화부터 다시 한다.
  - 이어서 실행한 횟수, 각 시점, 다시 실행한 대화 목록을 `driver_invocations.jsonl`·`resume_log.jsonl`에 남기고 보고서에 적는다.
- 실험이 도는 동안 같은 GPU나 SSD를 쓰는 다른 작업(단위 테스트 포함)을 돌리지 않는다. 하니스는 이미지 경계마다 GPU에 다른 프로세스가 있으면 그 이미지를 중단한다.

## 8. 보고 항목 (`results/mt_gqa_main_v2_<UTC>/report.md`, 드라이버가 끝에서 자동 생성; 측정하지 않은 값은 "미실행")

1. 본실험 표: 방법별 followup EM, T1·T2·T3 각각의 EM, 전체 EM(T1–T3 요청 전체), TTFT 평균/p50/p95, SSD MB/followup, 읽기 호출 수.
2. 짝지은 비교(95% CI, 정답률과 TTFT): Ours − IMPRESS, Ours − FullLoad, Ours − ReComp, IMPRESS − FullLoad, FullLoad − ReComp.
3. TTFT 비율: Ours/FullLoad, Ours/ReComp, Ours/IMPRESS, IMPRESS/FullLoad.
4. 기존 결과와의 대조: 방법별로 기존 실행(ReComp·FullLoad·Ours는 11/20, IMPRESS는 11/17)과 이번 실행의 followup EM과 TTFT, 생성 토큰이 달라진 요청 수, 정오가 바뀐 요청 수.
   Ours가 고른 layer별 토큰 목록이 기존과 같은지(이미지마다 Ours store의 인덱스 표와 layer별 행 수를 11/20 실행의 것과 비교).
5. MetaCompress 기본 모델(Acc1 63.83, Acc2 66.68, Acc3 67.94, 평균 66.15)과 이번 ReComp의 턴별 EM 대조.
6. IMPRESS 통계: fallback 비율, 평균 Jaccard, K·V chunk 읽기 비율(전체 / fallback이 아닌 layer만 / 재배치하지 않았을 경우의 계산값), 중요도 중 prefix 행이 차지하는 비율.
7. 재사용 턴 시간 분해: 읽기 대기와 GPU 적재 시간.
8. T1 저장 비용: 방법별 이미지당 저장 시간과 저장 크기, 실험 전체의 SSD 쓰기량.
9. 추가하거나 바꾼 코드 목록, 지시와 다르게 했거나 지시에 없던 것, 측정하지 않은 것과 한계.

보고서에서 방법은 ReComp, FullLoad, IMPRESS, Ours로만 부른다. 옵션 이름이나 클래스 이름을 방법 이름으로 쓰지 않는다.

## 9. 사전에 밝혀 두는 것 (지시에 없던 것, 지시와 다르게 되는 경우)

1. **gate 1의 프롬프트 집합**(6절 1번): 본실험 전에는 기존 본실험이 보낸 프롬프트 48,732개로 판정하고, 본실험에서는 요청마다 비교해 불일치 시 멈춘다.
2. **gate 2**는 토큰 ID 대조(60개)에 더해 LLaVA-1.5 본실험이 동결한 소스 파일이 그대로인지도 판정한다.
3. **gate 3의 `legacy` 단계**는 네 방법을 모두 돌린다. 판정은 ReComp만 한다.
4. **gate 4·5의 DEV 판정**은 별도 단계(`devgate`, 7 arm)로 돈다. DEV 파일럿(gate 8)을 본실험과 같은 4 arm으로 유지하기 위해서다.
5. 요청 단위 원본은 `runs/`에, 보고서·상태 파일은 `results/`에 둔다(5절).
6. 드라이버는 요청 단계가 오류로 끝나면 같은 실행 안에서 최대 3회까지 스스로 다시 시작한다(그때마다 이어서 실행으로 기록한다).
   토큰 ID 불일치, 이미지 감사 실패, 디스크 부족처럼 다시 해도 같은 곳에서 멈출 오류는 다시 시작하지 않고 멈춘다.
7. 드라이버는 GPU를 쓰는 단계를 시작하기 전에 다른 프로세스가 GPU를 쓰고 있으면 비워질 때까지 기다린다(30초 간격, 최대 2시간, 넘으면 오류를 기록하고 멈춘다).
8. 이어 실행에서 이미지 전체를 다시 하는 경우(7절)에는 그 이미지에서 이미 끝났던 대화도 다시 실행된다.
9. 모델을 올린 직후 합성 이미지 요청 1건으로 프로세스를 예열한다(기존 실행과 같음, 모든 지표에서 제외). 저장 KV의 warm-up은 없다.
10. FullLoad store에는 11/17 구현 그대로 probe key 사본이 함께 저장된다(읽지 않는다).
11. TTFT 비율과 기존 결과 대조의 EM 차이에 이미지 단위 bootstrap 95% CI를 함께 적는다. 기존 실행과의 TTFT는 서로 다른 날의 값이라 나란히 적기만 한다.
12. status.json의 예상 종료 시각은 근사값이다.
13. SMART(percentage used 등)는 root 권한이 없어 읽지 못한다. SSD 쓰기량은 커널의 장치 단위 쓰기 섹터 수(`/sys/block/nvme0n1/stat`)로 적는다.
14. 개발 리허설(측정 아님)을 freeze 전에 두 번 돌려 스크립트 오류와 이어 실행을 확인했다(단계당 이미지 1장, 본실험 단계 2장).
    확인한 것: 대화 도중 중단 → 완료된 대화 2개 유지, 끊긴 대화의 T1부터 재개; 이미지의 첫 대화 도중 중단 → 그 이미지를 첫 대화부터 다시; 잠금 파일이 두 번째 드라이버를 막음;
    gate가 끝나지 않은 상태와 본실험 완료 상태의 보고서 생성. 리허설 디렉터리는 확인 뒤 삭제했다. 그 값은 판정과 표에 쓰지 않는다.
    이 계약을 동결하기 전에 알려져 있던 것: gate 1의 계산을 기존 프롬프트 48,732개 전체에 미리 해 보았고 모두 일치했다(기존 경로는 0개 일치, 차이는 모두 `<image>` 직후의 `▁` 하나);
    gate 2의 계산도 60개 모두 일치했다; gate 이미지 1장에서 gate 3–6이 성립했고 기존 경로 옵션으로 네 방법 모두 기존 생성 토큰을 재현했다;
    그 이미지 1–2장의 정답률·TTFT를 봤다.
15. gate 7의 "이전부터 실패하던 테스트"에는 5.13.1 환경에서 `tests/test_convbench_main.py`의 2개가 더 있다(6절 7번). 기존 실행들의 목록(4.57.6에서 3개, 5.13.1에서 2개)보다 2개 많다.

## 10. 코드 변경 범위와 보호

- 새 파일: `mmimpress/mt_gqa_tokens.py`, `scripts/222_eval_mt_gqa_main_v2.py`, `scripts/223_validate_mt_gqa_main_v2.py`, `scripts/224_report_mt_gqa_main_v2.py`
  (보고; 기록만 읽으므로 소스 freeze에서 제외), `scripts/run_mt_gqa_main_v2.sh`, `tests/test_mt_gqa_main_v2.py`, 이 문서.
- 수정한 기존 파일: 없음.
- 소스 freeze: freeze 때 `mmimpress/**`, `scripts/*.py`(보고 스크립트 `scripts/<번호>_report_*.py` 제외), `tests/test_*.py`의 SHA256을 `config.json`에 기록한다.
  각 단계와 이어 실행은 그때 기록한 파일이 바뀌었으면 멈춘다. freeze 뒤에 새로 생긴 파일은 비교 대상이 아니다.
- 저장 계획: 이미지 하나씩 store 생성 → 측정 → 감사 → commit → run-local scratch 삭제(정확한 SHA256 allowlist). 삭제 범위는 `runs/<RUN>/scratch/**`뿐이다.
  예상 SSD 쓰기량은 11/17 본실험이 같은 이미지에서 잰 store 크기로 freeze가 계산해 `storage_plan.json`과 `results/<run>/prerun.json`에 기록한다
  (이미지당 FullLoad store 1회 + IMPRESS store 2회(쓰기 + 재배치 재기록) + Ours store 1회; `devgate`는 FullLoad store 없음).
- freeze 때 `results/`, `runs/`(이 run 제외), `kvstore*`의 모든 파일 (경로, 크기, mtime_ns) 목록과 그 SHA256을 남기고 보고 때 다시 비교한다.
- 커밋과 푸시는 하지 않는다.

## 11. 하지 않는 것

- 398장 전체를 기존 경로로 다시 돌리기, 본실험의 반복 실행, 본실험에서의 동기 arm, warm cache, 25% 외 예산, 다른 벡터, 다른 모델·데이터셋.
- LLaVA-NeXT × MT-VQA 경로의 수정·재실행(같은 토큰 결함이 있으나 이 작업의 범위가 아니다).
- O_DIRECT·io_uring, 워커 여러 개, SMART·NAND 수준 측정.
