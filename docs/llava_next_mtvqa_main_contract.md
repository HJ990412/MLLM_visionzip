# LLaVA-NeXT × MT-VQA 25k 본실험 계약 (FullLoad / IMPRESS / Ours 측정, ReComp는 v3 기록)

작성 2026-11-21. 이 문서는 실행 전 점검·gate·본실험을 시작하기 전에 SHA256으로 동결한다
(`runs/llava_next_mtvqa_main_<UTC>/config.json`의 `contract_sha256`, 사본 `CONTRACT.md`).
동결 뒤에는 방법 정의·gate 기준·통계 방법을 바꾸지 않는다. 측정 보고이며 GO/NO-GO 판정은 두지 않는다.

## 1. 목적

두 모델 × 두 데이터셋 표의 마지막 칸(LLaVA-NeXT × MT-VQA)을 채운다. 방법 구현은 모두 이미 있고 이 작업에서 방법 코드는 바꾸지 않는다.
새로 만드는 것은 이 모델·데이터셋 조합을 돌리는 실행기와 보고기다. FullLoad, IMPRESS, Ours를 25,000장 전체에서 한 실행 안에서 잰다.
ReComp는 SSD를 읽지 않아 방법이 그대로이므로 다시 재지 않고 v3 실행(`runs/mt_vqa_v2_reconstructed_25k_v3_20261021T100500Z`)의 요청별 기록을 쓴다.

보고서에서 방법은 ReComp, FullLoad, IMPRESS, Ours로만 부른다. 옵션 이름이나 클래스 이름을 방법 이름으로 쓰지 않는다(method id는 이 문서 3절에만 적는다).

## 2. 공통 조건

- 모델 `llava-hf/llava-v1.6-vicuna-7b-hf` revision `c916e6cdcd760b4cecd1dd4907f84ac649f93b23`, NF4 / BF16 / eager attention, FP16 SSD KV,
  greedy cap 16, batch 1, seed 1234, CPU thread 2, 기존 AnyRes 전처리. RTX 4090 1장 단독 사용.
- 환경: `/home/dblab/anaconda3/envs/mllm_ft/bin/python`, transformers 4.57.6, torch 2.5.1+cu121, numpy 2.2.6
  (11/17·11/20 실행과 같은 환경; freeze가 그 실행들의 기록과 대조하고 다르면 멈춘다).
- 이미지당 대화 1개 × 3턴. 모든 방법은 T1을 픽셀부터 계산한다(저장 KV의 warm-up 없음). T2–T3는 자기 방법의 생성 이력을 쓴다.
- cold SSD: 저장 KV를 읽는 요청 직전에 그 store 파일의 page cache를 비운다(`posix_fadvise(DONTNEED)`, TTFT 밖).
- 저장된 KV를 읽는 세 방법은 모두 비동기 읽기다. 백그라운드 워커 1개가 디스크를 읽고 GPU 적재는 요청 스레드가 한다. 읽기는 TTFT 타이머가 시작된 뒤에만 한다.
- store는 이미지마다 한 번, 그 store 종류를 쓰는 arm 중 회전 순서상 첫 arm의 T1 캡처로 만든다. 다른 arm의 T1은 픽셀부터 계산해 답만 낸다.
- TTFT: 타이머는 프롬프트 조립(이력, 이미지 파일 열기·디코드, 전처리) 직전에 시작해 첫 출력 토큰 직후에 끝난다. v3 기록의 ReComp TTFT와 같은 구간인지는 실행 전 점검에서 확인해 적는다(6절).

## 3. 방법 정의

| 표의 이름 | method id (본실험) | store | 구현의 출처 |
|---|---|---|---|
| ReComp | (v3 기록의 `recompute`) | 없음 — 매 턴 이미지부터 다시 계산 | v3 실행의 요청별 기록. 본실험에서는 돌리지 않고 파일럿 40장에서만 돌린다(`recompute`). |
| FullLoad | `fullload_async` | `raster`: Visual KV 전체, raster 순서 | `results/mt_gqa_impress_code_20261117T185832Z`의 FullLoad 그대로 |
| IMPRESS | `impress_official_async` | `impress_official`: Visual KV 전체 + probe key 사본, T1 통계로 layer별 재배치 | 같은 실행의 IMPRESS(`mmimpress/impress_official.py`)와 설정 그대로 |
| Ours | `ours_origorder_kv25_PYR-beta-raw_packed_async` | `packedorig_PYR-beta-raw`: 총 KV 예산 25% | `results/mt_gqa_ours_order_20261120T052725Z`의 "Ours(원래 순서)" 그대로 |

1. **FullLoad**: Visual KV 전체를 저장하고 전부 읽는다. `Server.request(mode="fullload", prefetch="layer_stream")`, 워커가 layer 순서로 K·V 전체를 읽는다.
2. **IMPRESS**: 정의는 `docs/mt_gqa_impress_code_contract.md` 5절을 따른다(공개 구현 HyperInfer `91df7a2` 방식): probe head 0·1·2, 전체 행(prefix + suffix)의 attention 열 합 중요도,
   상위 25%(k = ceil(0.25·N)), 세 head의 평균 pairwise Jaccard ≥ 0.3이면 투표 선택, 아니면 fallback으로 그 layer 전체를 읽어 mask 없이 사용, K/V 분리 저장, 64토큰 chunk,
   T1 요청의 통계로 layer별 [separator | 중요도 내림차순] 재배치, 다음 layer probe key 미리 읽기(`next_layer`). 원본과 다른 점의 표도 그 문서 6절과 같다.
3. **Ours**: image-only CLIP CLS→patch 중요도(질문을 쓰지 않음), PYR-beta-raw 벡터
   (`runs/layer_budget_kv25_pyrbeta_20261110T073604Z/vectors/layer_budget_llava_PYR-beta-raw.json`, ratio SHA256 `51f9b0db…`, 하한 없음), 총 KV 예산 25%
   (Σ k_l = 32·ceil(N/4)). layer l은 중요도 상위 k_l개 토큰을 **원래 토큰 인덱스 오름차순**으로 저장하고(`row_order="original"`, 중요도 순 재배열 없음),
   재사용 턴에서 각 행을 GPU cache의 원래 위치에 올리고 저장하지 않은 위치는 mask로 가린다. packed 파일 1개, 워커가 구간 순서로 읽는다(`layer_stream`).

방법 코드가 출처 실행과 같다는 것은 freeze에서 확인한다: 11/17 freeze의 소스 해시와 비교해 달라진 파일이
`mmimpress/packed_store.py`, `packed_serve.py`, `piggyback.py`(11/20의 원래 순서 옵션 추가; 11/20 실행이 FullLoad의 생성 토큰이 11/17과 12,183건 모두 같음을 확인)와
`mmimpress/llava15/store.py`, `serve.py`(LLaVA-1.5 전용, 이 실험이 쓰지 않음)뿐이어야 하고, 11/20 freeze와 비교해서는 `mmimpress/llava15/` 두 파일뿐이어야 한다. 그 밖의 파일이 달라졌으면 freeze가 멈춘다.
실행기는 `scripts/196_eval_mt_gqa_impress_code.py`를 import해 캡처·재배치·읽기 추적 함수를 그대로 부르고, FullLoad·IMPRESS 요청 호출과 store writer 호출의 인자는 그 스크립트와 같다.
Ours의 writer 호출과 요청 호출의 인자는 `scripts/202_eval_mt_gqa_ours_order.py`와 같다(단위 테스트가 호출 인자를 대조한다). 다른 것은 store 메타데이터에 적는 데이터셋 이름뿐이다.

검증 전용 arm(gate 4장 단계에만, 어떤 표에도 넣지 않는다): IMPRESS 동기 읽기(`impress_official`), IMPRESS 유지 100%(`impress_official_r100`, 동기),
같은 재배치 store를 선택·mask 없이 읽는 FullLoad loader(`fullload_reordered`, 동기), Ours 동기 읽기(`ours_origorder_kv25_PYR-beta-raw_packed`), Ours 두 번째 실행(`…_async_repeat`).

## 4. 데이터·프롬프트·채점

`docs/mt_vqa_v2_reconstructed_25k_contract.md`, `docs/llava_next_mtvqa_pyr_contract.md`와 같다.

- MT-VQA-v2-reconstructed-25k: 이미지 25,000장, 이미지당 대화 1개, 3턴 → 방법당 75,000요청. Ordered workload SHA256 `eb3c5e81…269cc8`
  (`data/mt_vqa_v2_reconstructed_25k/dialogues.json`에서 다시 계산하고 v3 main manifest의 dialogues와 같은지 확인). 이미지 파일 25,000개의 SHA256을 freeze에서 확인한다.
- 프롬프트: `USER: <image>\n[Q1.. A1.. 이력]Current question Q{t}: … Answer the question using a single word or phrase. ASSISTANT:` (v3와 같은 `prompt_factory`). 이력의 답은 자기 방법이 생성한 답이다.
- 채점: 공식 evaluator(GT-Vision-Lab/VQA `a013f0043c1e2cdc995922dfe257f7149aa9af06`, `mmimpress/mtvqa25k/scorer.py`)의 ten-answer leave-one-out VQA score.
  주 지표는 followup(T2–T3) 기준이다. 조건부 정규화(사람 답 10개가 모두 같으면 정규화하지 않음)를 그대로 따른다.

## 5. 실험 단계

실행 순서: 실행 전 점검 → gate(4장) → 파일럿(40장) → 본실험 → 보고서.

| 단계 | 이미지 | arm | 요청 수 |
|---|---|---|---:|
| gate | v3 smoke의 4장 | ReComp, FullLoad, IMPRESS(동기), IMPRESS, IMPRESS(유지 100%), FullLoad loader(재배치 store), Ours(동기), Ours, Ours(두 번째 실행) | 108 |
| pilot | v3 pilot40의 40장 | FullLoad, IMPRESS, Ours, ReComp | 480 |
| main | 25,000장 전체 (manifest 순서) | FullLoad, IMPRESS, Ours | 225,000 |

- gate·파일럿 요청은 본실험 분모에 넣지 않는다. gate·파일럿 이미지는 본실험에 다시 포함된다.
- arm 순서 회전: 한 이미지의 arm 순서 = 그 단계의 arm 목록을 (단계 안에서의 이미지 순번 % arm 수)만큼 회전한 것. 본실험은 3 arm, 파일럿은 4 arm, gate는 9 arm 기준이다.
- ReComp의 score와 TTFT는 v3 기록의 본실험(main) 단계 요청별 결과를 쓴다. 짝지은 비교는 같은 이미지·같은 턴끼리 짝지어 낸다.
- 통계: 이미지 단위 paired bootstrap 10,000회, seed 1234, 95% CI(`mmimpress.recompute_kv25.paired_bootstrap`; 이미지당 followup 2건의 합을 단위로 재표집).
- 결과 디렉터리: `results/llava_next_mtvqa_main_<UTC>/`(prerun.json, status.json, driver.log, PID 파일, 집계, report.md). 요청 단위 원본은 `runs/llava_next_mtvqa_main_<UTC>/`.
- 기존 store, 결과 디렉터리, 동결 파일은 수정하지 않는다. freeze 때와 보고 때 `results/`·`runs/`·`kvstore*`(이 실행 제외)의 파일 목록(경로, 크기, mtime)을 비교해 적는다.

이미지 하나의 수명주기: 세 방법의 T1(픽셀) → 각 store 쓰기 → T2·T3(cold 읽기) → 이미지 감사 → commit → **그 이미지의 store 삭제**. 동시에 존재하는 것은 한 이미지의 store 묶음뿐이다.
이미지 감사는 11/17·11/20 실행의 감사 항목(요청·이력·프롬프트 재구성, read syscall과 방법별 읽기 계약의 일치, store 구성, IMPRESS의 layer별 결정과 실제 읽기의 일치 등)에
공식 VQA score의 독립 재계산을 더한 것이다. 감사가 통과하지 못한 이미지는 commit되지 않는다.

## 6. 실행 전 점검 (`results/<run>/prerun.json`에 기록; 어떤 GPU 측정보다 먼저)

- 예상 SSD 쓰기량: v3 기록의 이미지별 geometry(visual token 수, separator 수)로 계산한 이미지당 store 쓰기량의 합
  (이미지당 FullLoad store + IMPRESS store 쓰기와 재배치 재기록 + Ours store). 예상 소요 시간, 여유 공간.
- SSD의 SMART 값(percentage used, data units written, 온도). 읽을 수 없으면 읽지 못한 이유와 root 없이 읽히는 값(NVMe 센서 온도, 커널의 장치 쓰기 카운터)을 적는다.
- **여유 공간이 이미지 하나의 store 묶음 최대 크기의 20배보다 작으면 시작하지 않는다.** 같은 기준을 매 이미지 시작 때 다시 확인한다.
- v3 기록에 ReComp의 요청별 결과(생성 토큰, score, TTFT)가 25,000장 × 3턴 모두 있는지 확인한다(v3의 이미지별 journal을 모두 열어 commit의 SHA256과 대조). **빠진 것이 있으면 시작하지 않고 보고한다.**
- v3 기록의 TTFT 측정 구간(타이머 시작과 끝)이 지금 실행기와 같은지 확인해 적는다: v3 실행이 동결해 둔 소스 사본과 지금 소스의
  `Server.recompute`(첫 토큰 시각), `normal_request`(타이머 시작), 요청 기록의 TTFT 필드를 문장 단위로 대조한다.

실행 전 점검이 막히면 드라이버는 아무것도 측정하지 않고 status.json과 report.md에 내용을 적은 뒤 종료한다.

## 7. Gate

1. **기존 테스트**에서 새 실패가 0개다(이전부터 실패하던 테스트는 목록만 적는다: transformers 4.57.6 환경의 GLM-4.6V 테스트 3개, 5.13.1 환경의 vision-capture 테스트 2개).
   실행 환경과 교차 확인용 5.13.1 환경 두 곳에서 돌린다. 새 테스트 파일(`tests/test_llava_next_mtvqa_main.py`)은 통과해야 한다.
2. **Ours**: 동기 읽기와 비동기 읽기의 생성 토큰·읽기 바이트가 일치하고(모든 요청), 같은 요청을 두 번 돌려 생성 토큰이 같다(모든 요청). — gate 4장
3. **IMPRESS**: 유지 비율 100%일 때의 생성 토큰이, 같은 재배치 store를 읽는 FullLoad loader의 생성 토큰과 일치한다(모든 요청).
   비동기와 동기의 생성 토큰·읽기 바이트(전체와 종류별)가 일치한다(모든 요청). — gate 4장
4. **연속성**: 파일럿 40장에서 ReComp를 함께 돌려, 생성 토큰이 v3 기록(본실험 단계)의 같은 요청과 모두(120건) 일치한다.
   세 방법의 T1 생성 토큰도 v3 기록의 같은 이미지 T1 토큰과 모두(방법당 40건) 일치한다.
5. **채점기**가 기존 검증(`scripts/135_validate_mtvqa25k_scorer.py`)을 통과한다(종료 코드 0, status PASS).
6. **파일럿 40장 표**를 만든다(FullLoad, IMPRESS, Ours, ReComp). 판정 항목이 아니다.

- 실행 전 점검과 gate 1–5가 통과하면 드라이버가 본실험까지 이어서 진행한다.
- gate가 실패하면 드라이버는 본실험을 시작하지 않고, 실패 내용과 원인을 status.json과 report.md에 적은 뒤 종료한다. 기준을 임의로 바꾸지 않는다. 판정이 기록된 gate는 다시 판정하지 않는다.
- gate 검사 프로그램이 결과를 내지 못하고 끝난 경우(판정이 아닌 실행 오류)와 요청 단계가 끝나지 못한 경우는 FAIL로 확정하지 않고 오류를 기록한 뒤 멈추며,
  같은 명령으로 그 지점부터 다시 실행된다. 이 경우에도 본실험은 gate가 모두 PASS로 기록되기 전에는 시작하지 않는다.
- 본실험 중에는 세 방법의 T1 생성 토큰이 v3 기록의 같은 이미지 T1 토큰과 일치하는지 이미지마다 확인한다. 불일치는 실행을 멈추지 않고 건수와 목록을 보고한다.

## 8. 실행 방식

- 실행 전 점검부터 보고서 생성까지를 드라이버 하나(`scripts/run_llava_next_mtvqa_main.sh runs/llava_next_mtvqa_main_<UTC>`)가 터미널·세션과 분리해 실행한다(setsid + nohup).
  드라이버에는 사람의 입력을 기다리는 지점이 없다. 같은 드라이버가 두 개 뜨지 않게 잠금 파일(`driver.lock`, flock)을 둔다.
- 상태 기록(`results/<run>/`): `status.json`(현재 단계, 완료 이미지 수 / 전체, 시작·최근 갱신 시각, 예상 종료 시각, 누적 SSD 쓰기량, SSD 온도, 마지막 오류),
  `driver.log`(표준 출력·오류 전체), `driver.pid`. 본실험 중에는 이미지를 commit할 때마다 갱신한다.
- 이어서 실행: 장비가 꺼졌다 켜져도 같은 명령으로 이어서 돈다.
  - 완료(commit)된 이미지는 건너뛴다. 끊긴 이미지는 세 방법 모두 T1부터 다시 한다.
  - 끊긴 이미지의 불완전한 기록은 결과에 넣지 않고(이전 시도 폴더에 `ABANDONED.json`을 남긴다), store는 지우고 다시 만든다.
  - 이어서 실행한 횟수와 시점(`driver_invocations.jsonl`), 다시 실행한 이미지 목록(`resume_log.jsonl`)을 보고서에 적는다.
- 실험이 도는 동안 같은 GPU나 SSD를 쓰는 다른 작업(단위 테스트 포함)을 돌리지 않는다.

## 9. 사전에 밝혀 두는 것 (지시에 없던 것, 지시와 다르게 되는 경우)

- gate 4장 단계에는 gate 2·3에 필요한 7개 arm에 ReComp와 FullLoad를 더해 9개 arm을 돌린다(본실험 방법 세 가지의 실행 경로를 파일럿 전에 한 번 통과시키기 위함; 판정에는 쓰지 않는다).
  4장 × 3턴이므로 gate 2·3의 비교는 arm 쌍마다 12요청(followup 8요청)이다.
- gate 4의 ReComp 대조 기준은 v3 기록의 본실험(main) 단계다(보고서의 ReComp 행이 쓰는 기록). v3의 pilot40 단계 기록과의 대조는 참고로 적는다.
- 본실험에서 한 이미지가 실행 오류나 감사 실패로 끝나지 않으면 그 이미지를 T1부터 한 번 더 시도하고, 그래도 끝나지 않으면 기록하고 건너뛴다.
  건너뛴 이미지가 25장을 넘거나 3장 연속이면 실행을 멈춘다. 건너뛴 이미지는 모든 표에서 빠지고 보고서에 건수와 목록을 적는다. gate·파일럿 단계에서는 건너뛰지 않고 멈춘다.
- 다른 프로세스가 GPU를 쓰는 것이 보이면(이미지 시작·끝에서 확인) 실행을 끝내지 않고 그 프로세스가 사라질 때까지 기다린다(최대 6시간). 그때 진행 중이던 이미지는 T1부터 다시 한다.
- 드라이버는 요청 단계가 오류로 끝나면 같은 실행 안에서 최대 5회까지 스스로 다시 시작한다(그때마다 이어서 실행으로 기록한다).
- store가 요청 중에 바뀌지 않았다는 확인: 본실험에서는 저장 직후와 요청 뒤의 파일 크기·수정 시각·inode를 비교한다(11/17 실행은 요청 뒤 전체 재해시). gate·파일럿에서는 전체 재해시도 한다.
  저장 직후의 전체 해시(무결성 봉인)와 활성화 때의 전체 확인은 11/17 실행과 같다. store 삭제는 그 이미지에 대해 기록한 파일 목록과 정확히 같을 때만 한다.
- 본실험에서는 이미지 감사를 통과한 뒤 IMPRESS의 T1 통계 증거 파일(이미지당 약 1.2 MB; 감사가 그 파일로 저장 순서를 다시 계산해 확인한 뒤)을 지우고
  store 메타데이터 사본을 gzip으로 보관한다(SHA256은 기록에 남긴다). gate·파일럿에서는 그대로 둔다.
- 모델을 올린 직후 합성 이미지 요청 1건으로 예열한다(11/17 실행과 같음, 모든 지표에서 제외).
- FullLoad store에는 11/17 구현 그대로 probe key 사본과 visual hidden 파일이 함께 저장된다(읽지 않는다).
- TTFT 비율에는 이미지 단위 bootstrap 95% CI를 함께 적는다.
- 실행 전 예상 소요 시간은 개발 리허설(이미지 몇 장, 측정 아님)의 이미지당 시간으로 적고, 파일럿이 끝나면 파일럿 실측값으로 다시 적는다(`prerun_pilot_estimate.json`, status.json).
- status.json의 예상 종료 시각은 근사값이다(현재 프로세스의 이미지당 평균 시간 × 남은 이미지 수).
- 이 계약을 동결하기 전에 개발 리허설을 두 번 돌려 스크립트 오류와 이어 실행 경로를 확인했다(`runs/llava_next_mtvqa_main_dev_20261121T061617Z`: 단계당 2–3장 완주;
  `runs/llava_next_mtvqa_main_dev_20261121T062504Z`: 본실험 중 드라이버를 종료한 뒤 같은 명령으로 이어 실행). 리허설은 측정이 아니며 그 값은 판정과 표에 쓰지 않는다.
  리허설에서 본 것: 본실험 이미지당 12.6–16.3 s, 이미지당 store 쓰기 3.9–4.8 GB, 모든 gate 통과(arm 쌍당 6요청).
- SSD 쓰기량은 커널의 장치 카운터(`/sys/block/<장치>/stat`, 장치 전체·모든 프로세스)와 store를 쓴 프로세스의 write 바이트로 적는다. NAND 수준의 쓰기량은 SMART 없이는 알 수 없다.

## 10. 보고 항목 (`results/llava_next_mtvqa_main_<UTC>/report.md`, 드라이버가 끝에서 자동 생성)

1. 본실험 표: 방법별 followup VQA score, T2·T3 각각의 score, TTFT 평균/p50/p95, SSD MB/followup, 읽기 호출 수. ReComp 행은 v3 기록의 값이며 다른 실행에서 잰 값임을 표시한다.
2. 짝지은 비교(95% CI) — score: Ours − IMPRESS, Ours − FullLoad, Ours − ReComp, IMPRESS − FullLoad, FullLoad − ReComp.
   TTFT: Ours − IMPRESS, Ours − FullLoad, IMPRESS − FullLoad (같은 실행의 세 방법만).
3. TTFT 비율: Ours/FullLoad, Ours/IMPRESS, IMPRESS/FullLoad. Ours/ReComp와 FullLoad/ReComp는 v3 기록의 ReComp TTFT로 계산하고 다른 실행의 값임을 표시한다.
4. ReComp TTFT 확인: 파일럿 40장에서 이번에 잰 ReComp TTFT와 v3 기록의 같은 요청 TTFT의 평균과 비율.
5. IMPRESS 통계: fallback 비율, 평균 Jaccard, K·V chunk 읽기 비율(전체 / fallback이 아닌 layer만 / 재배치하지 않았을 경우의 계산값), 중요도 중 prefix 행이 차지하는 비율.
6. 재사용 턴 시간 분해: 읽기 대기와 GPU 적재 시간.
7. T1 저장 비용: 방법별 이미지당 저장 시간과 저장 크기.
8. 실험 전체의 SSD 쓰기량(예상값과 실측값), SMART 값의 실행 전후 차이, 소요 시간.
9. 시간에 따른 TTFT 변화: 본실험을 10구간으로 나눈 방법별 평균 TTFT와 SSD 온도.
10. 연속성: 파일럿에서 ReComp 생성 토큰이 v3 기록과 일치한 건수, 본실험에서 T1 생성 토큰이 v3 기록과 일치한 건수,
    FullLoad의 followup score와 생성 토큰을 v3 기록의 FullLoad와 대조한 결과(참고).
11. 추가하거나 바꾼 코드 목록, 지시와 다르게 했거나 지시에 없던 것, 측정하지 않은 것과 한계.

측정하지 않은 값은 "미실행" 또는 "측정 못함"으로 적는다. CI가 0을 포함하면 "차이 미입증"이며 동등하다는 뜻이 아니다.

## 11. 새 파일

`scripts/209_eval_llava_next_mtvqa_main.py`(실행기), `scripts/210_validate_llava_next_mtvqa_main.py`(gate), `scripts/211_report_llava_next_mtvqa_main.py`(보고; 측정에 관여하지 않으므로 소스 freeze에서 제외),
`scripts/llava_next_main_status.py`(상태·SSD 기록), `scripts/run_llava_next_mtvqa_main.sh`(드라이버), `tests/test_llava_next_mtvqa_main.py`, 이 문서.
`mmimpress/**`와 기존 스크립트·테스트는 수정하지 않는다. 소스 freeze: freeze 때 `mmimpress/**`, `scripts/*.py`(보고 스크립트 `scripts/<번호>_report_*.py` 제외), `tests/test_*.py`의 SHA256을
`config.json`에 기록하고 각 단계와 이어 실행이 대조한다. freeze 뒤에 새로 생긴 파일은 비교 대상이 아니다(저장소의 다른 작업이 이 실험을 멈추지 않게 하기 위함).
커밋과 푸시는 하지 않는다.
