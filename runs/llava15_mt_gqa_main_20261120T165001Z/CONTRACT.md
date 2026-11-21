# LLaVA-1.5 × MT-GQA 본실험 계약 (ReComp / FullLoad / IMPRESS / Ours)

작성 2026-11-21. 이 문서는 gate와 본실험을 시작하기 전에 SHA256으로 동결한다
(`runs/llava15_mt_gqa_main_<UTC>/config.json`의 `contract_sha256`, 사본 `CONTRACT.md`).
동결 뒤에는 방법 정의·gate 기준·통계 방법을 바꾸지 않는다. 측정 보고이며 GO/NO-GO 판정은 두지 않는다.

## 1. 목적

Ours는 중요도 순 재배열을 뺀 방식으로 확정됐다(`results/mt_gqa_ours_order_20261120T052725Z`, 판정 "빼도 됨").
그 방식은 LLaVA-NeXT 경로(`mmimpress/packed_store.py`, `packed_serve.py`, `piggyback.py`의 `row_order="original"`)에만 있었다.
이를 LLaVA-1.5 경로(`mmimpress/llava15/`)에 옮기고, LLaVA-1.5 × MT-GQA를 네 방법으로 같은 실행 안에서 측정한다.

## 2. 공통 조건

- 모델 `llava-hf/llava-1.5-7b-hf`, HF revision `b234b804b114d9e37bb655e11cbbb5f5e971b7a9`, NF4 / BF16 / eager attention, FP16 SSD KV,
  greedy cap 16, batch 1, seed 1234, CPU thread 2. HF 기본 336×336 전처리, visual token 576개, separator 없음.
- 모든 방법은 매 대화 T1을 픽셀부터 계산한다(저장 KV의 warm-up 없음). T2–T3는 자기 방법의 생성 이력을 쓴다.
- cold SSD: 저장 KV를 읽는 요청 직전에 그 store 파일의 page cache를 비운다(`posix_fadvise(DONTNEED)`, TTFT 밖).
- 저장된 KV를 읽는 세 방법은 모두 비동기 읽기다. 백그라운드 워커 1개가 디스크 읽기만 하고 GPU 적재는 요청 스레드가 한다.
  미리 읽기는 TTFT 타이머가 시작된 뒤에만 한다.
- store는 이미지마다 그 이미지의 첫 대화 T1에서 한 번 만들고(각 store 종류를 쓰는 arm 중 회전 순서상 첫 arm의 T1 캡처),
  그 이미지의 모든 대화가 쓴다. 다른 대화의 T1은 픽셀부터 계산해 답만 낸다.
- 환경: 11/18 LLaVA-1.5 실행(`runs/llava15_mtvqa_impress_code_20261118T053221Z`)과 같은 환경
  (`/home/dblab/anaconda3/envs/mllm_ft/bin/python`, transformers 4.57.6, torch 2.5.1+cu121, numpy 2.2.6). freeze가 그 실행의 기록과 대조하고 다르면 멈춘다.

## 3. 방법 정의

1. **ReComp**: 매 턴 이미지부터 다시 계산한다.
2. **FullLoad**: Visual KV 전체를 저장하고 전부 읽는다. 11/18 실행의 FullLoad 구현을 그대로 쓴다
   (`mmimpress/llava15/raster.py`의 raster store, `Server.request(mode="fullload", prefetch="layer_stream")`).
3. **IMPRESS**: 11/18 실행의 구현과 설정을 그대로 쓴다. 코드를 바꾸지 않는다(`mmimpress/impress_official.py`,
   정의는 `docs/llava15_mtvqa_impress_code_contract.md`): 전체 Visual KV + probe key 사본 저장, probe head 0·1·2, 전체 행 열 합 중요도,
   상위 25%, 평균 Jaccard ≥ 0.3이면 투표 선택, 아니면 fallback으로 전부 읽어 사용, K/V 분리, 64토큰 chunk,
   첫 대화 T1 기준 layer별 재배치, 다음 layer probe key 미리 읽기.
4. **Ours**: 재배열을 뺀 방식.
   - 중요도: image-only CLIP CLS→patch attention(penultimate layer, head 합). 질문을 쓰지 않는다.
   - 계층별 배분: 기존 PYR-b20 벡터(총 KV 예산 25%, `runs/llava15_mt_gqa_20261112T062708Z/vectors/layer_budget_llava15_PYR-b20.json`,
     ratio SHA256 `51f9b0db…`, layer당 행 281, 272, 263, …, 16, 7, 합 4,608, payload 75,497,472 B).
   - layer l은 중요도 상위 k_l개 토큰을 뽑아 원래 토큰 인덱스 오름차순으로 저장한다. 뽑는 집합은 기존 LLaVA-1.5 재배열 store의 앞 k_l개와 같다
     (같은 `visionzip_repack_order` 순서의 앞 k_l개이므로 동점 처리도 같다).
   - 재사용 턴에서는 각 행을 GPU cache의 원래 토큰 위치에 올리고, 저장하지 않은 위치는 mask로 가린다.
   - 파일 구성(layer별 [K|V] 구간), 비동기 읽기 순서, pinned buffer 준비 시점(activation, TTFT 밖), 읽기 호출 수(hit당 preadv 32회)는
     기존 LLaVA-1.5 비동기 경로와 같다(같은 `PackedReader.stream_plan` / `Llava15StreamSelector` 코드).

FullLoad·IMPRESS의 코드가 11/18과 같다는 것은 freeze에서 확인한다: 11/18 freeze의 소스 해시와 비교해 달라진 파일이
`mmimpress/llava15/store.py`, `mmimpress/llava15/serve.py`(이 작업)와 `mmimpress/packed_store.py`, `packed_serve.py`, `piggyback.py`
(11/20의 LLaVA-NeXT 옵션 추가)뿐이어야 하고, 그 밖의 파일이 달라졌으면 freeze가 멈춘다. 실행 하니스는 11/18의
`scripts/199_eval_llava15_mtvqa_impress_code.py`를 import해 요청·캡처·재배치 함수를 그대로 호출한다.

## 4. 코드 변경 범위

- LLaVA-1.5의 기존 재배열 경로는 기본값으로 남기고, 원래 순서 경로는 옵션(`row_order="original"`)으로 추가한다
  (`mmimpress/llava15/store.py`, `mmimpress/llava15/serve.py`).
- LLaVA-NeXT 경로, 기존 store, 기존 결과 디렉터리, 동결 파일은 수정하지 않는다. 실행 전후로 `results/`·`runs/`·`kvstore*`의 파일 목록(경로, 크기, mtime)을 비교해 기록한다.
- 소스 freeze: freeze 때 `mmimpress/**`, `scripts/*.py`(보고서 스크립트 일부 제외), `tests/test_*.py`의 SHA256을 `config.json`에 기록한다.
  각 단계와 이어 실행은 그때 기록한 파일이 바뀌었으면 멈춘다. freeze 뒤에 새로 생긴 파일과 보고서 스크립트(`scripts/<번호>_report_*.py`, 기록만 읽는다)는 비교 대상이 아니다
  (저장소의 다른 작업이 이 실험을 멈추지 않게 하기 위함).
- 새 파일: `scripts/206_eval_llava15_mt_gqa_main.py`, `scripts/207_validate_llava15_mt_gqa_main.py`, `scripts/208_report_llava15_mt_gqa_main.py`,
  `scripts/llava15_main_status.py`, `scripts/run_llava15_mt_gqa_main.sh`, `tests/test_llava15_original_order.py`, 이 문서.

## 5. 실험

- 데이터: MT-GQA reconstructed 398 images / 4,061 dialogues × 3턴. 데이터·프롬프트·채점은 `docs/llava15_mt_gqa_contract.md`와 같다
  (INDEX_SHA `2c47cfad…`, WORKLOAD_SHA `0287e0c5…`, 멀티턴 프롬프트 `USER: <image>\n[Q1..A1..]Current question Q{t}: … ASSISTANT:`, strict normalized EM).
- 본실험 arm 4개 × 12,183요청 = 48,732요청. arm 순서는 대화 ordinal로 회전한다(`ordinal % arm 수`).
- 지표: followup(T2–T3) strict normalized EM, TTFT, SSD 읽기 바이트, 읽기 호출 수.
  TTFT 타이머는 프롬프트 구성 전에 시작해 첫 출력 토큰 직후에 끝난다(11/18 실행과 같은 경계). SSD 바이트·호출 수는 read syscall이 실제로 돌려준 값이다.
- 통계: 이미지 단위 paired bootstrap 10,000회, seed 1234, 95% CI(`mmimpress.recompute_kv25.paired_bootstrap`).
- 결과 디렉터리: `results/llava15_mt_gqa_main_<UTC>/`(집계·보고서·status.json·driver.log·PID 파일), 요청 단위 원본은 `runs/llava15_mt_gqa_main_<UTC>/`.

단계와 arm:

| 단계 | 이미지 | arm | 요청 수 |
|---|---|---|---:|
| gate | 4장(이전 MT-GQA 실행의 gate 이미지, 49 dialogues) | ReComp, FullLoad, IMPRESS(동기), IMPRESS, IMPRESS(유지 100%), FullLoad loader(재배치 store), Ours(동기), Ours, Ours(두 번째 실행), Ours(재배열) | 1,470 |
| devgate | DEV 40장(378 dialogues) | IMPRESS(동기), IMPRESS, IMPRESS(유지 100%), FullLoad loader(재배치 store), Ours(동기), Ours, Ours(두 번째 실행) | 7,938 |
| dev | DEV 40장 | ReComp, FullLoad, IMPRESS, Ours + 참고용 Ours(재배열) | 5,670 |
| main | 398장 | ReComp, FullLoad, IMPRESS, Ours | 48,732 |

DEV 40장은 `results/layer_budget_kv25_v2_20261106T073502Z/data_split.json`의 DEV 이미지다. gate·devgate·dev 요청은 본실험 분모에 넣지 않는다.
Ours(재배열)은 기존 LLaVA-1.5 재배열 store(11/18 실행의 Ours arm과 같은 method id)이며 gate 1과 DEV 참고값에만 쓴다.

## 6. 실행 방식

- gate(4장) → DEV 40장 → 본실험 → 보고서 생성을 드라이버 하나(`scripts/run_llava15_mt_gqa_main.sh <run>`)가 터미널·세션과 분리해 실행한다(setsid + nohup).
- 드라이버는 단계마다 스스로 판정한다. 사람의 입력을 기다리는 지점이 없다. 같은 드라이버가 두 개 뜨지 않게 잠금 파일(`driver.lock`, flock)을 둔다.
- 상태 기록(`results/<run>/`): `status.json`(현재 단계, 완료 요청 수 / 전체 요청 수, 시작·최근 갱신 시각, 예상 종료 시각, 마지막 오류), `driver.log`(표준 출력·오류 전체), `driver.pid`.
- 이어서 실행: 같은 명령을 다시 실행하면 이어서 돈다.
  - commit된 이미지와 완료된 대화는 건너뛴다. 중간에 끊긴 대화는 그 대화의 T1부터 다시 한다. 끊긴 대화의 불완전한 기록은 결과에서 버린다.
  - store는 이미지의 첫 대화 T1에서 만들므로, 끊긴 대화가 그 이미지의 첫 대화이거나 디스크의 store가 만들 때 기록한 SHA256과 맞지 않으면
    그 이미지의 store와 기록을 버리고 이미지의 첫 대화부터 다시 만든다.
  - 이어서 실행한 횟수, 각 시점, 다시 실행한 대화 목록을 `resume_log.jsonl`·`driver_invocations.jsonl`에 남기고 보고서에 적는다.
  - 이어서 실행한 직후의 첫 요청들이 TTFT에 영향을 주는지는 프로세스 시작 직후 요청과 나머지 요청의 TTFT를 비교해 적는다.
- 실험이 도는 동안 같은 GPU나 SSD를 쓰는 다른 작업(단위 테스트 포함)을 돌리지 않는다. 하니스는 이미지 경계마다 GPU에 다른 프로세스가 있으면 그 이미지를 중단한다.

## 7. Gate (본실험 전, 4장 → DEV 40장 순서)

1. **Ours 저장 내용 일치**: 같은 이미지에서 Ours store의 layer별 원래 인덱스 집합이 재배열 store의 앞 k_l개와 같고,
   같은 원래 토큰의 K·V 값이 비트 단위로 같으며, payload 총 바이트가 같다. (4장: gate 단계, DEV: dev 단계의 두 store를 디스크에서 읽어 비교)
2. **Ours의 선택과 배치가 질문과 무관하다**: 같은 이미지에 질문 3개를 줘서 store를 세 번 만들어도 layer별 인덱스 목록이 같다
   (행 수, byte table, 파일 구성, 요청 기록·payload 해시를 뺀 메타데이터도 같아야 하고, 그 단계가 commit한 store의 인덱스 표와도 같아야 한다).
   payload 바이트의 일치 여부는 gate가 아니라 보고 항목이다.
3. **Ours의 동기 읽기와 비동기 읽기**가 생성 토큰과 읽기 바이트에서 일치한다(모든 요청).
4. **Ours를 같은 요청에 두 번** 돌려 생성 토큰이 같다(모든 요청).
5. **IMPRESS**: 유지 비율 100%로 두었을 때의 생성 토큰이, 같은 재배치 store를 읽는 FullLoad loader의 생성 토큰과 일치한다.
   비동기와 동기의 생성 토큰·읽기 바이트(전체와 종류별)가 일치한다(모든 요청).
6. **ReComp**의 생성 토큰이 11/12 실행(`runs/llava15_mt_gqa_20261112T062708Z/main`)의 같은 요청(같은 대화, 같은 턴)과 일치한다.
7. **기존 테스트**에서 새 실패가 0개다(이 작업 전부터 실패하던 테스트는 목록만 적는다: transformers 4.57.6 환경의 GLM-4.6V 테스트 3개,
   5.13.1 환경의 vision-capture 테스트 2개). 실행 환경과 교차 확인용 5.13.1 환경 두 곳에서 돌린다. 새 테스트 파일은 통과해야 한다.
8. **DEV 40장 파일럿**을 4 arm으로 돌려 표를 만든다. 참고용으로 Ours(재배열)을 DEV에서만 함께 돌려 Ours와의 정답률·TTFT 차이를 적는다.
   이 값은 gate가 아니고 본실험 arm에도 넣지 않는다.

gate 3·4·5는 4장(gate 단계)과 DEV 40장(devgate 단계)에서, gate 1·6은 4장(gate 단계)과 DEV 40장(dev 단계)에서, gate 2는 4장과 DEV 40장에서 판정한다.

- gate 1–7이 4장과 DEV 40장에서 모두 통과하고 DEV 파일럿이 실행되면 드라이버가 본실험까지 이어서 진행한다.
- gate가 실패하면 드라이버는 본실험을 시작하지 않고, 실패 내용과 원인을 `status.json`과 `report.md`에 적은 뒤 종료한다.
  4장에서 실패하면 DEV 단계도 시작하지 않는다. 기준을 임의로 바꾸지 않는다. 판정이 기록된 gate는 다시 판정하지 않는다.
- gate 검사 프로그램이 결과를 내지 못하고 끝난 경우(판정이 아닌 실행 오류)와 요청 단계가 끝나지 못한 경우는 FAIL로 확정하지 않고
  오류를 기록한 뒤 멈추며, 같은 명령으로 그 지점부터 다시 실행된다. 이 경우에도 본실험은 gate가 모두 PASS로 기록되기 전에는 시작하지 않는다.

## 8. 사전에 밝혀 두는 것 (지시에 없던 것, 지시와 다르게 되는 경우)

- gate 판정용 arm은 본실험 arm과 별도다(위 표). DEV의 gate 3·4·5는 별도 단계(devgate)로 돈다.
- DEV 파일럿은 5 arm(네 방법 + Ours(재배열))을 한 단계에서 돌리므로 arm 회전이 5 arm 기준이다. 본실험은 4 arm 기준이다.
- 드라이버는 요청 단계가 오류로 끝나면 같은 실행 안에서 최대 3회까지 스스로 다시 시작한다(그때마다 이어서 실행으로 기록한다).
- 드라이버는 GPU를 쓰는 단계를 시작하기 전에 다른 프로세스가 GPU를 쓰고 있으면 비워질 때까지 기다린다(30초 간격, 최대 2시간, 넘으면 오류를 기록하고 멈춘다).
- 이어 실행에서 이미지 전체를 다시 하는 경우(6절)에는 그 이미지에서 이미 끝났던 대화도 다시 실행된다.
- 모델을 올린 직후 합성 이미지 요청 1건으로 예열한다(11/18 실행과 같음, 모든 지표에서 제외).
- FullLoad store에는 11/18 구현 그대로 probe key 사본이 함께 저장된다(읽지 않는다).
- TTFT 비율에는 이미지 단위 bootstrap 95% CI를 함께 적는다.
- status.json의 예상 종료 시각은 근사값이다.

## 9. 보고 항목 (`results/llava15_mt_gqa_main_<UTC>/report.md`, 드라이버가 끝에서 자동 생성)

1. 본실험 표: 방법별 followup EM, TTFT 평균/p50/p95, SSD MB/followup, 읽기 호출 수.
2. 짝지은 비교(95% CI, 정답률과 TTFT): Ours − IMPRESS, Ours − FullLoad, Ours − ReComp, IMPRESS − FullLoad, FullLoad − ReComp.
3. TTFT 비율: Ours/FullLoad, Ours/ReComp, Ours/IMPRESS, IMPRESS/FullLoad.
4. IMPRESS 통계: fallback 비율, 평균 Jaccard, K·V chunk 읽기 비율, 중요도 중 prefix 행이 차지하는 비율.
5. T1 저장 비용: 방법별 이미지당 저장 시간과 저장 크기, 실험 전체의 SSD 기록량.
6. 재사용 턴 시간 분해: 읽기 대기와 GPU 적재 시간.
7. DEV 참고값: Ours(재배열)과 Ours의 차이.
8. 코드에서 추가하거나 바꾼 부분 목록과 새 테스트.
9. 지시와 다르게 했거나 지시에 없던 것, 측정하지 않은 것과 한계.

보고서에서 방법은 ReComp, FullLoad, IMPRESS, Ours로만 부른다(DEV 참고값의 Ours(재배열) 제외). 옵션 이름이나 클래스 이름을 방법 이름으로 쓰지 않는다.
