# ConvBench × {LLaVA-NeXT, LLaVA-1.5}: HyperInfer 측정 계약

작성 2026-11-25 (실행 전 동결). 이 문서의 SHA256은 `results/convbench_hyperinfer_<UTC>/config.json`에 기록되며, 드라이버의 모든 단계가 시작할 때 다시 확인한다.

측정 보고이며 GO/NO-GO 판정은 두지 않는다. TTFT가 IMPRESS보다 빠른지 느린지는 gate가 아니며 결과 그대로 보고한다. 보고서에서 방법은 ReComp, FullLoad, IMPRESS, HyperInfer, Ours로만 부른다.

## 1. 배경

baseline을 IMPRESS에서 HyperInfer로 바꾸는 중이다. MT-GQA는 끝났고(`results/mt_gqa_hyperinfer_v2_20261125T034057Z`), 이 실행은 ConvBench에서 HyperInfer를 측정한다.

- 본실험에서 돌리는 방법은 HyperInfer 하나다(두 모델).
- ReComp, FullLoad, IMPRESS, Ours는 다시 돌리지 않는다. 기존 ConvBench 본실험(`results/convbench_main_20261122T103401Z`, 2026-11-22–24, 아래 "기존 본실험")의 요청별 기록을 읽어 같은 대화·같은 턴끼리 짝지어 비교한다.
- ConvBench는 이전 턴의 긴 답이 입력에 들어가 이미지 뒤의 텍스트가 길다. layer당 GPU 계산 시간이 MT-GQA(1.6–1.9 ms)보다 길 것이므로 prefetch의 효과가 MT-GQA와 다를 수 있다. 결과를 미리 가정하지 않는다.

## 2. 설정

### 2.1 ConvBench 설정 (기존 본실험과 같다)

`docs/convbench_main_contract.md` 2절과 `docs/convbench_baseline_alignment.md`의 값 그대로다: 두 모델의 checkpoint와 revision, NF4 4-bit + BF16 계산 + eager attention, SSD의 KV는 FP16, greedy, `max_new_tokens = 1024`(문맥 길이에 따른 조정 없음), `llava_v1` 템플릿, 원본 LLaVA와 같은 토큰열(요청마다 요청 전에 검사), 같은 방법이 생성한 A1·A2를 이력으로 사용, 프롬프트와 이력을 자르지 않음(입력이 4096 토큰을 넘어도 그대로 실행), HF 프로세서 기본 전처리, 동결한 판정기(Llama-3.1-8B-Instruct, 1차 출력 상한 2048, 공식 2차 추출, `Rating:` 3차, 분모 577, view `capN`).

실행기는 기존 본실험의 실행기 `scripts/217_eval_convbench_main.py`를 그대로 불러 쓴다. 대화 하나의 흐름, 요청 전 토큰 검사, T1을 픽셀부터 계산하는 것(미리 만든 KV 없음), 요청 전 page cache drop, 타이머 구간, store를 대화가 확정된 뒤에 지우는 것, 실패 처리, 이어서 실행하는 규칙이 모두 그 코드다. arm 표와 단계 표만 이 실행의 것으로 바꾼다(다른 코드가 쓰는 표는 건드리지 않고 이 실행 프로세스 안의 사본에만 넣는다).

### 2.2 HyperInfer (MT-GQA 실행과 같다)

구현과 설정은 `results/mt_gqa_hyperinfer_v2_20261125T034057Z`의 것이다(`docs/mt_gqa_hyperinfer_v2_contract.md` 3절, `docs/hyperinfer_port_contract.md`).

- 구현 `mmimpress/hyperinfer_v2.py`(수정하지 않는다). layer l의 계산이 시작될 때 worker가 layer l+1의 probe key를 읽고, 이어서 예측 chunk(layer l의 사용 집합 전체가 든 chunk)를 chunk 번호 오름차순으로 K chunk → V chunk 순서로 하나씩 읽는다. layer l의 GPU 계산이 끝나면(layer l+1 hook 첫머리의 `torch.cuda.synchronize()` 반환) 새 읽기를 시작하지 않고, 진행 중이던 읽기 하나만 끝낸다. 그 뒤 IMPRESS와 같은 선택을 하고, 선택된 토큰이 든 chunk 중 prefetch되지 않은 것만 동기로 읽는다. attention에는 선택 집합만 쓴다.
- 요청 함수는 MT-GQA 실행의 것을 그대로 부른다(`scripts/228`의 `hyper_hit`, `scripts/225`의 `hyper_request`·`dispatch`). 두 모델의 하네스 모듈(`scripts/209`, `scripts/206`)도 그 실행이 쓴 모듈과 같다.
- 설정: probe head 0·1·2, 유지 25 %, Jaccard 임계 0.3, 64토큰 chunk, worker 1개, 시간 예산·chunk 시간 상수 없음. freeze가 MT-GQA 실행의 `config.json`(arm 정의, IMPRESS 인자, 서버 인자, `hyperinfer_v2.py`의 SHA256)과 대조한다.
- 저장 형식은 IMPRESS와 같다(T1 attention으로 정한 layer별 재배치 store). HyperInfer는 자기 T1 요청의 KV로 이 store를 쓴다.
- 이 실행이 MT-GQA 실행과 다르게 주는 것은 ConvBench 실행기가 주는 것뿐이다: 프롬프트, 생성 상한 1024, store 메타데이터의 데이터셋 이름.

### 2.3 고정 확인 (freeze)

기존 본실험과 MT-GQA 실행이 동결한 소스 파일이 모두 그대로인지, 실행 환경(`mllm_ft`: transformers 4.57.6, torch 2.5.1)이 두 실행과 같은지, FullLoad·IMPRESS arm 정의와 서버 인자·IMPRESS 인자·읽기 방식이 기존 본실험의 `config.json`과 같은지 assert한다. 하나라도 다르면 측정을 시작하지 않는다.

## 3. arm과 단계

| 표의 이름 | role | 쓰임 |
|---|---|---|
| FullLoad | `fullload` | 파일럿 표, TTFT 연속성 |
| IMPRESS | `impress` | gate 1·2의 대조, 파일럿 표, TTFT 연속성 |
| HyperInfer | `hyperinfer` | 본실험 arm. gate 1, 4, 5, 6 |
| HyperInfer, prefetch 끔 | `hyperinfer_off` | gate 2 |
| HyperInfer, 두 번째 실행 | `hyperinfer_repeat` | gate 4, 5 |
| HyperInfer, 읽지 않은 행 NaN | `hyperinfer_nan` | gate 3, 5 |

단계 (모델마다, LLaVA-NeXT를 끝낸 뒤 LLaVA-1.5):

| 단계 | 대화 | arm | 요청 수 |
|---|---|---|---|
| `gate` | 4 (index 순번 72, 216, 360, 504) | 6 | 72 |
| `pilot` | 40 (순번 7 + 14i, i = 0…39) | FullLoad, IMPRESS, HyperInfer | 360 |
| `overflow` | 2 (파일럿에 4096 토큰을 넘는 HyperInfer 입력이 없을 때만) | HyperInfer | 6 |
| `main` | 577 전부 | HyperInfer | 1,731 |

- 4대화와 파일럿의 대화는 기존 본실험의 gate·파일럿과 같은 대화다. 본실험은 577대화 전부를 새로 돌리며, 4대화·파일럿의 결과는 본실험 표에 쓰지 않는다.
- 대화의 arm 순서 = 단계의 arm 목록을 (index 순번 mod arm 수)만큼 회전(기존 본실험과 같은 규칙).
- 대화 하나의 흐름: 모든 arm이 T1을 픽셀부터 계산 → store를 쓰는 arm이 자기 T1의 KV로 store를 씀 → T2, T3(자기 생성 이력, 요청 전 page cache drop) → 대화 확정 → 그 대화의 store 삭제.
- 실패 처리(기존 본실험과 같다): 요청이 예외로 끝나면 그 대화를 모든 arm에서 T1부터 한 번 더 실행하고, 다시 실패하면 실패를 기록한 채 확정한다. 분모는 577로 고정한다. 실패한 대화가 연속 5개 또는 누적 40개를 넘으면 그 단계를 멈춘다.
- 통계: 대화 단위 paired bootstrap 10,000회, seed 1234, 95% CI(percentile). TTFT는 대화별 T2·T3 차이의 평균, 비율은 합의 비, 품질은 대화별 (S1·S2·S3 승리의 평균 × 100)의 차이.
- 결과 디렉터리: `results/convbench_hyperinfer_<UTC>/`.

## 4. 품질 점수

본실험이 끝나면 모델마다 HyperInfer의 생성 토큰(T1–T3, 1,731요청)을 기존 본실험의 IMPRESS 기록과 요청 단위로 대조한다(`gates/<model>/identity.json`).

- 1,731요청이 모두 일치하면 그 모델은 판정을 다시 돌리지 않는다. 기존 본실험이 IMPRESS에 내린 판정 결과(S1·S2·S3, 대화별 승패)를 HyperInfer의 점수로 쓰고, 보고서에 그 사실을 적는다.
- 하나라도 다르면(실패한 요청도 불일치로 센다) 그 모델의 HyperInfer 답 전체(577대화)를 동결한 판정기로 판정한다. 불일치 요청의 목록(처음 달라진 토큰 위치, 같은 대화의 앞 턴이 일치했는지)과 원인으로 확인된 것을 보고한다.

## 5. Gate (모델마다, 4대화 → 파일럿 40대화 순서)

| Gate | 통과 기준 | 판정 위치 |
|---|---|---|
| 1 출력 동일성 | HyperInfer의 생성 토큰(T1–T3)과 layer별 선택 집합(T2–T3)이 같은 세션에서 돌린 IMPRESS와 모든 요청에서 일치 | 4대화, 파일럿 |
| 2 prefetch 끔 ≡ IMPRESS | prefetch를 끈 HyperInfer의 생성 토큰, 읽기 바이트(전체·probe·K·V), 읽기 호출 수가 같은 세션의 IMPRESS와 모든 요청에서 일치 | 4대화 |
| 3 읽지 않은 KV를 쓰지 않음 | MT-GQA 실행의 gate 3과 같은 방식: prefix cache의 모든 이미지 행을 NaN으로 채우고 시작해, 모든 layer의 attention 출력이 유한하고, mask되지 않은 이미지 열이 선택 집합과 같고, 사용한 행이 모두 읽은 행이며, 생성 토큰과 선택 집합이 IMPRESS와 같다 | 4대화 |
| 4 두 번 실행 | 같은 요청을 두 번 돌려 생성 토큰이 같다 | 4대화 |
| 5 멈춤 확인 | prefetch를 켠 arm의 모든 요청, 모든 layer에서 계산이 끝난 뒤에 새로 시작된 prefetch 읽기가 0건(읽기마다 기록한 판단 시각과 layer마다 기록한 멈춤 시각의 비교, 둘 다 같은 lock 안에서 잰 값). 멈춤 뒤에 끝난 읽기가 한 layer에 둘 이상이어도 실패 | 4대화(세 arm), 파일럿(HyperInfer) |
| 6 기존 기록과의 일치 | 파일럿 40대화에서 HyperInfer의 T1·T2·T3 생성 토큰이 기존 본실험의 IMPRESS 기록과 모두 일치 | 파일럿 |
| 7 기존 테스트 | 두 환경(transformers 4.57.6, 5.13.1)의 전체 테스트에서 새 실패 0개. 이전부터 실패하던 테스트는 목록만 적는다 | 모델 공통, 한 번 |
| 8 파일럿 표 (기록용) | 파일럿 40대화에서 FullLoad, IMPRESS, HyperInfer를 arm 순서 회전으로 함께 돌려 표를 만든다. 파일럿에 4096 토큰을 넘는 입력이 없으면 그런 대화 2개를 골라 HyperInfer로 따로 돌려 오류 없이 끝나는지 확인한다 | 파일럿 뒤 |

- 4대화 단계에서는 실패한 요청이 하나라도 있으면 FAIL이다. 파일럿에서 비교 대상 요청이 실패했으면 그 요청은 불일치로 센다.
- gate 1–7이 모두 PASS이면 드라이버가 그 모델의 본실험까지 이어서 진행한다. gate 8은 진입 조건이 아니다.
- 한 모델의 gate가 실패하면 그 모델의 본실험만 건너뛰고 다른 모델은 계속한다. gate 7이 실패하면 어느 모델도 본실험을 시작하지 않는다. 실패 내용과 원인은 `status.json`과 `report.md`에 적는다.
- 기록된 gate 결과는 다시 판정하지 않으며 기준을 바꾸지 않는다. gate 검사 자체가 결과를 내지 못하고 죽은 경우(실행 오류)는 gate 판정이 아니므로 같은 명령으로 다시 시도된다.

## 6. 실행 방식

- `scripts/run_convbench_hyperinfer.sh results/convbench_hyperinfer_<UTC>` 하나가 freeze → 실행 전 점검 → gate 7 → 모델별 [4대화 → gate 1–5 → 파일럿 → gate 1·5·6 → 파일럿 표·4096 초과 확인 → admission → 본실험 → 기존 IMPRESS 기록과 대조] → (필요한 모델만) 판정 → 사후 확인 → 보고서를 묶는다. setsid + nohup으로 터미널·세션과 분리하고, 사람의 입력을 기다리는 지점이 없다.
- 상태: `status.json`(현재 모델·단계, 완료 대화 수 / 전체, 예상 종료 시각, 누적 SSD 쓰기량, 마지막 오류), `driver.log`, `driver.pid`. 잠금 파일(`driver.lock`, flock)로 두 드라이버가 동시에 뜨지 않는다.
- 이어서 실행: 같은 명령. 끝난 단계와 확정된 대화는 건너뛴다. 끊긴 대화는 T1부터 다시 하며, 그 대화의 불완전한 기록은 `discarded/`로 옮겨 쓰지 않고 store는 지우고 다시 만든다. 이어서 실행한 횟수와 다시 실행한 대화는 `driver_invocations.jsonl`, `resume_log.jsonl`에 남고 보고서에 적는다.
- 실험이 도는 동안 같은 GPU나 SSD를 쓰는 다른 작업(단위 테스트 포함)을 돌리지 않는다. 다른 프로세스가 GPU를 쓰면 대화 시작 전에 기다리고, 대화 도중에 감지되면 그 대화를 버리고 다시 한다.
- 기존 store, 기존 결과 디렉터리, 동결 파일은 수정하지 않는다. 실행 전 점검 때와 종료 때 `results/`, `runs/`, `kvstore*`의 파일 목록(크기, 수정 시각)을 비교해 보고서에 적는다. 지우는 것은 이 실행의 `scratch/` 아래뿐이다.
- 실행 전 점검(`prerun.json`): 예상 SSD 쓰기량과 소요 시간(기존 본실험이 같은 대화에서 기록한 store 쓰기량과 요청 시간으로 계산), 여유 공간(가장 큰 대화 store 묶음의 20배 미만이면 시작하지 않음; 대화마다 다시 확인), SSD의 SMART 값(읽을 수 없으면 그 사유와 커널 쓰기 카운터·온도).

## 7. 보고 항목 (모델별, `report.md`)

1. TTFT 표: 다섯 방법의 T2·T3 각각과 합친 평균/p50/p95, SSD MB/followup. HyperInfer는 이번 실행, 나머지는 기존 본실험의 값이며 실행일을 함께 적는다.
2. 짝지은 비교(95% CI)와 비율: Ours 대 HyperInfer, HyperInfer 대 IMPRESS, HyperInfer 대 FullLoad, HyperInfer 대 ReComp. 같은 실행에서 잰 HyperInfer/IMPRESS 비율(파일럿 40대화)도 함께 적는다.
3. TTFT 연속성: 파일럿에서 다시 잰 FullLoad와 IMPRESS의 TTFT를 기존 본실험의 같은 요청과 비교한 평균과 비율.
4. 품질 표: 다섯 방법의 S1·S2·S3·평균. HyperInfer의 점수가 IMPRESS의 판정 결과를 쓴 것인지 새로 판정한 것인지 밝힌다. 짝지은 비교(95% CI): Ours − HyperInfer, HyperInfer − FullLoad, HyperInfer − ReComp.
5. 출력 동일성: 본실험 1,731요청 중 기존 IMPRESS 기록과 생성 토큰이 일치한 요청 수.
6. layer당 실측(턴별로): GPU 계산 시간, 다음 layer의 probe key 읽기 시간, chunk당 읽기 시간, 계산 시간 안에 읽기를 마친 chunk 수, 예측 chunk 수. MT-GQA 실행의 같은 값을 나란히 적는다.
7. prefetch 통계: 사용 집합 중 prefetch로 채워진 비율, prefetch된 것 중 실제로 쓴 비율, 읽었지만 쓰지 않은 바이트, IMPRESS 대비 총 읽기 바이트.
8. 임계 경로에 남은 시간 분해: probe key 대기, 진행 중이던 읽기의 마무리 대기, 선택 계산, 빠진 KV 동기 읽기, 이미지 뒤 텍스트의 계산 시간.
9. 텍스트 길이별 분석: 재사용 턴 요청을 이미지 뒤 텍스트의 토큰 수로 네 구간으로 나눠, 구간마다 요청 수, layer당 GPU 계산 시간, prefetch로 채워진 비율, HyperInfer TTFT, 기존 IMPRESS TTFT, 그 비율.
10. fallback 비율, 평균 Jaccard, K·V chunk 읽기 비율(기존 IMPRESS 값과 나란히), 인접 layer 선택의 Jaccard 유사도.
11. 4096 토큰을 넘는 입력의 수와 처리 결과.
12. T1 저장 비용, 실험 전체의 SSD 쓰기량(예상값과 실측값), 소요 시간.
13. 추가하거나 바꾼 코드 목록, 지시와 다르게 했거나 지시에 없던 것, 측정하지 않은 것과 한계.

## 8. 지시를 해석했거나 지시에 없던 것 (실행 전 고지)

1. **gate별 판정 위치.** gate 2·3·4는 추가 arm이 필요하므로 4대화에서만 판정한다. 파일럿은 지시대로 FullLoad, IMPRESS, HyperInfer 세 arm이고, 거기서 gate 1·5·6을 판정한다(gate 1과 5는 4대화와 파일럿 양쪽에서 본다). 기존 본실험 계약과 같은 방식이다.
2. **gate 7의 환경.** MT-GQA HyperInfer 실행과 같이 두 환경의 전체 테스트를 본다. 이 실행의 새 테스트(`tests/test_convbench_hyperinfer.py`)도 그 묶음에 들어 있다.
3. **4대화·파일럿에서의 store.** IMPRESS와 HyperInfer arm들은 저장 형식이 같으므로 한 대화 안에서 store 하나를 함께 읽는다. 그 store는 arm 회전에서 먼저 오는 arm이 자기 T1의 KV로 쓴다(모든 arm은 T1을 픽셀부터 계산한다). 본실험에서는 HyperInfer가 자기 T1으로 쓴다. T1 생성 토큰이 arm 사이에서 같은지는 gate 결과에 함께 적는다.
4. **"warm-up 없음".** 기존 본실험과 같이, 미리 만든 KV 없이 T1을 픽셀부터 계산한다는 뜻으로 읽는다. 모델을 올린 직후의 프로세스 warm-up 요청 1건(측정 제외)은 기존 본실험과 같이 둔다.
5. **4096 토큰 초과 대화의 선택.** 파일럿의 HyperInfer 입력에 4096 초과가 없으면, 기존 본실험의 IMPRESS 기록에서 입력이 가장 긴 2대화를 고른다. LLaVA-1.5는 기존 기록에 4096을 넘는 입력이 없으므로 입력이 가장 긴 2대화로 대신한다(기존 본실험과 같은 처리). 본실험 577대화에도 같은 대화가 들어 있으므로, 4096 초과 입력의 수와 결과는 본실험 기록으로도 보고한다.
6. **판정이 필요한 경우.** 그 모델의 HyperInfer 답 전체를 빈 cache에서 새로 판정한다(기존 판정 cache를 읽지 않는다). 판정기 재현 확인(기존 본실험의 gate 6)은 다시 하지 않고, 대신 세 답이 IMPRESS와 글자까지 같은 대화에서 새 판정의 승패가 기존 IMPRESS 판정과 같은 건수를 보고한다. 실패한 요청의 판정 처리는 기존 본실험과 같다(그 턴과 뒤 단계는 0승).
7. **기존 기록과의 비교 조건.** HyperInfer는 이번 실행, 나머지 네 방법은 2026-11-22–24의 실행이다. 서로 다른 날의 TTFT 차이에는 실행 간 변동이 섞인다. 그 크기는 보고 항목 3(파일럿에서 다시 잰 FullLoad·IMPRESS 대 기존 기록)으로 보인다. 기존 본실험은 4 arm 회전, 이번 본실험은 HyperInfer 단독이라는 차이도 있다.
8. **layer당 GPU 계산 시간.** prefetch 제출 시각부터 다음 layer hook 첫머리에서 GPU 동기화가 끝난 시각까지다(MT-GQA 실행과 같은 정의). 이미지 뒤 텍스트의 계산 시간은 기존 본실험의 정의(prefill 구간 − layer hook 시간)와, HyperInfer에서만 잴 수 있는 layer별 계산 시간 창의 합을 함께 적는다.
9. **텍스트 길이 구간.** 그 모델의 HyperInfer 재사용 턴 요청(T2·T3 합쳐서)의 이미지 뒤 토큰 수 사분위로 나눈다. 이미지 뒤 토큰 수 = 입력 토큰 수 − 저장된 prefix 길이.
10. **짝짓기에서 빠지는 대화.** 어느 한쪽에 실패한 요청이 있는 대화는 그 비교에서 빼고 건수를 적는다.
11. **대화가 끝날 때마다 GPU 메모리 cache를 비운다**(측정 구간 밖, 기존 본실험과 같다).
