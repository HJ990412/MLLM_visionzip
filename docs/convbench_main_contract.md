# ConvBench 본실험 계약 (LLaVA-NeXT, LLaVA-1.5 × ReComp, FullLoad, IMPRESS, Ours)

작성 2026-11-22 (실행 전 동결). 이 문서의 SHA256은 `results/convbench_main_<UTC>/config.json`에 기록되며, 드라이버의 모든 단계가 시작할 때 다시 확인한다.

측정 보고이며 GO/NO-GO 판정은 두지 않는다. 보고서에서 방법은 ReComp, FullLoad, IMPRESS, Ours로만 부른다.

## 1. 배경

기준 점수 점검(`results/convbench_alignment_20261121T083134Z`)에서 ReComp의 S1·S2·S3가 두 모델 모두 MetaCompress 기본 모델과 정해 둔 범위 안에서 맞았다. 그때 동결한 설정(`docs/convbench_baseline_alignment.md`)을 네 방법 실행기에 옮겨 본실험을 한다. 방법 코드(`mmimpress/`)와 기존 스크립트의 동작은 바꾸지 않는다.

## 2. 동결 설정 (점검 문서 그대로)

| 항목 | 값 |
|---|---|
| 모델 | LLaVA-NeXT-7b `llava-hf/llava-v1.6-vicuna-7b-hf`, LLaVA-1.5-7b `llava-hf/llava-1.5-7b-hf` (기준 실행과 같은 revision) |
| 정밀도 | NF4 4-bit(double quant), BF16 계산, eager attention. SSD의 KV는 FP16 |
| 디코딩 | greedy |
| 생성 길이 상한 | `max_new_tokens = 1024`, 문맥 길이에 따른 조정 없음 |
| 프롬프트 | `llava_v1` 템플릿: `{system} USER: <image>\n{Q1} ASSISTANT: {A1}</s>USER: {Q2} ASSISTANT: …` |
| 프롬프트 토큰열 | 원본 LLaVA `tokenizer_image_token`과 같은 토큰 ID. LLaVA-NeXT는 `<image> \n`, `</s> USER`로 렌더링. 요청마다, 요청을 보내기 전에(시간 측정 밖) 일치를 검사한다 |
| 대화 이력 | 같은 방법이 생성한 A1, A2. 이미지는 첫 턴에만 |
| 문맥 한계 | 프롬프트·이력을 자르지 않는다. 입력이 4096 토큰을 넘어도 그대로 실행한다(점검 실행과 같음) |
| 이미지 전처리 | HF 프로세서 기본값 |
| 판정 | Llama-3.1-8B-Instruct(BF16, SDPA, greedy), 공식 프롬프트, 공식 pairwise.npy, 1차 출력 상한 2048, 공식 2차 추출, 공식 `Rating:` 3차, 판정 불가는 0승, 분모 577. 보고하는 값은 채택한 view(`capN`) |

렌더러·토큰 검사·상한은 `scripts/213_convbench_recomp_generate.py`의 함수와 상수를 import해 쓴다. 판정은 `scripts/214_convbench_alignment_judge.py`의 함수를 import해 쓴다.

## 3. 방법 정의

공통: 모든 방법이 T1을 픽셀부터 계산한다(미리 만들어 둔 KV 없음). T2–T3는 자기 방법의 생성 이력을 쓴다. 저장된 KV를 읽는 요청마다 요청 전에 그 store의 page cache를 비운다(cold SSD). 저장된 KV를 읽는 세 방법은 모두 비동기 읽기다.

| 방법 | LLaVA-NeXT | LLaVA-1.5 |
|---|---|---|
| ReComp | 매 턴 이미지부터 다시 계산 | 같음 |
| FullLoad | `results/mt_gqa_impress_code_20261117T185832Z`의 구현과 설정: T1에서 이미지 KV 전체를 원래 순서로 저장, 재사용 턴에 layer 단위 비동기 읽기 | `results/llava15_mt_gqa_main_20261120T165001Z`의 구현과 설정 |
| IMPRESS | 같은 실행의 구현과 설정: 공개 구현(HyperInfer 91df7a2)의 선택 절차 — probe head 0·1·2, 유지 25 %, Jaccard 임계 0.3 미만이면 그 layer 전체, 64토큰 chunk, T1 attention으로 정한 layer별 재배치 store, 다음 layer probe 비동기 읽기 | 같은 기준 실행의 구현과 설정 |
| Ours | `results/mt_gqa_ours_order_20261120T052725Z`의 "Ours(원래 순서)": 중요도 순 재배열 없음, 총 KV 예산 25 %, layer 배분 PYR-beta-raw, layer 단위 비동기 읽기 | 기준 실행의 Ours: 재배열 없음, 총 KV 예산 25 %, layer 배분 PYR-b20 |

구현 방식: LLaVA-NeXT는 `scripts/209_eval_llava_next_mtvqa_main.py`, LLaVA-1.5는 `scripts/206_eval_llava15_mt_gqa_main.py`를 import해 그 요청·저장·활성화 함수를 그대로 부른다(두 스크립트가 위 기준 실행들의 호출이다). 동결 단계에서 서버 인자, IMPRESS 인자, 읽기 방식, arm 정의, layer 배분 벡터의 SHA256이 기준 실행의 `config.json`과 같고 소스 파일이 가장 최근 기준 실행의 동결과 같음을 assert한다.

이 실행기가 기준 실행과 다르게 주는 것은 세 가지뿐이다: 프롬프트 문자열(2절), 서버 객체의 생성 상한(16 → 1024), store 메타데이터의 데이터셋 이름.

ConvBench 프롬프트는 이미지 앞에 system 문장이 있고(이미지 토큰의 시작 위치가 달라짐) 턴 사이에 종료 토큰이 있다. 세 방법이 이 배치에서 올바르게 동작하는지는 5절의 gate로 확인한다.

## 4. 실험

- 데이터: ConvBench 577 대화 × 3턴(`data/convbench/index.json`, 기존 index 그대로).
- 모델 2 × 방법 4 × 577 대화 × 3턴 = 13,848 요청. 대화의 arm 순서 = [ReComp, FullLoad, IMPRESS, Ours]를 (index 순번 mod 4)만큼 회전.
- 한 드라이버가 LLaVA-NeXT를 끝낸 뒤 LLaVA-1.5를 돌린다. 판정은 두 모델의 생성이 모두 끝난 뒤 실행한다.
- 대화 하나의 흐름: 네 방법이 각자 T1을 픽셀부터 계산 → store를 쓰는 방법은 자기 T1 요청의 KV로 자기 store를 씀 → T2, T3 → 대화 확정(기록을 원자적으로 씀) → 그 대화의 store 삭제.
- 본실험은 577 대화 전부를 새로 돌린다. 4대화·파일럿 40대화의 결과는 본실험 표에 쓰지 않는다.
- 실패 처리: 요청이 예외로 끝나면 방법·턴·원인을 기록한다. 대화에서 처음 실패가 나면 그 대화를 네 방법 모두 T1부터 한 번 더 실행한다. 다시 실패하면 실패를 기록한 채 대화를 확정하고, 그 방법의 뒤 턴은 실행하지 않고 실패로 적는다. 대화를 빼지 않는다. 분모는 577로 고정한다. 실패한 대화가 연속 5개 또는 누적 40개를 넘으면 그 단계를 멈춘다(체계적 오류로 본다).
- 판정에서의 실패한 요청: 그 턴의 단계와 뒤 단계는 "생성 실패"로 0승. 앞 단계는 빠진 뒤 답을 빈 문자열로 두고 판정하고 건수를 보고한다(공식 판정 프롬프트는 단계마다 세 턴의 답을 모두 넣는다).
- 통계: 대화 단위 paired bootstrap 10,000회, seed 1234, 95% CI(percentile). 품질은 대화별 (S1·S2·S3 승리의 평균 × 100)의 차이, TTFT는 대화별 T2·T3 차이의 평균, 비율은 합의 비.
- 결과 디렉터리: `results/convbench_main_<UTC>/`.

## 5. Gate (모델마다)

순서: 4대화(9 arm) → 파일럿 40대화(4 arm). 4대화 = index 순번 72, 216, 360, 504. 파일럿 = 순번 7 + 14i (i = 0…39).

4대화 단계의 9 arm: 네 방법 + IMPRESS 동기 읽기, IMPRESS 유지 비율 100 %, IMPRESS의 재배치 store를 읽는 FullLoad loader, Ours 동기 읽기, Ours 두 번째 실행.

| Gate | 통과 기준 | 판정 위치 |
|---|---|---|
| 1 | 기존 테스트(`python -m unittest discover -s tests`, 본실험 환경)에서 새 실패 0개. 이전부터 실패하던 테스트(이 환경에서는 GLM 관련 3개)는 목록만 적는다 | 모델 공통, 한 번 |
| 2 | 본실험 실행기의 ReComp가 점검 실행의 최종 ReComp의 같은 요청과 모두 일치. 점검 실행은 생성 토큰 ID 목록을 남기지 않았으므로 답 문자열, 생성 토큰 수, 첫 토큰 ID, 입력 토큰 수가 모두 같아야 한다 | 4대화(12요청), 파일럿(120요청) |
| 3 | Ours: 동기 읽기와 비동기 읽기의 생성 토큰·읽기 바이트가 같고, 같은 요청을 두 번 돌려 생성 토큰이 같다 | 4대화 |
| 4 | IMPRESS: 유지 비율 100 %의 생성 토큰이 같은 재배치 store를 읽는 FullLoad loader의 생성 토큰과 같다. 비동기와 동기의 생성 토큰·읽기 바이트가 같다 | 4대화 |
| 5 | T1 생성 토큰이 네 방법에서 모두 같다 | 4대화, 파일럿 |
| 6 | 판정기: 점검 실행의 최종 ReComp 답 8대화(24 단계 판정)를 빈 cache에서 다시 판정해 승자와 해석 단계가 그때의 기록과 모두 같다(판정문 글자 일치 건수도 적는다) | 모델마다 그 모델의 점검 답으로, 생성 전에 |
| 7 | 파일럿 40대화를 4 arm으로 돌려 표를 만든다. 파일럿에 4096 토큰을 넘는 입력이 없으면 그런 대화 2개(점검 실행에서 ReComp 입력이 가장 긴 2대화)를 네 방법으로 따로 돌려 오류 없이 끝나는지 확인한다. 기록용이며 진입 조건이 아니다 | 파일럿 뒤 |

- 4대화 단계에서는 실패한 요청이 하나라도 있으면 FAIL이다.
- gate 1–6이 모두 PASS이면 드라이버가 그 모델의 본실험까지 이어서 진행한다.
- gate가 FAIL인 모델은 본실험을 시작하지 않는다. 실패 내용과 원인을 `status.json`과 `report.md`에 적는다. 드라이버는 다른 모델(그 모델 자신의 gate부터), 판정, 보고서를 마친 뒤 종료한다. gate 1 또는 6이 FAIL이면 두 모델 모두 본실험을 시작하지 않는다.
- 기록된 gate 결과는 다시 판정하지 않으며 기준을 바꾸지 않는다. gate 검사 자체가 결과를 내지 못하고 죽은 경우(실행 오류)는 gate 판정이 아니므로 같은 명령으로 다시 시도된다.

## 6. 실행 전 점검 (`prerun.json`)

예상 SSD 쓰기량(이미지별 geometry로 계산), 예상 소요 시간, 여유 공간(가장 큰 대화 store 묶음의 20배 미만이면 시작하지 않음; 대화마다 다시 확인), SSD의 SMART 값(읽을 수 없으면 그 사유와 커널 쓰기 카운터·온도), 점검 실행 기록의 완전성. 대화 하나를 끝내면 그 대화의 store를 지운다.

## 7. 실행 방식

- `scripts/run_convbench_main.sh results/convbench_main_<UTC>` 하나로 실행 전 점검부터 판정·보고서까지 묶는다. setsid + nohup으로 터미널·세션과 분리하고, 사람의 입력을 기다리는 지점이 없다.
- 상태: `status.json`(현재 모델·단계, 완료 대화 수/전체, 시작·최근 갱신 시각, 예상 종료 시각, 누적 SSD 쓰기량, 마지막 오류), `driver.log`, `driver.pid`. 잠금 파일(`driver.lock`, flock)로 두 드라이버가 동시에 뜨지 않는다.
- 이어서 실행: 같은 명령. 끝난 단계·확정된 대화·끝난 판정은 건너뛴다. 끊긴 대화는 네 방법 모두 T1부터 다시 하며, 그 대화의 불완전한 기록은 `discarded/`로 옮겨 쓰지 않고 store는 지우고 다시 만든다. 이어서 실행한 횟수·시점·다시 실행한 대화는 `driver_invocations.jsonl`, `resume_log.jsonl`에 남고 보고서에 적는다.
- 실험이 도는 동안 같은 GPU나 SSD를 쓰는 다른 작업(단위 테스트 포함)을 돌리지 않는다. 다른 프로세스가 GPU를 쓰면 대화 시작 전에 기다리고, 대화 도중에 감지되면 그 대화를 버리고 다시 한다.
- 기존 store, 기존 결과 디렉터리, 동결 파일은 수정하지 않는다. 지우는 것은 이 실행의 `scratch/` 아래뿐이다.

## 8. 보고 항목 (모델별, `report.md`)

1. 품질 표(S1, S2, S3, 평균, 판정 불가 건수; MetaCompress 기본 모델과 점검 실행 ReComp를 참고 행으로). 2. 품질의 짝지은 비교(Ours − IMPRESS, Ours − FullLoad, Ours − ReComp, IMPRESS − FullLoad, FullLoad − ReComp; FullLoad − ReComp는 잡음 수준 참고값). 3. TTFT 표(T2·T3·합친 평균/p50/p95, 짝지은 차이, 비율 Ours/FullLoad, Ours/ReComp, Ours/IMPRESS, IMPRESS/FullLoad). 4. SSD MB/followup, 읽기 호출 수. 5. IMPRESS 통계. 6. 재사용 턴 시간 분해. 7. T1 저장 비용. 8. 입력 길이·생성 길이. 9. SSD 쓰기량(예상·실측), SMART 전후, 소요 시간. 10. 판정 진단. 11. 0단계 결과. 12. 코드 목록, 지시와 다르게 했거나 지시에 없던 것, 측정하지 않은 것과 한계.

## 9. 지시를 해석했거나 지시에 없던 것 (실행 전 고지)

1. gate 2는 점검 실행에 토큰 ID 목록이 없어 답 문자열·생성 토큰 수·첫 토큰 ID·입력 토큰 수로 판정한다.
2. gate 3·4는 추가 arm이 필요하므로 4대화에서만 판정한다. 파일럿은 지시대로 4 arm이다.
3. gate가 실패한 모델만 본실험을 건너뛰고 드라이버는 다른 모델·판정·보고서까지 진행한 뒤 종료한다.
4. "warm-up 없음"은 미리 만든 KV 없이 모든 방법이 T1을 픽셀부터 계산한다는 뜻으로 읽었다. 기준 실행과 같은 프로세스 warm-up 요청 1건(모델을 올린 직후, 측정 제외)은 그대로 둔다.
5. 4절의 실패 처리 규칙(한 번 더 실행, 연속 5·누적 40에서 정지)과 판정에서의 "생성 실패" 처리.
6. 판정 cache: gate 6 통과 뒤, 판정 프롬프트가 점검 실행과 글자까지 같은 경우 점검 실행의 cache를 읽기 전용으로 읽는다(그 디렉터리에는 쓰지 않는다).
7. LLaVA-1.5는 점검 실행에 4096을 넘는 입력이 없다. gate 7의 2대화는 입력이 가장 긴 대화로 고른다.
8. 대화가 끝날 때마다 GPU 메모리 cache를 비운다(측정 구간 밖).
9. TTFT 비율의 bootstrap CI, 품질 비교의 단계별 차이와 판정 결과가 다른 대화 수를 함께 보고한다.
