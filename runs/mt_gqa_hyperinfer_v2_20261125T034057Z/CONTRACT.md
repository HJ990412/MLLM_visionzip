# MT-GQA × {LLaVA-NeXT, LLaVA-1.5}: HyperInfer 재측정 실행 계약 (prefetch를 논문 IV-C대로)

작성 2026-11-25. 실행 전에 동결한다. 드라이버의 freeze 단계가 이 문서와 원본 대조표의 SHA256을 `config.json`에 기록하고, 이후 모든 단계가 시작할 때 다시 확인한다.

## 1. 배경과 목적

2026-11-24 실행(`results/mt_gqa_hyperinfer_20261124T171039Z`, 아래 "이전 실행")에서 HyperInfer의 TTFT가 IMPRESS보다 1.07배(LLaVA-NeXT), 1.04배(LLaVA-1.5) 느렸다. prefetch가 현재 layer의 계산이 끝난 뒤에도 계속되어 다음 layer가 그것을 기다렸기 때문이다(요청당 107.9 ms, 41.9 ms). 이전 실행은 공개 코드처럼 예상 시간으로 읽을 양을 미리 정했다.

논문 IV-C는 prefetch가 현재 layer의 계산 시간 안에서만 이루어진다고 쓴다. "CPU는 GPU 상태를 보면서 읽기 요청을 계속 내다가, GPU가 현재 layer의 계산을 끝내는 즉시 새 요청을 내지 않는다." Fig. 10(c)에서는 그 뒤 token selection을 하고 빠진 KV만 읽는다. 이 실행은 prefetch를 멈추는 방식만 이 서술대로 바꿔 MT-GQA를 다시 측정한다.

- 본실험에서 돌리는 방법은 HyperInfer 하나다.
- ReComp, FullLoad, IMPRESS, Ours는 다시 돌리지 않고 기존 본실험의 요청별 기록을 쓴다.
  - LLaVA-NeXT: `runs/mt_gqa_main_v2_20261124T051939Z` (2026-11-24, 공식 토큰열)
  - LLaVA-1.5: `runs/llava15_mt_gqa_main_20261120T165001Z` (2026-11-21)
- 이전 실행의 HyperInfer 기록(`runs/mt_gqa_hyperinfer_20261124T171039Z`)은 비교 대상으로 쓴다.
- 측정 보고다. GO/NO-GO 판정은 없다. TTFT가 IMPRESS보다 빠른지 느린지는 gate가 아니며 결과 그대로 보고한다.

## 2. 참조와 환경

- 논문: `papers/HyperInfer.md` IV-C, Fig. 10(c).
- 공개 코드: https://github.com/ISCS-ZJU/HyperInfer, commit `91df7a2fc12581c6780a6485c5b50e7fd9ef9d0e`. 코드를 복사하지 않는다.
- 원본 대조표: `docs/hyperinfer_port_contract.md` (2026-11-25 개정: 자르기는 논문의 서술을 따름, 공개 코드는 예상 시간으로 미리 자름). 이전 실행 때의 표는 `runs/mt_gqa_hyperinfer_20261124T171039Z/PORT_TABLE.md`에 그대로 있다.
- 환경: 기존 본실험과 같은 `mllm_ft` (transformers 4.57.6, torch 2.5.1+cu121). freeze가 기존 본실험의 `config.json`과 대조한다.
- 기존 본실험 두 개와 이전 실행이 동결한 소스 파일은 하나도 바꾸지 않는다. freeze가 세 실행의 `source_sha256`과 현재 파일을 대조해 다르면 멈춘다.

## 3. 바꾼 것: prefetch를 멈추는 방식

구현은 새 파일 `mmimpress/hyperinfer_v2.py`다. 이전 이식본 `mmimpress/hyperinfer.py`는 수정하지 않고, 바뀌지 않는 부분(예측 집합 → chunk 번호 계산, chunk 하나를 읽는 함수, 디버그용 NaN 검사)은 거기서 그대로 불러 쓴다. IMPRESS 이식본(`mmimpress/impress_official.py`)의 선택기를 상속해 선택 코드를 그대로 실행하는 구조도 같다.

재사용 턴(T2–T3)에서의 동작:

1. **제출.** layer l의 선택·읽기·GPU 적재가 끝나고 layer l의 계산이 시작되기 직전에, worker에게 layer l+1용 작업을 준다. 예측 집합은 layer l의 사용 집합 G_l 전체다(이전과 같다).
2. **worker.** (1) layer l+1의 probe key를 읽는다. (2) 예측 chunk를 layer l+1의 저장 순서에서 chunk 번호 오름차순(= 중요도가 높은 순)으로, chunk마다 K chunk → V chunk를 읽는다. 읽기 하나는 pread 한 번이고 K chunk 하나 또는 V chunk 하나다. worker는 읽기 하나를 시작하기 전마다 멈춤 표시를 확인한다.
3. **멈춤.** layer l+1의 hook 첫머리에서 요청 thread가 `torch.cuda.synchronize()`로 layer l의 GPU 계산이 실제로 끝나기를 기다린 뒤, 곧바로 멈춤을 표시한다. 추정값을 쓰지 않는다. 멈춤 표시와 worker의 "읽기 시작" 판단은 같은 lock 안에서 하고 시각도 그 안에서 기록한다. 그래서 허용된 읽기의 판단 시각은 항상 멈춤 시각보다 앞선다.
   - 멈춤 뒤에는 새 chunk 읽기를 시작하지 않는다. 이미 시작한 읽기 하나만 끝낸다.
   - 그때까지 읽기를 마친 K chunk와 V chunk가 prefetch된 집합 P_{l+1}이다.
4. **선택.** IMPRESS와 같은 방법으로 G_{l+1}을 구한다. 선택에는 probe key가 반드시 필요하므로, probe key 읽기가 아직 안 끝났으면 요청 thread가 그것만 기다린다(probe key 대기).
5. **복구.** G_{l+1}의 토큰이 든 chunk 중 prefetch되지 않은 것만 IMPRESS의 읽기 함수로 동기로 읽는다. K와 V를 따로 본다. K만 prefetch된 chunk는 V만 읽는다.
   - 진행 중이던 읽기 하나는 선택 계산과 겹쳐 흐른다. 요청 thread는 선택을 끝내고 자기 첫 KV 읽기를 하기 직전에 worker가 끝났는지 확인하고, 아직이면 기다린다(진행 중이던 읽기의 마무리 대기).
6. **출력.** attention에는 G_{l+1}만 쓴다. GPU에는 G_{l+1}의 토큰이 든 chunk만 올리고, 나머지는 IMPRESS의 mask로 가린다. prefetch됐지만 필요 없는 chunk는 버린다. 출력은 IMPRESS와 같아야 한다.
7. 마지막 layer에서는 제출하지 않는다. layer 0에는 prefetch가 없다. 첫 prefetch(layer 0 → 1)도 위 규칙을 그대로 따른다.

쓰지 않는 것: 시간 예산, chunk당 읽기 시간 상수, 앞 절반 규칙. 상수 측정 단계도 없다.

바꾸지 않은 것: 예측 집합, 읽는 단위(K chunk와 V chunk), 선택, fallback, 저장 형식(IMPRESS 이식본과 같은 store), 검증·복구, 출력(G_l만 사용), worker 1개(디스크 읽기와 host 배열 변환만, GPU 복사는 요청 thread), 캐시 계층 없음, cold SSD, warm-up 없음, prefetch를 끄면 IMPRESS 선택기가 그대로 도는 것.

두 대기 시간(probe key 대기, 진행 중이던 읽기의 마무리 대기)은 TTFT에 들어가고 요청마다 layer별로 기록한다.

## 4. 데이터, 방법, 단계

- 데이터: MT-GQA 398 images / 4,061 dialogues × 3턴 = 모델당 12,183요청. 프롬프트, 생성 설정(greedy, 최대 16토큰, seed 1234), 채점(strict normalized EM), 대화 순서, page cache 처리, 타이머 구간은 이전 실행과 같다(같은 하네스 `scripts/222`, `scripts/206`을 같은 방식으로 불러 쓴다). LLaVA-NeXT는 공식 토큰열 경로다.
- 매 대화 T1은 픽셀부터 계산하고, T2–T3는 자기 생성 이력을 쓴다. 이미지 하나씩: 첫 대화의 T1에서 store를 만들고 → 서빙 → 감사 → 커밋 → store 삭제.
- 실행 arm:

| 표의 이름 | method id | 쓰임 |
|---|---|---|
| FullLoad | `fullload_async` | gate 7의 표 |
| IMPRESS | `impress_official_async` | gate 1, 2, gate 7의 표, I/O ratio 대조값 |
| HyperInfer | `hyperinfer_async` | 본실험 arm. gate 1, 5 |
| HyperInfer, prefetch 끔 | `hyperinfer_off` | gate 2 |
| HyperInfer, 두 번째 실행 | `hyperinfer_async_repeat` | gate 4, 5 |
| HyperInfer, 읽지 않은 행 NaN | `hyperinfer_async_nan` | gate 3, 5 |

- 단계 (모델마다, LLaVA-NeXT를 끝낸 뒤 LLaVA-1.5):

| phase | 이미지 | 대화 | arm | 요청 수 |
|---|---|---|---|---|
| `gate` | 기존과 같은 gate 이미지 4장 | 49 | 6 | 882 |
| `dev` | DEV 40장 | 378 | 6 | 6,804 |
| `main` | 398장 전체 | 4,061 | HyperInfer 1 | 12,183 |

- arm 순서는 대화의 데이터셋 내 순번으로 회전한다(기존과 같은 규칙). gate와 DEV 이미지는 본실험 398장에 포함된다.

## 5. Gate

두 모델 각각, 4장 → DEV 40장 순서로 본다. 기준은 아래에 고정하며 실행 중에 바꾸지 않는다.

1. **출력 동일성.** `hyperinfer_async`의 모든 요청에서 생성 토큰(T1–T3)과 layer별 사용 집합(T2–T3)이 같은 세션의 IMPRESS arm과 같다.
2. **prefetch 끔 ≡ IMPRESS.** `hyperinfer_off`의 모든 요청에서 생성 토큰, 읽기 바이트(전체, probe, K, V), 읽기 호출 수가 IMPRESS arm과 같다.
3. **읽지 않은 KV를 쓰지 않음.** 이전 실행의 gate 5와 같은 방식이다. `hyperinfer_async_nan`은 prefix cache의 모든 이미지 행을 NaN으로 채우고 시작하고, 읽어 온 행만 실제 값으로 덮인다. 디버그 래퍼는 mask된 열에 한해 K와 V를 0으로 넘긴다(0은 정상 요청에서 읽지 않은 행이 갖는 값이다). 통과 조건: 모든 layer의 attention 출력이 유한하고, mask되지 않은 이미지 열이 G_l과 같고, 사용한 행이 모두 읽은 행이며, 생성 토큰과 사용 집합이 IMPRESS와 같다.
4. **두 번 실행.** `hyperinfer_async_repeat`의 생성 토큰이 `hyperinfer_async`와 모든 요청에서 같다. prefetch된 양은 실행마다 달라질 수 있고, 얼마나 달랐는지 함께 적는다.
5. **멈춤 확인.** prefetch를 켠 세 arm의 모든 요청, 모든 layer에서 계산이 끝난 뒤에 새로 시작된 prefetch 읽기가 0건이다. 판정은 읽기마다 기록한 판단 시각과 layer마다 기록한 멈춤 시각의 비교다(둘 다 같은 lock 안에서 잰 값). 멈춤 뒤에 끝난 읽기가 한 layer에 둘 이상이어도 실패다(진행 중이던 읽기는 하나뿐이어야 한다).
6. **기존 테스트.** 두 환경(transformers 4.57.6, 5.13.1)의 전체 테스트에서 새 실패가 0개다. 이전부터 실패하던 테스트는 목록만 적는다.
7. **DEV 40장 표 (gate 아님, 보고 항목).** IMPRESS, FullLoad, HyperInfer를 arm 순서 회전으로 함께 돌려 followup EM, TTFT, 읽기량의 표를 만든다. 같은 실행에서 잰 HyperInfer / IMPRESS TTFT 비율이 여기서 나온다.

admission: gate 6과 그 모델의 `gate`·`dev` 단계 판정(gate 1–5)이 모두 PASS일 때만 그 모델의 본실험을 시작한다. 한 모델의 gate가 실패하면 그 모델의 본실험만 건너뛰고 다른 모델은 계속한다. gate 6이 실패하면 어느 모델도 측정하지 않는다. 실패 내용과 원인은 `status.json`과 `report.md`에 적는다.

## 6. 보고 항목 (모델별, `results/<run>/report.md`)

1. 결과 표: 다섯 방법의 followup EM, TTFT 평균/p50/p95, SSD MB/followup. HyperInfer는 이번 실행, 나머지는 기존 값이며 실행일을 함께 적는다.
2. 짝지은 비교(이미지 단위 paired bootstrap 10,000회, seed 1234, 95% CI)와 비율: Ours 대 HyperInfer, HyperInfer 대 IMPRESS, HyperInfer 대 FullLoad, HyperInfer 대 ReComp. 같은 실행에서 잰 HyperInfer / IMPRESS 비율(DEV 40장)도 함께 적는다.
3. 이전 실행과의 비교: TTFT, prefetch 대기, prefetch로 채워진 비율.
4. layer당 GPU 계산 시간, probe key 읽기 시간, chunk당 읽기 시간의 실측 분포와, 한 layer의 계산 시간 안에 읽기를 마친 chunk 수(layer별 평균). GPU 계산 완료를 판단하는 방법과 그 비용.
5. prefetch 통계: 사용 집합 중 prefetch로 채워진 비율, prefetch된 것 중 실제로 쓴 비율, 읽었지만 쓰지 않은 바이트, IMPRESS 대비 총 읽기 바이트, 인접 layer 선택의 Jaccard 유사도.
6. 임계 경로에 남은 시간 분해: probe key 대기, 진행 중이던 읽기의 마무리 대기, 선택 계산, 빠진 KV 동기 읽기. I/O ratio(IMPRESS는 DEV 값).
7. 출력 동일성: 본실험 12,183요청 중 기존 IMPRESS 기록과 생성 토큰이 일치한 요청 수.
8. 바꾼 코드 목록과 새 테스트, 지시와 다르게 했거나 지시에 없던 것, 한계.

보고서에서 방법은 ReComp, FullLoad, IMPRESS, HyperInfer, Ours로만 부른다.

## 7. 실행 방식

- 드라이버 하나(`scripts/run_mt_gqa_hyperinfer_v2.sh runs/mt_gqa_hyperinfer_v2_<UTC>`)가 freeze → gate → DEV → 본실험 → 보고서를 묶어, 새 세션(setsid + nohup)에서 돈다. 사람의 입력을 기다리는 지점이 없다.
- `results/<run>/`에 `status.json`(현재 모델·단계, 완료 요청 수 / 전체, 예상 종료 시각, 누적 SSD 쓰기량, 마지막 오류), `driver.log`, `driver.pid`를 계속 갱신한다.
- 같은 명령으로 이어서 돈다. 커밋된 이미지는 건너뛰고, 끊긴 이미지 안에서는 완료된 대화를 유지하고 끊긴 대화를 T1부터 다시 한다. 이어서 실행한 횟수와 다시 실행한 대화 목록을 보고서에 적는다. 잠금 파일로 중복 실행을 막는다.
- 실험이 도는 동안 같은 GPU나 SSD를 쓰는 다른 작업을 돌리지 않는다.
- 결과: 행과 증거는 `runs/<run>/llava_next/`, `runs/<run>/llava15/`, 판정은 `runs/<run>/gates/`, 보고서와 상태는 `results/<run>/`.
- 기존 store, 결과 디렉터리, 동결 파일은 수정하지 않는다. freeze 때와 종료 때 `results/`, `runs/`, `kvstore*`의 파일 목록(크기, 수정 시각)을 비교해 보고서에 적는다. 삭제하는 것은 이 실행의 scratch store뿐이다.
- 예상 SSD 쓰기량과 소요 시간은 freeze가 계산해 `prerun.json`에 적는다(대략 1.4 TB, 7시간).

## 8. 지시에 없던 것과 해석한 것

1. **읽기 하나의 단위.** K chunk 하나 또는 V chunk 하나를 읽기 하나로 본다(pread 한 번). 그래서 멈춘 시점에 따라 마지막 chunk는 K만 읽혀 있을 수 있다. 그 K는 버리지 않고 쓰며 V만 동기로 읽는다. "prefetch로 채워진 토큰"은 K와 V가 모두 prefetch된 chunk의 토큰으로 센다.
2. **진행 중이던 읽기를 기다리는 위치.** 멈춤 직후가 아니라 선택 계산이 끝난 뒤, 요청 thread의 첫 KV 읽기 직전에 기다린다. 읽기가 선택 계산과 겹쳐 흐르므로 대기가 줄어든다. HyperInfer에 유리한 쪽이다.
3. **GPU 계산 완료의 판단.** layer l+1의 hook 첫머리의 `torch.cuda.synchronize()`가 돌아온 시점이다. layer l의 마지막 연산을 GPU에 넘긴 직후에 호출되므로, 실제 완료와 이 시점 사이에는 요청 thread가 깨어나는 시간만 있다. 비용은 (a) 아무 일도 없을 때 동기화 호출 한 번의 시간(단계 시작 때 2,000번 재서 기록), (b) 요청 thread가 동기화에서 실제로 막힌 시간(layer별 기록)으로 보고한다. (b)는 직전 layer의 계산을 기다린 시간이고, 선택이 그 출력을 필요로 하므로 IMPRESS에서도 선택의 첫 GPU 연산에서 치르는 시간이다.
4. **"읽기 시작" 시각의 정의.** worker가 lock 안에서 멈춤 표시가 없음을 확인한 시각이다. 실제 pread 호출은 lock을 놓은 직후에 나가므로 멈춤 시각보다 수 µs 늦을 수 있다. 그 차이의 최댓값을 gate 5 결과에 함께 적는다.
5. **probe key 읽기.** 멈춤이 먼저 와도 probe key 읽기는 시작하고 끝낸다(선택에 반드시 필요하다).
6. **gate 1의 비교 대상.** 같은 세션에서 함께 돌린 IMPRESS arm이다(layer별 사용 집합은 그 arm에서만 기록된다). DEV 40장에서는 기존 본실험의 IMPRESS 기록과의 토큰 일치도 세어 보고한다(판정 아님).
7. **gate 2의 범위.** 지시의 세 항목(생성 토큰, 읽기 바이트, 읽기 호출 수)으로 판정한다. layer별 사용 집합의 일치는 참고로 함께 적는다.
8. **gate 5의 대상.** prefetch를 켠 세 arm(HyperInfer, 두 번째 실행, NaN 검사) 전부다.
9. **단계 구성.** 4장과 DEV 40장을 각각 6개 arm으로 한 번씩 돈다. gate 1–5와 gate 7의 표를 같은 실행에서 본다. gate 7의 FullLoad·IMPRESS TTFT는 6개 arm 회전 속에서 잰 값이다.
10. **공개 코드 대조 gate.** 이번에는 두지 않는다. 멈추는 방식을 공개 코드와 다르게(논문대로) 했기 때문이다. 바뀌지 않은 부분(예측 → chunk 번호)의 공개 코드 대조 테스트는 `tests/test_hyperinfer.py`에 그대로 있고 gate 6의 테스트 묶음에 포함된다.
11. **파일.** 이전 이식본과 이전 실행의 스크립트는 그 실행이 동결한 파일이라 수정하지 않고 새 파일을 만들었다. 원본 대조표 문서(`docs/hyperinfer_port_contract.md`)는 지시대로 수정했다. 이 문서는 이전 실행도 SHA256을 기록해 둔 파일이고, 이전 실행 때의 내용은 그 실행의 `PORT_TABLE.md`에 남아 있다.
12. **이전 실행과의 비교.** 같은 이미지·대화·턴의 요청끼리 짝짓는다. 서로 다른 날의 실행이므로 TTFT 차이에는 실행 간 변동이 섞인다.
