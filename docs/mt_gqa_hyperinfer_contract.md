# MT-GQA × {LLaVA-NeXT, LLaVA-1.5}: HyperInfer 측정 실행 계약

작성 2026-11-24. 실행 전에 동결한다. 드라이버의 freeze 단계가 이 문서, 원본 대조표, 상수 파일의 SHA256을 `config.json`에 기록하고, 이후 모든 단계가 시작할 때 다시 확인한다.

## 1. 목적

비교 대상을 IMPRESS에서 HyperInfer로 바꾼다. HyperInfer(IEEE TC 2026)는 IMPRESS에 "다음 layer의 중요 KV를 현재 layer 계산 중에 미리 읽는" 기법을 더한 것이다. 이 실행은 HyperInfer를 우리 MLLM Visual KV 조건에 이식해 구현을 검증하고, MT-GQA에서 두 모델로 측정한다.

- 본실험에서 돌리는 방법은 HyperInfer 하나다.
- ReComp, FullLoad, IMPRESS, Ours는 다시 돌리지 않고 기존 본실험의 요청별 기록을 쓴다.
  - LLaVA-NeXT: `runs/mt_gqa_main_v2_20261124T051939Z` (2026-11-24, 공식 토큰열, 네 방법)
  - LLaVA-1.5: `runs/llava15_mt_gqa_main_20261120T165001Z` (2026-11-21, 네 방법)
- 측정 보고다. GO/NO-GO 판정은 없다. TTFT가 IMPRESS보다 빠른지 느린지는 gate가 아니며 결과 그대로 보고한다.

## 2. 참조와 환경

- 논문: `papers/HyperInfer.md` (IV-A, IV-B, IV-C, VI-A, VI-D, VI-F).
- 공개 코드: https://github.com/ISCS-ZJU/HyperInfer, commit `91df7a2fc12581c6780a6485c5b50e7fd9ef9d0e`, 체크아웃 `/home/dblab/hj/_external_ref/HyperInfer` (프로젝트 밖). LICENSE 파일이 없어 코드를 복사하지 않고 동작만 대조한다.
- 원본 대조표(0단계): `docs/hyperinfer_port_contract.md`. 보고서에 최종본을 싣는다.
- 환경: 기존 두 본실험과 같은 `mllm_ft` (transformers 4.57.6, torch 2.5.1+cu121). freeze가 두 본실험의 `config.json`과 대조한다.
- 기존 본실험이 동결한 소스 파일은 하나도 바꾸지 않는다. freeze가 두 본실험의 `source_sha256`과 현재 파일을 대조해 다르면 멈춘다.

## 3. 이식 정의

구현은 `mmimpress/hyperinfer.py`다. IMPRESS 이식본의 선택기(`OfficialImpressSelector`)를 상속해 선택 코드를 그대로 실행하고, 그 선택기가 store를 읽는 지점만 prefetch buffer를 거치게 바꾼다. IMPRESS 이식본(`mmimpress/impress_official.py`)의 선택, fallback, 저장 형식, 재배치는 바꾸지 않는다.

재사용 턴(T2–T3)에서 layer l을 처리하는 순서:

1. l > 0이면 GPU 동기화로 layer l−1의 계산이 끝난 시각을 잡고(계산 시간 측정), layer l용으로 제출해 둔 prefetch가 끝나기를 기다린다. 기다린 시간은 TTFT에 들어간다.
2. layer l의 probe key로 IMPRESS와 같은 방법으로 사용 집합 G_l을 구한다(선택된 이미지 토큰 + 항상 유지하는 separator; 평균 Jaccard ≥ 0.3이면 투표 선택, 아니면 fallback으로 전체).
3. prefetch된 chunk 집합과 비교해, G_l의 토큰이 든 chunk 중 prefetch되지 않은 chunk만 동기로 읽는다(G_l \ P_l).
4. attention에는 G_l만 쓴다. GPU에는 G_l의 토큰이 든 chunk만 올리고(IMPRESS와 같은 chunk 집합), G_l 밖의 토큰은 IMPRESS의 mask로 가린다. prefetch됐지만 G_l 토큰이 없는 chunk는 버린다. 따라서 HyperInfer의 출력은 IMPRESS의 출력과 같아야 한다.
5. layer l의 계산 직전에 layer l+1용 prefetch를 제출한다.
   - 예측 집합: G_l 전체.
   - 단위: G_l을 layer l+1의 저장 순서로 옮겨 chunk 번호를 구하고, 오름차순으로 K chunk → V chunk를 chunk 하나씩 통째로 읽는다.
   - 자르기: 공개 코드의 시간 예산. 예산 = 직전 layer의 (계산 시간 + 선택 시간). chunk마다 (K chunk 읽기 시간 + V chunk 읽기 시간) 상수를 빼 가며 예산을 넘는 첫 chunk 앞에서 자른다. 예산이 0이면 예측 chunk의 앞 절반을 읽는다.
   - 예산은 공개 코드처럼 요청 사이에 이어진다(프로세스 안에서 arm별로 하나). 예산이 0인 경우는 프로세스의 첫 요청뿐이다.
6. 마지막 layer에서는 제출하지 않는다.

고정 사항:

- chunk당 읽기 시간 상수는 4절의 값이다.
- 다음 layer의 probe key 미리 읽기는 IMPRESS 이식본과 같이 유지한다. worker는 probe key를 먼저 읽고 이어서 예측 chunk를 읽는다. probe 읽기 시간은 예산에서 빼지 않는다. 공개 코드와 다르고 baseline에 유리한 차이다.
- 저장 형식은 IMPRESS 이식본과 같다(전체 Visual KV + probe key 사본, K/V 분리, 64토큰 chunk, 첫 대화 T1 기준 layer별 재배치). 새 형식을 만들지 않는다. HyperInfer와 IMPRESS arm은 같은 store를 읽는다.
- 캐시 계층 없음, cold SSD(요청 전 page cache drop), warm-up 없음. 이미지 토큰만 대상으로 한다.
- worker는 1개이고 디스크 읽기와 host 배열 변환만 한다. GPU 복사와 cache 기록은 요청 thread가 한다(FullLoad, IMPRESS, Ours의 비동기 읽기와 같은 규칙). 공개 코드와 다르다(대조표 m).
- prefetch를 끄는 옵션(`kv_prefetch="off"`)에서는 이 모듈의 재정의가 하나도 실행되지 않고 IMPRESS 선택기가 그대로 돈다.
- layer l → l+1 위치 표는 store의 `meta.json`에서 만들며, 요청 타이머 밖에서 store마다 한 번 만든다.

## 4. 동결한 chunk 읽기 시간 상수

측정 방법: MT-GQA에 없는 이미지 4장(MT-VQA 세트의 COCO val2014 이미지, 대화 목록의 앞에서부터)으로 모델마다 IMPRESS 형식 store를 실제 writer로 만든다. store의 page cache를 비운 뒤 layer 0부터 순서대로, 각 layer의 모든 chunk를 오름차순으로 K chunk → V chunk로 한 번씩 읽는다(prefetch worker와 같은 읽기 함수, pread 1회씩). 이 과정을 이미지당 3번 한다. 64토큰이 다 찬 chunk의 읽기 시간 평균을 K와 V 각각의 상수로 쓴다. GPU는 모델만 올려 둔 채 쉬는 상태다. 측정 결과 전체는 `results/<run>/calibration_<model>.json`에 있다.

<!-- CHUNK_TIME_CONSTANTS:BEGIN -->
| 모델 | K chunk 읽기 시간 (ms) | V chunk 읽기 시간 (ms) | 표본 수 | 측정 시각 (UTC) |
|---|---|---|---|---|
| LLaVA-NeXT-7B | 0.19613 | 0.20938 | 14,208 | 2026-11-24T17:11:17Z |
| LLaVA-1.5-7B | 0.25588 | 0.20804 | 3,456 | 2026-11-24T17:11:34Z |

상수 파일: `docs/mt_gqa_hyperinfer_chunk_times.json` (SHA256은 `config.json`의 `constants_sha256`). 값은 표본 평균이다. 공개 코드의 디스크 상수는 0.40818 ms다.
<!-- CHUNK_TIME_CONSTANTS:END -->

## 5. 데이터, 방법, 단계

- 데이터: MT-GQA 398 images / 4,061 dialogues × 3턴 = 모델당 12,183요청. 프롬프트, 생성 설정(greedy, 최대 16토큰, seed 1234), 채점(strict normalized EM), 대화 순서는 기존 본실험과 같다. LLaVA-NeXT는 공식 토큰열 경로를 쓰고 요청마다 원래 LLaVA 토큰 id와 대조한다.
- 매 대화 T1은 픽셀부터 계산하고, T2–T3는 자기 생성 이력을 쓴다. page cache 처리, 타이머 구간, store 삭제 시점, 요청 사이의 처리는 기존 본실험 하네스(`scripts/222`, `scripts/206`)의 함수를 그대로 불러 쓴다. 이미지 하나씩: 첫 대화의 T1에서 store를 만들고 → 모든 대화를 서빙하고 → 감사하고 → 커밋하고 → store를 지운다.
- 실행 arm (표의 이름 / method id):

| 표의 이름 | method id | 쓰임 |
|---|---|---|
| FullLoad | `fullload_async` | gate 8 (TTFT 연속성) |
| IMPRESS | `impress_official_async` | gate 2, 3, 8과 보고서의 I/O ratio 대조값 |
| HyperInfer | `hyperinfer_async` | 본실험 arm |
| HyperInfer, prefetch 끔 | `hyperinfer_off` | gate 2 |
| HyperInfer, prefetch 동기 실행 | `hyperinfer_sync` | gate 6 |
| HyperInfer, 두 번째 실행 | `hyperinfer_async_repeat` | gate 6 |
| HyperInfer, 읽지 않은 행 NaN | `hyperinfer_async_nan` | gate 5 |

- 단계 (모델마다, LLaVA-NeXT를 끝낸 뒤 LLaVA-1.5):

| phase | 이미지 | 대화 | arm | 요청 수 |
|---|---|---|---|---|
| `gate` | 기존 본실험과 같은 gate 이미지 4장 | 49 | 7 | 1,029 |
| `dev` | DEV 40장 | 378 | 7 | 7,938 |
| `main` | 398장 전체 | 4,061 | HyperInfer 1 | 12,183 |

- arm 순서는 대화의 데이터셋 내 순번으로 회전한다(기존과 같은 규칙).
- gate와 DEV 이미지는 본실험 398장에 포함된다(기존 본실험과 같다).

## 6. 구현 검증 (gate)

두 모델 각각, 4장 → DEV 40장 순서로 본다. 기준은 아래에 고정하며 실행 중에 바꾸지 않는다.

1. **로직 대조.** `tests/test_hyperinfer.py`가 공개 코드의 `PrefixKVLayer.prefetch_by_chunk`와 `_prefetch_chunks_direct_pinned`를 직접 호출해, 같은 입력에서 저장 순서 변환, chunk 번호, 예산 자르기(앞 절반 규칙 포함), K → V 읽기 순서, prefetch된 토큰 집합이 이식본과 같은지 본다. 예측 집합과 빠진 토큰 계산은 공개 코드에서 1,300줄짜리 layer 함수 안의 한 줄씩이라 모델 없이 부를 수 없다. 그 연산을 테스트에 그대로 적어 비교한다. 통과 조건: 테스트 전부 통과, 공개 코드 대조 테스트가 건너뛰어지지 않음, 체크아웃 HEAD가 위 commit.
2. **prefetch 끔 ≡ IMPRESS.** `hyperinfer_off`의 모든 요청에서 생성 토큰(T1–T3), layer별 사용 집합, 읽기 바이트(전체, probe, K, V), 읽기 호출 수가 같은 세션의 IMPRESS arm과 같다.
3. **출력 동일성.** `hyperinfer_async`의 모든 요청에서 생성 토큰(T1–T3)과 layer별 사용 집합(T2–T3)이 IMPRESS arm과 같다.
4. **기존 기록과의 일치 (DEV 40장).** `hyperinfer_async`의 T1·T2·T3 생성 토큰이 기존 본실험의 IMPRESS 기록의 같은 요청과 모두 같다.
5. **읽지 않은 KV를 쓰지 않음.** `hyperinfer_async_nan`은 요청의 prefix cache에서 모든 이미지 행을 NaN으로 채운 뒤 시작한다. 읽어 온 행만 실제 값으로 덮인다. 통과 조건: 모든 layer의 attention 출력이 유한하고, mask되지 않은 이미지 열이 G_l과 같고, 사용한 행이 모두 읽은 행이며, 생성 토큰과 사용 집합이 IMPRESS와 같다(3번과 같은 결과).
   - mask는 더하는 방식이라 mask된 열의 NaN도 그대로 두면 softmax가 NaN이 된다. 그래서 디버그 래퍼가 mask된 열에 한해 K와 V를 0으로 넘긴다. 0은 정상 요청에서 읽지 않은 행이 실제로 갖는 값이다. mask되지 않은 열은 건드리지 않으므로, 읽지 않은 행이 하나라도 사용되면 출력이 NaN이 되어 실패한다.
6. **결정성.** `hyperinfer_sync`(prefetch를 요청 thread에서 실행)와 `hyperinfer_async_repeat`(같은 요청을 한 번 더)의 생성 토큰이 `hyperinfer_async`와 모든 요청에서 같다.
7. **기존 테스트.** 두 환경(transformers 4.57.6, 5.13.1)의 전체 테스트에서 새 실패가 0개다. 이전부터 실패하던 테스트는 목록만 적는다.
8. **TTFT 연속성 (gate 아님, 보고 항목).** DEV 40장에서 FullLoad와 IMPRESS의 TTFT를 기존 본실험의 같은 요청과 비교한다(평균과 비율).

admission: gate 7, gate 1, 그리고 그 모델의 `gate`·`dev` 단계 판정이 모두 PASS일 때만 그 모델의 본실험을 시작한다. 실패하면 그 모델의 본실험을 시작하지 않고 실패 내용을 `status.json`과 `report.md`에 적는다.

## 7. 보고 항목 (모델별, `results/<run>/report.md`)

1. 결과 표: 다섯 방법의 followup EM, T1·T2·T3 EM, TTFT 평균/p50/p95, SSD MB/followup, 읽기 호출 수. HyperInfer는 이번 실행, 나머지는 기존 본실험 값이며 실행일을 함께 적는다.
2. 짝지은 비교(이미지 단위 paired bootstrap 10,000회, seed 1234, 95% CI; 정답률과 TTFT): Ours − HyperInfer, HyperInfer − IMPRESS, HyperInfer − FullLoad, HyperInfer − ReComp. TTFT 비교는 서로 다른 실행의 값임을 표시한다.
3. TTFT 비율: Ours/HyperInfer, HyperInfer/IMPRESS, HyperInfer/FullLoad, HyperInfer/ReComp.
4. 출력 동일성: 본실험 12,183요청에서 HyperInfer의 생성 토큰이 기존 IMPRESS 기록과 일치한 요청 수. 불일치가 있으면 목록과 원인.
5. TTFT 연속성: gate 8의 결과.
6. prefetch 통계(요청 평균과 layer별):
   - prefetch된 토큰 중 실제 사용 집합에 든 비율, 사용 집합 중 prefetch로 채워진 비율. 사용 집합은 선택된 이미지 토큰 + separator다.
   - 인접 layer의 선택 집합(이미지 토큰만) Jaccard 유사도: fallback이 아닌 layer 쌍만, 그리고 전체.
   - 예산에 걸려 잘린 chunk 비율, 첫 prefetch(layer 0 → 1)에서 읽은 chunk 수, 앞 절반 규칙이 적용된 요청 수.
   - 읽었지만 쓰지 않은 바이트, IMPRESS 대비 총 읽기 바이트.
   - 임계 경로에 남은 시간: prefetch 대기, layer 0의 probe key 읽기, 선택 계산, 빠진 KV 동기 읽기.
   - I/O ratio = (prefetch 대기 + 임계 경로의 probe 읽기 + 빠진 KV 동기 읽기) / (worker 포함 전체 pread 시간). IMPRESS의 값은 DEV 40장에서 함께 잰 것으로, (요청 thread가 읽기로 막힌 시간) / (전체 pread 시간)이다.
7. fallback 비율, 평균 Jaccard, K·V chunk 읽기 비율(기존 IMPRESS 값과 나란히).
8. 재사용 턴 시간 분해: 읽기 대기와 GPU 적재 시간.
9. T1 저장 비용, 실험 전체의 SSD 쓰기량, 소요 시간.
10. 원본 대조표의 최종본과 동결한 chunk 읽기 시간 상수.
11. 추가하거나 바꾼 코드 목록과 새 테스트, 지시와 다르게 했거나 지시에 없던 것, 측정하지 않은 것과 한계.

보고서에서 방법은 ReComp, FullLoad, IMPRESS, HyperInfer, Ours로만 부른다.

## 8. 실행 방식

- 드라이버 하나(`scripts/run_mt_gqa_hyperinfer.sh runs/mt_gqa_hyperinfer_<UTC>`)가 상수 기록 → freeze → gate → DEV → 본실험 → 보고서를 묶어, 새 세션(setsid + nohup)에서 돈다. 사람의 입력을 기다리는 지점이 없다.
- `results/<run>/`에 `status.json`(현재 모델·단계, 완료 요청 수 / 전체, 예상 종료 시각, 누적 SSD 쓰기량, 마지막 오류), `driver.log`, `driver.pid`를 계속 갱신한다.
- 같은 명령으로 이어서 돈다. 커밋된 이미지는 건너뛰고, 끊긴 이미지 안에서는 완료된 대화를 유지하고 끊긴 대화를 T1부터 다시 한다. store를 SHA256으로 확인할 수 없으면 그 이미지를 처음부터 다시 한다. 이어서 실행한 횟수와 다시 실행한 대화 목록을 보고서에 적는다. 잠금 파일로 중복 실행을 막는다.
- 실험이 도는 동안 같은 GPU나 SSD를 쓰는 다른 작업을 돌리지 않는다. GPU 단계는 다른 프로세스가 GPU를 쓰고 있으면 비워질 때까지 기다린다.
- 결과: 행과 증거는 `runs/<run>/llava_next/`, `runs/<run>/llava15/`, 판정은 `runs/<run>/gates/`, 보고서와 상태는 `results/<run>/`.
- 기존 store, 결과 디렉터리, 동결 파일은 수정하지 않는다. freeze 때와 종료 때 `results/`, `runs/`, `kvstore*`의 파일 목록(크기, 수정 시각)을 비교해 보고서에 적는다. 삭제하는 것은 이 실행의 scratch store뿐이다.
- 예상 SSD 쓰기량과 소요 시간은 freeze가 기존 본실험의 이미지별 store 크기로 계산해 `prerun.json`에 적는다(대략 LLaVA-NeXT 1.2 TB, LLaVA-1.5 0.3 TB, 합쳐 8–9시간).

## 9. 지시에 없던 것과 해석한 것

1. **gate 실패 시 동작.** 한 모델의 gate가 실패하면 그 모델의 본실험만 건너뛰고 다른 모델은 그대로 진행한다. 지시는 "그 모델의 본실험을 시작하지 않고 … 종료한다"인데, 무인 실행에서 나머지 모델까지 버리지 않도록 이렇게 해석했다. gate 7이나 gate 1(두 모델 공통)이 실패하면 어느 모델도 측정하지 않고 종료한다.
2. **DEV 단계 하나.** DEV 40장을 7개 arm으로 한 번만 돈다(gate 2–6과 gate 8을 같은 실행에서 본다). SSD 쓰기량과 시간을 줄이기 위해서다. gate 8의 FullLoad·IMPRESS TTFT는 7개 arm 회전 속에서 잰 값이다.
3. **gate 5의 방법.** 6절 5번에 적은 대로 mask된 열에만 0을 넘긴다.
4. **예산의 범위.** 대조표 c의 정정대로 공개 코드를 따른다. arm마다 따로 두는 것은 gate용 arm이 본 arm의 예산에 영향을 주지 않게 하기 위해서다.
5. **계산 시간의 측정.** 공개 코드는 CUDA event로 재지만 여기서는 HuggingFace 모델의 forward 안에서 돌기 때문에, 제출 시각부터 다음 layer 시작의 GPU 동기화까지를 잰다. prefetch를 켠 arm은 layer마다 GPU 동기화가 한 번 더 들어간다(끈 arm에는 없다).
6. **worker의 읽기 단위.** worker는 chunk 하나에 pread 한 번씩 읽는다(공개 코드의 chunk별 반복과 같다). 동기 복구 읽기는 IMPRESS 이식본처럼 인접 chunk를 합친다. 상수는 worker의 읽기 방식 그대로 잰 값이다.
7. **IMPRESS arm의 사용 집합 기록.** gate 2, 3에서 layer별 사용 집합을 비교하려면 IMPRESS arm의 선택 결과가 필요하다. 기존 하네스는 이를 행에 남기지 않으므로, IMPRESS 요청 함수 바깥에 통과만 시키는 기록기를 두고 요청이 끝난 뒤 요약값(layer별 SHA256 앞 12자)을 행에 적는다. IMPRESS의 코드 경로는 바뀌지 않는다.
8. **상수 측정 조건.** GPU 계산과 동시에 읽는 실제 상황이 아니라 GPU가 쉬는 상태에서 잰다. 모든 chunk 위치의 평균이라 layer의 첫 chunk만 보면 이보다 느릴 수 있다. 위치별 값은 측정 파일에 있다.
9. **probe 읽기 시간.** 예산에서 빼지 않는다(3절). 넘친 시간은 대기로 드러나고 그대로 보고한다.
10. **하네스 재사용 방식.** 두 모델의 기존 본실험 스크립트를 import해서 이미지 단위 실행·이어서 실행·커밋 코드를 그대로 쓰고, 이 실행의 프로세스 안에서만 arm 표, 단계 표, 요청 분기, 이미지 감사, 상태 기록기를 바꿔 끼운다. 기존 스크립트 파일은 수정하지 않는다.
