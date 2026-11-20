## 예상과 다른 점

- **검증 4가 실패했다.** 과제와 계약은 "유지 100%의 IMPRESS와 FullLoad가 다르면 FullLoad의 상위 두 후보 logit이 동점인 경우뿐"이라고 가정했다.
  실제 불일치 1건(image 23351, 턴 3, "How many animals are there?")은 FullLoad에서는 동점이 아니다: "4" 19.375, "3" 19.25로 BF16 한 단계(0.125) 차이다.
  같은 KV를 재배치된 순서로 놓고 계산하면 두 후보가 19.375로 정확히 같아지고 "3"이 선택된다.
  진단 결과 재배치 store는 raster store의 행을 순서만 바꿔 bit 단위로 그대로 담고 있고, FullLoad loader(선택·마스크 없음)를 재배치 store에 써도 IMPRESS(100%)와 같은 토큰이 나온다.
  즉 IMPRESS 선택·마스크·읽기 경로의 차이가 아니라 key 순서에 따른 BF16 반올림 차이다. 그래도 사전 고정 기준 9.4(a)("FullLoad에서 동점")에는 해당하지 않으므로 FAIL로 두고, 계약대로 본실행을 하지 않았다.
- 개발 리허설(측정 아님)에서도 같은 요청이 같은 방식으로 달라졌다. 본 run의 gate는 그 뒤 같은 기준으로 다시 측정한 값이다(계약 13절).
- Pilot 40장(본실행 아님)에서 IMPRESS의 followup TTFT는 FullLoad의 2.05배, ReComp의 1.98배였다. IMPRESS가 읽은 양은 FullLoad의 80.5%(243.1 / 302.0 MB)이고,
  layer의 32.6%가 fallback이었으며 요청 스레드가 읽기를 기다린 시간이 평균 104 ms였다(FullLoad 6.7 ms, Ours 0.7 ms).
- Pilot 40장에서 FullLoad(비동기)는 ReComp보다 느리지 않았다(0.966×). 34절 MT-GQA의 동기 100% packed 읽기는 ReComp의 1.65배였다. 실행·데이터·읽기 방식이 달라 두 값을 직접 비교하지 않는다.
  gate 44장에서 FullLoad의 동기 → 비동기 차이는 followup 평균 −91 ms였다(IMPRESS −1.7 ms, Ours −10.7 ms; 판정 기준 아님).
- Pilot 40장의 VQA score 차이는 모든 쌍에서 95% CI가 0을 포함한다(차이 미입증; 동등성의 증거는 아니다). 40장으로는 방법 간 정확도를 말할 수 없다.

## 계약·과제와 관련해 알아둘 점

- 계약 11절의 probe key 사본 크기 "28,311,552 bytes"는 오기다. 실제 크기는 14,155,776 bytes(32 layer × 576 token × 3 head × 128 × 2)이며 실행 코드와 `storage_plan.json`은 이 값으로 계산했다.
  freeze된 계약 파일은 고치지 않았다. 판정 기준과는 무관하다.
- 검증 2(경로 보존)는 MT-GQA 이미지 2장(`n229548`, `n181355`)의 대화 33개 × 3턴 × 34절의 5 arm = 495요청으로 했다.
- GPU 단독 사용 확인은 요청마다가 아니라 이미지마다 시작·끝에서 했다(계약 5절).
- FullLoad·IMPRESS store에는 rater용 hidden state(`v_hidden.pt`)를 쓰지 않았다. H2O 중요도는 이를 쓰지 않는다. LLaVA-NeXT 경로의 store와 다른 점이다.
- Ours의 비동기 읽기는 hit당 preadv 32회다(separator가 없어 LLaVA-NeXT의 33회와 다르다). 동기 경로(hit당 1회)는 기본값으로 그대로 있다.
- 개발 리허설 `runs/llava15_mtvqa_impress_dev_20261117T0900Z`과 그 결과 디렉터리는 측정이 아니며 어떤 표에도 쓰지 않았다.

## 하지 않은 것

- 25,000장 본실행(300,000요청), 같은 실행 1,000장 구간 표, 25,000장 기준의 paired bootstrap CI·IMPRESS 내부 통계·저장 비용: 미실행.
- 검증 4의 기준이나 표본을 바꾼 재실행, 다른 Ours 변형, 25% 외 유지 비율, warm cache, 반복 실행, 다른 모델·데이터셋.
- README 갱신, 커밋, 푸시.

## 본실행을 하려면 필요한 결정

검증 4의 기준을 그대로 두면 이 실행은 여기서 끝난다. 기준을 바꾸려면 새 계약·새 run으로 처음부터 다시 검증해야 한다(코드는 그대로 쓸 수 있다). 생각할 수 있는 기준은 다음과 같다.

1. 비교 대상을 "재배치 store에 FullLoad loader를 쓴 결과"로 바꾼다. IMPRESS 100% 경로가 선택·마스크 없이 전부 읽는 것과 같은지를 key 순서 효과와 분리해 확인한다(이번 진단의 (다) = (나)).
2. 허용 범위를 "FullLoad의 상위 두 후보 logit 차이가 BF16 한 단계 이내"로 넓힌다.
3. 지금 기준을 유지하고 본실행을 하지 않는다.

Pilot 실측으로 본실행은 이미지당 5.4초, 약 37시간이 걸리고 SSD에 약 27 TB를 쓴다(전체 KV store 2개 + IMPRESS 재배치 재기록 + packed store, 이미지당 약 1.07 GB).
