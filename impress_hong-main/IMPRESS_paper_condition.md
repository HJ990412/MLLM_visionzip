# IMPRESS 논문 §6.1 조건 재현 실험 — 설계 및 결과

작성: 2026-07-08. 실행 스크립트: `flexllmgen/impress/verify_paper_condition.py`
원칙: **하드웨어(RTX 4090 24GB vs A100 80GB)만 물리적 차이를 인정**하고, 나머지
파라미터는 각각 독립적으로 논문 §6.1 명시값을 그대로 사용한다. 파라미터 간
연쇄 조정(예: 캐시 비율을 맞추려고 shot 수를 늘리는 것)은 금지.

## 1. 파라미터 표 (실행 전 확정)

분류: **[동일]** 논문과 동일하게 설정 가능 / **[축소]** 하드웨어 제약으로 축소
(비율 명시) / **[추정]** 논문에 명시 안 됨, 합리적으로 추정.

| # | 파라미터 | 논문 §6.1 근거 문장 (원문 인용) | 이번 설정 | 분류 |
|---|---|---|---|---|
| 1 | 데이터셋 | "We select four representative datasets from the standard LM-Evaluation-Harness benchmark: PIQA, RTE, COPA, and OpenBookQA." | **RTE + COPA** (retention 프로토콜이 다른 두 개; PIQA/OpenBookQA 미실행) | [축소] 4→2개 (실행 시간; 파이프라인은 동일) |
| 2 | Few-shot 개수 | "by prepending **two to ten** few-shot examples as system prompts before each query" | prefix당 shot 수 = **uniform(2, 10)** | [동일] |
| 3 | Prefix 평균 길이 | "The average number of tokens in the request prefixes across the four datasets ranges from **4.8k to 5.7k**." | prefix 길이 ~ uniform(4.6k, 5.9k), **평균 ≈ 5.25k** | [동일]* (단, 도달 방식은 #3b [추정]) |
| 3b | 2~10 shot으로 4.8k+ 도달 방법 | 명시 없음. 단서: "to test TTFT with long prefixes as in [21] ..., **we extend the prefixes** to a maximum length of ..." — 즉 표준 QA 템플릿(shot당 30~90 tokens, 10-shot ≤ 0.9k)만으로는 불가능하며 논문 스스로 prefix를 '확장'했다고 명시. 인용 [17](InfiniGen)의 few-shot 방식을 따른다고 했으므로 **각 shot이 긴 컨텍스트를 동반하는 형태**로 추정 | shot = "Context: <장문 passage> + Q + A". passage는 **wikitext-2-raw-v1 train(외부 텍스트)** 의 연속 슬라이스를 한 번씩만 소비(반복 없음) — 데이터셋 자체 텍스트는 총량 부족(RTE train ≈7.5만 tokens < 필요 ~11만). shot당 passage 길이로 prefix 총길이를 4.6k~5.9k에 맞춤. **shot 수는 2~10 유지, shot 반복 없음**. 외부 텍스트 사용의 해석 영향은 §2.6 참조 | [추정] |
| 4 | 최대 prefix 길이 캡 | "we extend the prefixes to a maximum length of 4K for OPT-30B and **10K for the other OPT models**" | OPT-6.7B → **10K 캡** (`pos_extend` target 10240; PI 보간, Stage 11) | [동일] (단, 논문의 position 확장 방법은 명시 없음 → PI는 [추정]) |
| 5 | Prefix KV pool 총량 | "On average, PIQA, RTE, COPA, and OpenBookQA require 55GB, **57GB**, **64GB**, and 65GB of storage for prefix KVs" | **RTE 57GB, COPA 64GB** 그대로 목표 (prefix 개수 = 목표GB ÷ prefix당 KV ≈ 21개/23개; 실측치는 결과에 기록) | [동일] |
| 6a | Retention (accuracy 실험) | "vary the prefix KV retention ratios **from 50% to 5%** to observe accuracy changes" (Fig 15: 50/25/10/5%) | **50%, 25%, 10%, 5%** 각각 별도 측정 | [동일] |
| 6b | Retention (TTFT 실험) | §6.2: "We set the KV retention ratio to **50% for COPA** and **25% for the other three datasets**" | COPA 50%, RTE 25% 고정 | [동일] |
| 7 | Chunk size | "Uniformly, each chunk holds keys or values from **64 tokens**" | 64 tokens | [동일] |
| 8a | GPU 캐시 | "we allocate **10GB of GPU cache** ... for prefix KVs, leaving the remaining GPU ... memory for storing model weights, ..." | **10GB 우선 시도**. 논문은 80GB HBM이라 weights(13.3GB)+10GB가 여유. 4090(23.64GB 가용)은 weights 13.3 + 캐시 10 = 23.3GB로 runtime 여유가 0.3GB뿐 → OOM 예상. OOM 확인 시: max 캐시 = 23.64 − 13.34(weights) − ~2.3(runtime: attention workspace+logits+단편화) ≈ **8GB (논문 대비 80%)** | 시도 후 결정 → **결과 참조** |
| 8b | CPU 캐시 | "and **32GB of CPU cache**" | **32GB 그대로** (DRAM 125GB) | [동일] |
| 8c | 캐시/pool 비율 (사후 확인) | 논문: (10+32)/55~65GB = **65~76%** | RTE: (10 or 8+32)/57 = 74% or 70% / COPA: 66% or 63% → 논문 범위 내, "캐시 < pool → 매 요청 disk I/O 발생" 조건 유지 여부를 결과에 기록 | 확인 항목 |
| 9 | 평가 지표 | "We measure model generation quality using **accuracy**, as in prior works" | accuracy만 (RTE: True/False logit 비교, COPA: 선택지 loglikelihood 비교 — lm-eval 방식) | [동일] |
| 10 | 모델 | "three open-source OPT models (OPT-6.7B, OPT-13B, OPT-30B)" | **OPT-6.7B 단독** | [축소] 3→1 모델 |
| 11 | 하드웨어 | "one NVIDIA A100 GPU with 80GB HBM, ... 128 GB DRAM, ... Intel SSD (~5GB/s read)" | RTX 4090 24GB / DRAM 125GB / NVMe(~5GB/s) — GPU만 실질 차이 | [축소] (전제된 차이) |
| 12 | Prefix 재사용 빈도 | "These system prompts are shared across different queries, with **reuse frequency following a normal distribution**." | prefix 배정 ~ 정규분포 가중 샘플링 | [동일] |
| 13 | 표본 수 | 명시 없음 (validation set 사용 관례) | RTE 120문항 / COPA 100문항(=validation 전체) — 소표본 노이즈 완화 요구 반영 | [추정] |
| 14 | KV reordering 주기 | §4.4.1 "Scheduled at regular intervals (e.g., every 10 minutes)" | 측정 창 사이 동기 pass 1회 (10분 주기의 실험적 등가) | [추정] |

기대 비교 대역 (논문 abstract/§6.2): prefix KV **I/O 시간 1.5~3.8× 감소**,
**TTFT 1.2~2.8× 개선**, accuracy 하락 1% 미만(retention 6b 설정 기준).

## 2. 실행 계획

```bash
# RTE (pool 57GB, retention-TTFT 25%)
$PY -m flexllmgen.impress.verify_paper_condition --dataset rte

# COPA (pool 64GB, retention-TTFT 50%)
$PY -m flexllmgen.impress.verify_paper_condition --dataset copa
```

각 실행: pool 구축 → 워밍업(전 문항 1회) → 동기 reorder 1 pass →
(a) retention {50,25,10,5}% accuracy 스윕 → (b) 고정 retention TTFT 3-모드
(IMPRESS→[GPU 캐시 해제]→FullLoad→ReComp; ReComp accuracy가 (a)의 기준선).
GPU 캐시는 10GB로 먼저 시도, OOM 시 `--keep-tree`로 pool 재사용하며 8GB 재실행.

## 2.5 실행 중 확정된 사항 (2026-07-08)

- **GPU 캐시 10GB 시도 결과**: RTE에서 pool 구축·워밍업까지 진행 후 retention
  스윕 도중 **OOM** (paper의 A100 80GB 전제). 8GB도 PyTorch 할당 단편화로
  OOM → `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + **8GB**로 확정
  (논문 대비 80%). 캐시(8+32)/pool = RTE 70%(논문 74%), COPA 62%(논문 66%)
  — "캐시 < pool → warm 상태에서도 매 요청 disk I/O 발생" 조건 유지 ✓.
- **(a)와 (b)의 prefix 분리** (§6.1 재해석): 확장 prefix(평균 5.2k)로 accuracy
  스윕을 돌리자 ReComp가 49.2%(chance)로 붕괴하고 곡선이 역전됨(retention
  낮을수록 상승) — PI 보간 구간의 zero-shot 저하 때문. §6.1 원문은 accuracy
  프로토콜 서술 **후에** "Additionally, **to test TTFT** with long prefixes
  ..., we extend the prefixes"라고 하므로, **확장은 TTFT 실험에만 적용**되고
  accuracy 실험은 자연 길이(2~10 shot 그대로)로 수행한 것으로 해석. (a)는
  자연 prefix(RTE 164~712 / COPA 29~170 tokens)로 실행. [추정/해석]

## 2.6 워크로드 생성 상세 — 자연 prefix vs 확장 prefix의 차이 (명시 기록)

두 실험의 few-shot 구성 방식은 **서로 다르며**, 이는 §2.5의 §6.1 재해석
(확장은 TTFT 전용)에 따른 의도된 차이다. 코드: `verify_paper_condition.py`.

### (a) accuracy 실험 — 자연 prefix (`build_natural_prefixes`)

| 항목 | 내용 |
|---|---|
| 템플릿 | RTE: **lm-evaluation-harness 기본 템플릿 그대로** — `"{sentence1}\nQuestion: {sentence2} True or False?\nAnswer:"` + 정답 `" True"/" False"`. **COPA: lm-eval 기본형이 아님(커스텀)** — 구조(premise 끝 마침표 제거 + 접속사 + 선택지 첫 글자 소문자화)는 lm-eval과 같으나, effect 접속사가 lm-eval 기본값 `" therefore"` 대신 **`" so"`**이고 shot 끝에 `"."`를 추가함. 접속사 차이는 선택지 loglikelihood에 영향을 줄 수 있어 §4 이탈 목록 #17에 등재 |
| shot 출처 | **train split** (RTE 2,490 / COPA 400). `np.random.RandomState(seed+1)`로 permutation 후 순차 소비 — **중복 없음** |
| shot 수 | prefix당 `randint(2, 11)` = **uniform 2~10** (논문 값) |
| prefix 수 | 8개 (실측 길이: RTE 164~712 / COPA 29~170 tokens — 확장 없음) |
| 평가 문항 | **validation split**에서 `RandomState(seed)`로 비복원 추출 (RTE 120 / COPA 100=전체) |
| prefix 배정 | `RandomState(seed+2)`, 정규분포 가중 |
| 시드 | 전부 `--seed`(기본 0)에서 파생 — **완전 재현 가능** |

### (b) TTFT 실험 — 확장 prefix (`build_workload`)

| 항목 | 내용 |
|---|---|
| shot의 QA 부분 | (a)와 동일 템플릿·동일 train split (`RandomState(seed)` permutation, 순차 소비) |
| **장문 passage** | **wikitext-2-raw-v1 train** — 데이터셋과 **무관한 외부 텍스트**. `PassageCursor`가 연속 슬라이스를 한 번씩만 소비(전 pool에 걸쳐 반복 없음). shot 구조: `"Context: {wikitext 슬라이스}\n{QA}"` |
| passage를 외부 텍스트로 한 이유 | 데이터셋 자체 텍스트(premise 등)는 총량이 부족(RTE train 전체 ≈ 7.5만 tokens < 필요 ~11만)하고 문장 재사용은 '반복 금지' 원칙 위반. 논문이 인용한 [17] InfiniGen류의 "shot이 문서형 컨텍스트를 동반"하는 형태로 추정 구현 |
| prefix 수/길이 | pool GB 목표(57/64GB)를 채울 때까지 생성 (RTE 21 / COPA 24개, 평균 5.2k/5.1k tokens) |

### 무관한 외부 passage가 결과 해석에 미치는 영향 (유의)

1. **ITF importance가 실제보다 '쉬운' 문제가 됨**: 질문과 무관한 wikitext
   토큰은 attention을 거의 받지 못해, 중요 토큰이 QA 라인(+BOS sink)에
   강하게 집중된다 → probe head 간 합의가 쉬워 fallback이 낮아지고(RTE(b)
   2.7%), reorder 후 중요 chunk 밀집도가 높아져 chunk-skip이 잘 작동한다.
   **RAG처럼 컨텍스트가 실제로 관련 있는 워크로드보다 I/O 감소가 과대**할
   수 있음 — RTE(b)의 8.46×(대역 초과)에 기여하는 요인 중 하나로 §3.3의
   원인 목록과 함께 봐야 한다.
2. **(b)의 accuracy 해석**: 무관 컨텍스트는 distraction으로 작용하고 PI
   보간 저하와 겹쳐 확장 prefix에서 전 모드의 절대 accuracy가 낮다(특히
   RTE). 세 모드가 동일한 prefix를 쓰므로 모드 간 상대 비교는 유효하나,
   accuracy 결론은 (a)에서만 내린다는 원칙(§3.3 각주)이 여기서도 적용된다.
3. (a)는 passage가 전혀 없으므로 이 영향과 무관하다.

## 3. 결과 (2026-07-08, OPT-6.7B, GPU 캐시 8GB / CPU 32GB)

### 3.1 Pool 구축 (논문 §6.1 목표 그대로)

| 데이터셋 | prefix 수 | 길이(평균) | 계획 KV | 실측 디스크 | 논문 목표 |
|---|---|---|---|---|---|
| RTE | 21 | 4520~5849 (**5202**) | 57.3 GB | 60.0 GB(sidecar 포함) | 57 GB ✓ |
| COPA | 24 | 4643~5840 (**5130**) | 64.5 GB | 67.3 GB | 64 GB ✓ |

평균 길이 모두 논문 대역(4.8~5.7k) 내 ✓. shot 수는 전부 2~10 ✓.

### 3.2 (a) accuracy vs retention (자연 prefix, 문항 120/100)

| retention | RTE | COPA |
|---|---|---|
| ReComp 기준 | 55.0% | 86.0% |
| 50% | 58.3% | 85.0% |
| 25% | 58.3% | 82.0% |
| 10% | 50.0% | 77.0% |
| 5% | 50.0% | 72.0% |

**Figure 4/15 방향 재현 ✓**: 완만한 하락(COPA: 50%→5%에서 −13%p 점진;
RTE: 25%까지 기준선 이상 유지 후 10%부터 chance로) — 급락은 낮은 ratio에서만.
50/25%에서 IMPRESS ≥ ReComp인 것도 논문 §6.2의 "중요 토큰 집중이 때로 품질을
높인다"는 관찰과 일치.

### 3.3 (b) TTFT 3-모드 (확장 prefix, §6.2 retention: RTE 25% / COPA 50%)

| | RTE | COPA |
|---|---|---|
| IMPRESS TTFT (disk) | **336.1 ms** (79.0) | **521.1 ms** (183.0) |
| FullLoad TTFT (disk) | 981.5 ms (668.4) | 963.7 ms (695.6) |
| ReComp TTFT | 1468.4 ms | 1632.7 ms |
| **I/O 시간 감소** (vs FullLoad) | **8.46×** (대역 1.5~3.8 초과) | **3.80×** (대역 상단 ✓) |
| TTFT vs FullLoad | 2.92× (≈대역 상단) | **1.85×** (대역 내 ✓) |
| TTFT vs ReComp | 4.37× (초과) | 3.13× (소폭 초과) |
| accuracy 차 (vs ReComp) | +3.3%p* | **+0.0%p ✓** |
| ITF fallback | 2.7% | 37.3% |

\* RTE(b)의 accuracy는 확장 prefix = PI 구간이라 전 모드가 chance 부근
(ReComp 49.2%) — accuracy 결론은 (a)를 근거로 해야 함. COPA(b)는 78.0%로
전 모드 동일(차 0%p, 논문의 <1%p 충족).

**대역 초과 원인 조사** (요구사항 #3): retention(25/50%)·pool·캐시 비율이
의도값과 일치함을 재확인함. 초과의 원인은 파라미터가 아니라 비교군 강도:
1. **FullLoad baseline이 논문 baseline보다 약함** — 논문의 1.5~3.8×는
   AS+H2O(중요 value만 비동기 로드 + 캐시)를 상대로 한 수치인데, 우리
   FullLoad는 프리페치·캐시 없는 naive 동기 로더라 분모가 큼. RTE처럼
   fallback이 낮은(2.7%) 설정에서 특히 IMPRESS 쪽 분자가 작아져 8.46×까지
   벌어짐. COPA는 retention 50% → threshold 0.517 → fallback 37.3% →
   I/O 감소가 정확히 논문 상단(3.80×)에 위치.
2. **vs-ReComp는 하드웨어 비율 차이** — 4090에서 FlexGen의 비최적화 dense
   prefill(5.2k, 대형 임시 텐서 3개)이 상대적으로 느려 ReComp 분모가 큼.
   비교 가능성이 높은 vs-FullLoad TTFT는 두 데이터셋 모두 대역 내/상단.
3. disk 절대 시간은 page-cache warm이라 압축되어 있으며, 반복 접근되는
   IMPRESS의 선택 chunk가 비대칭적으로 유리(§5 유의).

## 4. 논문과 다른 지점 전체 목록

| # | 항목 | 논문 | 이번 실험 | 성격 |
|---|---|---|---|---|
| 1 | GPU | A100 80GB | RTX 4090 24GB | 전제된 하드웨어 차이 |
| 2 | GPU 캐시 | 10GB | **8GB (80%)** — 10GB는 OOM 실증, 8GB도 expandable_segments 필요 | 하드웨어 제약 축소 |
| 3 | 캐시/pool 비율 | 74%(RTE)/66%(COPA) | 70%/62% | GPU 캐시 축소의 파생(−4%p) |
| 4 | 모델 | OPT-6.7B/13B/30B | OPT-6.7B만 | 축소 |
| 5 | 데이터셋 | 4개 | RTE+COPA 2개 | 축소 |
| 6 | prefix 확장 방식 | "we extend the prefixes" (방식 명시 없음) | shot당 wikitext 문서 컨텍스트 | 추정 |
| 7 | position 2048 초과 지원 | 방식 명시 없음 | PI 선형 보간 (zero-shot) | 추정 — 확장 구간 절대 accuracy는 논문과 비교 불가 |
| 8 | accuracy 실험의 prefix 길이 | 명시적 언급 없음 (문장 순서상 자연 길이로 해석) | 자연 2~10-shot (164~712/29~170 tokens) | 해석 |
| 9 | 비교 baseline | ReComp, AS-like, AS+H2O+LRU/LFU | ReComp, naive FullLoad (AS+H2O 미구현) | 축소 — 비율 과대의 주원인 |
| 10 | prefix 평균 길이 | 4.8~5.7k | 5.2k/5.1k | 동일 ✓ |
| 11 | pool 총량 | 57/64GB | 57.3/64.5GB (디스크 60/67GB) | 동일 ✓ |
| 12 | retention 프로토콜 | 스윕 50→5%; TTFT 25%(RTE)/50%(COPA) | 동일 | 동일 ✓ |
| 13 | chunk 64, 2~10 shot, 정규분포 재사용, 10K 캡, accuracy 지표 | — | 전부 논문 값 | 동일 ✓ |
| 14 | 표본 수 | 명시 없음 | RTE 120 / COPA 100 | 추정 |
| 15 | disk read 온도 | 실 SSD (5GB/s) | NVMe + page-cache warm | 측정 조건 차이 |
| 16 | reordering 주기 | "예: 10분마다" | 측정 창 사이 동기 1 pass | 추정 (등가 프로토콜) |
| 17 | COPA 프롬프트 접속사 | lm-eval 기본: cause→" because", effect→**" therefore"** | effect→**" so"** (+shot 끝 ".") — 커스텀 | 이탈 — COPA accuracy 절대값이 lm-eval 표준과 다를 수 있음(모드 간 상대 비교는 동일 프롬프트라 유효). "therefore"로 교체 시 COPA 전체 재실행 필요 |

### 결론 (위 표를 전제로)

- **COPA는 세 지표 모두 논문과 정합**: I/O 감소 3.80×(대역 상단), TTFT
  vs FullLoad 1.85×(대역 내), accuracy 차 0.0%p(<1%p ✓). retention-accuracy
  곡선도 Fig 15 방향.
- **RTE는 방향은 동일하나 비율이 대역을 초과**(I/O 8.46×, vs ReComp 4.37×)
  — 파라미터 이탈이 아니라 #9(약한 baseline)와 #1/#2(하드웨어)의 결과로
  판단. 논문 대역과의 엄밀 비교를 원하면 AS+H2O baseline 구현이 다음 단계.
- accuracy 주장은 (a)(자연 prefix)로, TTFT/I/O 주장은 (b)(확장 prefix)로
  근거를 분리하는 것이 이 재현의 올바른 사용법이다.
