# Qwen2.5-VL 보존 비율 sweep (25 / 30 / 40 / 50 / 100%) — MT-GQA HOLD 실행 계약

2026-11-11 UTC, 어떤 gate·smoke·main 측정보다 먼저 freeze. 저장소 `/home/dblab/hj/mllm_v2` (사용 가능한 `.git` 없음; 소스 보호는 SHA256 manifest).
Run root `runs/qwen_budget_sweep_20261111T152143Z`, 결과 `results/qwen_budget_sweep_20261111T152143Z`.
읽기 전용 입력: `runs/qwen_layer_budget_pyr_20261111T075124Z` (Qwen PyramidKV run; manifest·model 설정·core pair·packed 코드의 기준),
`runs/qwen_mt_gqa_prefixstore3_20261002T074856Z` (Qwen 3-arm 본실험). 이 계약의 SHA256은 `runs/<RUN>/config.json`의 `contract_sha256`에 기록한다.

## 1. 목적

Qwen2.5-VL은 인코더에서 패치 4개를 1개로 합쳐 MT-GQA 이미지당 content 토큰이 260–529개뿐이다. 균등 25%에서 ReComp 대비 followup 손실이 −3.5 pp로
LLaVA 균등 25%(−2.30 pp)보다 컸다. 이 실행은 보존 비율을 25 / 30 / 40 / 50 / 100%로 올렸을 때 **정확도와 TTFT가 어떻게 맞바뀌는지 곡선으로 측정**한다.

- 이미지별 image-only saliency 순서(stable order)의 앞 `k = ceil(비율 × N)`개 content 행을 **모든 layer에서 똑같이** 남긴다. 구조 행은 항상 전부 남긴다.
- 질문은 선택에 쓰지 않는다. layer별 배분은 쓰지 않는다(Qwen에서 NO-GO였다).
- 같은 순서의 앞 k개이므로 큰 비율의 집합이 작은 비율의 집합을 포함한다.
- **판정(GO/NO-GO)은 없다.** 곡선 전체를 보고한다. 데이터로 무엇을 맞추거나 고르는 과정은 없다. 요약 기준선 두 개는 실행 전에 고정한다(10절).
- 범위 밖: MT-VQA, GLM, LLaVA.

## 2. Arm (이 순서로 freeze, 7개 전부 필수)

1. `recompute`
2. `fullload` — 기존 경로 그대로: layer별 파일 57개, 64행 chunk 단위, CPU 조립(`run_cache(budget_ratio=1.0, budget_unit="chunk")`)
3. `qwen_ours_kv_U25_packed`
4. `qwen_ours_kv_U30_packed`
5. `qwen_ours_kv_U40_packed`
6. `qwen_ours_kv_U50_packed`
7. `qwen_ours_kv_U100_packed`

하나라도 실행할 수 없으면 임의로 빼지 않고 BLOCKED로 멈춘다.
U25는 같은 실행 안의 기준점이자 PyramidKV run과의 결정성 확인용이다. U100은 "손실 없는 끝점"을 packed 경로로 잰 값이며 FullLoad와 KV 내용이 같다.

## 3. 벡터 (데이터 불필요, `runs/<RUN>/vectors/`)

`L = 28`, 벡터 = `[r]·28`, `r ∈ {0.25, 0.30, 0.40, 0.50, 1.00}`.
`mmimpress.layer_budget.budget_document(..., num_layers=28, total_ratio=r, r_min=0.0, r_max=1.0)`로 저장했다.

| 벡터 | r (정수 분수) | 합 28·r | ratio SHA256 |
|---|---|---:|---|
| U25 | 0.25 (1/4) | 7.0 | `1343c3ea849a9e70c965cea28530125c1072d841fc23cfd019f4282e11c7fc76` |
| U30 | 0.30 (3/10) | 8.4 | `b726c756cd0680a018200d934536542ea2a31effbe086535577e6f895984a0e7` |
| U40 | 0.40 (2/5) | 11.2 | `d14e8bb77c002045dad35ce55ba2b4b31301af1fbde74383244139a6b949e464` |
| U50 | 0.50 (1/2) | 14.0 | `40302d54a7eba3a7ef3fcc11e150ddef996c4e96a55e8df52af97cb1336d750b` |
| U100 | 1.00 (1/1) | 28.0 | `7becfed787365c80b3e3cf60051c8abd84029c23be7cf7ef53dcc59df702daf9` |

- U25–U50의 SHA는 프롬프트의 기대값과 일치한다. U100의 SHA는 Phase 0에서 계산해 `vectors/vectors_sha256.json`에 기록한 값이다.
  U25의 ratio SHA는 PyramidKV run의 U와 같다(문서는 `r_max`, `alloc_id`만 다르다).
- k는 두 가지로 계산해 비교한다: `layer_keep_counts(r, N, num_layers=28, total_ratio=r, r_min=0.0, r_max=1.0)`(부동소수)와
  정수 분수 `ceil(p·N/q)`. N = 1–3,000 전부와 HOLD의 모든 N에서 일치한다(`phase0/k_tables.json`). writer·validator도 요청마다 두 값을 비교해 다르면 거부한다.
- **`mmimpress/layer_budget.py`의 기본값(NUM_LAYERS=32, R_MIN=0.10, R_MAX=0.60, TOTAL_RATIO=0.25)은 LLaVA용이며 이 실행의 어떤 호출도 기본값에 기대지 않는다**(SG1이 검사).
- HOLD 398장: N = 260 / 345 / 529 (최소 / 실제 이미지 중간값 / 최대, 서로 다른 N 22개).

| N | U25 | U30 | U40 | U50 | U100 |
|---:|---:|---:|---:|---:|---:|
| 260 | 65 | 78 | 104 | 130 | 260 |
| 345 | 87 | 104 | 138 | 173 | 345 |
| 529 | 133 | 159 | 212 | 265 | 529 |

  프롬프트의 기대값과 일치한다. packed 파일 크기 = (k + 21) × 57,344 B. HOLD 이미지 평균: 6.35 / 7.38 / 9.41 / 11.47 / 21.68 MB.

## 4. 데이터와 실행 조건 (Qwen PyramidKV run과 동일)

- MT-GQA-reconstructed 398 images, 4,061 dialogues × 3 turns = 12,183 요청/arm → 7 arm = **85,281 요청**.
  INDEX_SHA `2c47cfad2a7ccbb673042b400304d7f3ca03d6fbe59d04fa83db50708c924224`, WORKLOAD_SHA `0287e0c57813800c781633b969c5cff336b3a3c1a1bdcdbb56d63f6ddab0ca62`.
  dialogues·image identity는 PyramidKV run manifest와 같음을 freeze에서 assert한다.
- integration dialogue = smoke dialogue = `mtgqa_001937`, `mtgqa_000285`, `mtgqa_002643`, `mtgqa_000001` (이전 run의 id 재사용, 4 images).
  core pair = 이전 run의 `correctness_manifest.json` 고정 10개. **이들은 정합성 fixture이지 품질 holdout이 아니다.**
- 모델 `Qwen/Qwen2.5-VL-7B-Instruct` revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`, NF4 double / BF16 compute / SDPA, 28 layers × 4 KV heads × 128,
  min/max pixels 200704 / 802816, seed 1234, batch 1, greedy max 16, CPU threads 2. backend·정밀도·해상도는 바꾸지 않는다.
- 각 arm 자기 full-pixel T1, 첫 source dialogue의 T1에서만 capture·store 생성, method-local generated history(gold 미사용), strict normalized EM,
  image-atomic attempt, 이미지별 독립 감사 후 run-local scratch store만 정확한 SHA allowlist로 삭제, 중단 시 `--resume`.
- dialog별 arm 순서 회전: `i = ordinal % 7; METHODS[i:] + METHODS[:i]`.
- timed 요청 중 다른 GPU 작업·무거운 감사 금지(요청마다 `nvidia-smi` 단독 점유 확인).

## 5. 변경 범위

기존 소스(PyramidKV run에서 추가한 파일 포함)는 수정하지 않는다. 기존 store·run·results는 수정·삭제하지 않는다. 새 파일만 추가한다.

- `mmimpress/qwen25/ratio_packed.py` — 비율을 문서에서 받는 packed writer / validator / reader / session / hit wrapper
- `scripts/qwen_sweep_support.py` — 공통 상수·helper
- `scripts/172_freeze_qwen_budget_sweep.py` — Phase 0 (벡터, k 표, freeze, protection before)
- `scripts/173_validate_qwen_budget_sweep.py` — SG1–SG5
- `scripts/174_eval_qwen_budget_sweep.py` — 7-arm runner (integration = SG6, smoke, main). 169를 얇게 옮긴 것
- `scripts/175_audit_qwen_budget_sweep.py` — 독립 감사(생산 코드·layer_budget을 import하지 않음). 170을 얇게 옮긴 것
- `scripts/176_report_qwen_budget_sweep.py` — 보고 (freeze 대상 runtime source에서 제외; 측정 뒤 실행하는 분석 스크립트)
- `tests/test_qwen_budget_sweep.py`, `docs/qwen_budget_sweep_contract.md`(이 문서)

왜 새 모듈이 필요한가: PyramidKV run의 `packed_store.py`는 총 비율을 0.25로 고정했다(`TOTAL_RATIO = 0.25`, 행 수 합 = `28·ceil(N/4)` 검사, `retention_ratio == 0.25` 검사).
기존 B 경로(`store.py`의 prefix25)도 25% 초과를 거부한다. 그래서 B 경로는 쓰지 않고, packed를 새 모듈에서 일반화했다.

- writer: `packed_store.write_qwen_packed_store`와 **같은 행 연산·같은 순서**. 행 수만 문서의 `total_ratio`에서 나온다. 25%에서 payload가 비트 동일해야 한다(SG2).
- reader: `RatioPackedStore`는 `QwenPackedStore`를 상속하고 activation(메타 검증)만 바꾼다. `plan` / `read_all`(preadv 1회) / `rows` / `assemble_cpu` / DONTNEED는 상속한 그대로다.
- hit: `run_cache_ratio`는 `layerwise.run_cache_layerwise`(PyramidKV run의 hit 경로)를 **그대로 호출**한다. 계획 계산·`preadv`·H2D·GPU 조립·layer별 mask hook 설치·suffix prefill·decode가 모두 그 함수 안이다.
  wrapper는 요청이 끝난 뒤 결과 필드 이름만 고친다. 균등 벡터에서는 모든 layer 길이가 같아 hook이 mask를 바꾸지 않는다(PyramidKV run의 U_packed와 같은 동작).
- 요청 생성·history·채점·RequestTrace 계측·image-atomic commit·감사·cleanup·arm 순서 회전(ordinal % arm 수) 규칙은 169/170과 같다.
  바뀌지 않는 helper(해시, atomic write, DONTNEED trace, RequestTrace, cleanup, capacity guard, runner load)는 110에서 import한다. 110/111/169/170은 실행하지 않는다.
- store 형식 이름: `qwen25_bf16_prefix_ratio_packed_v1`. 메타에 `retention_ratio`, `retention_fraction`, `layer_budget_total_ratio`, `uniform_content_count`를 기록하고
  validator가 벡터가 균등인지, 비율·clip·layer 수·행 수·바이트 표가 서로 맞는지 검사한다. 두 packed 형식의 reader는 서로의 store를 거부한다.

## 6. Gate (모두 PASS여야 main. 실패 gate 우회·임계값 변경·표본 교체 금지)

검증 표본: core pair 10개(각각 hit prompt 2종: history 없음 / 1턴 history + "What is visible?") + integration dialogue 4개.

- **SG1 (CPU)**: 다섯 벡터 SHA·합(28·r); N = 1–3,000 전부에서 비율별 k가 정수 분수 계산과 일치, Σ = 28·k;
  packed validator가 다른 비율의 meta·잘린 파일·SHA 변조·행 수 변조를 거부; `layer_budget` 기본값에 기대는 호출 없음 —
  (a) 기본값을 독으로 바꾼 상태에서 28 layer·r_min 0.0·r_max 1.0·total_ratio 0.3 벡터로 write→open→plan→assemble 통과,
  (b) meta의 total_ratio만 0.25로 바꾸면 거부, (c) 새 소스의 모든 `layer_budget` 호출이 인자를 명시하는지 AST로 검사.
- **SG2 25% 회귀 (GPU, core pair 10개)**: U25_packed의 payload가 같은 capture로 PyramidKV run 코드가 만든 U packed payload와 비트 동일
  (meta는 문서 필드 차이만 허용하고 차이 목록을 기록), 조립 K/V bitwise 동일, 첫 logits bitwise 동일, 생성 ID 전부 동일(추가로 28 layer SDPA 입력 bitwise 동일).
- **SG3 100% 동치 (GPU, 같은 10개)**: U100_packed의 layer별 K/V가 FullLoad 경로(`load_prefix(1.0, chunk)`) 조립 결과와 행 집합·순서·값 bitwise 동일,
  첫 logits bitwise 동일, 생성 ID 전부 동일(추가로 28 layer SDPA 입력 bitwise 동일). 다르면 원인을 분석하고 멈춘다.
- **SG4 포함 관계 (CPU)**: HOLD 398장 전부에서 선택된 원래 content ID 집합이 U25 ⊂ U30 ⊂ U40 ⊂ U50 ⊂ U100(= 전체), 크기 = `ceil(p·N/q)`.
  점수는 PyramidKV run이 각 이미지의 T1에서 capture한 image-only saliency(선택 artifact)를 쓴다. 이 실행에서 실제로 capture한 점수로는 이미지별 독립 감사가 같은 검사를 다시 한다.
  질문 불변: 선택 함수에 질문 인자가 없고(검사), store는 이미지의 T1 capture에서 한 번만 쓰이며 이후 모든 hit가 그 파일을 읽는다.
  실제 불변은 SG5(hit prompt 2종)와 main의 이미지별 감사(같은 이미지의 모든 dialogue·turn)에서 확인한다.
- **SG5 I/O (GPU)**: packed hit당 preadv 1회·pread 0회, 반환 바이트 = 파일 크기 = 구조 바이트 + 28·2·k·row_bytes, vision 호출 0·score 호출 0,
  질문만 바꿔도 read range와 선택 ID 불변, 다섯 비율의 요청별 선택 집합이 포함 관계. 다른 비율·다른 형식·다른 이미지의 store는 거부.
- **SGR 참조 검증 (GPU, 요구 사항에 더한 것)**: 다섯 비율 각각에서 production hit의 28 layer SDPA 입력·첫 logits·생성 ID가 store·reader를 거치지 않는 독립 in-memory 참조,
  반복 실행과 bitwise 동일. 30/40/50%는 FP64 dense-vs-compact 의미 검증(atol = rtol = 1e-12)도 한다. legacy FP32 1e-5 비교는 진단값으로만 기록한다.
- **SG6 통합**: integration dialogue에서 7 arm generated-history 3턴 실행(84 요청) + 독립 감사 PASS. 같은 history에서 U100_packed와 FullLoad의 첫 logits·생성 ID 일치,
  stored arm 6개의 T1 capture KV 비트 동일, packed arm 5개의 saliency 비트 동일을 함께 확인한다.
- **기존 CPU 테스트 전체 통과.** GLM-4.6V 테스트 3건은 이 변경 전부터 설치 환경의 transformers API 때문에 실패하며 새 파일을 import하지 않는다.
  PyramidKV run과 같이 분리 실행해 그대로 보고하고, PASS 판정에서는 non-GLM 전체(새 테스트 포함)의 returncode 0을 요구한다.
- 실패 시: 원인·수정을 `freeze_amendment_*.json`으로 남기고 재freeze한다. 이전 config와 그 config로 만든 gate·phase 기록은 `superseded_N/`으로 옮겨 보존한다.

## 7. Smoke → main

smoke: 이전 run의 smoke dialogue 4개 × 3 turns × 7 arm = 84 요청. main: 398 images, 4,061 dialogues × 3 turns, arm당 12,183 요청, 총 **85,281 요청**.
요청 수 완주·실패 0·독립 감사 PASS여야 final이다.

## 8. Timing 경계 (이전 Qwen run과 동일)

- True TTFT는 history·chat prompt·입력 준비 **전에** 시작해 첫 token materialization + CUDA sync까지다.
  pixel 요청은 이미지 파일 open/RGB decode·processor·vision·full prefill을 포함한다.
  hit는 tokenization, (packed) 계획 계산·`preadv`·H2D·GPU 조립·hook 설치 또는 (FullLoad) 57회 pread·CPU 조립·H2D, suffix prefill을 포함한다.
- Request E2E는 generation 완료까지. capture가 있는 source T1은 score·clone 후처리 tail까지 포함한다.
- timer 밖: image hash, 모델 load/warmup, activation(메타·payload SHA 검증·pinned buffer 할당), DONTNEED conditioning, 감사·hash·logging.
  setup(writer wall + activation)은 이미지당 1회 실측해 session accounting에 넣는다.
- DONTNEED는 OS page-cache hint이며 NAND/controller cold나 O_DIRECT를 뜻하지 않는다. 절대 TTFT는 실행 간 비교하지 않는다.

## 9. 정합성 사후 확인 (보고 전 필수)

- U100_packed와 fullload의 예측이 12,183 요청에서 토큰 단위로 일치해야 한다. 아니면 불일치 요청을 전부 덤프하고 처음 갈라진 지점을 분석해 REPORT에 싣고 상태를 `PENDING_REVIEW`로 둔다.
- recompute, fullload, U25_packed 예측의 PyramidKV run 대비 일치율(그 run의 recompute, fullload, U_packed 요청별 raw와 비교). raw가 없으면 NOT_MEASURED.

## 10. 보고 규칙 (판정 없음, 실행 전에 고정)

모든 비율에 대해 HOLD followup(T2–T3) 기준으로 한 표에 낸다(이미지 단위 paired bootstrap 95% CI, 10,000회, seed 1234):

- 정확도와 vs ReComp Δ·CI, vs FullLoad Δ·CI, vs U25 Δ·CI, T1/T2/T3별 정확도
- TTFT mean/p50/p95, ReComp 대비 비율과 단축률, SSD MB/followup, 구간별 ms(read·H2D·조립·suffix prefill), GPU cache KV 바이트
- persistence ms/이미지, store MB/이미지, shared-stream / standalone-equivalent(DERIVED) 3-turn session

사전 고정한 요약 기준선 두 개(HOLD를 보고 바꾸지 않는다, 기술 통계로만 쓴다). 후보는 U25, U30, U40, U50, U100 순서다.

- **(a) "손실 미검출"**: ReComp 대비 Δ(비율 − ReComp, followup 정확도)의 95% CI가 0을 포함하는 가장 작은 비율. 없으면 NONE.
- **(b) "LLaVA 25% 수준"**: ReComp 대비 손실 점추정(ReComp − 비율)이 2.30 pp 이하인 가장 작은 비율(LLaVA 균등 25%의 v2 HOLD 손실). 없으면 NONE.

둘 다 HOLD에서 관찰한 값이며 다른 데이터에서 확인되지 않았음을 명시한다. CI가 0을 포함하면 "차이 미입증"으로 쓰고 동등하다고 쓰지 않는다.
비율이 커지는데 정확도가 떨어지는 구간이 있으면 숨기지 않고 CI와 함께 표시한다.

그림: `fig_tradeoff_qwen.png` — 가로축 followup TTFT(ms), 세로축 followup 정확도(%), 점 = 각 arm(ReComp·FullLoad 포함), 정확도 95% CI 오차 막대, 점마다 비율 표기.
같은 데이터로 가로축을 SSD MB로 바꾼 `fig_tradeoff_qwen_ssd.png`.

`summary.json`에 곡선 수치와 기준선 a·b를 기계적으로 기록한다.

## 11. 디스크 (실측)

계약 작성 시점 여유 공간 279.2 GB (260.0 GiB; `storage_plan.json`에 freeze 시점 값을 다시 기록). 필요량: reserve 30 GiB + 작업 4 GiB + 새 payload 상한 20 GiB + raw 10 GiB + 검증 8 GiB = **72 GiB**.
이미지 1장의 live store는 FullLoad + packed 5개로 N = 529에서 약 123 MB이고 감사·commit 뒤 삭제한다. raw·evidence는 PyramidKV 6-arm main(약 3 GB)에 비례해 약 3.5 GB로 예상한다.
부족하면 BLOCKED_STORAGE로 멈추며 기존 산출물은 삭제하지 않는다.
삭제 대상은 `runs/<RUN>/scratch/**`(정확한 SHA allowlist)와 `runs/<RUN>/gates/gpu/*/stores/**`(case별)뿐이다.

## 12. 소스 보호

freeze에서 `protection_before.json`을 만든다: `results/`, `docs/`, `mmimpress/`, `scripts/`, `tests/`, `README.md`의 전체 SHA256(이 실행의 새 경로 제외),
`kvstore*`와 다른 모든 `runs/*`의 (size, mtime_ns, inode) inventory. 보고 단계에서 after와 비교한다.
PyramidKV run이 freeze한 runtime source 전체(기존 Qwen core + 그 run의 새 파일)가 그대로인지 freeze에서 확인한다(`source_modifications.json`).
이 실행의 runtime source 전체 SHA256은 `config.json`의 `source_sha256`이다.

## 13. 예상 소요 시간 (smoke 이후)

PyramidKV 6-arm main은 73,098 요청에 2.42시간(요청당 0.119초; conditioning·감사·cleanup 포함)이었다.
7 arm 85,281 요청은 arm 수와, 더 큰 store를 읽는 arm(U100은 U25의 약 3.4배 바이트)을 감안해 **약 2.9–3.8시간**으로 예상한다.

## 14. REPORT에 반드시 적을 것

- 모든 packed arm은 같은 중요도 순서의 앞 k개를 모든 layer에 똑같이 쓰고 비율만 다르다.
- U100_packed와 fullload의 TTFT 차이는 packed 경로(파일 1개·preadv 1회·GPU 조립)의 효과이며 토큰 선택과 무관하다.
- Qwen의 N(260–529)과 비율별 layer당 남는 토큰 수(중간값 이미지 기준)를 LLaVA 25%(중간값 이미지 552개)와 나란히 적는다.
- MT-VQA·GLM·LLaVA는 실행하지 않았다.
- 실제 요청 수, 실패·스킵, 소요 시간, 프롬프트와 다르게 한 점, NOT_MEASURED 항목.
