# Qwen2.5-VL Layer-wise Budget — PyramidKV(β=20) + PackedPrefixStore, MT-GQA HOLD 실행 계약

2026-11-11 UTC, 어떤 gate·smoke·main 측정보다 먼저 freeze. 저장소 `/home/dblab/hj/mllm_v2` (사용 가능한 `.git` 없음; 소스 보호는 SHA256 manifest).
Run root `runs/qwen_layer_budget_pyr_20261111T075124Z`, 결과 `results/qwen_layer_budget_pyr_20261111T075124Z`.
부모 run(읽기 전용 입력): `runs/qwen_mt_gqa_prefixstore3_20261002T074856Z` (Qwen 3-arm 본실험).
실행 프롬프트: `papers/qwen_layer_budget_pyr_prompt.md`. 이 계약의 SHA256은 `runs/<RUN>/config.json`의 `contract_sha256`에 기록한다.

## 1. 목적

기존 Qwen PrefixStore-KV25 [B]는 모든 layer가 같은 `k = ceil(N/4)`개의 content 행을 쓴다. 이 실행은 총 행 수 `28·ceil(N/4)`(= 읽는 바이트)를 그대로 두고
layer별 행 수 `k_l`만 PyramidKV 공개 코드 식으로 바꿨을 때 MT-GQA HOLD followup 정확도가 얼마나 달라지는지 측정한다.
"MHA(LLaVA)와 GQA(Qwen) 모두에서 된다"를 보이기 위한 두 번째 모델 실험이다.

- 이미지별 중요도 순서(image-only saliency의 stable order)는 그대로이고 layer l은 그 순서의 앞 `k_l`개를 쓴다. 질문은 선택에 쓰지 않는다.
- 저장은 이미지당 packed 파일 하나, hit당 읽기 1회(`preadv`)로 통일한다.
- 데이터로 벡터를 만들거나 고르는 과정은 없다. 따라서 holdout 겹침 문제가 없다.
- 원 PyramidKV의 attention 기반 token 선택은 쓰지 않는다. **layer별 자르는 지점만** PyramidKV 식을 따른다.
- 범위 밖: MT-VQA, GLM, LLaVA 재실행.

## 2. Arm (이 순서로 freeze, 6개 전부 필수)

1. `recompute`
2. `fullload`
3. `qwen_ours_prefixstore_kv25` — 기존 B, layer별 파일 57개, 기존 코드 경로(`run_cache`, `persist`, `condition_cache`) 그대로
4. `qwen_ours_prefixstore_kv25_U_packed` — 균등 벡터, packed (대조군)
5. `qwen_ours_prefixstore_kv25_PYR-b20_packed` — PyramidKV β=20, 하한 없음, packed ← **주 판정 대상**
6. `qwen_ours_prefixstore_kv25_PYR-b2p5_packed` — PyramidKV β=2.5, packed ← 민감도, 보고만

하나라도 실행할 수 없으면 임의로 빼지 않고 BLOCKED로 멈춘다.

## 3. 벡터 (데이터 불필요, `runs/<RUN>/vectors/`)

`L = 28`. `U = [0.25]·28`. PYR: `lo = 0.25/β; hi = 0.5 − lo; step = (hi − lo)/27; r_i = hi − i·step` (`scripts/158` `pyramid_beta`와 같은 식, clip 없음).
`mmimpress.layer_budget.budget_document(..., num_layers=28, total_ratio=0.25, r_min=0.0, r_max=0.60)`로 저장했다.

| 벡터 | 첫 layer | 마지막 layer | 합 | ratio SHA256 |
|---|---:|---:|---:|---|
| U | 0.25 | 0.25 | 7.0 | `1343c3ea849a9e70c965cea28530125c1072d841fc23cfd019f4282e11c7fc76` |
| PYR-b20 (β=20, 주 판정) | 0.4875 | 0.0125 | 7.0 | `58dcb86e2f8b1de4ead4fe1672f36516c6472b835f545f45da5e7b4008b7a52b` |
| PYR-b2p5 (β=2.5, 민감도) | 0.40 | 0.10 | 7.0 | `5e5c3ef263fbf691e2678ea1deefd81fc6f1516147f679611e01fb0f75d4f6e0` |

- 세 SHA는 프롬프트의 기대값과 일치한다(`vectors/vectors_sha256.json`).
- 독립 검산(`phase0/pyramidkv_verification.json`): 유리수로 따로 계산한 식과의 layer별 최대 차 < 1e-16, 합 = 7 (유리수로 정확히 7).
- PyramidKV 공개 코드 **REVERIFIED**: `Zefan-Cai/PyramidKV` `pyramidkv/pyramidkv_utils.py`, commit `c97f9d706d63d1b56e980f8b81b2a924c21162b3`
  (이 파일의 최신 변경 commit)과 ref `68cd9551`에서 받은 파일이 같다(파일 SHA256 `a8a1e8139a8c24a0…`). **L360–370**:
  `min_num = (max_capacity_prompt − window_size) // beta`, `max_num = (…)*2 − min_num`, `steps = (max_num − min_num) // (num_hidden_layers − 1)`,
  `max_capacity_prompt = max_num − layer_idx * steps`. `PyramidKVCluster.__init__` 기본값 `beta = 20` (L324). GQA용 분기(L449–457)도 같은 식이다.
  여기서는 window 0, 정수 내림 없이 비율로 쓰고 `num_hidden_layers = 28`이다.
- layer별 행 수는 `layer_keep_counts(r, N, num_layers=28, total_ratio=0.25, r_min=0.0, r_max=0.60)` (Hamilton 최대 잉여). Σ k_l = 28·ceil(N/4).
  **`mmimpress/layer_budget.py`의 기본값(NUM_LAYERS=32, R_MIN=0.10)은 LLaVA용이며 이 실행의 어떤 호출도 기본값에 기대지 않는다**(QG1이 검사).
- HOLD 398장: N = 260 / 345 / 529 (최소 / 실제 이미지 중간값 / 최대; 산술 중간값 345.0, 서로 다른 N 22개). `phase0/k_tables.{json,csv}`.

| N | 균등 k | PYR-b20 첫 / 마지막 | PYR-b2p5 첫 / 마지막 |
|---:|---:|---|---|
| 260 | 65 | 127 / 3 | 104 / 26 |
| 345 | 87 | 169 / 5 | 139 / 35 |
| 529 | 133 | 259 / 7 | 212 / 54 |

  k_l = 0인 layer는 HOLD의 어떤 N에서도 없다(최소 PYR-b20 3행, PYR-b2p5 26행). 하나라도 있었다면 BLOCKED였다. writer·validator도 0행 layer를 거부한다.

## 4. 데이터와 실행 조건 (부모 Qwen 본실험과 동일)

- MT-GQA-reconstructed 398 images, 4,061 dialogues × 3 turns = 12,183 요청/arm → 6 arm = **73,098 요청**.
  INDEX_SHA `2c47cfad2a7ccbb673042b400304d7f3ca03d6fbe59d04fa83db50708c924224`, WORKLOAD_SHA `0287e0c57813800c781633b969c5cff336b3a3c1a1bdcdbb56d63f6ddab0ca62`.
  dialogues·image identity는 부모 manifest와 같음을 freeze에서 assert한다.
- integration dialogue = smoke dialogue = `mtgqa_001937`, `mtgqa_000285`, `mtgqa_002643`, `mtgqa_000001` (부모 run의 id 재사용, 4 images).
  core pair = 부모 `correctness_manifest.json`의 고정 10개. **이들은 정합성 fixture이지 품질 holdout이 아니다.**
- 모델 `Qwen/Qwen2.5-VL-7B-Instruct` revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`, NF4 double / BF16 compute / SDPA, 28 layers × 4 KV heads × 128,
  min/max pixels 200704 / 802816, seed 1234, batch 1, greedy max 16, CPU threads 2. backend·정밀도·해상도는 바꾸지 않는다.
- 각 arm 자기 full-pixel T1, 첫 source dialogue의 T1에서만 capture·store 생성, method-local generated history(gold 미사용), strict normalized EM,
  image-atomic attempt, 이미지별 독립 감사 후 run-local scratch store만 정확한 SHA allowlist로 삭제, 중단 시 `--resume`.
- dialog별 arm 순서 회전: 110의 규칙을 arm 수 6에 맞춘다 — `i = ordinal % 6; METHODS[i:] + METHODS[:i]`.
- timed 요청 중 다른 GPU 작업·무거운 감사 금지(요청마다 `nvidia-smi` 단독 점유 확인, 110과 같음).

## 5. 변경 범위

기존 소스(`mmimpress/qwen25/{runner,store,vision}.py`, `mmimpress/layer_budget.py`, 기존 scripts·tests 전부)는 수정하지 않는다. 기존 store·run·results는 수정·삭제하지 않는다.
새 파일만 추가한다.

- `mmimpress/qwen25/packed_store.py` — packed writer / validator / reader
- `mmimpress/qwen25/layerwise.py` — layer별 mask hook, packed persistence·activation·hit 경로
- `scripts/qwen_pyr_support.py` — 공통 상수·helper
- `scripts/167_freeze_qwen_layer_budget_pyr.py` — Phase 0 (벡터, 검산, k_l 표, freeze, protection before)
- `scripts/168_validate_qwen_layerwise_packed.py` — QG1–QG6
- `scripts/169_eval_qwen_layer_budget_pyr.py` — 6-arm runner (integration = QG7, smoke, main)
- `scripts/170_audit_qwen_layer_budget_pyr.py` — 독립 감사(생산 코드·layer_budget을 import하지 않음)
- `scripts/171_report_qwen_layer_budget_pyr.py` — 판정·보고 (freeze 대상 runtime source에서 제외; 측정 뒤 실행하는 분석 스크립트)
- `tests/test_qwen_layerwise_packed.py`, `docs/qwen_layer_budget_pyr_contract.md`(이 문서)

110/111은 `METHODS` 3개, `ordinal % 3`, pread 57회 assert가 박혀 있으므로 **실행하지 않는다.** 169는 110의 요청 생성·history·채점·RequestTrace 계측·
image-atomic commit·cleanup 규칙을 6 arm으로 옮겼고, 바뀌지 않는 helper(해시, atomic write, DONTNEED trace, RequestTrace, cleanup, capacity guard, runner load)는 110에서 import한다.
arm 1–3은 기존 runner 메서드(`run_pixels`, `run_cache`, `persist`, `condition_cache`)를 110과 같은 인자로 그대로 호출한다. 170은 111의 감사 규칙을 6 arm으로 옮기고 packed용 검사를 더했다.

## 6. ragged cache 설계 — A 확정

근거와 줄 번호는 `runs/<RUN>/design_ragged_cache.md` (SHA256은 `config.json`의 `design_sha256`). 요약:

- 설치된 `transformers 4.57.6`의 Qwen2.5-VL text model은 forward마다 4D mask를 **한 번**, **layer 0의 cache 길이**로 만들어 같은 객체를 모든 layer에 넘긴다.
  prefill(q > 1)에서는 bool `[1,1,T,len0+T]` 텐서, decode step(q = 1)에서는 None이다. SDPA wrapper는 mask를 key 길이에 맞춰 **오른쪽만** 자른다.
- `DynamicCache`는 layer별로 길이가 달라도 된다. position·rope 상태는 cache 길이에 의존하지 않는다(우리는 `position_ids`를 항상 명시).
- **설계 A**: layer l cache = 구조 행 + 순서표 앞 `k_l`개(원래 위치 순). 각 decoder layer의 `self_attn`에 forward pre-hook(with_kwargs)을 걸어
  mask를 그 layer 길이에 **오른쪽 정렬**한다(prefix 열 전부 보임, suffix 열은 HF가 만든 causal 부분 그대로, None이면 그대로).
  hook은 요청마다 걸고 prefill·모든 decode step 동안 유지한 뒤 `finally`에서 제거한다. 남은 hook 검사·길이 검사·호출 횟수 검사를 한다.
- 설계 B(padding + additive mask)는 쓰지 않는다. A가 가능하다는 근거가 있기 때문이다.

규범적 의미: hit 요청의 layer l에서 suffix·생성 토큰은 정확히 (1) 구조 행 S개 전부, (2) 원래 index가 `full_importance_permutation[:k_l]`에 속하는 content 행,
(3) 자기보다 앞선 suffix·생성 토큰만 본다. K/V 값은 T1 capture 값(BF16, post-MRoPE key) 그대로이고 다시 회전하지 않는다. suffix logical position은 기존과 같다.

## 7. PackedPrefixStore (Qwen) — `store_format = "qwen25_bf16_prefix25_packed_v1"`

- 이미지당 payload 파일 1개 `kv_packed.bin`: `[구조 K/V (2,L,S,H,D)][L0 K rows 0..k_0)[L0 V][L1 K]…[L27 V]`. BF16, padding 없음.
  구조 블록은 기존 `structural_kv.bin`과, layer 블록은 기존 writer(prefix25)의 `layer_xxx/{k,v}.bin`과 같은 연산으로 만들어 비트 동일하다(U에서 QG2로 확인).
- 메타(`meta.json`): layer별 `(k_offset, v_offset, rows)`, `layer_keep_ratios`, `layer_keep_counts`, `layer_budget_id`, `layer_budget_sha256`,
  `layer_budget_r_min/r_max`, 벡터·clip·layer 수를 묶는 `layer_budget_binding_sha256`, `num_layers`, `full_importance_permutation`,
  기존 B 메타의 identity 필드(`_REQUIRED_EXTRA`) 전부, `packed_sha256`, `packed_code_revision`.
- hit: `os.preadv` 1회로 파일 전체를 activation 때 잡아 둔 page-locked host buffer에 읽는다. 그 뒤 buffer를 한 번 H2D하고 GPU에서 행 index 한 번으로
  layer별 `[구조 행 + 앞 k_l행, 원래 위치 순]`을 만든다(기존 `load_prefix`와 같은 의미, QG2에서 텐서 bitwise 비교).
  실제 호출 수·반환 바이트는 IOCounter와 별도로 `os.pread`/`os.preadv` trace로 기록한다.
- validator(fail closed): 파일 크기 = 구조 바이트 + Σ(2·k_l·row_bytes), offset 단조·빈틈 없음, k_l을 벡터·N에서 재계산해 일치, 0행 layer 거부,
  벡터 SHA·binding digest·기대 벡터 일치, identity digest, payload SHA256(activation에서 전체 검증).
- persistence: packed writer의 write/fsync 시간을 기존 B와 같은 경계(writer wall + completion seal, 이미지당 1회)로 기록한다.
- DONTNEED는 기존과 같이 요청 timer 밖에서 packed 파일에 적용한다(hit마다 1회, 성공 기록).

## 8. Gate (모두 PASS여야 main. 실패 gate 우회·임계값 변경·표본 교체 금지)

검증 표본: core pair 10개(각각 hit prompt 2종: history 없음 / 1턴 history + "What is visible?") + integration dialogue 4개.

- **QG1 (CPU)**: 벡터 SHA·합; HOLD 398장의 N과 N ∈ {1,3,4,5,63,64,65,260,345,529,1024}에서 Σ k_l = 28·ceil(N/4) 정확히 일치;
  U가 N = 1..2000 전부에서 k_l = (N+3)//4; packed validator가 변조된 meta/SHA/행 수/길이/r_min/num_layers를 거부;
  `layer_budget` 기본값(32, 0.10)에 기대는 호출 없음 — (a) 기본값을 독으로 바꾼 상태에서 28-layer·r_min 0.0 벡터로 write→open→plan→assemble 통과,
  (b) meta의 r_min만 0.10으로 바꾸면 거부, (c) 새 소스의 모든 `layer_budget` 호출이 인자를 명시하는지 AST로 검사.
- **QG2 U 회귀 (GPU)**: U_packed의 구조·layer별 slice가 같은 capture로 만든 기존 B store의 payload와 비트 동일, 총 payload 바이트 동일,
  조립된 layer별 K/V 텐서가 기존 `load_prefix` 결과와 bitwise 동일, 첫 토큰 logits bitwise 동일, 생성 token ID 전부 동일(추가로 28 layer SDPA 입력 bitwise 동일).
- **QG3 layerwise 구현 (GPU, PYR-b20과 PYR-b2p5 각각)**: (a) packed의 layer l slice가 독립적으로 stable-rank한 canonical capture의 앞 k_l행과 28 layer 전부 bitwise 동일,
  (b) prefill 28 layer에서 실제 SDPA에 들어간 Q/K/V/mask가 독립적으로 만든 같은 shape의 in-memory 참조와 bitwise 동일, layer l의 key 수 = S + k_l + suffix 길이, decode step마다 +1,
  (c) 실제 SSD 경로 / 독립 in-memory 같은 shape 참조 / 반복 실행의 첫 logits bitwise 동일·생성 ID 전부 동일.
- **QG4 의미 검증 (FP64)**: 실제 production Q와 suffix K/V, 독립 canonical prefix로 "layer별 가시 집합을 적용한 dense 논리 참조"와 compact 참조를 만들고,
  입력을 FP64로 바꾼 뒤 QK·scale·softmax·value 집계를 계산해 모든 layer·head·suffix query에서 atol = rtol = 1e-12. dense에서 가려진 위치의 확률은 0.
  기존 FP32 dense-vs-compact 1e-5 비교(legacy G7)는 연산 shape에 따른 반올림 차이로 FAIL이 나는 것이 규명되어 있으므로 gate로 쓰지 않고 진단값으로만 기록한다.
- **QG5 negative fixture**: 한 layer의 k_l ±1, 두 layer의 행 뒤바꿈, 가려야 할 행 1개 노출, 구조 행 누락, 이전 요청의 hook 잔존(및 hook 누락),
  잘린 packed 파일, 다른 벡터 SHA — 각각 validator가 거부하거나 attention 출력이 달라짐을 확인.
- **QG6 I/O**: packed hit당 preadv 1회·pread 0회, 반환 바이트 = 파일 크기, vision 호출 0·score 호출 0, 질문만 바꿔도 read range와 layer별 선택 ID 불변.
  세 packed 벡터의 payload 바이트가 서로 같고 기존 B의 payload 바이트와 같음.
- **QG7 통합**: integration dialogue에서 6 arm generated-history 3턴 실행(72 요청) + 독립 감사 PASS. 같은 history에서 U_packed와 B의 첫 logits·생성 ID 일치를 함께 확인.
- **기존 CPU 테스트 전체 통과.** GLM-4.6V 테스트 3개는 이 변경 전부터 설치 환경의 transformers API 때문에 실패하며(새 파일이 없던 baseline과 11/10 LLaVA run에서 같은 3건),
  이 변경이 추가한 것을 import하지 않는다. 분리 실행해 그대로 보고하고 PASS 판정에서는 non-GLM 전체(새 테스트 포함)의 returncode 0을 요구한다.
- 실패 시: 원인·수정을 `freeze_amendment_*.json`으로 남기고 재freeze한다. 이전 config와 그 config로 만든 gate·phase 기록은 `superseded_N/`으로 옮겨 보존한다.

## 9. Timing 경계 (기존 Qwen 계약과 동일)

- True TTFT는 history·chat prompt·입력 준비 **전에** 시작해 첫 token materialization + CUDA sync까지다.
  pixel 요청은 이미지 파일 open/RGB decode·processor·vision·full prefill을 포함한다.
  hit는 tokenization, (packed) 계획 계산·`preadv`·H2D·GPU 조립·hook 설치 또는 (B/FullLoad) 57회 pread·CPU 조립·H2D, suffix prefill을 포함한다.
- Request E2E는 generation 완료까지. capture가 있는 source T1은 score·clone 후처리 tail까지 포함한다(110과 같음).
- timer 밖: image hash, 모델 load/warmup, activation(메타·payload SHA 검증·pinned buffer 할당), DONTNEED conditioning, 감사·hash·logging.
  setup(writer wall + activation)은 이미지당 1회 실측해 session accounting에 넣는다.
- DONTNEED는 OS page-cache hint이며 NAND/controller cold나 O_DIRECT를 뜻하지 않는다. 절대 TTFT는 실행 간 비교하지 않는다.

## 10. 판정 규칙 (사전 고정, LLaVA 실험과 같은 규칙)

주 판정: Δ = HOLD followup(T2–T3) 정확도(PYR-b20_packed − U_packed), 같은 실행 안, 이미지 단위 paired bootstrap 95% CI (10,000회, seed 1234).

- **GO**: Δ ≥ +1.0 pp & CI 0 제외 & SSD MB/followup ≤ U_packed × 1.02 & TTFT 평균 ≤ U_packed × 1.05
- **PARTIAL**: Δ ≥ +0.5 pp이고 CI 0 포함, 또는 GO에서 MB/TTFT 중 정확히 하나 위반
- **NO-GO**: 그 외 (Δ가 +0.5~1.0 pp이고 CI가 0을 제외하면 규칙상 NO-GO이며 숫자를 그대로 함께 적는다)

`decision.json`에 기계적으로 기록한다.

β 처리: 주 판정은 β=20뿐이다. PYR-b2p5 − U_packed와 PYR-b20 − PYR-b2p5는 보고만 한다. β=20이 GO가 아니고 β=2.5가 더 좋게 나와도 판정을 바꾸지 않으며,
그 경우 "HOLD에서 본 뒤의 사후 관찰이므로 다른 데이터에서 확인이 필요하다"고 명시한다. CI가 0을 포함하는 차이는 "차이 미입증"으로 쓰고 동등하다고 쓰지 않는다.

정합성 사후 확인(판정 전 필수): U_packed와 기존 B arm은 같은 KV를 쓰므로 12,183 요청의 예측이 토큰 단위로 일치해야 한다.
100%가 아니면 불일치 요청을 전부 덤프하고 처음 갈라진 지점을 분석해 REPORT에 싣고, decision은 `PENDING_REVIEW`로 둔다.

요청 수 완주(73,098)·실패 0·독립 감사 PASS여야 final이다.

부 판정(보고만): packed 효과(U_packed vs B의 TTFT 평균/p50/p95, 구간별 ms, 읽기 호출 수, persistence ms), 손실 회복(각 packed arm vs ReComp·FullLoad의 followup Δ·CI와 TTFT 비율, T1/T2/T3별 정확도),
질문 유형별 Δ(LLaVA 보고서와 같은 regex 분류, spatial 강조, 탐색적), layer별 k_l·바이트 표, GPU cache KV 바이트, peak GPU allocated, shared-stream / standalone-equivalent(DERIVED) session,
결정성 참고(recompute·fullload·B 예측이 10/02 Qwen 본실험의 같은 요청과 일치하는 비율).

## 11. 디스크 (실측)

freeze 직전 여유 공간 282.4 GB (263.0 GiB; `storage_plan.json`에 freeze 시점 값을 다시 기록). 필요량: reserve 30 GiB + 작업 4 GiB + 새 payload 상한 20 GiB + raw 10 GiB + 검증 8 GiB = **72 GiB**.
이미지 1장의 live store는 FullLoad + B + packed 3개로 N = 529에서 약 70 MB이고 감사·commit 뒤 삭제한다. 부족하면 BLOCKED_STORAGE로 멈추며 기존 산출물은 삭제하지 않는다.
삭제 대상은 `runs/<RUN>/scratch/**`(정확한 SHA allowlist)와 `runs/<RUN>/gates/gpu/*/stores/**`(case별)뿐이다.

## 12. 소스 보호

freeze에서 `protection_before.json`을 만든다: `results/`, `docs/`, `mmimpress/`, `scripts/`, `tests/`, `README.md`의 전체 SHA256(이 실행의 새 경로 제외),
`kvstore*`와 다른 모든 `runs/*`의 (size, mtime_ns, inode) inventory. 보고 단계에서 after와 비교한다.
Qwen core 소스가 부모 본실험 freeze와 같은지도 freeze에서 확인한다(`source_modifications.json`). runtime source 전체의 SHA256은 `config.json`의 `source_sha256`이다.

## 13. 예상 소요 시간 (smoke 이후)

부모 3-arm main은 36,549 요청에 4,863초(요청당 0.133초; conditioning·감사·cleanup 포함)였다. 6 arm 73,098 요청은 **약 2.7–3.4시간**으로 예상한다
(stored arm이 2개에서 5개로 늘어 이미지당 capture·persist·activation이 더 든다).

## 14. REPORT에 반드시 적을 것

- PYR-b20 = PyramidKV 공개 코드 식(β=20)의 layer 비율 그대로이며 하한이 없다. 원 PyramidKV의 attention 기반 token 선택은 쓰지 않고 image-only saliency 순서에서 layer별 자르는 지점만 따랐다.
- Qwen은 N이 작아(260–529) β=20의 마지막 layer가 3–7행이다.
- 모든 packed arm은 같은 바이트·같은 I/O 경로·같은 token 순서를 쓰고 layer별 자르는 지점만 다르다.
- 선택한 ragged cache 설계(A)와 그 검증 범위. MT-VQA·GLM·LLaVA는 실행하지 않았다.
- 실제 요청 수, 실패·스킵, 소요 시간, 프롬프트와 다르게 한 점, NOT_MEASURED 항목.

## 15. freeze 전에 한 개발 리허설 (측정 아님)

freeze 전에 코드 경로가 끝까지 도는지 보려고 별도 run root `runs/qwen_layer_budget_pyr_dev_20261111T0900Z`에서 gate(QG1–QG6)·integration·smoke·부분 감사를 한 번씩 돌려 보았다
(protection inventory만 생략한 같은 freeze 코드, 정합성 fixture인 core pair 10개와 integration/smoke dialogue 4개만 사용).
이 디렉터리는 개발 기록으로 보존하며 이 실행의 gate 판정·수치·판정 규칙 어디에도 쓰지 않는다. 벡터·β·설계·임계값은 리허설 전에 이미 정해져 있었고 리허설 뒤에 바꾸지 않았다.
리허설에서 고친 것은 QG1 정적 검사의 대상 범위(보고 스크립트와 단위 테스트 파일 제외) 하나뿐이다. 공식 gate·integration·smoke는 freeze 뒤 이 run root에서 처음부터 다시 실행한다.
그보다 앞서 설계 A의 기능 점검을 이미지 1장으로 한 번 했다(`design_ragged_cache.md` 5절).
