# IMPRESS 실험 명령어 치트시트 (OPT-6.7B)

구현: `/home/dblab/hj/FlexGen/flexllmgen/impress/` (Stage 1~10)
사전조사: [IMPRESS_flexgen_survey.md](IMPRESS_flexgen_survey.md)

## 0. 공통 준비

```bash
PY=/home/dblab/anaconda3/envs/hj/bin/python
cd /home/dblab/hj/FlexGen
```

- 모델: `facebook/opt-6.7b` (기본값, 가중치 `~/opt_weights/opt-6.7b-np` 준비됨)
- GPU: RTX 4090 24GB, batch 1, 가중치 전부 GPU 상주
- 데이터셋(GLUE-RTE, PIQA)은 최초 1회 자동 다운로드 후 캐시됨

## 1. 회귀 테스트 (CPU, 모델 불필요, ~1분)

코드 수정 후 항상 먼저 실행. 전부 `ALL PASSED` / `Part 1 PASSED` 여야 함.

```bash
for t in test_prefix_kv_roundtrip test_radix_tree test_prefix_reuse_unit \
         test_sim_guided_unit test_reordering test_token_cache; do
  echo "== $t =="; $PY -m flexllmgen.impress.$t | tail -1
done
$PY -m flexllmgen.impress.verify_selective_loading --parts 1
```

| 테스트 | 검증 내용 |
|---|---|
| test_prefix_kv_roundtrip | Stage 1: chunk 저장/로드 bit-exact |
| test_radix_tree | Stage 2: Figure 13 시나리오 R/NR, split, 영속화 |
| test_prefix_reuse_unit | Stage 4: prefix-reuse attention == 원본 mha |
| test_sim_guided_unit | Stage 5: Figure 9 벡터 수(160/76), threshold, Jaccard |
| test_reordering | Stage 6: mapping(m0=[0,2,3,1]), 비동기, chunk 감소 |
| test_token_cache | Stage 7: Figure 14 score 예시, invariant, vs LRU |
| verify_selective_loading --parts 1 | Stage 10: chunk-skip == 전체로드 (allclose), 미선택 chunk 미접근 |

## 2. GPU 실험 (단계별, 그대로 복붙)

```bash
# Stage 1 — chunk 저장/로드 bit-exact + decoding 동일성 (~2분)
$PY -m flexllmgen.impress.verify_stage1 --prompt-len 512 --gen-len 8

# Stage 3 — importance metric (layer x head x token score) (~2분)
$PY -m flexllmgen.impress.verify_importance --prompt-len 32

# Stage 4 — retention별 accuracy (RTE, Figure 4 방향) (~5분)
$PY -m flexllmgen.impress.verify_selection_rte \
    --num-samples 50 --num-shots 5 --ratios 1.0,0.5,0.25,0.1

# Stage 5 — similarity-guided ITF (PIQA; fallback/로드벡터/accuracy) (~7분)
$PY -m flexllmgen.impress.verify_sim_guided_piqa \
    --num-samples 50 --num-shots 10 --retention 0.25 --alpha 0.6 --probe-heads 3

# Stage 8 — 통합 서빙 3-모드 TTFT (IMPRESS vs ReComp vs FullLoad) (~5분)
$PY -m flexllmgen.impress.verify_impress_e2e \
    --num-samples 40 --num-shots 21 --retention 0.25

# Stage 9 — 하이퍼파라미터 스윕 (alpha/probe/chunk → CSV+PNG) (~6분)
$PY -m flexllmgen.impress.sweep_impress --num-samples 20

# Stage 10 — 선택적 chunk 로딩 물리 I/O (~8분)
#   Part 2: retention 50/25/10% vs FullLoad (파일수/바이트/disk시간 단조성)
#   Part 3: alpha vs disk read time 단조성 (+PNG)
#   Part 4: TTFT + disk I/O breakdown (Figure 17 형태)
$PY -m flexllmgen.impress.verify_selective_loading --parts 2,3,4

# Stage 11 — position embedding 확장 (PI 보간, 최대 8192 토큰) (~10분)
#   Part 1: 보간 identity(target=2048 == 원본) + verify_stage1 재실행 무변화
#   Part 2: 확장 설치 후에도 2048 이하 입력 logits torch.equal (길이 스위치)
#   Part 3: 3k/5k/7k forward 정상 (8k는 dense recompute의 24GB VRAM 한계로 OOM)
#   Part 4: PIQA 품질 — 1.8k prefix 85% vs 4.5k prefix 80% (붕괴 없음)
$PY -m flexllmgen.impress.verify_pos_extend
```

Stage 11 사용법: 긴 prefix 실험 전에 `pos_extend.install_pos_extension(8192)` 호출
(총길이 ≤2048이면 자동으로 원본 테이블 사용 → 기존 결과 무변화).
방법: OPT는 learned absolute embedding이라 RoPE용 NTK/PI·ALiBi가 적용 불가 →
학습된 2048개 벡터를 선형 보간(Position Interpolation의 learned-table 대응)하는
zero-shot 방식 채택. fine-tuning 없음.

### 논문 §6.1 스케일 재검증 (prefix ~4.4k tokens, --pos-extend 8192)

```bash
# 3-모드 TTFT + accuracy (55-shot = ~4.4k tokens)
$PY -m flexllmgen.impress.verify_impress_e2e \
    --num-samples 40 --num-shots 55 --pos-extend 8192 \
    --tree-dir /home/dblab/hj/impress_kv_store/stage8_long

# alpha sweep(disk time) + TTFT breakdown
$PY -m flexllmgen.impress.verify_selective_loading \
    --parts 3,4 --num-shots 55 --pos-extend 8192
```

2026-07-07 측정 결과 (드디어 논문 체제 재현 — 긴 prefix에서 IMPRESS 최속):
```
e2e (캐시 0.6/1.0): IMPRESS 411ms > ReComp 755ms(1.83x) > FullLoad 918ms(2.23x)
breakdown(캐시 on): IMPRESS 88ms (disk 38ms) vs ReComp 767ms => 8.7x
alpha sweep(캐시 off): disk 578->335 ms/req, TTFT 873->555 ms 단조 감소
cache-off에서 selective loading의 disk 절감 210ms = FullLoad 대비 TTFT 격차의 84%
```
주의: 4.4k는 전 모드가 PI 보간 영역이라 RTE 절대 accuracy가 흔들림
(ReComp 45%, IMPRESS 62.5% — 중요 토큰만 남기는 필터링이 오히려 도움;
논문 §6.2도 "IMPRESS가 ReComp보다 약간 나은 경우" 언급). accuracy 비교는
2048 이하 실험을, TTFT/I/O 비교는 이 장문 실험을 근거로 쓰는 것을 권장.

### 논문 §6.1 데이터셋 스케일 재현 (prefix POOL ~29GB) — 대표 실험

논문의 55~57GB prefix KV는 prefix 1~2개가 아니라 **여러 few-shot prompt 세트의
풀**이며, 쿼리가 정규분포 빈도로 재사용함. 4090(24GB)은 A100-80GB와 달리
가중치(13.3GB) 옆에 GPU 캐시 10GB를 못 두므로, **비율 보존 축소**로 재현:
prefix 12개(~4~4.8k tokens) = 29.3GB 데이터셋, GPU 5GB(18% ≈ 논문 10/57),
CPU 16GB(57% ≈ 논문 32/57).

```bash
$PY -m flexllmgen.impress.verify_impress_e2e \
    --num-samples 80 --num-shots 55 --num-prefixes 12 --pos-extend 8192 \
    --gpu-cache-gb 5 --cpu-cache-gb 16 \
    --tree-dir /home/dblab/hj/impress_kv_store/stage8_pool
```

프로토콜(pool 모드 자동): 데이터셋 구축(12 cold insert, ~43s) → 워밍업 80요청
(importance 축적+캐시 적재) → **동기 reorder pass 1회**(논문의 "10분마다"에
해당하는 측정 창 사이 pass) → steady-state 80요청 측정.

2026-07-07 결과:
```
IMPRESS : acc 58.8% | TTFT 328.3 ms (disk 78.6 ms, pcie 593 MB/req)
FullLoad: acc 61.3% | TTFT 844.1 ms (disk 570.5 ms)
ReComp  : acc 61.3% | TTFT 754.8 ms
→ IMPRESS가 ReComp 대비 2.30x, FullLoad 대비 2.57x (논문 보고 1.2~2.8x 대역)
  disk I/O는 FullLoad의 1/7.3; accuracy 차 -2.5%p (80문항 중 2문항)
```

구현 노트(이 스케일에서 필요했던 수정): ① reorder_node가 순서가 실제로 바뀐
노드만 재작성(변경 감지 — 안 하면 GB급 재작성이 lock을 잡아 서빙 정체),
② importance의 tree.json 영속화를 서빙 경로에서 지연(reorder pass/close 때만),
③ 모드 전환 시 GPU 캐시 tier 해제(dense prefill 트랜지언트와 VRAM 경쟁 방지).

긴 실험은 백그라운드 실행 권장:
```bash
nohup $PY -m flexllmgen.impress.sweep_impress --num-samples 20 > sweep.log 2>&1 &
tail -f sweep.log
```

## 3. TTFT / Accuracy 측정 방법 (무엇을 어떻게 재는가)

### 3.1 데이터셋과 워크로드 구성

| 데이터셋 | 사용 스크립트 | 역할 |
|---|---|---|
| **GLUE-RTE** (HF `glue/rte`) | verify_selection_rte, verify_impress_e2e, sweep_impress, verify_selective_loading | train split에서 few-shot 예제를 뽑아 **공유 prefix**를 만들고, validation split에서 평가 질문을 뽑음 |
| **PIQA** (HF `ybisk/piqa`) | verify_sim_guided_piqa | 위와 동일 구조 (논문 Figure 11이 PIQA 기준이라 Stage 5 검증에 사용) |

워크로드 생성 방식 (논문 §6.1 "few-shot examples를 system prompt로 prepend, 여러
query가 공유" 재현):
- `--num-shots N`개의 train 예제(정답 포함)를 `\n\n`으로 이어붙여 prefix 생성.
  e2e/sweep/stage10은 **prefix 2개**(마지막 2-shot만 다르고 앞부분 공유 → radix
  tree의 매칭/분기 실행)를 만들고 요청이 두 prefix를 번갈아 사용.
- 각 요청 = `prefix + "\n\n" + 질문`. `--seed`로 shot/질문 샘플링 고정(재현성).

### 3.2 Accuracy 측정

정답 생성이 아니라 **선택지 logit 비교** 방식 (lm-eval-harness와 같은 원리):
- **RTE** (2지선다, 단일 토큰): prefill 마지막 토큰의 logits를 hook으로 캡처
  (`opt_output_embed` 패치, `controller.last_logits`)한 뒤
  `logit(" True") > logit(" False")` 이면 entailment로 예측. 정답 label과 비교해
  정확도 산출.
- **PIQA** (2지선다, 다중 토큰 선택지): 선택지별로 `질문+선택지`를 prefill하고
  전체 위치 logits(`controller.full_logits`)에서 선택지 토큰들의
  log-softmax 합(loglikelihood)을 구해 큰 쪽을 예측.
- 비교 기준: 같은 질문 집합에 대한 **ReComp**(전체 재계산, 근사 없음)의 accuracy.
  논문 주장은 "retention 25%에서 ReComp 대비 <1% 하락" — 소표본(20~50문항)이라
  1문항 = 2~5%p 노이즈가 있음을 감안해 방향성으로 판단.

### 3.3 TTFT 측정

- **정의**: 요청 1건 처리 시간 = radix tree 매칭 + (선택적) chunk 로드/조립 +
  prefill + 첫 토큰 logits 산출. `gen_len=1`이므로 `generate()` 전체가 곧 TTFT.
  `time.perf_counter()`로 `server.request()` 전 구간을 감쌈 (NR 저장/트리 삽입/
  importance 기록은 응답 이후 처리라 **TTFT에 미포함**).
- **warm만 집계**: 각 prefix의 첫 요청(cold: NR 전체 계산 + 삽입)은 별도 표기,
  통계는 warm 요청의 평균(e2e) 또는 중앙값(stage 10).
- **비교 모드 3종** (같은 요청 집합, 같은 모델 인스턴스에서 순차 측정):
  - `IMPRESS`: 전체 파이프라인 (ITF + TokenCache + reordering + selective loading)
  - `ReComp`: prefix+query 전체 재계산 (논문의 ReComp baseline)
  - `FullLoad`: prefix KV 전량을 disk에서 로드, 필터링·캐시 없음 (AS-like naive)
- **I/O 분해** (stage 10 Part 4): `PhysicalIOCounter`가 요청당 disk read
  wall-clock을 따로 재서 `TTFT = disk I/O + 나머지`로 분해 (논문 Figure 17 형태).

### 3.4 바로 실행 — 대표 측정 두 가지

```bash
# TTFT + accuracy 종합 (3-모드 비교; 결과 표가 곧 최종 수치)
$PY -m flexllmgen.impress.verify_impress_e2e --num-samples 40 --num-shots 21

# 논문 §6.1 캐시 비율 재현 (GPU 10GB/57GB=17.5%, CPU 32GB/57GB=56%):
# 캐시가 working set보다 작아 warm 상태에서도 disk read가 발생하는 조건
$PY -m flexllmgen.impress.verify_impress_e2e --num-samples 40 \
    --gpu-cache-frac 0.175 --cpu-cache-frac 0.56

# TTFT의 disk I/O 기여 분해 + retention별 물리 I/O
$PY -m flexllmgen.impress.verify_selective_loading --parts 2,4
```

논문 비율(0.175/0.56) 실행 결과 예시 (2026-07-07 측정):
```
IMPRESS : acc 57.5% | warm TTFT 257.7 ms  (disk 40.8 ms/req, pcie 318 MB/req)
FullLoad: acc 57.5% | TTFT 368.1 ms       (disk 224.7 ms/req)
ReComp  : acc 62.5% | TTFT 229.5 ms
```
→ disk I/O는 IMPRESS가 FullLoad의 1/5.5 (40.8 vs 224.7 ms), TTFT 1.43×.
단 ReComp에는 짐 — prefix가 pos-embed 한계로 1.8k tokens에 묶여 재계산이
싼 반면(논문은 4.8~5.7k), 캐시 부족분의 PCIe 전송(318MB/req)이 커서임.
넉넉한 캐시(기본 0.6/1.0)에서는 IMPRESS가 ReComp도 이김(94ms vs 229ms).

출력 읽는 법 (e2e 예시):
```
IMPRESS : acc 60.0% | warm TTFT 94.0 ms (n=38), cold 519.6 ms (n=2)
FullLoad: acc 55.0% | TTFT 480.1 ms
ReComp  : acc 62.5% | TTFT 229.2 ms
```
→ accuracy는 ReComp와의 차이(여기선 -2.5%p = 1문항)로, TTFT는 warm 평균의
배수(2.44× vs ReComp, 5.11× vs FullLoad)로 보고하면 됨.

## 4. 결과물 위치

| 경로 | 내용 |
|---|---|
| `/home/dblab/hj/impress_sweep/` | `sweep_results.csv`, `sweep_{alpha,probe,chunk}.png`, `selective_alpha.png` |
| `/home/dblab/hj/impress_kv_store/` | 실험용 KV 저장소 (sweep/selective는 자동 정리; 남은 것 삭제 무방) |
| `~/flexllmgen_offload_dir/` | FlexGen TorchDisk 작업 디렉토리 |

## 5. 결과 해석 시 전제 (요약)

1. **prefix 길이 ≤ ~1900 tokens** (OPT pos-embedding 2048 한계; `--num-shots 21`이 상한).
   짧은 prefix(<500)에서는 ReComp가 이기는 게 정상 (논문 Figure 3과 일치).
2. **cold/warm**: 각 prefix의 첫 요청(트리 삽입 포함)은 TTFT 통계에서 제외, warm만 측정.
3. **chunk-skip은 KV reordering 이후에만 유효** — Stage 10은 8요청마다 재정렬,
   13번째 요청부터 측정하는 프로토콜.
4. **물리 I/O 측정(Stage 10 Part 2-3)은 캐시 OFF** (capacity 0) 조건.
5. **disk 시간 절대값은 page-cache warm이라 낙관적** — 파일 수/바이트 계측과
   상대 비교(단조성, 배수)가 결론의 근거.
6. gen_len=1 (TTFT 중심). 선택된 prefix KV의 decode 캐시 배선은 미구현.

## 6. 주요 CLI 노브

| CLI | 기본 | 의미 |
|---|---|---|
| `--retention` | 0.25 | 유지할 important token 비율 |
| `--alpha` | 0.6 | similarity threshold = j^alpha |
| `--probe-heads` | 3 | 레이어당 probe head 수 |
| `--num-shots` | 21 (e2e/sweep), 5~10 (stage 4/5) | prefix 길이 (shot당 ~85 tokens) |
| `--num-samples` | 20~50 | 평가 문항 수 |
| `--gpu-cache-frac` (e2e) | 0.6 | TokenCache GPU 용량 (전체 KV 대비) |
| `--seed` | 0 | 워크로드 샘플링 시드 |
