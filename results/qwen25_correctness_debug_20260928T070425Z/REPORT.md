# Qwen2.5-VL GPU correctness: 최초 분기 조사

실행일 2026-09-28. 이 문서는 최초 Qwen port의 실패 기록을 수정하지 않고, 동일 checkpoint·NF4·BF16·prompt·image policy·64-token chunk·nominal 25%·greedy 16 tokens·기존 tolerance에서 한 이미지의 수치 분기점을 조사한 결과다. **Production adapter의 구현 오류는 확인되지 않아 adapter를 변경하지 않았다. GPU correctness는 여전히 FAIL이고 pilot 재실행 준비 상태는 NO다.** 새 디버깅 결과는 모두 `runs/qwen25_correctness_debug_20260928T070425Z/`에 있다.

## A. 최초 divergence

기존 SDPA validation에서 두 failure가 모두 난 질문 `201751740`을 primary fixture로 고정했다. 이미지 `n355567`, content SHA256 `6b09d6ab2c13d108951fd6d53522fcb80e6cfd5bb3e1f2925e1001919343a39f`, 질문은 “Does the person to the left of the lady look happy and modern?”이다. Full input IDs 394개, image grid `[[1,30,46]]`, visual 345개, prefix 경계 exclusive 366, suffix 28개, 선택 visual 128개/2 chunks이다. 전체 `input_ids`, physical first-k와 original indices, original→stored mapping, revisions는 `primary_fixture.json` 및 `sdpa_q2/fixture_201751740.json`에 보존했다. R0/R1, P0/P1/P2 모두 첫 생성 token `9693`, 생성 token sequence `[9693,151645]`가 같고, 비교한 중간 텐서와 logits에 NaN/Inf가 없다.

### A1. FullLoad 대 pixel ReComp

R0는 `pixels + full prompt` 정상 prefill이고, R1은 **같은 R0 forward에서 만든 prefix KV를 직접 복제**한 뒤 SSD를 전혀 읽지 않고 suffix만 forward한다. 새 15-gate 검증에서 R1과 SSD FullLoad(R2)의 첫 logits은 세 질문 모두 최대 차이 **0**으로 재확인했다. 따라서 R0/R1 차이는 SSD read·직렬화 이전에 존재한다.

| Layer 0 suffix 단계, R0 대 R1 | bitwise 동일? | 최대 절댓값 차이 | 평균 절댓값 차이 | 첫 mismatch index |
|---|---|---:|---:|---|
| Embedding / layer input | YES | 0 | 0 | 없음 |
| Input RMSNorm / attention input | YES | 0 | 0 | 없음 |
| Q projection, MRoPE 전 | YES | 0 | 0 | 없음 |
| **K projection, MRoPE 전** | **NO** | **0.5** | **0.001541** | **[0,0,10]** |
| V projection | NO | 0.01171875 | 0.000493 | [0,0,0] |
| Q after MRoPE | YES | 0 | 0 | 없음 |
| K after MRoPE | NO | 0.5 | 0.001660 | [0,0,0,8] |
| Attention output | NO | 0.03125 | 0.000552 | [0,0,0] |
| 최종 first-token logits | NO | 0.4375 | 0.031024 | [0] |

**최초 수치 분기는 layer 0의 NF4 K projection 출력이다.** 입력 BF16 hidden state와 RMSNorm 값은 동일하다. 별도 재호출에서 같은 RMSNorm suffix를 28-token shape으로 `k_proj`에 넣으면 R0 대비 4,114 BF16 값이 다르고 최대 차이 0.5다. V에서는 7,836값/최대 0.01171875가 다르다. 같은 suffix를 앞쪽 366개 dummy row로 채워 **394-token matrix shape**으로 projection한 뒤 suffix만 취하면 Q/K/V 세 projection 결과가 모두 R0와 bitwise 동일하다. Full matrix 재호출도 원본과 bitwise 동일하다. 이 실험은 prefix 내용 차이 없이 GEMM 입력 shape에 따른 NF4/BF16 projection 수치 경로가 최초 분기임을 분리한다. 본래 R1에 dummy rows를 넣는 코드는 production에 적용하지 않았다.

R0의 stock SDPA는 mask `None`, `is_causal=True`, native 4-KV-head GQA다. R1은 Boolean mask `[1,1,28,394]`, `is_causal=False`, 4 KV heads를 28 query heads로 repeat한다. 해당 추가 수치 경로 차이도 있다. 동일한 논리적 causal pattern을 가진 **명시적 R0 mask**만 강제한 별도 control은 원래 R0와 logits 차이 0.21875를 만들고 layer 0 attention output부터 달라졌다. 이 control과 R1의 차이는 0.25이며 최초 분기는 여전히 K projection이다. 그러므로 SDPA branch는 부가 요인이지만 최초 원인은 아니다.

### A2. Prefix25 dense 대 compact

P0는 원래 prefix 366 slots에 선택 K/V만 두고 미선택 visual 217개를 mask한다. P1은 SSD first-k의 physical importance 순서를 GPU compact cache에도 유지하는 **새 진단 경로**다. P2는 선택 K/V를 원래 token 순서로 정렬해 149-slot compact cache에 두는 **기존 Ours25 구현**이다. P1/P2는 같은 단 한 번의 first-k SSD 읽기(visual 7,340,032 B, 총 `pread` 57회)에서 갈라졌다. 현재 Ours25가 P1이라는 전제는 사실과 다르다: `store.py`의 `sort_order`/`index_select`가 기존 경로를 이미 P2로 만든다.

세 경로는 같은 128개 선택 token과 21개 structural token을 쓰며 P0/P1/P2 suffix의 full logical MRoPE가 같다. CPU invariant audit 20/20 PASS: 모든 layer의 선택 KV raw bits가 canonical full store와 같고, inverse permutation, selected tuple, stock `get_rope_index`, stock `create_causal_mask`의 original-index 대응 Boolean 가시 pattern이 exact하다. 첫 suffix query는 선택 prefix 149개+현재 token 1개=150개를 보고, 마지막은 149+28=177개를 본다. 미선택 visual·미래 suffix key는 0개 보인다.

| 비교 | first-token logits 최대 절댓값 차이 | 사전 `allclose` | 첫 수치 분기 |
|---|---:|---|---|
| P0 dense 대 P1 physical | 0.25 | FAIL | Layer 0 attention output |
| **P0 dense 대 P2 logical (기존 경로)** | **0.15625** | **FAIL** | **Layer 0 attention output** |
| P1 physical 대 P2 logical | 0.25 | FAIL | Layer 0 attention output |

P0/P2에서는 suffix embedding, layer input, RMSNorm, Q/K/V projection 및 Q/K after MRoPE가 전부 bitwise 동일했다. Layer 0 attention output이 처음 달라지며 최대 차이는 0.0078125다. P0/P2의 BF16 SDPA는 모두 명시적 Boolean mask, repeated 28-head KV, `is_causal=False`다. 차이는 key tensor 길이 **394 대 177** 및 masked holes/layout이다. P1/P2는 길이 177로 같지만 key 물리 순서가 다르다. P2가 P0에 더 가까워, 기존 original-order 복원은 수치 차이를 줄이는 방향이다.

같은 layer 0 Q/K/V와 같은 가시 key set을 별도 FP32 attention math로 계산하면 P0/P2 attention output 최대 차이는 **`3.58e-7`**로 줄었다. BF16 SDPA에서는 **0.0078125**였고, FP32 출력을 BF16으로 내린 후에는 서로 다른 값이 3개, 최대 0.00012207뿐이었다. 이 결과는 P0/P2의 첫 분기가 선택·MRoPE·causal semantics보다 BF16 attention kernel의 key 길이/감산 순서 민감도에서 발생함을 뒷받침한다. P1/P2의 차이도 physical ordering에 따른 수치 감산 순서 차이이며, source store bytes나 선택 set은 동일하다.

## B. 원인 판정과 위치·상태 검사

| 비교 | 판정 | 배제·확인한 근거 |
|---|---|---|
| R0 대 R1 | **confirmed backend/BF16 numerical sensitivity**: layer 0 NF4 K/V projection의 matrix shape | 전체 input IDs와 suffix embeddings exact; R0 prefix K/V와 R1에 넘긴 복제본 28 layers bitwise exact; stock/manual MRoPE 및 논리 mask exact; 394-row padded projection control이 K/V exact 복원 |
| P0 대 기존 P2 | **confirmed backend/BF16 numerical sensitivity**: layer 0 SDPA attention의 key length/layout | 같은 selected set·Q/K/V·MRoPE·Boolean 가시성 exact; FP32 attention control에서 첫 출력 차이가 `3.58e-7`로 감소 |
| P1 대 P2 | **backend/BF16 numerical sensitivity**: 물리 key order | 같은 SSD read와 key set/가시성, 서로 다른 attention reduction order. 기존 production은 P2 |
| confirmed implementation bug / MRoPE mismatch / causal-mask mismatch / state leakage | **이번 fixture에서 확인되지 않음** | 자세한 assertion과 trace는 아래 artifact 참조 |

R0/R1의 complete expanded input IDs가 같고, Qwen stock `prepare_inputs_for_generation()`의 4축 output 중 temporal/height/width 3축이 수동 full-prompt `get_rope_index()`와 exact하다. 두 경로의 suffix 첫 위치는 `[44,44,44]`, 끝은 `[71,71,71]`, 첫 생성 token은 `[72,72,72]`, `rope_deltas=[[-322]]`다. Stock text axis는 full `[0..393]`, suffix `[366..393]`다. R1 `cache_position`은 366부터 393까지이고, P0는 366부터, P1/P2는 149부터 시작한다. 이는 storage slot이며 MRoPE 좌표로 재번호화되지 않는다. R0/R1 mask를 full logical key index로 대응하면 28×394 Boolean pattern의 다른 cell이 0개다. 첫/마지막 suffix row의 visible keys는 367/394개다. 모든 독립 request 시작 시 trace의 `model.rope_deltas`는 cleared이며, 새 15-gate `request_isolation`도 다른 이미지 실행 후 같은 hit 결과가 exact함을 재확인했다. 설치된 Qwen attention에서 K에 MRoPE를 적용한 뒤 `past_key_values.update`에 기록하고, hit 경로는 그 post-MRoPE key에 rotary를 재적용하지 않는다.

## C. backend control

고정 질문 `201751740`, 동일 checkpoint/NF4/BF16 model/입력/seed에서 R0 대 R1이다. FP32는 decoder attention Q/K/V만 FP32 math backend로 계산하고 출력을 BF16으로 되돌린 별도 진단이다. Production/backend 변경이 아니다.

| Backend | R0 대 R1 max logit diff | 기존 tolerance 통과? | 첫 token 동일? | 전체 생성 동일? | 최초 분기 |
|---|---:|---|---|---|---|
| SDPA BF16 (production) | 0.4375 | NO | YES | YES | layer 0 K projection |
| Eager BF16 | 0.515625 | NO | YES | YES | layer 0 K projection |
| SDPA + FP32 decoder attention diagnostic | 0.25 | NO | YES | YES | layer 0 K projection |

Eager로 차이가 줄지 않고 FP32 attention에서도 gate를 통과하지 않는다. 이는 attention precision 변경만으로 layer 0 projection shape 차이를 없앨 수 없음을 보여 준다. 세 control은 각각 독립 run/provenance를 보존한다.

## D. physical order test

P0/P1/P2 logits 표는 A2에 있다. 예시 selected visual index 100은 stored index 2, original prompt index 120, MRoPE `(t,h,w)=(20,24,28)`, P1 compact index 22, 기존 P2 compact index 68이다. 모든 128 selected token의 tuple은 `cpu_invariants.json`과 `sdpa_q2/fixture_201751740.json`에 있다. P1과 P2의 first token 및 전체 generated sequence가 P0와 같다. P1을 production으로 바꾸면 기존 0.15625가 0.25로 커지므로 이 진단 결과는 P1 전환을 뒷받침하지 않는다.

## E. 변경 내용

Production `mmimpress/qwen25/{runner,store,vision}.py`, 기존 validator·pilot·dataset·store·validation·raw는 **변경하지 않았다**. 확인된 것은 수치 경로 민감도이며, prompt/MRoPE/mask/assembly의 최소 수정 대상이 되는 구현 오류는 찾지 못했다. 새로운 파일은 `scripts/81_debug_qwen25_correctness.py`와 debug 전용 `quick_layer0_probe.py`, `projection_shape_control.py`, `attention_shape_control.py`, `cpu_invariants.py`, 각 JSON 및 본 보고서다. Debug 하네스는 모든 decoder layer의 suffix slice 통계만 기록하고 전체 tensor dump를 저장하지 않는다. 하네스 SHA256 `e3143be60c932e60efc61d57939e23a189d2416a077b1086e20896c99c945c33`이다. Production adapter 세 파일 SHA256은 이전 frozen validation과 동일하다.

## F. validation 재실행

Production 수정이 없다는 점을 명시한 채 동일 SDPA·동일 tolerance(`atol=0.125,rtol=0.02`)로 기존 15개 GPU gate를 새 `full_validation_sdpa_unchanged/validation.json`에 다시 실행했다. Exit code 1은 현재의 실제 FAIL 결과다. 모델·processor·tokenizer revision과 source/sample IDs, adapter code revision도 이전과 같다.

| Gate | Before | After |
|---|---|---|
| CPU score reference | PASS | PASS |
| Runtime load | PASS | PASS |
| Source capture | PASS | PASS |
| Geometry | PASS | PASS |
| GPU score reference | PASS | PASS |
| Query independence | PASS | PASS |
| Persistence | PASS | PASS |
| BF16 SSD round-trip | PASS | PASS |
| **FullLoad SSD 대 ReComp logits** | **FAIL** | **FAIL** |
| RepackedFull100 | PASS | PASS |
| **Prefix25 compact 대 dense logits** | **FAIL** | **FAIL** |
| Capture | PASS | PASS |
| I/O | PASS | PASS |
| Request isolation | PASS | PASS |
| History | PASS | PASS |

FullLoad 세 질문의 차이는 before/after 모두 `0.1328125, 0.4375, 0.4375`; Prefix25는 `0.125, 0.15625, 0.3125`로 동일하다. FullLoad SSD 대 in-memory 및 RepackedFull100 대 canonical은 여전히 각각 최대 차이 0이다. 최종 CPU 전체 suite는 **351/351 PASS**였다. `protected_before.json`/`protected_after.json` 비교에서 기존 Qwen artifact **9,917개/4,154,349,211 bytes**, 기존 LLaVA·data 등 legacy 보호 파일 **8,883개**가 모두 bitwise unchanged이며 추가·삭제·변경은 0개였다.

| 최종 상태 | 판정 |
|---|---|
| GPU CORRECTNESS | **FAIL** (13/15) |
| FULLLOAD PATH | **FAIL** (ReComp logits 근접성) |
| PREFIX25 PATH | **FAIL** (dense logits 근접성) |
| READY FOR PILOT RERUN | **NO** |

15/15가 아니므로 3~5 image smoke와 GQA 720/MT 360 pilot은 실행하지 않았다.

## G. 남은 문제와 재현

이 조사는 **최초 수치 분기의 연산과 수치적 원인**을 특정했다. 그러나 사전 허용치의 logits equivalence를 만족하는 production 계산 경로는 아직 없다. 다음 조사 위치는 (1) bitsandbytes `Linear4bit`의 394-row 대 28-row K/V projection kernel/accumulation dispatch를 추적하고, (2) 동일 BF16·NF4 조건에서 shape-independent projection 가능성을 검증하며, (3) dense/compact key shape에 대해 수치적으로 일관된 attention 연산을 설계하고 그 비용을 측정하는 것이다. 이 과정에서 모델·budget·chunk·score·dataset·tolerance를 바꾸지 말아야 한다. Numeric control을 근거로 기존 FAIL을 PASS로 재라벨링하지 않는다. 그런 production 경로가 실제 구현·검증되기 전까지 pilot 재실행은 막는다.

주요 파일: `runs/qwen25_correctness_debug_20260928T070425Z/primary_fixture.json`, `cpu_invariants.json`, `quick_layer0_probe.json`, `projection_shape_control.json`, `attention_shape_control.json`, `sdpa_q2/trace_201751740.json`, `eager_q2/trace_201751740.json`, `fp32_attention_q2/trace_201751740.json`, `sdpa_all3_r0r1/summary.json`, `full_validation_sdpa_unchanged/validation.json`, `protected_before.json`, `protected_after.json`. 동일한 한 질문 trace 재현 명령:

```bash
cd /home/dblab/hj/mllm_v2
PY=/home/dblab/anaconda3/envs/mllm_ft/bin/python
TAG=$(date -u +%Y%m%dT%H%M%SZ)
$PY scripts/81_debug_qwen25_correctness.py --attn sdpa --forced-mask-control --out-dir "runs/qwen25_correctness_debug_${TAG}_sdpa"
$PY scripts/81_debug_qwen25_correctness.py --attn eager --no-prefix --out-dir "runs/qwen25_correctness_debug_${TAG}_eager"
$PY scripts/81_debug_qwen25_correctness.py --attn sdpa --fp32-attention --no-prefix --out-dir "runs/qwen25_correctness_debug_${TAG}_fp32"
$PY scripts/78_validate_qwen25.py --attn sdpa --out-dir "runs/qwen25_correctness_debug_${TAG}_validation"
```
