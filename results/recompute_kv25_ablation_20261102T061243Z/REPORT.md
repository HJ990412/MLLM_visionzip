# Full-context KV25 vs. reduced-token Recompute KV25 (LLaVA, GQA Stage 1)

Run: `runs/recompute_kv25_ablation_20261102T061243Z`. Contract: [CONTRACT.md](CONTRACT.md) (frozen; G2 revision 2 was set after the dev samples n355567/n9181 and before measurement; revision 3 after the rev.-2 Stage-1 run stopped at n51002 and before any of its quality results were opened — see the amendment history, [dev_g2_revision1_failure/](dev_g2_revision1_failure/) and [dev_g2_revision2_failure/](dev_g2_revision2_failure/)). The stopped rev.-2 run is kept as a record only and is not analysed here.

Workload: 40 GQA images × `questions[4:10]` (manifest `40e23fa7…`), T1 = `questions[4]` with full-token inference for every arm, 200 cache-hit requests (T2–T6) per arm. LLaVA-NeXT-Vicuna-7B NF4/BF16/eager, FP16 KV, 64-token chunks, separator sidecar, greedy max 16, cold page cache (DONTNEED) outside timers. All three KV25 stores share S, k, order and payload bytes (G1); PosShift reuses the Ours store.

## Gates

| Gate | Result |
|---|---:|
| G1 selection/bytes/embeds/T1 identity | PASS |
| G1 equal hit read bytes/preads (4 KV25 arms) | PASS |
| G2 rev. 3: structural bit-identity vs truncation control + layer-0 S/sep (all 40 images) | PASS |
| G3 positions (all KV25 hits; persistence) | PASS |
| G4 GPU correctness (5 pairs) | PASS |
| G5 protection before/after | PASS |
| G6 reference reproduction (report gate A) | Ours 100.00% sequence agreement |
| G7 sentinel non-leakage (smoke) | PASS |

## Cache-hit quality (T2–T6)

| Arm | Hit acc. | All acc. | First-token agr. vs FullLoad | Prediction agr. vs FullLoad |
|---|---:|---:|---:|---:|
| ReComp | 63.00% | 62.50% | 98.00% | 98.00% |
| FullLoad | 63.00% | 62.50% | 100.00% | 100.00% |
| Ours-KV25 | 56.50% | 57.08% | 82.00% | 82.00% |
| Recomp-KV25-Orig | 56.00% | 56.67% | 78.00% | 77.50% |
| Recomp-KV25-Compact | 60.50% | 60.42% | 84.50% | 84.50% |
| Ours-KV25-PosShift | 0.00% | 10.00% | 0.00% | 0.00% |

PosShift T1 is the Ours T1 (shared store, not re-executed).

### Paired differences (image-cluster bootstrap, 10,000 resamples, seed 1234)

| Difference | Mean | 95% CI | a-only / b-only correct | Pre-registered reading |
|---|---:|---:|---:|---:|
| Ours-KV25 − Recomp-KV25-Orig | +0.50 pp | [-2.50 pp, +3.50 pp] | 7 / 6 | difference not shown (CI includes 0; not equivalence) |
| Ours-KV25 − Recomp-KV25-Compact | -4.00 pp | [-8.00 pp, +0.00 pp] | 6 / 14 | difference not shown (CI includes 0; not equivalence) |
| Ours-KV25 − Ours-KV25-PosShift | +56.50 pp | [+48.50 pp, +64.00 pp] | 113 / 0 | Ours-KV25 higher (CI lower bound > 0) |
| Recomp-KV25-Orig − Recomp-KV25-Compact | -4.50 pp | [-8.00 pp, -0.50 pp] | 7 / 16 | Recomp-KV25-Compact higher (CI upper bound < 0) |

## Decisions (rules fixed in the contract before results)

- **Ours − Recomp-Orig**: CI includes 0 → no quality difference shown. The paper uses only the cost argument (no extra prefill).
- **Ours − PosShift**: Ours-KV25 higher (CI lower bound > 0). This is evidence that original-input text positions are needed.
- **Ours − Recomp-Compact** (VisionZip-style alternative): difference not shown (CI includes 0; not equivalence). The CI upper bound is exactly +0.00 pp (mean -4.00 pp, Compact-only correct 14 vs Ours-only 6); by the fixed rule (upper bound < 0) this is not a shown difference, and it is reported as a boundary case.
- **Recomp-Orig − Recomp-Compact** (no pre-registered decision; reported as is): Recomp-KV25-Compact higher (CI upper bound < 0) (mean -4.50 pp, CI [-8.00 pp, -0.50 pp]).

## Quality interpretation and numerical noise floor

Rows whose context is identical still change under a different prefill shape because BF16 GEMM kernels and RoPE rounding depend on sequence length. Controls (G4 pairs, recorded, not gated), each compared with the T1 cache of the same request: prefix-only drops only the (causally irrelevant) question; Orig is the actual reduced prefill; same-length / half-length truncations cut the original sequence to the reduced length `L` / `L/2`, so the structural rows keep context and position and only the shape changes.

| Image | Prefix-only last-layer S K / V | Orig last-layer S K / V | Context / prefix-only floor (K, V) | Orig sys max-layer K / V |
|---|---:|---:|---:|---:|
| n355567 | 0.009 / 0.020 | 0.214 / 0.420 | 23.3× / 21.0× | 0.0080 / 0.0145 |
| n9181 | 0.010 / 0.021 | 0.219 / 0.419 | 22.6× / 20.1× | 0.0080 / 0.0145 |
| n390187 | 0.010 / 0.023 | 0.213 / 0.416 | 20.6× / 18.0× | 0.0080 / 0.0145 |
| n133585 | 0.010 / 0.021 | 0.225 / 0.442 | 23.3× / 21.0× | 0.0080 / 0.0145 |
| n272098 | 0.010 / 0.022 | 0.222 / 0.422 | 21.7× / 19.3× | 0.0073 / 0.0134 |

Single-row (common-prefix) drift, max over layers ≥ 1, K / V:

| Image | Common rows | Orig reduced | Same-length trunc. | Half-length trunc. | Context / single-row (K, V) |
|---|---:|---:|---:|---:|---:|
| n355567 | 0 | — | — | — | — |
| n9181 | 0 | — | — | — | — |
| n390187 | 0 | — | — | — | — |
| n133585 | 0 | — | — | — | — |
| n272098 | 1 | 0.0140 / 0.0270 | 0.0140 / 0.0270 | 0.0165 / 0.0297 | 15.9× / 15.6× |

Dev observations behind the G2 revisions: (rev. 2, n355567/n9181) prefix-only last-layer S difference 0.010 / 0.021 (K / V) versus the Recomp-Orig context effect 0.21 / 0.42, about 21× / 20×; (rev. 3, n51002/n272098/n331357) single-row V drift 0.0308 / 0.0270 / 0.0190 for both the Orig reduced prefill and the same-length truncation (identical), 0.0329 / 0.0297 / 0.0202 for the half-length truncation, 0 for prefix-only; K 0.0105–0.0165. Context effect 0.21 / 0.40 (K / V) is 13–20× (K) and 12–21× (V) the single-row drift.

Stage 1 shape-only drift of common-prefix rows (truncation control vs Ours, max over layers ≥ 1) on 16 images with a common prefix: K 0.0099–0.0165, V 0.0179–0.0321; the last-layer S context effect on the same images is K 0.195–0.232, V 0.371–0.446.

**Because BF16 shape-dependent rounding alone measurably changes deep-layer KV of context-identical rows and can flip a boundary first token, differences below 1 pp are not interpreted as established.** With 40 image clusters a CI that includes 0 is also not evidence of equivalence.

G2 rev. 3 over all 40 images: structural rows bit-identical to the same-length truncation control at every layer (K and V) for Orig 40/40 and Compact 40/40 images (5–7 rows). Maximum layer-0 K oracle ratio 1.063 (bound 1.5). Report-only rev.-2 values: maximum deep-layer structural rel L2 vs Ours Orig sys 0.0152 / common 0.0321 (V), Compact sys 0.0152 / common 0.0321 (V); images outside the old bound: Orig 2, Compact 2. Truncation control wall time (outside every timer, not in persistence or TTFT): Orig 89.5 ms, Compact 89.6 ms per image.

## Per-layer KV difference vs Ours (mean over 40 images, S rows)

| Layer | Orig K | Orig V | Compact K raw | Compact K realigned | Compact V |
|---|---:|---:|---:|---:|---:|
| 0 | 0.0023 | 0.0021 | 1.1739 | 0.0043 | 0.0021 |
| 1 | 0.0308 | 0.0244 | 1.0248 | 0.0262 | 0.0193 |
| 2 | 0.0275 | 0.0340 | 0.9335 | 0.0262 | 0.0326 |
| 4 | 0.0678 | 0.0972 | 0.9756 | 0.0461 | 0.1067 |
| 8 | 0.1232 | 0.2171 | 0.9998 | 0.2145 | 0.3872 |
| 12 | 0.2103 | 0.3801 | 0.9785 | 0.3233 | 0.5214 |
| 16 | 0.2280 | 0.3648 | 0.9762 | 0.3535 | 0.4789 |
| 20 | 0.2162 | 0.3380 | 0.9296 | 0.3191 | 0.4294 |
| 24 | 0.2112 | 0.3573 | 0.9088 | 0.2959 | 0.4498 |
| 28 | 0.2173 | 0.4015 | 0.9283 | 0.2739 | 0.5000 |
| 31 | 0.2158 | 0.4184 | 1.0333 | 0.2590 | 0.5211 |

Full per-layer table (S and separator rows, rel L2 and cosine): [kv_layer_diff.csv](kv_layer_diff.csv).

## Cache-hit latency and SSD reads

| Arm | TTFT mean / p50 / p95 (ms) | E2E mean (ms) | Read MB/hit | preads/hit |
|---|---:|---:|---:|---:|
| ReComp | 518.09 / 525.16 / 589.02 | 543.82 | 0.000 | 0.0 |
| FullLoad | 691.16 / 675.56 / 879.58 | 718.12 | 1165.073 | 64.0 |
| Ours-KV25 | 265.71 / 263.16 / 324.68 | 292.55 | 306.289 | 65.0 |
| Recomp-KV25-Orig | 262.17 / 259.72 / 321.33 | 289.39 | 306.289 | 65.0 |
| Recomp-KV25-Compact | 266.27 / 263.74 / 321.08 | 293.17 | 306.289 | 65.0 |
| Ours-KV25-PosShift | 265.85 / 263.35 / 329.90 | 549.52 | 306.289 | 65.0 |

The four KV25 arms read identical bytes and preads on every hit: yes. Recomp-KV25-Orig − Ours TTFT -3.54 ms [-7.34, +0.47]; Recomp-KV25-Compact − Ours TTFT +0.56 ms [-3.44, +4.60]; Ours-KV25-PosShift − Ours TTFT +0.14 ms [-1.28, +1.53].

## Persistence cost per image (mean ms; not part of cache-hit TTFT)

| Arm | Saliency | Permutation | Plan | Embed gather | Reduced prefill | Synthetic assembly | Repack | Write | fsync | Seal | Total | File MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FullLoad | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 407.28 | 398.89 | 629.26 | 1631.66 | 1260.58 |
| Ours-KV25 | 0.02 | 0.65 | 0.00 | 0.00 | 0.00 | 0.00 | 132.11 | 103.31 | 168.90 | 179.06 | 608.77 | 309.02 |
| Recomp-KV25-Orig | 0.02 | 0.60 | 0.91 | 0.11 | 87.70 | 2.25 | 132.82 | 104.08 | 166.58 | 179.42 | 698.42 | 309.02 |
| Recomp-KV25-Compact | 0.02 | 0.59 | 0.91 | 0.11 | 87.68 | 2.24 | 132.90 | 104.20 | 171.01 | 177.93 | 701.43 | 309.02 |

**Primary metric — prefill-only Δ** (embed gather + CUDA-synchronized reduced prefill), amortized over requests per image:

| Arm | Prefill-only Δ mean / p95 | ÷1 | ÷3 | ÷5 | ÷10 | Full Δ vs Ours [95% CI] | Full Δ ÷1/3/5/10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Recomp-KV25-Orig | 87.81 / 114.83 | 87.81 | 29.27 | 17.56 | 8.78 | +89.6 [+79.0, +99.9] | +89.6 / +29.9 / +17.9 / +9.0 |
| Recomp-KV25-Compact | 87.79 / 114.64 | 87.79 | 29.26 | 17.56 | 8.78 | +92.7 [+77.9, +110.2] | +92.7 / +30.9 / +18.5 / +9.3 |

The full Δ includes the full-size synthetic-cache assembly and plan recomputation that come from reusing the unchanged prefix25 writer, and wall-time noise of write/fsync; the prefill-only Δ is the cost a native implementation cannot avoid. The reduced prefill also requires keeping the T1 `inputs_embeds` (~24 MB BF16) until persistence.

## Reference reproduction (report gate A)

| New arm | Reference arm | First-token agr. | Prediction agr. | Sequence agr. | New hit acc. | Reference hit acc. |
|---|---:|---:|---:|---:|---:|---:|
| Ours-KV25 | B | 100.00% | 100.00% | 100.00% | 56.50% | 56.50% |
| FullLoad | fullload | 100.00% | 100.00% | 100.00% | 63.00% | 63.00% |
| ReComp | recompute | 100.00% | 100.00% | 100.00% | 63.00% | 63.00% |

## Limits

- 40 image clusters from the existing development set (not a holdout); 200 hits per arm. Stage 2 (MT-GQA) not run.
- LLaVA only; Qwen2.5-VL (M-RoPE, compact cache) not ported — see the contract extension note.
- Recomp arms reuse the arm's own T1 `inputs_embeds`; no VisionZip contextual-token merge (to keep S identical). Results therefore isolate KV context and position rule, not the full VisionZip system.
- G2 was revised twice: rev. 2 after two dev samples before measurement; rev. 3 after the rev.-2 Stage-1 stop at n51002, before any of that run's quality results were opened (Stage 1 restarted from G4 in this run). All versions and failure logs are kept.
- Latency is single-GPU, batch 1, page-cache DONTNEED; no NAND/controller coldness claim.

## PosShift 출력 유형 (사후 기술 분석, 2026-11-03)

이 절은 135 report 생성 후 수동으로 추가한 기술(descriptive) 분석이다. 사전 판정은 바꾸지 않는다.
135를 다시 실행하면 이 절은 사라진다. 입력은 Ours-KV25-PosShift의 cache-hit 200 requests(40 images × T2–T6)이고,
원본 raw `runs/recompute_kv25_ablation_20261102T061243Z/llava/gqa/raw_requests.jsonl`은 읽기만 했다.
T1은 Ours T1을 공유하므로 제외했다. 스크립트, 추출본, 로그, sha256은 `posshift_output_types/`와 REPRODUCE.md에 있다.

### 1. 전체 특징

| 항목 | 값 |
|---|---|
| Hit 요청 | 200 |
| 정답 (strict EM) | 0 / 200 |
| 16 토큰 cap 도달 | 200 / 200 (EOS 0개) |
| 서로 다른 출력 문자열 | 90 |
| 첫 토큰이 개행 `\n` (id 13) | 156 / 200 (78%) |
| 나머지 첫 토큰 | `▁in` 12, `▁on` 8, `\|` 7, `***` 4, `==` 2, `[[` 1 등 |

### 2. 출력 유형

| 유형 | 개수 | 비율 | 토큰 패턴 |
|---|---:|---:|---|
| A. 개행만 | 96 | 48% | `\n` × 16 (문자열은 `''`) |
| C. 전치사·기능어 반복 | 60 | 30% | `\n`×k 뒤 `▁on ▁on ▁on…`, `in in in…`, `to to to… in in` |
| B. 기호 반복 | 36 | 18% | `\| \n \| \n …`, `== \n == …`, `[[\| \n \|` |
| D. 기호 + 조각 | 5 | 2.5% | `\n \| \n \| … d : d : d : pol` |
| E. 기타 혼합 | 3 | 1.5% | `\n` 사이에 `p`, `____` 등이 섞임 |

유형 분류 규칙은 `posshift_output_types/posshift_types.py`의 `kind()`에 있다. 규칙은 출력을 본 뒤에 정한 사후 규칙이다.

### 3. 유형별 실제 예시

| 유형 | 질문 | Gold | Ours (같은 요청) | PosShift 생성 토큰 |
|---|---|---|---|---|
| A | Is the motorcycle near the jacket black and metallic? | no | No | `\n` × 16 |
| A | What's the wheelchair in front of? | vest | Curb | `\n` × 16 |
| B | Does the person to the left of the lady look happy and modern? | yes | Yes | `== \n == \n \n …` |
| B | How large is the wire net? | small | Small | `\n == \n == \n == \n == …` |
| C | Are there any boys to the right of the fat person? | yes | Yes | `\n`×5 + `▁on` × 11 |
| C | Is the person to the left of the boy wearing jeans? | no | No | `\n`×3 + `▁on` × 13 |
| D | What kind of baked good is it? | donuts | Donut | `\n \| \n \| \n \| \n \| \n d : d : d : pol` |
| E | What is the color of the frisbee that flies above the ground? | green | Green | `\n`×8 `p \n p \n …` |

### 4. 턴별 분포

| Turn | A 개행 | C 기능어 | B 기호 | D | E |
|---|---:|---:|---:|---:|---:|
| T2 | 17 | 16 | 6 | 1 | 0 |
| T3 | 21 | 11 | 6 | 1 | 1 |
| T4 | 20 | 11 | 8 | 1 | 0 |
| T5 | 18 | 12 | 8 | 1 | 1 |
| T6 | 20 | 10 | 8 | 1 | 1 |

### 5. 이미지 단위 경향

| 이미지당 "개행만" 출력 수 (5턴 중) | 이미지 수 |
|---|---:|
| 0 | 18 |
| 2 | 3 |
| 4 | 5 |
| 5 (모든 턴) | 14 |

같은 이미지의 5턴 출력이 문자열까지 모두 같은 이미지는 14개다.

### 해석 범위

- 사전 판정(Decisions 절)은 그대로다. Ours − PosShift +56.50 pp [+48.50, +64.00]로 Ours-KV25가 더 높고, 이는 원래 입력의 text position이 필요하다는 증거다.
- 이 절은 그 0% 결과가 어떤 출력 형태였는지만 기술한다. 오답 답변이 아니라 EOS 없이 16 토큰 cap까지 개행, 기호, 기능어가 반복된 형태였다(유효한 답 형태 0/200).
- 붕괴 원인(예: attention 분포)은 측정하지 않았다. 유형 분류는 사후 규칙이고 통계 검정 대상이 아니다.

