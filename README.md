# Storage-aware visual KV selection for image-prefix IMPRESS

IMPRESS (FAST'25) 의 disk 기반 prefix-KV 재사용을 **이미지 prefix** 로 옮기고,
중요 visual token 판별을 여러 방식으로 교체·비교한 구현.

- 모델: `llava-hf/llava-v1.6-vicuna-7b-hf` (LLaVA-NeXT, 32L x 32H MHA), 4-bit NF4, eager
- 데이터: GQA `testdev_balanced` — 이미지 40장 / 질문 1177개 (이미지당 평균 29개)
- 이미지 1장 = prefix 1개 = **visual KV 약 1.1 GB** (2144 visual tokens x 512 KiB)
- 환경: `conda activate mllm_ft` (torch 2.5.1, transformers 4.57.6), RTX 4090 24GB
- 평가 subset·budget·attention backend 는 모든 방법에서 동일 (eager, 동일 240문항)

## 1. 기존 SparseVLM 방식의 문제

> **visual-token importance identification overhead exceeds the saved SSD I/O
> latency, while token-level sparsity does not directly translate into
> chunk-level I/O sparsity.**

두 문장 모두 측정으로 확인된 것이다.

`scripts/05_profile_sparsevlm.py` (n=12 요청) — per-layer hook 723.2 ms 의 내역:

| stage | ms | 내용 |
|---|---|---|
| chunk_io | 377.9 | 선택 chunk SSD read |
| fallback | 124.8 | Jaccard 미달 layer(16.7%)가 layer 전체를 read |
| scatter | 88.1 | 읽은 row 를 GPU prefix cache 에 기록 |
| probe_io | 53.5 | probe-head sidecar read |
| qk_proj | 21.1 | hook 안에서 suffix Q/K 재투영 (layer 가 곧 또 함) |
| jaccard | 19.3 | probe 합의 + 투표 |
| score | 8.7 | rater x visual softmax |

**online selector compute (hook − disk) = 306.3 ms.** 순수 SparseVLM 수식은
49.1 ms(qk_proj+jaccard+score)에 불과하고, 나머지는 전부 *선택을 저장소에
적용하는 비용* 이다 — fallback 전량 read, chunk 낭비, GPU scatter.

`scripts/03_diagnose.py` — token→chunk mismatch:

| 지표 | 값 |
|---|---|
| 질문 간 top-k Jaccard | **0.712** |
| touched chunk fraction (25% token budget) | **0.49** |
| floor (k / chunk_size / n_chunks) | 0.254 |
| chunk size 16 / 32 / 64 / 128 / 256 의 byte fraction | 0.463 / 0.499 / 0.535 / 0.556 / 0.556 |

같은 이미지의 서로 다른 질문은 중요 토큰을 **상당히 공유한다(0.712)**. 즉
질문마다 다시 계산할 이유가 약하다. 그런데 25% 토큰을 고르면 chunk 는 49% 를
건드린다 — chunk 크기를 바꿔도 0.46~0.56 으로 갇혀 있다. 토큰 단위 top-k 는
chunk 단위 I/O 로 번역되지 않는다.

## 2. 새 방법의 핵심

> **precompute reusable image-level visual saliency, perform only lightweight
> optional query correction, and directly rank storage chunks rather than
> selecting individual tokens first.**

세 가지가 동시에 바뀐다.

1. **재사용**: saliency 를 이미지당 한 번 계산해 sidecar 에 저장하고 그 이미지의
   모든 질문이 재사용한다 (GQA 는 이미지당 평균 29문항).
2. **hook 제거**: 선택이 LLM forward *이전* 에 끝난다. importance 목적의
   `output_attentions=True` 도, attention forward hook 도 없다.
3. **chunk 우선**: budget 이 token 비율이 아니라 **chunk/byte budget** 이라
   touched chunk fraction 이 budget 과 정의상 같아진다.

### selector 목록

| `--selectors` | 내용 |
|---|---|
| `sparsevlm` | 기존 baseline. 코드·조건 변경 없음 |
| `visionzip_static_chunk` | VisionZip static saliency 만. 질문별 계산 없음 |
| `cvpr25_hybrid_chunk` | + PACT 형태의 경량 query correction |
| `cvpr25_hybrid_diverse` | + DivPrune MaxMin 을 chunk 사이 tie-break 로만 |

## 3. Reference 구현 매핑

clone 위치: `/home/dblab/hj/reference_repos/cvpr25/` (원본 수정 없음)

### VisionZip — `JIA-Lab-research/VisionZip`

| 참고한 것 | 파일 / 함수 |
|---|---|
| penultimate CLIP layer 의 CLS→patch attention 을 head 합으로 모아 patch saliency 로 사용 | `visionzip/clip_encoder.py:43-53` — `attn_weights = image_forward_outs.attentions[-2]`, `cls_attention = attn_weights[:, :, 0, 1:]`, `cls_attention_sum = cls_attention.sum(dim=1)` |
| per-patch descriptor 로 penultimate layer key 평균을 쓰는 발상 | `visionzip/utils.py:89` `return ..., raw_key_states.mean(1)` / `:124` `self.metric = metric` |

**가져오지 않은 것**: VisionZip 은 dominant/contextual token 을 뽑아 visual token
sequence 를 그 자리에서 prune·merge 한다 (`clip_encoder.py:57-83`). 우리는
token index ↔ SSD offset mapping 을 유지해야 하므로 **토큰을 하나도 지우지 않고
점수만 쓴다**. 즉 VisionZip 의 *scoring* 만 쓰고 *action* 은 쓰지 않는다.
우리 구현: `mmimpress/cvpr25.py:clip_cls_patch_saliency`, `anyres_token_scores`.

### PACT — `orailix/PACT`

| 참고한 것 | 파일 / 함수 |
|---|---|
| attention matrix 를 만들지 않고 K·Q 축소만으로 importance 를 내는 인터페이스 | `transformers/PACT/utils.py:650-696` `custom_pruning(k_image, q_image)` — `(k_image * q_image).sum(-1)` 후 `mean(dim=1)` (head 평균) |
| rotary 를 양쪽에 동일 적용할지의 선택지 | 같은 파일 `use_cosine_in_token_pruning` 주석 (`:679`) |

**우리의 변경**: PACT 는 layer 안에서 매번 K,Q 를 쓴다. 우리는 **image 쪽 key 를
오프라인으로 밀어냈다** — store 빌드 때 layer-0 `k_proj(input_layernorm(v_hidden))`
를 chunk 평균으로 저장하고(`image_chunk_keys`), 온라인에서는 질문 토큰에 대한
`q_proj` 한 번과 `(n_chunks x hidden)` dot 만 한다(`query_chunk_scores`).
rotary 는 양쪽 모두 미적용해 공간을 일치시켰다.
우리 구현: `mmimpress/cvpr25.py:image_chunk_keys`, `query_chunk_scores`,
`mmimpress/serve.py:CVPR25ChunkSelector._query_scores`.

### DivPrune — `vbdi/divprune`

| 참고한 것 | 파일 / 함수 |
|---|---|
| cosine distance 위의 greedy MaxMin (선택집합까지의 최소거리가 최대인 후보를 추가) | `LLaVA/llava/model/llava_arch.py:151-170` `DivPrune()` (`torch.min(m2, dim=0).values` → `torch.argmax`) |

**우리의 변경**: DivPrune 은 **전체 visual token** 에 diversity 를 적용한다.
그대로 하면 선택이 이미지 전역으로 흩어져 touched chunk fraction 이 오히려
커진다 — 이 프로젝트가 없애려는 실패 모드 그 자체다. 그래서 **chunk 사이에서만**,
그것도 budget 의 뒤쪽 일부를 채우는 tie-break 로만 쓴다.
우리 구현: `mmimpress/cvpr25.py:maxmin_diverse`, `CVPR25ChunkSelector` 의
`mode="diverse"`.

### PyramidDrop — `Cooperx521/PyramidDrop` (참고만, 사용하지 않음)

`llava/model/modeling_llama_pdrop.py:414` 에서
`attn_weights = torch.matmul(query_states, key_states.transpose(2,3))` 로
**full attention matrix 를 만들어** 마지막 instruction token 의 attention 으로
visual token 을 랭킹하고 stage 경계마다 점진적으로 버린다.

쓰지 않은 이유 두 가지. (1) 우리가 제거하려는 비용(full attention + eager) 을
정확히 요구한다. (2) layer-progressive drop 은 KV **retrieval** 과 맞지 않는다 —
저장소는 어떤 layer 도 실행되기 전에 무엇을 읽을지 정해야 한다.
"layer 마다 다른 양을 남긴다" 는 발상만 인지하고, 우리는 layer 별 chunk 예산을
동일하게 두었다.

## 4. 유지한 IMPRESS 요소

| IMPRESS | 유지 여부 |
|---|---|
| 64-token chunk 저장 (§4.2, §6.1) | 유지 — chunk 가 곧 selection 단위가 됨 |
| KV reordering + per-layer mapping list (§4.4.1) | 유지 — static score 도 각 layer 의 stored order 로 매핑 |
| probe-head sidecar (§6.5) | `sparsevlm` baseline 전용으로 유지. 새 selector 는 probe 자체가 없어 불필요 |
| radix tree (§4.1) | 이미지 prefix 는 전부 공유 아니면 전무 → image_id 조회로 축약 (R/NR 분할 없음) |
| decoding 미변경 | 유지 — prefill 에서 mask 고정 |

## 5. storage-aware 변경점 (reference 들과 다른 점)

- reference 3편은 전부 **token 을 줄여 LLM 연산을 줄이는** 것이 목적이고, 줄인
  token 은 되돌릴 필요가 없다. 우리는 **SSD 에서 무엇을 읽을지** 를 정하는 것이
  목적이라 token index ↔ byte offset mapping 이 반드시 살아 있어야 한다.
  그래서 어떤 방법도 token sequence 를 실제로 prune/merge 하지 않는다.
- budget 의 단위가 다르다: reference 는 token 개수, 우리는 **chunk(=byte)**.
  chunk 를 통째로 읽으므로 선택된 chunk 안의 token 은 전부 유지한다 (마스킹해도
  바이트가 줄지 않으므로 정확도만 손해다). 그래서 logical KV ratio 를 가정하지
  않고 실측해 보고한다.
- LLaVA-NeXT 의 row separator(`image_newline`) 는 구조 토큰이다. 실측상 이걸
  버리면 모델이 질문에 답하지 않고 캡션을 생성하기 시작한다. 그렇다고 separator
  가 든 chunk 를 강제로 사면 chunk budget 의 1/3 이 날아간다(34개 중 11~16개).
  그래서 **separator KV 만 따로 작은 sidecar** 로 빼서 항상 읽는다
  (이미지 KV 의 약 1.5%, 순차 read 1회). reference 에는 없는 storage-aware 결정.

## 6. 실행

```bash
conda activate mllm_ft && cd ~/hj/mllm_v2
python -c "from mmimpress.dataset import build_index; build_index(40)"
python scripts/01_build_store.py                                  # ~47 GB
python scripts/02_reorder.py --order importance --calib-questions 4
python scripts/06_build_static.py                                 # static sidecar
python scripts/05_profile_sparsevlm.py --questions 3 --limit 4    # root cause
python scripts/03_diagnose.py --questions 8 --limit 4             # token/chunk mismatch
python scripts/04_eval.py --questions 6 --skip 4 --ratio 0.25 --budget 0.25
python scripts/04_eval.py --questions 6 --skip 4 --ratio 0.50 --budget 0.50 --no-recompute
```

## 7. 구조

```
mmimpress/
  sparsevlm.py   [baseline, 미변경] rater 선택 + per-head visual 점수
  cvpr25.py      [신규] VisionZip saliency / anyres 매핑 / chunk 집계 /
                 byte budget 선택 / PACT query correction / DivPrune MaxMin
  store.py       token-major chunk 저장 + probe sidecar, pread 리더, I/O 카운터
  reorder.py     IMPRESS 4.4.1 KV reordering (importance / Morton)
  serve.py       LayerSelector(SparseVLM, hook 사용) +
                 CVPR25ChunkSelector(hook 없음, forward 이전에 선택·read 완료)
  model.py       LLaVA-NeXT 로더, anyres 그리드/separator 위치
  dataset.py     GQA 인덱스 + accuracy
scripts/
  01_build_store  02_reorder  03_diagnose  04_eval  05_profile_sparsevlm  06_build_static
```

## 8. 이전 구현에서 측정으로 되돌린 결정 (기록)

- **head-major 저장 → token-major 로 복귀.** head 를 통째로 스킵할 수 있는
  배치를 먼저 만들었으나, IMPRESS consensus 는 *하나의 토큰 집합을 모든 head 에*
  적용하므로 실제 read 단위는 "이 토큰들, 전 head" 다. head-major 는 요청당
  pread 를 64 → 3116 으로 늘려 syscall 이 byte 절감을 다 먹었다.
- **`DynamicCache.update()` 는 텐서를 복사한다.** 캐시를 미리 채우고 원본을
  나중에 수정하는 방식은 조용히 무시되어 모델이 0 을 attend 한다.
  `PrefixCache.new_request()` 는 캐시가 소유한 텐서에 바인딩한다.
- **재정렬된 저장소는 복원하지 않는다.** attention 은 key 집합에 순열 불변이고
  RoPE 는 저장 시점에 반영돼 있어, 서빙은 전부 stored position 공간에서 동작한다.

## 9. 결과 (GQA 40 images / 240 held-out questions, cold page cache)

보정에 쓴 앞 4문항은 평가에서 제외(`--skip 4`). 모든 방법이 동일 subset, 동일
eager backend, 동일 budget. 원시 데이터: `results/eval_b25.json`,
`results/eval_b50.json`, `results/abl_*.json`, `results/final_div_*.json`.

### 9.1 budget 25%

| method | acc | TTFT mean | p50 | p95 | selector | disk MB | disk ms | touched chunk | logical KV | fallback |
|---|---|---|---|---|---|---|---|---|---|---|
| ReComp | 62.5% | 537.1 | 543.4 | 624.4 | — | — | — | — | 1.000 | — |
| FullLoad | 62.1% | 721.7 | 726.1 | 833.6 | 629.3 | 1165.1 | 429.7 | 1.000 | 1.000 | 0.00 |
| SparseVLM (baseline) | **60.0%** | 786.7 | 795.3 | 903.4 | **675.4** | 928.1 | 406.2 | **0.743** | 0.246 | 0.16 |
| visionzip_static_chunk | 46.7% | **337.8** | 326.1 | 470.6 | **4.0** | **306.1** | **153.8** | **0.243** | 0.262 | 0.00 |
| cvpr25_hybrid_chunk | 44.2% | 390.2 | 373.2 | 565.2 | 6.7 | 306.1 | 196.2 | 0.243 | 0.263 | 0.00 |
| cvpr25_hybrid_diverse | **58.3%** | 392.9 | 388.4 | **458.3** | 17.3 | 300.4 | 214.4 | 0.243 | 0.244 | 0.00 |

FullLoad 의 `selector` 열은 hook 시간(전량 read 를 포함)이라 selector 비용이
아니다. SparseVLM 만 hook 을 쓰므로 그 값이 곧 identification 경로 비용이다.

| vs FullLoad | bytes | I/O time | TTFT | vs ReComp TTFT |
|---|---|---|---|---|
| SparseVLM | 0.797x | 1.06x | 0.92x | 0.68x |
| visionzip_static_chunk | **0.263x** | **2.79x** | **2.14x** | **1.59x** |
| cvpr25_hybrid_chunk | 0.263x | 2.19x | 1.85x | 1.38x |
| cvpr25_hybrid_diverse | 0.258x | 2.00x | 1.84x | 1.37x |

### 9.2 budget 50%

| method | acc | TTFT mean | p50 | p95 | selector | disk MB | disk ms | touched chunk | logical KV |
|---|---|---|---|---|---|---|---|---|---|
| FullLoad | 62.1% | 726.0 | 717.2 | 873.4 | 633.4 | 1165.1 | 427.8 | 1.000 | 1.000 |
| SparseVLM | 62.9% | 849.3 | 849.2 | 976.9 | 750.6 | 1121.4 | 454.1 | 0.909 | 0.491 |
| visionzip_static_chunk | 55.4% | **518.5** | 514.4 | 682.8 | **4.8** | **603.8** | **265.5** | 0.496 | 0.517 |
| cvpr25_hybrid_chunk | 51.2% | 601.2 | 592.0 | 774.4 | 7.7 | 603.9 | 325.4 | 0.496 | 0.519 |
| cvpr25_hybrid_diverse | **61.3%** | 639.3 | 642.9 | 717.5 | 24.7 | 594.4 | 360.3 | 0.496 | 0.493 |

vs FullLoad: static `bytes 0.518x / TTFT 1.40x`, diverse `bytes 0.510x / TTFT 1.14x`,
SparseVLM `bytes 0.962x / TTFT 0.85x`.

### 9.3 static metadata 비용 (first-use vs amortised)

`results/static_build.json`, 40 images:

| 항목 | 값 |
|---|---|
| CLIP CLS saliency (penultimate layer) | 47.6 ms / image |
| chunk score + PACT key descriptor + separator KV sidecar | 543.8 ms / image |
| **first-use total** | **591.4 ms / image** |
| 이미지당 질문 수 (GQA) | 29.4 |
| **amortised** | **20.1 ms / question** |
| sidecar 크기 | 29.3 MB / image (KV 의 약 2.6%) |

online selector 4.0 ms + amortised static 20.1 ms = **질문당 24.1 ms** 로,
SparseVLM 의 675.4 ms 대비 **28배** 낮다. 두 번째 질문부터는 4.0 ms 다.

### 9.4 ablation (20 images / 120 questions, 별도 subset)

| config | acc | TTFT | selector |
|---|---|---|---|
| diverse, `diverse_frac=0.25` (기본) | — | — | — |
| diverse, `diverse_frac=0.50` | 58.3% | 413.3 | 21.0 ms |
| diverse, `diverse_frac=1.00` | 60.0% | 428.2 | 25.0 ms |
| hybrid, `lam_query=0.3` | 49.2% | 360.6 | 6.7 ms |
| (같은 subset FullLoad) | 64.2% | 736.3–762.0 | — |

`diverse_frac=1.0` 은 subset 에서 앞섰으나 전체 240문항 재검증에서 budget 별로
갈렸다 (`results/final_div_b{25,50}.json`):

| budget | `diverse_frac` | acc | TTFT | selector | touched chunk |
|---|---|---|---|---|---|
| 25% | 0.25 | **58.3%** | 392.9 | 17.3 ms | 0.243 |
| 25% | 1.00 | 57.5% | 416.1 | 24.2 ms | 0.243 |
| 50% | 0.25 | 61.3% | 639.3 | 24.7 ms | 0.496 |
| 50% | 1.00 | **62.5%** | 680.3 | 43.8 ms | 0.496 |

25% 에서는 0.25 가, 50% 에서는 1.00 이 낫다. 예산이 넉넉하면 순수 diversity 가
이기고, 빠듯하면 saliency 로 seed 를 잡아주는 편이 낫다는 뜻이다. 기본값은
25% 기준인 0.25 로 두었다.

## 10. 판정 (요구된 A~E 기준)

| 기준 | 결과 |
|---|---|
| **A.** online selector overhead 를 675 ms 보다 압도적으로 감소 | **통과** — 4.0 ms (static) / 17.3 ms (diverse). first-use 상환 포함해도 24.1 ms |
| **B.** 25% budget 에서 touched chunk fraction 이 0.49 보다 감소 | **통과** — 0.743(측정된 SparseVLM 실값) → **0.243**, budget 과 정의상 일치 |
| **C.** SSD bytes 가 FullLoad 대비 유의미 감소 | **통과** — 0.797x → **0.255~0.263x** |
| **D.** FullLoad 대비 TTFT 개선 | **통과** — 0.92x → **2.14x** (static), 1.84x (diverse). ReComp 도 1.59x 로 상회 |
| **E.** accuracy 가 SparseVLM 이상 | **budget 25% 미달 / 50% 사실상 동률** — 25%: diverse 58.3% vs SparseVLM 60.0% (**-1.7pp**). 50%: diverse(`diverse_frac=1.0`) **62.5%** vs SparseVLM 62.9% (**-0.4pp**), FullLoad 62.1% 은 상회 |

### E 미달에 대한 수치적 원인

1. **static saliency 만으로는 부족하다.** CLS→patch attention 은 "눈에 띄는"
   영역에 몰려 chunk 선택이 공간적으로 뭉친다. 같은 budget·같은 chunk fraction
   에서 MaxMin diversity 를 넣으면 46.7% → **58.3%** (+11.6pp) 로 회복된다.
   chunk 단위 선택에서는 saliency 보다 **coverage** 가 지배적이다.
2. **PACT 형태의 query correction 이 현재 형태로는 해롭다.** 44.2%
   (`lam_query=1.0`), 49.2% (`0.3`) 로 static 단독(46.7%)과 비슷하거나 낮다.
   원인 후보는 layer-0 `k_proj`/`q_proj` 로 만든 descriptor 가 실제 layer 별 K 와
   다르고, RoPE 를 양쪽 모두 생략한 근사라는 점이다. 현재 구현으로는 정당화되지
   않으므로 기본 파이프라인에서 빼는 것이 옳다.
3. **budget 을 올리면 격차가 사라진다**: 50% + `diverse_frac=1.0` 에서
   **62.5%** 로 SparseVLM 62.9% 와 -0.4pp 차이이고 FullLoad 62.1% 는 넘는다.
   즉 남은 손실은 방법이 아니라 25% 라는 예산 자체에서 온다.

### 다음 병목 (수치)

budget 25%, `cvpr25_hybrid_diverse` 의 TTFT 392.9 ms 내역:
selector 17.3 ms + disk 214.4 ms + 나머지(cache write / prefill / decode).
**disk 214 ms 가 여전히 55%** 다. bytes 는 이미 budget 과 같으므로 더 줄이려면
budget 자체를 낮춰야 하고, 그러면 accuracy 가 더 떨어진다.
따라서 다음 레버는 selector 가 아니라
(1) chunk read 의 비동기화/prefetch (현재 완전 동기),
(2) separator sidecar 처럼 항상 필요한 조각의 상시 상주,
(3) accuracy 를 지키면서 budget 을 낮출 수 있는 더 나은 chunk scoring 이다.

## 11. 성능 조건 준수 확인

- 새 selector 는 `output_attentions=True` 를 **importance 목적으로 사용하지 않는다**.
  CLIP 쪽 `output_attentions` 는 오프라인 static 생성 1회에만 쓰이고 LLM 과 무관하다.
- 새 selector 에는 **LLM attention forward hook 이 없다**. 선택·read·cache 채우기가
  전부 forward *이전* 에 끝난다 (`CVPR25ChunkSelector.prepare`). SparseVLM baseline
  만 기존대로 hook 을 쓴다.
- attention backend 는 baseline 과 동일한 eager 로 고정했다. 새 selector 는 eager 를
  요구하지 않으므로 SDPA 로도 동작 가능하지만, backend 를 방법마다 바꾸면 비교가
  깨지므로 그렇게 하지 않았다.
- 분리 측정: `select_ms` / `query_ms` / `chunk_io_ms` / `scatter_ms` / `model_ms` /
  `prepare_ms` / `ttft` 를 각각 `torch.cuda.synchronize()` 로 감싸 기록한다
  (`mmimpress/serve.py:CVPR25ChunkSelector.stats`, `Server.request_cvpr25`).
- FLOPs: 이 구현은 key 를 **마스킹** 할 뿐 짧게 만들지 않으므로 실제 attention
  연산량은 줄지 않는다. `scripts/04_eval.py:logical_attn_gflops` 는 "마스크된 key 를
  실제로 건너뛸 수 있는 엔진이라면" 의 값을 계산하며, 측정값과 구분해 표기한다.

---

# Ablation: What Actually Makes Chunk Selection Work?

`cvpr25_hybrid_diverse` 는 네 요소가 섞여 있었다 — VisionZip static importance,
PACT query correction, DivPrune diversity, SSD chunk-level 직접 선택. 어느 것이
실제로 정확도를 만들었는지 분리해서 측정했다.

조건은 전부 동일하다: GQA 40 images / **240 held-out questions** (`--skip 4`),
동일 LLaVA-NeXT 체크포인트, eager backend, `max_new_tokens=16`, greedy,
chunk 64 tokens, separator sidecar, cold page cache. **selector 만 다르고 SSD
read / cache 재구성 / 마스킹 경로는 모든 방법이 같은 코드를 탄다**
(`mmimpress/cvpr25.py:choose_chunks` 만 갈라지고, 그 뒤는
`CVPR25ChunkSelector.prepare` 공통).

Budget 은 chunk 개수 기준 `round(total_chunks * budget)` 으로 모든 방법·모든
budget 에 동일 적용 (`cvpr25.py:budget_chunk_count`) — 34 chunk 기준
25% -> 8, 37.5% -> 13, 50% -> 17. 표의 `touched chunk` 가 모든 selector 에서
같은 값인 것이 그 확인이다.

**Sanity check**: `cvpr25_hybrid_diverse` 는 이전 실행과 동일하게 **58.3%** 로
재현되었다 (`results/eval_b25.json` vs `results/ablation_25/all_25.json`).

## Table A — 25% budget ablation (n=240)

| method | Accuracy | ΔAcc vs FullLoad | TTFT (ms) | Speedup | Selector (ms) | SSD MB | SSD ratio | Touched chunk |
|---|---|---|---|---|---|---|---|---|
| FullLoad | 62.1% | — | 706.4 | 1.00x | — | 1165.1 | 1.000 | 1.000 |
| Random Chunk (5 seeds) | **40.5% ± 3.5** | −21.6 | 433.6 ± 11.1 | 1.63x | 7.0 | 302.9 | 0.260 | 0.243 |
| VisionZip Static | 46.7% | −15.4 | **330.2** | **2.14x** | **4.1** | 306.1 | 0.263 | 0.243 |
| Diverse Only | 58.3% | −3.8 | 405.2 | 1.74x | 18.1 | 296.6 | 0.255 | 0.243 |
| **Static + Diverse** | **60.8%** | **−1.2** | 353.8 | 2.00x | 15.2 | 300.4 | 0.258 | 0.243 |
| Hybrid (static+query) | 44.2% | −17.9 | 383.9 | 1.84x | 6.8 | 306.1 | 0.263 | 0.243 |
| Hybrid + Diverse | 58.3% | −3.8 | 385.1 | 1.83x | 17.0 | 300.4 | 0.258 | 0.243 |

Random 시드별 accuracy: 42.9 / 36.7 / 41.2 / 37.1 / 44.6 %.
참고로 같은 조건의 SparseVLM baseline 은 60.0% / TTFT 786.7 ms / touched 0.743.

## Table B — Budget sweep (n=240 each)

| Budget | Method | Accuracy | TTFT (ms) | SSD MB | Touched chunk |
|---|---|---|---|---|---|
| 25% | Random | 40.5% | 433.6 | 302.9 | 0.243 |
| 25% | VisionZip Static | 46.7% | 330.2 | 306.1 | 0.243 |
| 25% | Diverse Only | 58.3% | 405.2 | 296.6 | 0.243 |
| 25% | **Static + Diverse** | **60.8%** | 353.8 | 300.4 | 0.243 |
| 25% | Hybrid + Diverse | 58.3% | 385.1 | 300.4 | 0.243 |
| 37.5% | Random | 52.1% | 556.9 | 463.7 | 0.381 |
| 37.5% | VisionZip Static | 52.1% | 436.1 | 468.8 | 0.381 |
| 37.5% | Diverse Only | 60.4% | 563.9 | 457.3 | 0.381 |
| 37.5% | **Static + Diverse** | **60.8%** | 486.4 | 460.2 | 0.381 |
| 37.5% | Hybrid + Diverse | 60.4% | 531.5 | 460.0 | 0.381 |
| 37.5% | SparseVLM | 60.8% | 828.1 | 1045.0 | 0.843 |
| 50% | Random | 55.0% | 639.5 | 597.4 | 0.496 |
| 50% | VisionZip Static | 55.4% | 510.8 | 603.8 | 0.496 |
| 50% | Diverse Only | 62.1% | 662.6 | 591.5 | 0.496 |
| 50% | **Static + Diverse** | **62.5%** | 580.0 | 594.2 | 0.496 |
| 50% | Hybrid + Diverse | 61.3% | 623.0 | 594.4 | 0.496 |
| — | FullLoad (100%) | 62.1% | ~700 | 1165.1 | 1.000 |

## Figures

`results/figures/` — light-surface PNG. 5 개 hue 는 데이터-viz 검증기의
**all-pairs** 게이트를 통과하는 슬롯(worst CVD ΔE 13.0, worst normal-vision
ΔE 16.3)이며, 마커 모양이 2차 인코딩으로 붙어 값이 겹쳐도 구분된다. 진단용
`Hybrid` 는 6번째 hue 를 만들면 게이트가 깨지므로(CVD 3.2 / normal 12.9)
중립 잉크로 접었다.

- `fig1_ssd_vs_accuracy.png` — SSD read ratio vs accuracy
- `fig2_ttft_vs_accuracy.png` — TTFT vs accuracy
- `fig3_budget_vs_accuracy.png` — budget vs accuracy (5개 방법 선)

## 다섯 질문에 대한 답 (측정값만)

**1. Random 보다 좋은가?** — 그렇다, 큰 차이로. 25% 에서 Static+Diverse 60.8%
vs Random 40.5% ± 3.5 → **+20.3pp**, 시드 표준편차의 약 5.8배. 5개 시드 전부
(42.9~44.6%) 가 Static+Diverse 보다 16pp 이상 낮다. 선택 자체가 의미 있다.

**2. Diversity 만으로 충분한가?** — 거의, 하지만 아니다. Diverse Only 58.3% 는
Random 대비 **+17.8pp** 로 이 방법의 정확도 대부분을 설명한다. 그러나
Static+Diverse 60.8% 에 **2.5pp** 못 미친다.

**3. Static importance 가 추가로 필요한가?** — 그렇다. Diverse Only → Static+Diverse
는 25% 에서 +2.5pp, 37.5% 에서 +0.4pp, 50% 에서 +0.4pp. 게다가 static seed 가
MaxMin 을 빨리 수렴시켜 **TTFT 도 405.2 → 353.8 ms 로 줄어든다**. 단
Static 단독(46.7%)은 Diverse 단독(58.3%)보다 훨씬 나쁘다 — **주 성분은
diversity 이고 static 은 보정**이다.

**4. Query correction 이 실제 도움이 되는가?** — **아니다, 해롭다.** diversity
없이: Static 46.7% → Hybrid 44.2% (**−2.5pp**). diversity 와 함께:
Static+Diverse 60.8% → Hybrid+Diverse 58.3% (**−2.5pp**). 두 조건 모두 같은 크기로
악화되고, 50% budget 에서도 62.5% → 61.3% 로 같은 방향이다. 게다가 selector 시간을
15.2 → 17.0 ms 로 늘린다. 최종 방법에서 제거한다.

**5. 어느 budget 이 가장 좋은 trade-off 인가?** — **25%**. Static+Diverse 는
25% 와 37.5% 에서 **똑같이 60.8%** 다. 즉 37.5% 는 정확도를 전혀 사주지 못하면서
TTFT 를 353.8 → 486.4 ms (**+133 ms**), SSD 를 300 → 460 MB 로 늘린다.
50% 는 +1.7pp (62.5%) 를 사지만 TTFT 를 +226 ms 지불한다. FullLoad 와의 차이가
25% 에서 이미 **−1.2pp** 뿐이므로 25% 가 운영점이다.

## 해석 규칙 적용

- **Case 1 성립** (`Static + Diverse 60.8% > Hybrid + Diverse 58.3%`) → PACT-style
  query correction 을 최종 방법에서 제거한다.
- **Case 3 성립** (`Static + Diverse 60.8% > Diverse Only 58.3%`) → static 과
  diversity 두 요소 모두 필요하다.
- **Case 2 불성립** — Diverse Only 는 Static+Diverse 와 2.5pp 차이로 동등하지 않다.
- **Case 4 불성립** — 37.5% 는 25% 대비 정확도 이득 0.0pp 이므로 main operating
  point 가 아니다. 25% 가 그 자리다.
- **Case 5 불성립** — Random 과의 격차가 20.3pp 로 크다. 선택 방법의 의미가 약하다는
  증거는 없다.

## 결론

**`Static + Diverse should be the final method`** — budget 25%.

한 문장 정의:

> **이미지에서 한 번만 계산해 재사용하는 VisionZip CLS saliency 로 SSD chunk 를
> 채점하고, 그 상위 chunk 를 seed 로 DivPrune MaxMin 을 돌려 서로 다른 시각
> 정보를 담은 chunk 를 budget 만큼 직접 고른 뒤, 질문마다 재계산하지 않고 그
> chunk 만 SSD 에서 읽는다.**

25% budget 기준 실측: accuracy **60.8%** (FullLoad −1.2pp, SparseVLM +0.8pp),
TTFT **353.8 ms** (FullLoad 대비 **2.00x**, SparseVLM 대비 **2.22x**),
SSD read **300.4 MB / 0.258x**, touched chunk **0.243**,
online selector **15.2 ms** (SparseVLM 675.4 ms 대비 **44x** 감소),
static metadata 는 이미지당 591.4 ms 1회로 질문 29.4개에 상환 (20.1 ms/question,
`ImageContext` 생성 시 1회 로드이므로 질문별 TTFT 밖).

---

# Large-Scale and Cross-Dataset Validation

앞의 ablation 은 GQA 40 images / 240 questions 에서 `static_diverse_chunk` 를
최종 후보로 골랐다. 이 절은 알고리즘을 **하나도 바꾸지 않고** 그 결론이
(a) 더 큰 평가에서, (b) 다른 VQA 데이터셋에서, (c) 실제 SSD I/O 감소로
유지되는지만 검증한다.

## 0. 실험 구조 (요구된 형태 그대로)

이미지마다 visual KV 를 **한 번** 만들어 SSD 에 저장하고, 그 이미지의 모든
질문이 같은 저장본을 재사용한다. 질문이 오면 `image_id` 로 그 이미지의
metadata 만 불러오고, 그 이미지의 chunk 중에서만 고른다.

```
question -> image_id -> 그 이미지의 meta.json + static.pt
         -> 그 이미지의 chunk 중 Static+Diverse 로 25% / 50% 선택
         -> 선택된 chunk 의 K/V 만 pread
         -> GPU prefix cache 에 scatter -> LLaVA-NeXT prefill
```

여러 이미지의 KV 를 한 pool 로 섞지 않는다 —
`CVPR25ChunkSelector` 는 `ImageContext` 하나(= 이미지 하나)의 chunk 만 본다.
FullLoad 는 같은 `image_id` 의 모든 chunk 를 읽고, ReComp 는 SSD 를 쓰지 않고
픽셀에서 다시 계산한다.

이미지 하나의 visual KV 가 약 1.2 GB 라 데이터셋 전체를 동시에 올릴 수 없다.
`scripts/11_run_large_eval.py` 가 **이미지 샤드**(80장 = 약 95 GB)로 나눠
build -> reorder -> static -> evaluate -> 삭제 순으로 돌린다. 재사용 성질은
그대로다: 샤드 안에서 한 이미지의 KV 와 metadata 는 한 번만 만들어지고 그
이미지의 모든 질문이 쓴다. 바뀌는 것은 SSD 상주 시간뿐이다.

## 1. 먼저 확인한 것 — KV reordering 이 없으면 방법이 성립하지 않는다

TextVQA 는 이미지당 질문이 최대 2개다. 기존 파이프라인은 importance
reordering 에 **이미지당 4개의 calibration 질문**을 쓰므로 그대로는 옮길 수
없다. 그래서 store 순서를 바꿔가며 먼저 측정했다 (GQA 40 images / 240
questions, budget 25%, 다른 조건 전부 동일):

| store 순서 / calibration | Static+Diverse 정확도 |
|---|---|
| importance, 4 calib (앞 절의 검증값) | **60.8%** |
| importance, 2 calib | 57.5% |
| importance, 1 calib | 58.8% |
| morton (calibration 불필요, 공간 Z-order) | **33.3%** |
| raster (reorder 없음) | **37.1%** |

**reordering 자체가 방법의 전제**다. query-독립 순서(morton)나 원본
순서(raster)로는 60.8% -> 33~37% 로 무너진다. 반면 calibration 질문 수는
1~4 사이에서 57.5~60.8% 로, n=240 의 노이즈 폭 안에 들어간다.

따라서 세 데이터셋 공통 프로토콜로 **calib = 1** 을 쓴다 (TextVQA 가 감당할 수
있는 최대). calibration 에 쓴 질문은 평가에서 제외했고, 누수 0건을 확인했다
(`results/gqa_large/shard_000.json` 의 question_id 와 index 의 첫 질문 교집합 = 0).
**이는 앞 절의 60.8% 가 calib=4 였다는 점에서 소규모/대규모 비교의 교란 요인**
이므로, GQA 에 calib=4 arm 을 따로 돌려 분리한다 (§6).

## 2. 평가 subset

seed 1234, 이미지 단위 샘플링. 질문만 늘리면 소수 이미지에 몰리므로 이미지
다양성을 우선했다.

| dataset | split | images | questions/image | 평가 질문 | metric |
|---|---|---|---|---|---|
| GQA-large | testdev_balanced | 395 | 5 (1 calib + 4) 중 3 평가 | **1,185** | exact match |
| VQAv2 | validation (streamed) | 250 | 5 (1 calib + 4 평가) | **1,000** | VQA acc (10 annotators) |
| TextVQA | validation | 500 | 2 (1 calib + 1 평가) | **500** | VQA acc (10 annotators) |

총 **2,685 questions / 1,145 images**. 40/240 결과는 지우지 않고
`results/ablation_25/`, `results/eval_b*.json` 에 그대로 둔다.

SSD 에 쓴 양: GQA 484 GB, VQAv2 308 GB, TextVQA 632 GB (샤드로 순환).
이미지당 평균 **1.17~1.21 GB KV + 0.06 GB sidecar**, visual token 2,232~2,303 개,
layer 당 chunk 35.3~36.4 개.

## 3. Main tables

동일 checkpoint(`llava-hf/llava-v1.6-vicuna-7b-hf`, 4-bit NF4), 동일 eager
backend, 동일 prompt/tokenizer/`max_new_tokens=16`/greedy, 동일 chunk 64,
동일 SSD 경로, cold page cache. 모든 방법이 같은 질문을 푼다.

### GQA-large (n = 1,185)

| Method | Budget | Score | Δ vs FullLoad | TTFT | p50 | p95 | Speedup | SSD Read | Ratio | Selector |
|---|---|---|---|---|---|---|---|---|---|---|
| ReComp | — | 63.8% | +0.4 | 539.1 | 540.9 | 617.9 | 1.31x | — | — | — |
| FullLoad | 100% | 63.4% | 0.0 | 708.9 | 703.5 | 838.4 | 1.00x | 1170.1 MB | 1.000 | — |
| SparseVLM | 25% tok | 63.3% | −0.1 | 797.8 | 802.4 | 933.0 | 0.89x | 932.4 MB | 0.797 | 675 ms† |
| **Static+Diverse** | **25%** | **60.4%** | **−3.0** | **390.5** | 388.5 | 469.9 | **1.82x** | **301.2 MB** | **0.257** | **14.9 ms** |
| Static+Diverse | 50% | 63.0% | −0.4 | 616.5 | 616.6 | 726.4 | 1.15x | 595.6 MB | 0.509 | 21.9 ms |

### VQAv2 (n = 1,000)

| Method | Budget | Score | Δ vs FullLoad | TTFT | p50 | p95 | Speedup | SSD Read | Ratio | Selector |
|---|---|---|---|---|---|---|---|---|---|---|
| ReComp | — | 81.6% | +0.2 | 544.7 | 539.0 | 723.9 | 1.31x | — | — | — |
| FullLoad | 100% | 81.3% | 0.0 | 715.9 | 708.4 | 854.2 | 1.00x | 1176.5 MB | 1.000 | — |
| SparseVLM | 25% tok | 81.0% | −0.3 | 786.2 | 790.3 | 926.2 | 0.91x | 909.5 MB | 0.773 | 675 ms† |
| **Static+Diverse** | **25%** | **80.0%** | **−1.3** | **393.7** | 386.7 | 491.9 | **1.82x** | **304.0 MB** | **0.258** | **14.7 ms** |
| Static+Diverse | 50% | 81.5% | +0.1 | 623.0 | 617.0 | 786.2 | 1.15x | 600.0 MB | 0.510 | 22.0 ms |

### TextVQA (n = 500)

| Method | Budget | Score | Δ vs FullLoad | TTFT | p50 | p95 | Speedup | SSD Read | Ratio | Selector |
|---|---|---|---|---|---|---|---|---|---|---|
| ReComp | — | 61.9% | +0.0 | 590.5 | 568.8 | 781.7 | 1.25x | — | — | — |
| FullLoad | 100% | 61.9% | 0.0 | 736.7 | 723.5 | 921.5 | 1.00x | 1207.5 MB | 1.000 | — |
| SparseVLM | 25% tok | 61.9% | −0.1 | 829.8 | 826.1 | 998.3 | 0.89x | 913.7 MB | 0.757 | 675 ms† |
| **Static+Diverse** | **25%** | **60.1%** | **−1.8** | **432.1** | 419.5 | 547.2 | **1.70x** | **314.4 MB** | **0.260** | **15.0 ms** |
| Static+Diverse | 50% | 62.6% | +0.7 | 669.4 | 648.6 | 850.1 | 1.10x | 616.4 MB | 0.510 | 22.4 ms |

† SparseVLM 의 selector 값은 앞 절에서 측정한 hook 시간이다. 이 표에서는
per-layer hook 이 SSD read 를 포함하므로 selector 열에 분리해 싣지 않았고,
40/240 측정치(675.4 ms)를 참고로 적는다.

## 4. Accuracy 95% CI — paired bootstrap (10,000 resamples)

같은 질문을 두 방법에 대해 함께 재표집한 paired bootstrap. McNemar 는
score >= 0.5 로 이진화한 불일치 수(정확 이항검정).

| dataset | method | score (95% CI) | Δ vs FullLoad (95% CI) | ours-only / FL-only | p |
|---|---|---|---|---|---|
| GQA | SD 25% | 60.4% [57.6, 63.2] | **−3.0pp [−4.5, −1.4]** | 24 / 59 | 0.00016 |
| GQA | SD 50% | 63.0% [60.2, 65.7] | −0.4pp [−1.3, +0.4] | 10 / 15 | 0.42 |
| GQA | SparseVLM | 63.3% [60.5, 66.1] | −0.1pp [−1.0, +0.8] | 14 / 15 | 1.0 |
| VQAv2 | SD 25% | 80.0% [77.7, 82.3] | **−1.3pp [−2.6, −0.1]** | 22 / 31 | 0.27 |
| VQAv2 | SD 50% | 81.5% [79.3, 83.7] | +0.1pp [−0.6, +0.9] | 12 / 10 | 0.83 |
| VQAv2 | SparseVLM | 81.0% [78.8, 83.2] | −0.3pp [−1.2, +0.5] | 12 / 14 | 0.85 |
| TextVQA | SD 25% | 60.1% [55.9, 64.3] | −1.8pp [−3.9, **+0.3**] | 12 / 21 | 0.16 |
| TextVQA | SD 50% | 62.6% [58.4, 66.7] | +0.7pp [−0.6, +2.0] | 7 / 4 | 0.55 |
| TextVQA | SparseVLM | 61.9% [57.7, 66.0] | −0.1pp [−1.8, +1.7] | 10 / 12 | 0.83 |

해석은 CI 로만 한다.

- **25% 는 세 데이터셋 모두에서 FullLoad 보다 낮다.** GQA 에서만 CI 가 0 을
  넘지 않아 명확히 유의하고(−3.0pp), VQAv2 는 경계(−1.3pp, 상한 −0.1),
  TextVQA 는 CI 가 0 을 포함해 유의하지 않다(−1.8pp, [−3.9, +0.3]).
- **50% 는 세 데이터셋 모두 FullLoad 와 구분되지 않는다.** VQAv2 +0.1pp,
  TextVQA +0.7pp 로 FullLoad 보다 높게 나온 값이 있지만 CI 가 0 을 포함하므로
  "better than FullLoad" 가 아니라 **comparable to FullLoad / no meaningful
  degradation observed** 로만 읽는다.
- **소규모에서 본 "SparseVLM 대비 +0.8pp" 우위는 대규모에서 사라졌다.**
  25% 는 SparseVLM 보다 GQA −2.9pp, VQAv2 −1.0pp, TextVQA −1.8pp 낮다.

## 5. SSD read 와 TTFT

| dataset | SD 25% SSD ratio | SSD read 시간 (평균 / p50 / p95) | FullLoad SSD 시간 |
|---|---|---|---|
| GQA | 0.257 | 214 / 214 / 270 ms | 421 / 412 / 505 ms |
| VQAv2 | 0.258 | 216 / 212 / 273 ms | 426 / 416 / 513 ms |
| TextVQA | 0.260 | 222 / 218 / 294 ms | 429 / 389 / 528 ms |

touched chunk fraction 은 세 데이터셋 모두 **0.242~0.244** (25%) 와
**0.496** (50%) 로 예산과 정의상 일치한다. 선택 chunk 수는 layer 당 8.6~8.9
(전체 35.3~36.4) 이다.

**SSD 재사용이 재계산을 실제로 이기는 구성은 25% 하나뿐이다:**

| vs ReComp | GQA | VQAv2 | TextVQA |
|---|---|---|---|
| FullLoad | 0.76x (느림) | 0.76x (느림) | 0.80x (느림) |
| SparseVLM | 0.68x (느림) | 0.69x (느림) | 0.71x (느림) |
| **Static+Diverse 25%** | **1.38x** | **1.38x** | **1.37x** |
| Static+Diverse 50% | 0.87x (느림) | 0.87x (느림) | 0.88x (느림) |

FullLoad 와 SparseVLM 은 **재계산보다 느리다**. 즉 이 규모의 이미지 prefix
(약 1.2 GB)에서는 naive 한 SSD 재사용이 순손해이고, 읽는 양을 1/4 로 줄인
25% 예산만이 손익분기를 넘는다.

## 6. Metadata 비용 (offline vs online 분리)

| 항목 | 값 |
|---|---|
| first-use: CLIP CLS saliency + chunk score + PACT key + separator sidecar | **591 ms / image** (1회) |
| repeated-question: online selector | **14.7~15.0 ms** (p50 15.1, p95 18.4) |
| sidecar 크기 | 약 0.06 GB / image (KV 의 약 5%) |

`results/generalization_summary/fig3_metadata_reuse.png` 가 같은 이미지의
k 번째 질문이 내는 비용을 보여준다 — 측정된 online selector 는 평평하고(15 ms),
1회성 metadata 는 질문 수로 나뉘어 줄어든다.

## 7. GQA 오류 분석 — 25% 선택이 놓치는 정보

FullLoad 와 Static+Diverse 25% 가 다르게 답한 GQA 질문 **83건**
(FullLoad-only 59 / ours-only 24) 을 질문 유형별로 분류했다
(`results/generalization_summary/errors_gqa_large.json`, 전체 케이스 수록).

| 질문 유형 | n | FullLoad | ours | Δ | FL-only | ours-only |
|---|---|---|---|---|---|---|
| color | 67 | 43.3% | **35.8%** | **−7.5pp** | 6 | 1 |
| yes/no | 530 | 78.5% | 75.1% | −3.4pp | 31 | 13 |
| object | 293 | 50.5% | 47.4% | −3.1pp | 12 | 3 |
| other | 36 | 50.0% | 47.2% | −2.8pp | 1 | 0 |
| material/attr | 64 | 67.2% | 65.6% | −1.6pp | 3 | 2 |
| **spatial** | 195 | 49.7% | **49.2%** | **−0.5pp** | 6 | 5 |

패턴이 분명하다. **공간 관계는 거의 손상되지 않고(−0.5pp), 세밀한 외형 정보가
깎인다** — color 가 최대 낙폭이고, object 오답도 "무엇 옆에 있나: chair ->
Table", "접시 위 음식: mashed potatoes -> Meat", "반팔 옷: jersey -> Shirt"
처럼 **어느 물체인지는 알지만 어떤 것인지 구별하지 못하는** 형태다.

25% chunk 선택은 장면의 배치(어디에 무엇이 있는지)를 보존하지만, 특정 패치의
색·재질·정확한 물체 정체성을 결정하는 국소 디테일을 잃는다.

TextVQA 가 오히려 GQA 보다 덜 떨어진 것(−1.8 vs −3.0pp)은 사전 예상과
반대였다. 작은 글자가 있는 영역이 CLIP CLS saliency 에서 높은 점수를 받아
25% 안에 남는 것으로 보이지만, 이 실험만으로 단정할 수는 없다.

## 7b. Calibration 을 분리한 대조 실험 (GQA, 같은 395 images)

§1 에서 세 데이터셋 공통 프로토콜로 calib=1 을 골랐는데, 앞 절의 60.8% 는
calib=4 였다. 따라서 GQA-large 의 −3.0pp 가 **규모 때문인지 calibration 축소
때문인지** 구분되지 않는다. 같은 395 이미지에서 calib=4 로 한 번 더 돌려
분리했다 (질문 5개 중 4개를 calibration, 남은 1개를 평가 -> n=395).

| GQA, 395 images | calib=1 (n=1,185) | calib=4 (n=395) |
|---|---|---|
| FullLoad | 63.4% | 64.6% |
| SparseVLM | 63.3% (−0.1) | 63.5% (−1.0) |
| **Static+Diverse 25%** | **60.4% (−3.0 [−4.5, −1.4])** | **64.3% (−0.3 [−2.5, +2.0])** |
| Static+Diverse 50% | 63.0% (−0.4) | 63.8% (−0.8) |
| SD 25% McNemar (ours-only / FL-only) | 24 / 59, p=0.00016 | 10 / 11, p=1.0 |

**대규모 자체는 문제가 아니었다.** calibration 을 원래대로 4개 주면 같은
395 이미지에서 Static+Diverse 25% 는 FullLoad 와 구분되지 않고(−0.3pp, CI 가
0 을 포함), SparseVLM(63.5%)보다도 높다. 40/240 소규모의 −1.2pp 와 같은 크기다.
40/240 에서 측정한 calibration 민감도(calib 4 -> 1 에서 −2.0pp)와도 방향·크기가
맞는다.

주의: 두 arm 은 **같은 이미지지만 서로 다른 평가 질문**을 쓰므로 paired 비교가
아니고, calib=4 arm 은 n=395 라 CI 가 [−2.5, +2.0] 으로 넓다. 그래도 −3.0pp 가
규모의 결과가 아니라 calibration 축소의 결과라는 점은 분명하다.

VQAv2 는 이미지당 5문항이라 calib=4 를 줄 수 있지만 이번엔 통일성 때문에 1로
돌렸다 — 그 조건의 결과는 측정하지 않았다. **TextVQA 는 이미지당 2문항이라
구조적으로 calib=1 이 상한**이다.

## 8. 현재 확인된 limitation

1. **importance reordering 없이는 성립하지 않는다** (60.8% -> 33~37%). 이는
   이미지당 최소 1개의 사전 질문을 요구하므로 **cold-start 이미지(첫 질문)에는
   적용할 수 없다.** 이번 실험은 모두 calibration 질문을 소비한 뒤의 상태를
   측정한 것이다.
2. **정확도가 calibration 질문 수에 민감하다.** calib=1 에서 GQA −3.0pp
   [−4.5, −1.4] 로 유의하고, calib=4 에서 −0.3pp [−2.5, +2.0] 로 사라진다
   (§7b). 즉 "1~2pp 이내" 목표는 calib=4 에서만 충족된다. TextVQA 는 이미지당
   2문항이라 calib=1 이 구조적 상한이므로 이 데이터셋에서는 목표를 보장할 수
   없다.
3. **SparseVLM 대비 정확도 우위는 없다.** 대규모에서는 세 데이터셋 모두
   SparseVLM 보다 낮다. 우리 방법의 이점은 정확도가 아니라 **속도와 I/O** 다
   (selector 675 -> 15 ms, SSD 0.80 -> 0.26, TTFT 0.89x -> 1.82x).
4. **50% 는 정확도를 회복하지만 속도 이점의 대부분을 잃는다** (1.15x, ReComp
   보다 느림).
5. 평가 규모는 2,685 questions 로 요구된 최소치(1,000)는 넘겼지만 5,000 에는
   못 미친다. 이미지당 1.2 GB 의 KV 를 샤드로 순환시키는 비용이 상한이었다.
6. VQAv2 는 validation 스트림의 앞 8,000 행에서 표집한 250 이미지로, 공식
   전체 split 이 아니다.

## 9. 다음 병목 (기록만, 이번에 코드 수정 없음)

SD 25% 의 TTFT 390 ms 내역: selector 15 ms + **SSD read 214 ms (55%)** +
scatter 52 ms + model forward 70 ms. bytes 는 이미 예산과 같으므로 더 줄이려면
예산을 낮춰야 하고 그러면 정확도가 더 떨어진다.

**next bottleneck = SSD I/O.** 다음 단계는 async read / prefetch / CUDA stream
overlap 이지만, 이번 검증 범위 밖이라 **구현하지 않았다.**

## 10. 판정

요구된 성공 기준별 실측 결과:

| 기준 | 결과 |
|---|---|
| **A.** GQA large 에서 FullLoad 대비 accuracy 감소가 1~2pp 이내 | **조건부 충족** — calib=4 에서 −0.3pp [−2.5, +2.0] (충족), calib=1 에서 −3.0pp [−4.5, −1.4] (미충족) |
| **B.** SparseVLM 이상의 accuracy | **미충족** — calib=1 에서 GQA −2.9 / VQAv2 −1.0 / TextVQA −1.8pp. calib=4 GQA 에서만 +0.8pp |
| **C.** SSD read 가 FullLoad 대비 25~30% | **충족** — 0.257 / 0.258 / 0.260 (세 데이터셋) |
| **D.** TTFT 가 FullLoad·SparseVLM 보다 유의하게 낮음 | **충족** — FullLoad 대비 1.70~1.82x, SparseVLM 대비 2.0x 이상. ReComp 보다 빠른 유일한 구성 |
| **E.** VQAv2 / TextVQA 에서도 같은 경향 | **충족** — 세 데이터셋 모두 동일한 I/O·TTFT 이득, 25% 는 하락 / 50% 는 FullLoad 와 comparable |

### 결론

**`Static+Diverse generalizes and should remain the proposed method.`**

근거와 단서를 함께 적는다.

- **시스템 측 주장은 세 데이터셋에서 그대로 재현된다**: touched chunk
  0.242~0.244 (예산과 정의상 일치), SSD read 0.26x, online selector 15 ms,
  TTFT 1.70~1.82x. 이 값들은 데이터셋에 거의 영향을 받지 않는다.
- **정확도 주장도 프로토콜을 맞추면 규모에서 유지된다**: 같은 395 이미지에서
  calib=4 로 −0.3pp [−2.5, +2.0] 이며, 이는 40/240 소규모의 −1.2pp 와 같은
  크기다. 즉 소규모 결과가 대규모에서 깨진 것이 아니다.
- **다만 무조건적이지 않다.** (1) 이미지당 사전 질문이 1개뿐이면 25% 는
  GQA 에서 −3.0pp 로 유의하게 떨어진다. (2) 대규모에서 SparseVLM 보다 정확하지
  않다 — 이 방법의 이점은 정확도가 아니라 selector 비용(675 -> 15 ms)과
  I/O(0.80 -> 0.26)와 TTFT(0.89x -> 1.82x)다. (3) reordering 이 필수라
  cold-start 이미지에는 쓸 수 없다.
- 정확도를 우선한다면 **50% 예산**이 세 데이터셋 모두에서 FullLoad 와
  comparable 하지만(no meaningful degradation observed), 속도 이점의 대부분을
  잃고 ReComp 보다 느려진다.

