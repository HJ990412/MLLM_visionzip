# Qwen2.5-VL layer별 길이가 다른 cache(ragged cache) 설계 — 설계 A 확정

작성: 2026-11-11 (UTC), 본 측정(gate·smoke·main) 전에 확정. 이 문서의 SHA256은 `config.json`의 `design_sha256`에 기록된다.

근거 소스는 설치된 패키지를 직접 읽었다. `transformers 4.57.6`, `torch 2.5.1+cu121` (10/02 Qwen 본실험 기록과 같은 버전).
파일 SHA256은 `phase0/transformers_sources.json`에 있다.

| 파일 (site-packages/transformers/) | SHA256 앞 16자리 |
|---|---|
| `models/qwen2_5_vl/modeling_qwen2_5_vl.py` | `f5051523300145f6` |
| `masking_utils.py` | `7a963feed8173b82` |
| `integrations/sdpa_attention.py` | `dc5abe49a98dec3b` |
| `cache_utils.py` | `f4e11caf14c75bde` |

## 1. 소스에서 확인한 사실

### 1.1 4D causal mask는 forward당 한 번, layer 0의 cache 길이로 만들어진다

- `Qwen2_5_VLTextModel.forward` (`modeling_qwen2_5_vl.py` L860–876): 2D `attention_mask`가 dict가 아니면 `create_causal_mask(**mask_kwargs)`를
  **한 번** 호출해 `causal_mask_mapping["full_attention"]`에 넣는다. Qwen2.5-VL-7B는 `use_sliding_window=False`라서
  `has_sliding_layers`(L784)가 False이고 mask는 이것 하나뿐이다.
- L889–899: decoder layer loop에서 **같은 mask 객체**가 `attention_mask=causal_mask_mapping[decoder_layer.attention_type]`로 28개 layer 모두에 전달된다.
- `create_causal_mask` (`masking_utils.py` L783–786): hybrid cache가 아니면 `layer_idx = 0`.
  `_preprocess_mask_arguments` L727: `kv_length, kv_offset = past_key_values.get_mask_sizes(cache_position, layer_idx)`.
- `DynamicCache.get_mask_sizes` (`cache_utils.py` L814) → `DynamicLayer.get_mask_sizes` (L123–128):
  `kv_length = self.get_seq_length() + query_length`, `kv_offset = 0`. 즉 **past 길이는 layer 0 기준**이다.
- 설치된 torch가 2.5.1이므로 `sdpa_mask = sdpa_mask_older_torch` (L472). 반환값은 bool `[1, 1, q, kv_length]`이고 (L458–461)
  `causal_mask_function`(L74–78, `kv_idx <= q_idx`)을 `cache_position` × `arange(kv_length)`에 적용한 뒤 2D padding mask를 곱한다.
  우리 hit 경로는 `cache_position = arange(len0, len0 + T)`이므로 mask는 `[prefix len0열 전부 True | suffix T×T causal]`이다.
- 2D mask가 `kv_length`보다 짧으면 `prepare_padding_mask`(L208)가 **False로 채운다**. 따라서 2D mask 길이는 `len0 + T` 이상이어야 한다
  (hit 경로는 정확히 `len0 + T`개의 1을 넘긴다).

### 1.2 SDPA 경로에서 mask가 None이 되는 조건

- `_ignore_causal_mask_sdpa` (`masking_utils.py` L219–262, 조건 L247): tracing이 아니고, `query_length == 1` 또는 `kv_length == query_length`이며,
  padding mask가 전부 True이면 `None`을 반환한다 (호출부 L446). `allow_is_causal_skip`은 DynamicCache에서 True다 (L803).
- hit prefill은 `q = T > 1`이고 `kv_length = len0 + T ≠ T`이므로 **mask가 실제 텐서로 만들어진다**. decode step은 `q = 1`이라 **mask가 None**이다.
- `sdpa_attention_forward` (`integrations/sdpa_attention.py`): mask가 None이 아니면 `repeat_kv`로 K/V를 28 head로 늘리고 (L66–68),
  **L73에서 `attention_mask[:, :, :, : key.shape[-2]]`로 오른쪽만 자른다.** `is_causal = query.shape[2] > 1 and attention_mask is None` (L81)이라
  decode step은 `is_causal=False`, mask 없음, `enable_gqa=True`로 실행된다 (L45).

### 1.3 DynamicCache는 layer별로 다른 길이를 허용한다

- `DynamicLayer.update` (`cache_utils.py` L98–121): layer마다 독립적으로 `torch.cat([self.keys, key_states], dim=-2)` (L119). layer 사이 길이 검사는 없다.
- `DynamicCache.get_seq_length(layer_idx=0)` (L798–802): 기본값이 layer 0이다. `Qwen25Runner._decode`는 이 값을 `cache_position`으로만 쓰고,
  decode step에서는 mask가 None이므로 이 값은 attention 결과에 영향을 주지 않는다.
- `Qwen2_5_VLAttention.forward` (`modeling_qwen2_5_vl.py` L658–660): layer 자신의 `layer_idx`로 `past_key_values.update(...)`를 부른다.

### 1.4 position·rope 상태는 cache 길이에 의존하지 않는다

- hit 경로는 `position_ids`(3축 MRoPE, full-prompt logical position)를 **항상 명시적으로 넘긴다**. 그래서
  `Qwen2_5_VLModel.forward` L1272–1303의 `get_rope_index` / `rope_deltas + cache_position` 분기는 실행되지 않는다.
- `Qwen2_5_VLTextModel.forward` L881: `position_embeddings = self.rotary_emb(hidden_states, position_ids)` — 입력 `position_ids`만 쓴다.
  L831의 `past_seen_tokens`는 `cache_position is None`일 때만 쓰이며 우리는 항상 넘긴다.
- `apply_multimodal_rotary_pos_emb`(L545)는 새로 들어온 suffix의 Q/K에만 적용된다. cache에 있는 K는 T1 capture 때 이미 회전된 값(post-MRoPE)이고 다시 회전하지 않는다.
- `self.model.model.rope_deltas`는 기존 runner와 같이 요청 전후에 None으로 되돌린다.

### 1.5 hook 없이 layer별 길이만 다르게 하면 틀린다

layer l의 cache 길이 `len_l < len0`이면 L73의 오른쪽 자르기 때문에 layer l이 받는 mask는 `[True × len0 | causal]`의 앞 `len_l + T`열이 된다.
그러면 suffix 열 중 앞쪽 `len0 − len_l`개가 "prefix(전부 보임)"로 취급되어 **미래 suffix 토큰이 보인다.** (`len_l > len0`이면 mask가 key보다 짧아 SDPA가 실패한다.)
따라서 layer별 mask 처리가 반드시 필요하다. 이 경우는 QG5의 `mask_hook_missing` negative fixture로 실제 출력이 달라짐을 확인한다.

## 2. 확정한 설계: A (ragged compact + forward pre-hook)

- **cache**: layer l = 구조 행 S개 전부 + 중요도 순서표(`full_importance_permutation`)의 앞 `k_l`개 content 행, 원래 위치 순으로 정렬.
  길이 `len_l = S + k_l`. K/V는 T1 capture 값(BF16, post-MRoPE key) 그대로.
- **mask**: 각 decoder layer의 `self_attn`에 `register_forward_pre_hook(..., with_kwargs=True)`를 건다 (`mmimpress/qwen25/layerwise.py::LayerMaskHooks`).
  hook은 HF가 만든 mask `[.., q, len0 + q]`를 **오른쪽 정렬**로 `[.., q, len_l + q]`에 맞춘다.
  - 마지막 `q`열(HF가 만든 causal 부분)은 건드리지 않는다.
  - prefix 부분은 전부 True이므로 `len_l ≤ len0`이면 왼쪽 `len0 − len_l`열을 잘라낸 view, `len_l > len0`이면 True 열을 왼쪽에 붙인다.
  - `len_l == len0`이면 같은 객체를 그대로 둔다(균등 벡터 U는 모든 layer가 이 경우라서 기존 B 경로와 SDPA 입력이 같은 연산이 된다).
  - mask가 None(decode step)이면 그대로 둔다.
- **수명**: hook은 요청마다 새로 걸고 prefill과 모든 decode step 동안 유지하며 `finally`에서 반드시 제거한다.
  - 걸기 전에 남아 있는 hook이 있으면 거부한다. 요청 후 `assert_no_layer_hooks`로 0개임을 확인한다.
  - hook은 호출될 때마다 "그 layer의 현재 cache 길이 == 이 요청의 `len_l` + 지금까지 붙인 토큰 수"를 검사한다. 이전 요청의 hook이 남아 다른 cache에 적용되면 여기서 실패한다.
  - 요청이 끝나면 모든 layer가 forward 횟수만큼 호출되었는지 확인한다.
- **2D mask / cache_position**: `ones(1, len0 + T)`, `arange(len0, len0 + T)` — layer 0 기준(기존 compact 정책과 같은 의미). suffix `position_ids`는 full-prompt logical position.
- **기존 코드 수정 없음**: `runner.py`·`store.py`는 그대로이고 `_cache_hit_ids`, `_decode`, `_result`를 그대로 호출한다. hook 설치 시간은 TTFT 안에 포함된다.

설계 B(모든 layer를 `S + max k_l`로 padding하고 additive mask로 가림)는 쓰지 않는다. A가 설치된 버전에서 가능하고(1.1–1.4),
B는 padding 행만큼 GPU cache 바이트와 attention 연산이 늘며 연산 shape이 달라진다.

## 3. 규범적 의미 (gate가 확인하는 것)

hit 요청의 layer l에서 suffix·생성 토큰은 정확히 다음만 본다.
1. 구조 행 S개 전부
2. 원래 index가 `full_importance_permutation[:k_l]`에 속하는 content 행
3. 자기보다 앞선 suffix·생성 토큰(자기 자신 포함, causal)

## 4. 검증 범위

| 무엇 | gate |
|---|---|
| U_packed가 기존 B와 payload·조립 텐서·SDPA 입력·logits·생성 ID 모두 bitwise 동일 | QG2 |
| 실제 SDPA에 들어간 Q/K/V/mask가 독립 in-memory 참조(store·reader·hook을 쓰지 않고 SDPA 호출 지점에서 명시적 mask 주입)와 28 layer bitwise 동일, key 수 = S + k_l + suffix, decode step마다 +1 | QG3 |
| layer별 가시 집합을 적용한 dense 논리 참조와 compact 참조가 FP64에서 atol = rtol = 1e-12, 가려진 위치 확률 0 | QG4 |
| k_l ±1, 두 layer 행 뒤바꿈, 가려야 할 행 노출, 구조 행 누락, 이전 요청 hook 잔존, hook 누락, 잘린 파일, 다른 벡터 SHA 검출 | QG5 |
| hit당 preadv 1회·pread 0회, vision 0·score 0, 질문이 바뀌어도 read range·선택 불변 | QG6 |
| 6 arm generated-history 3턴 + 독립 감사 | QG7 |

검증 표본은 기존 고정 core pair 10개와 integration dialogue 4개(정합성 fixture)이다. MT-VQA·GLM·LLaVA에서는 이 설계를 실행·검증하지 않는다.

## 5. 이 문서를 쓰기 전에 한 일 (정직한 기록)

이 문서를 디스크에 쓰기 전에, 설계 A가 실제로 동작하는지 보려고 HOLD 첫 dialogue의 이미지 1장으로 기능 점검을 한 번 실행했다
(scratchpad, 저장소 밖; U_packed와 B의 첫 token logits bitwise 일치, hook 없이 실행하면 logits이 달라짐을 확인).
정확도·TTFT 판정에 쓰는 측정이 아니고 벡터나 설계 선택에 영향을 주지 않았다(설계 A는 프롬프트의 선호안이며 1절의 소스 분석으로 정했다).
프롬프트가 요구한 "GPU 측정 전에 문서로 남긴다"의 순서를 이 점검 1회만큼 지키지 못했음을 REPORT에도 적는다.
