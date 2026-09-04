# IMPRESS 구현을 위한 FlexGen 사전 조사 리포트

- 대상 저장소: `/home/dblab/hj/FlexGen` (commit `004ffef82b46e8dc8685c55d0cdda650bdaf1269`, 2024-10-27)
- 대상 논문: IMPRESS (FAST'25), `fast25-chen-weijian-impress.txt`
- 참고: 저장소는 `flexgen`이 아니라 `flexllmgen` 패키지명을 사용함. 핵심 코드는 단 2개 파일에 집중되어 있음:
  - `flexllmgen/pytorch_backend.py` (906줄) — 텐서/디바이스 추상화 + 실제 attention 연산
  - `flexllmgen/flex_opt.py` (1327줄) — OPT 모델 레이어 정의 + 실행 루프 + 캐시 관리

논문 §5(Implementation)는 "mha 함수를 prefix 재사용용으로 수정하고, attn_weight의 값으로 KV 중요도를 평가했으며, KV reordering용 `PrefixKVLayer` 클래스와 score 기반 정책의 `TokenCache` 클래스를 구현했다"고 명시함. 아래 조사는 이 수정 지점들을 FlexGen 코드에서 정확히 짚는 것을 목표로 함.

---

## 1. Attention (Q,K,V 계산 및 attention weight)이 일어나는 위치

### 1.1 Prefill phase: `TorchDevice.mha` — `pytorch_backend.py:298-365`

IMPRESS가 타깃으로 하는 TTFT(prefill)의 attention 전체가 이 함수 하나에 들어 있음.

| 단계 | 위치 | 내용 |
|---|---|---|
| Q,K,V projection | `pytorch_backend.py:315-317` | `q = F.linear(hidden, w_q) * scaling`, `k`, `v` |
| head 분리/reshape | `pytorch_backend.py:319-328` | `(b, s, h)` → `(b*n_head, s, head_dim)` |
| attention score | `pytorch_backend.py:331` | `attn_weights = torch.bmm(q, k)` shape `(b*n_head, s, s)` |
| causal mask 적용 | `pytorch_backend.py:334-341` | 마스크 위치에 `-1e4` |
| **softmax** | `pytorch_backend.py:342` | `attn_weights = F.softmax(attn_weights, dim=2)` |
| V 곱 | `pytorch_backend.py:344` | `value = torch.bmm(attn_weights, v)` |
| KV cache 반환 | `pytorch_backend.py:354-365` | k `(s, b*n_head, head_dim)`, v 를 permute 후 반환 |

호출부: `SelfAttention.forward`의 prefill 분기 `flex_opt.py:444-449`.

### 1.2 Decoding phase: `TorchDevice.mha_gen` — `pytorch_backend.py:367-469`

- attention weight 계산은 헬퍼 `_attention_weights`(`pytorch_backend.py:471-481`)로 분리되어 있음. softmax는 `:480`.
- dense 경로: `_attention_value` `pytorch_backend.py:483-487`
- **sparse 경로: `_sparse_attention_value` `pytorch_backend.py:489-521`** — attention weight에서 top-k를 뽑아 중요한 토큰의 **V만** 선택적으로 로드하는 코드가 이미 존재 (`:493-495`에서 `topk`, `:509-511`에서 `general_copy`로 v_cache 부분 로드). IMPRESS 논문 Challenge 1이 지적하는 "value만 선택 로드하는 방식은 K 전체를 로드해야 하므로 한계가 있다"에 해당하는 구현이며, 선택적 KV 로딩의 좋은 코드 레퍼런스임.
- GPU/CPU 분할 attention: `_mixed_device_attention` `pytorch_backend.py:523-567`

호출부: `SelfAttention.forward`의 decoding 분기 `flex_opt.py:450-457`.

---

## 2. Attention weight 텐서(softmax 이후, V 곱 이전) 접근 지점

### 2.1 Prefill (IMPRESS의 주 수정 지점)

**`pytorch_backend.py:342`(softmax)와 `:344`(V 곱) 사이**가 정확한 삽입 지점.

```python
# pytorch_backend.py:342
attn_weights = F.softmax(attn_weights, dim=2)   # (b*n_head, s, s)
# <-- 여기서 attn_weights 접근 가능. IMPRESS importance 계산 지점.
value = torch.bmm(attn_weights, v).view(b, n_head, s, head_dim)  # :344
```

- 논문 §4.1(Importance metric): H2O 방식대로 **attention weight 행렬의 열(column) 합**을 토큰 중요도로 사용 → 여기서 `attn_weights.view(b, n_head, s, s).sum(dim=2)` 형태로 per-head 토큰 중요도를 얻을 수 있음 (shape `(b, n_head, s)`).
- head 차원이 `b*n_head`로 접혀 있으므로 probe head(논문 §4.3: 각 레이어 첫 3개 head)만의 weight를 보려면 `view(b, n_head, s, s)` 후 head 인덱싱하면 됨.
- 주의: 마스킹이 `-inf`가 아닌 `-1e4`(`:340`)라 softmax 후 마스크 위치가 정확히 0이 아님. 열 합 기반 importance 계산 시 무시 가능한 수준이지만 인지 필요.

### 2.2 Decoding

`_attention_weights`가 softmax 결과를 반환하므로(`pytorch_backend.py:480-481`), 반환값을 받는 `_attention_value:485`, `_sparse_attention_value:492` 진입 직후가 접근 지점. decoding 중 중요도 갱신이 필요하면 이곳 사용.

---

## 3. Prefix KV (KV cache) 관리 구조

### 3.1 핵심 사실: **FlexGen에는 cross-request prefix 재사용이 없음**

- KV cache는 **요청(generate 호출) 단위로 생성되고 종료 시 삭제됨**: `OptLM.generate`에서 `init_cache`(`flex_opt.py:874-876`) → 생성 종료 후 `delete_cache`(`flex_opt.py:904-906`, 구현 `:722-726`).
- radix tree, 프롬프트 매칭, chunk 개념 모두 없음. `grep -rn "prefix"` 결과 prefix 재사용 관련 코드 전무.
- 따라서 논문 §4.1의 radix tree 기반 prefix 검색·재사용(dataflow)은 **전부 신규 구현 대상**.

### 3.2 KV cache 자료구조

- 컨테이너: `ValueHolder`(`utils.py:171-186`, store/pop/clear만 있는 단순 홀더).
- 레이어 j, GPU 배치 k별 3종 버퍼 — `OptLM.__init__` `flex_opt.py:628-630`:
  - `cache_home[j][k]`: KV의 영구 저장 위치 (GPU/CPU/disk/mixed)
  - `cache_read_buf[j][k]`: 연산 직전 로드 버퍼
  - `cache_write_buf[j][k]`: 연산 직후 새 KV 저장 버퍼
- 캐시 텐서 shape: **`(prompt_len + gen_len - 1, b * n_head, head_dim)`** — 토큰이 dim 0. K, V 각각 별도 `TorchTensor`.

### 3.3 캐시 생명주기 함수 (모두 `SelfAttention` 클래스)

| 함수 | 위치 | 역할 |
|---|---|---|
| `init_cache_one_gpu_batch` | `flex_opt.py:321-336` | policy 퍼센트에 따라 GPU(100%)/CPU/disk/mixed 디바이스 선택 후 할당 |
| `load_cache` | `flex_opt.py:338-403` | home → read_buf 로드. path 0: 직접 복사(`:359-373`), path 1: CPU workspace 경유(`:374-385`), path 2: GPU+CPU 분할(`:386-401`) |
| `store_cache` | `flex_opt.py:405-422` | write_buf → home. prefill이면 전체(`:413-415`), decoding이면 마지막 위치 1칸(`:416-419`). `general_copy` 사용 |
| `forward` | `flex_opt.py:427-459` | prefill: `mha` 호출 후 `cache_write_buf.store` (`:449`), decoding: `cache_read_buf.pop` 후 `mha_gen` (`:452-456`) |

디바이스별 할당 구현:
- GPU/CPU: `TorchDevice.init_cache_one_gpu_batch` `pytorch_backend.py:287-296`
- Disk: `TorchDisk.init_cache_one_gpu_batch` `pytorch_backend.py:667-674`
- Mixed(3-tier 분할): `TorchMixedDevice.init_cache_one_gpu_batch` `pytorch_backend.py:738-760`
- 압축: `TorchCompressedDevice.init_cache_one_gpu_batch` `compression.py:51`

### 3.4 실행 루프에서의 스케줄링

- `OptLM.load_cache`/`store_cache` 래퍼: `flex_opt.py:680-698`, `:700-720` — 각각 전용 CUDA stream(`load_cache_stream`, `store_cache_stream`, 정의 `:618-620`)에서 비동기 실행.
- 파이프라인 루프: `generation_loop_overlap_single_batch` `flex_opt.py:1011-1032` (j 레이어 계산 중 j+1 레이어 weight/cache를 미리 로드). IMPRESS 논문 Figure 10의 "I/O stream vs GPU stream" 오버랩 구조가 이미 여기 있음 — probe head 키 선로딩을 이 루프에 끼워 넣는 것이 자연스러움.

---

## 4. GPU/CPU/Disk 3-tier offloading 메커니즘

### 4.1 디바이스 추상화 (데이터 플레인 — 최대 재사용 대상)

| 클래스 | 위치 | 내용 |
|---|---|---|
| `TorchTensor` | `pytorch_backend.py:55-156` | 통합 텐서 래퍼. GPU/CPU면 `data`=torch.Tensor, **disk면 `data`=파일 경로(str)**, mixed면 (tensors, seg_points) 튜플. `copy/smart_copy/move` (`:127-152`) |
| `TorchDevice` | `pytorch_backend.py:159-618` | GPU/CPU. `allocate` `:184-191` (CPU는 기본 pinned memory) |
| `TorchDisk` | `pytorch_backend.py:621-698` | 디스크 텐서 = **numpy memmap 파일 1개** (`allocate` `:656-661`, `np.lib.format.open_memmap`). 비동기 copy thread 4개 (`:640-647`) + `copy_queue` |
| `TorchMixedDevice` | `pytorch_backend.py:704-760` | 하나의 논리 텐서를 GPU/CPU/disk 세그먼트로 분할. **분할 축은 `SEG_DIM = 1`(`:702`) = `b*n_head` 차원** — 즉 head/배치 단위 분할이지 토큰 단위 분할이 아님 |
| `TorchLink` | `pytorch_backend.py:763-788` | 디바이스 간 대역폭 모델 (비용 모델용) |

### 4.2 데이터 이동: `general_copy` — `pytorch_backend.py:791-855`

모든 디바이스 조합의 복사를 처리하는 단일 진입점. `dst[dst_indices] = src[src_indices]` 시맨틱, **비동기**.
- mixed 재귀 분해: `:799-826`
- 압축 텐서: `:827-830` (→ `compression.py`의 `general_copy_compressed`)
- **disk ↔ any: copy thread에 위임** `:831-836` → `copy_worker_func` `pytorch_backend.py:878-906` (1GB pinned CPU 버퍼를 릴레이로 사용, 전용 CUDA stream)
- disk 텐서의 부분 읽기: `map_to_torch_tensor` `pytorch_backend.py:866-875` — memmap을 열어 인덱스 슬라이싱하므로 **파일의 일부 구간만 읽는 partial read가 이미 가능**.

### 4.3 배치 정책

`Policy` dataclass `flex_opt.py:34-79`: `cache_gpu_percent / cache_cpu_percent / cache_disk_percent` 등 6개 퍼센트로 weight/cache/activation의 tier 배치를 **정적으로** 결정. CLI `--percent` `flex_opt.py:1291-1299`. 실행 환경은 `ExecutionEnv`(`utils.py:35-52`)가 gpu/cpu/disk/mixed 4개 디바이스를 묶음(`flex_opt.py:1192-1195`).

### 4.4 재사용 가능성 판단

**그대로 재사용 가능 (IMPRESS의 data plane으로 충분):**
1. `TorchTensor` + `TorchDevice`/`TorchDisk` + `general_copy` — 논문 Figure 6의 3-tier 저장/전송 계층과 정확히 대응. 특히 disk 비동기 copy thread와 pinned buffer 릴레이는 그대로 쓰면 됨.
2. disk 텐서가 "파일 1개 = 텐서 1개"이므로, **논문의 chunk(기본 64토큰의 K 또는 V)를 각각 독립된 disk 텐서로 할당**하면 chunk 단위 I/O가 자연스럽게 구현됨. memmap partial read(`map_to_torch_tensor`)로 probe head 키만 읽는 것도 가능하나, 논문 §6.5처럼 probe head 키를 별도 파일로 중복 저장(전체의 1.2% 공간)하는 편이 read amplification 없이 깔끔함.
3. prefill/decoding 실행 루프와 CUDA stream 오버랩 구조(`flex_opt.py:1011-1059`).
4. `attn_weights`가 white-box로 노출되어 있어(§2.1 지점) importance 계산 hook이 한 줄 거리임 — 논문이 FlexGen을 선택한 이유 그대로.

**신규 구현 필요 (gap):**
| IMPRESS 구성요소 (논문 §) | FlexGen 현황 | 구현 방향 |
|---|---|---|
| Prefix KV 영속화 + radix tree 검색 (§4.1) | 없음. generate 종료 시 캐시 삭제 (`flex_opt.py:722-726`) | 요청 간 유지되는 저장소 신설. `delete_cache` 경로 우회 |
| Chunk 단위 저장 (§3.2, 64-token) | 레이어당 단일 거대 텐서 | chunk별 `TorchDisk.allocate` |
| 토큰 축 tier 분할 | `TorchMixedDevice`는 `SEG_DIM=1`(head축) 분할이라 **토큰 단위 중요도 분할에 못 씀** | chunk 리스트로 대체 (mixed device 불필요해질 가능성 큼) |
| Similarity-guided ITF (§4.3): probe head 3개, Jaccard, threshold t=j^0.6 | 없음 (유일한 유사물: `_sparse_attention_value`의 V top-k) | `mha`를 2-pass로 분리: probe head K 로드→중요 토큰 식별→선택 로드→나머지 연산 |
| `mha`의 prefix 재사용 (prefill에 기존 KV 주입) | `mha`는 항상 전체 시퀀스를 처음부터 계산 | `mha` 시그니처 확장: 재사용 KV + 신규 토큰만 projection (논문 §5의 "modified the mha function") |
| KV reordering + mapping list (§4.4.1, `PrefixKVLayer`) | 없음 | 신규 클래스. 주기적 비동기 재배치, radix tree 노드 내부로 제한 |
| Score 기반 캐시 관리 (§4.4.2, `TokenCache`, min-heap 2개) | 없음. 정적 percent 배치만 존재, 동적 교체/승격 로직 전무 | 신규 클래스. score = 접근빈도 × 중요 KV 비율, GPU/CPU 캐시 상호 배타, disk에 전체 복제본 유지 |

**결론**: FlexGen은 논문 주장대로 데이터 플레인(3-tier 텐서/비동기 I/O)과 attention white-box 접근성은 훌륭하게 제공하지만, IMPRESS의 컨트롤 플레인(prefix 영속화·radix tree·chunk·ITF·reordering·score 캐시)은 전부 새로 얹어야 함. 수정의 중심은 `pytorch_backend.py`의 `mha`(:298-365)와 `flex_opt.py`의 `SelfAttention` 캐시 3함수(:321-422), 그리고 `OptLM.generate`의 캐시 생명주기(:872-908)임.

---

## 5. 참고: 기타 파일

- `flexllmgen/opt_config.py` — OPT 모델별 `n_head`/`num_hidden_layers` 등 (OPT-6.7B: 32 head/32 layer, 13B: 40/40, 30B: 48/56 — 논문 §4.2의 32/40/48 head와 대응). `get_opt_config` 및 캐시 크기 계산 `cache_bytes` 포함.
- `flexllmgen/compression.py` — 4-bit group quantization (`TorchCompressedDevice`, `compress:87`, `decompress:146`). 논문 §7이 quantization은 IMPRESS와 orthogonal·상보적이라 언급. 초기 구현에서는 비활성(`compress_cache=False`) 권장 — `mha` 수정 시 분기 단순화.
- `flexllmgen/dist_flex_opt.py` — 멀티 GPU 파이프라인 버전. IMPRESS 논문은 단일 A100 기준이므로 조사 범위 제외.
