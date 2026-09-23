# ReKV-Chunk25 (adapted): source and implementation contract

This contract is frozen **before** the ReKV implementation and the GQA pilot. The
reference is Di et al., *Streaming Video Question-Answering with In-context Video
KV-Cache Retrieval*, ICLR 2025 ([official paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/67a9b444cbcd647572c88194619f72d5-Paper-Conference.pdf),
Section 3 and Section 4.2), and the [official ReKV repository at commit
`1fd9a3dbf5dbff7f27069ae2f4463674c495e830`](https://github.com/Becomebright/ReKV/tree/1fd9a3dbf5dbff7f27069ae2f4463674c495e830).
The local transcriptions are `papers/rekv.md` (SHA-256
`d13e2ec5628900a6d44010a9b99926773d1a50adc25dbad1cabfe89284880c43`)
and `papers/rekv.pdf` (SHA-256
`d869e5c0602c44a2ad991e2c82ae17f63ef99dec572aa3d32330bc2a02c353fe`).
The PDF was consulted where the Markdown transcription dropped equations.
All code line references below point to this pinned commit, not a moving branch.

The method ID is `rekv_chunk25`; the publication label is **ReKV-Chunk25
(adapted)**. It tests internal KV retrieval on a single static image whose
visual KV payload resides on SSD. It is not a reproduction of the entire
StreamingVQA pipeline or of the paper's reported performance.

| Item | Paper description | Pinned official code | This implementation | Difference and reason |
| --- | --- | --- | --- | --- |
| Representative K | Section 3: mean key vector per frame; heads concatenated. | [`ContextManager._append_global()` L569–611](https://github.com/Becomebright/ReKV/blob/1fd9a3dbf5dbff7f27069ae2f4463674c495e830/model/attention/kv_cache_manager.py#L569-L611): mean of **raw** `global_remainder` K over block tokens, expand GQA heads where needed, flatten all heads. | Mean pre-RoPE K across valid spatial rows of each 64-token physical chunk, then flatten every KV head. | Video frame becomes SSD chunk. Exclude newline/separator/padding from mean and divide by the actual valid count. Current backbone has 32 Q and 32 KV heads; no QA probe-head subset. |
| Question Q | Section 3: mean query over question tokens; heads concatenated. | [`_calc_block_topk()` L422–427](https://github.com/Becomebright/ReKV/blob/1fd9a3dbf5dbff7f27069ae2f4463674c495e830/model/attention/kv_cache_manager.py#L422-L427) averages pre-RoPE projected `global_q` over token axis and flattens all heads. | Current question tokens only; mean pre-RoPE Q at each decoder layer, flatten all heads. | Use current LLaVA-NeXT tokenizer semantics, not the Qwen2 wrapper's exact special token IDs. Record token IDs/hash. |
| Similarity | Section 3 and Fig. 2 describe cosine similarity, with temperature 1 for internal retrieval. | Despite its name, [`VectorTensor.get_cosine_similarity()` L172–180](https://github.com/Becomebright/ReKV/blob/1fd9a3dbf5dbff7f27069ae2f4463674c495e830/model/attention/kv_cache_manager.py#L172-L180) casts K and Q to FP32 and performs **unnormalized dot product**. The not-yet-offloaded remainder branch at L441–444 also uses dot product, without this explicit FP32 cast. | Main pilot fixes `similarity_mode=official_code_dot`: FP32 dot of resident representative K and current Q. | Preserve the official vector-cache path; disclose paper/code mismatch. Never choose cosine after seeing results. |
| Head aggregation | Section 3 concatenates heads, without per-head ranking. | `_append_global()` L602–606 and `_calc_block_topk()` L425–427 flatten `heads × head_dim`. | One score per layer and chunk from all 32 heads concatenated. | No per-head Top-k or voting; GQA head expansion would need an explicit contract if backbone changes. |
| Layer-wise retrieval | Section 3 says each self-attention layer retrieves independently. | [`rekv_attention_forward()` L34–71](https://github.com/Becomebright/ReKV/blob/1fd9a3dbf5dbff7f27069ae2f4463674c495e830/model/attention/rekv_attention.py#L34-L71) computes this layer's Q and invokes its `ContextManager`; [`patch.py` L97–114](https://github.com/Becomebright/ReKV/blob/1fd9a3dbf5dbff7f27069ae2f4463674c495e830/model/patch.py#L97-L114) passes the resulting hidden states to the next layer. | Each layer scores and selects independently during the question-only model traversal. | No text-only precomputation of all layer Q; next-layer Q must depend on selected-KV attention in the previous layer. |
| Retrieval forward | Section 3 Eq. 3 uses retrieved KV as question attention context. | [`question_answering()` L44–52](https://github.com/Becomebright/ReKV/blob/1fd9a3dbf5dbff7f27069ae2f4463674c495e830/model/llava_onevision_rekv.py#L37-L69) runs a question-only language-model forward; `rekv_attention_forward()` L77–136 concatenates retrieved and current Q K/V, performs attention, then output projection. | Stage A traverses all decoder layers with the current question, reads selected all-head K/V from SSD, assembles compact context, performs actual attention and MLP. | Static image and SSD reads replace streaming-video CPU/GPU payload cache; no vision forward on hits. |
| Retrieved-cache handoff | Section 3: retrieved KV reused to answer. | `rekv_attention_forward()` L71 and L91–92 returns `(past_k,past_v)` for retrieval, **excluding Stage-A question K/V**; `llava_onevision_rekv.py` L51–69 passes these raw compact tuples into a separate answer prefill. | Keep Stage-A selected raw K/V in request-local GPU memory and pass the same tensors/data to Stage B; zero Stage-B visual payload rereads. | Do not persist retrieval-question K/V, duplicate source prefix, or run retrieval again during answer prefill. |
| KV compaction | Section 3 and positional-encoding paragraph treat selected KV as consecutive tokens. | [`get_retrieved_kv()` L343–418](https://github.com/Becomebright/ReKV/blob/1fd9a3dbf5dbff7f27069ae2f4463674c495e830/model/attention/kv_cache_manager.py#L343-L418) copies sorted selected blocks after initial KV into one GPU `global_buffer`; `_calc_block_topk()` L458–466 sorts IDs. | Pack initial context, selected spatial rows, and required separators in relative source order with **no holes for unselected image rows**. Track actual per-layer lengths. | Partial chunks and separators require variable lengths and masks beyond official fixed frame blocks. Existing QA full-length scatter/mask is unsuitable. |
| RoPE/position policy | Section 3: disregard original retrieved positions; treat retrieved tokens as consecutive. An all-at-one-position variant was worse. | [`RotaryEmbeddingESM.forward()` L107–112](https://github.com/Becomebright/ReKV/blob/1fd9a3dbf5dbff7f27069ae2f4463674c495e830/model/attention/rope.py#L107-L112) rotates raw compact K at consecutive indices and right-aligns Q to the compact key length. `rekv_attention_forward()` L94–129 additionally has local-window and initial-prefix branches. | Store raw K; assign compact attention positions after ordered packing, rotate once for attention. Record both `source_token_position` and `compact_attention_position`. Reapply the proper compact-position rotation from raw K on Stage B without SSD reread. | Original SSD source positions identify payload/order, not attention RoPE gaps. Preserve and test init/local branches; a simple full-context path is allowed only when shown equivalent for the active compact lengths. |
| Initial/system KV | Section 3 retains an initial context under the streaming attention policy. | [`ContextManager.init()` L302–325](https://github.com/Becomebright/ReKV/blob/1fd9a3dbf5dbff7f27069ae2f4463674c495e830/model/attention/kv_cache_manager.py#L302-L325) sets `init_k/v`; `get_retrieved_kv()` prepends it; `rekv_attention_forward()` L82–129 handles `n_init` and `n_local`. The LLaVA-OV wrapper tokenizes a Qwen-style system prefix and defaults `n_init` to its token count (L109–126). | Capture the current source prompt's necessary initial/system prefix in the same Turn-1 forward, retain it once, and freeze actual `n_init`, `n_local`, and branch behavior before the pilot. | Vicuna prompt differs from Qwen-style wrapper. A post-RoPE source prefix cannot be mixed into a raw-K compact cache. |
| Storage offloading | Section 3: old video KV may be offloaded to RAM or disk. | [`MemoryUnit` L31–118](https://github.com/Becomebright/ReKV/blob/1fd9a3dbf5dbff7f27069ae2f4463674c495e830/model/attention/kv_cache_manager.py#L31-L118) holds CPU payload, optional pinned memory, CUDA block cache, asynchronous copies; `ContextManager` uses LRU GPU blocks. It does not implement this project's SSD `pread` layout. | Canonical-order raw K/V in a distinct SSD store, 64-token physical granularity. Keep no unselected full-image payload in CPU/GPU request cache; coalesce adjacent selected ranges. | Common payload-cold SSD-serving comparison. Do not present SSD timing as official warm-cache ReKV latency. |
| Metadata residency | Paper leaves representative-vector placement implicit. | [`VectorTensor` L121–180](https://github.com/Becomebright/ReKV/blob/1fd9a3dbf5dbff7f27069ae2f4463674c495e830/model/attention/kv_cache_manager.py#L121-L180) is a dynamically growing GPU vector cache. | Active image's K representatives are GPU resident and metadata-ready before cache-hit timing. Report dtype, bytes/image, build and activation time, actual GPU/CPU memory, and calculated 100-image capacity. | Metadata residency is separate from visual payload residency; metadata activation outside TTFT is reported separately. |
| Cross-request payload cache | Paper allows GPU/RAM/disk management. | `MemoryUnit.load()` L67–101 and `ContextManager._remove_lru_blocks()` L221–240 may keep hot selected blocks on GPU; all blocks have CPU copies. | Evict prior-request image payload outside timer. Same-request Stage-A selected K/V stays resident through Stage B and decode. | Required payload-cold common condition, explicitly an adaptation of original cache hierarchy. |
| Backbone/precision | Section 4.2: LLaVA-OV 0.5B/7B, FP16 on A100; internal defaults `b=1`, `r=64` frames. | [`llava_onevision_rekv.load_model()` L109–140](https://github.com/Becomebright/ReKV/blob/1fd9a3dbf5dbff7f27069ae2f4463674c495e830/model/llava_onevision_rekv.py#L109-L140) uses LLaVA-OV Qwen2 7B FP16, Flash attention, 196 frame tokens. | Frozen local `llava-hf/llava-v1.6-vicuna-7b-hf`, 4-bit NF4, BF16 compute, FP16 KV storage, eager attention and greedy cap from the validated local pilot. | Model/backend/precision/workload differ; record exact checkpoint, processor/tokenizer revision, seed and cap in run config. |
| Retrieval block size | Section 4.2: one frame per retrieval block by default; frame has its own visual tokens. | Wrapper sets `block_size=n_frame_tokens=196`, `chunk_size=1` frame group (L110–125); `_calc_block_topk()` L452–460 groups blocks by its `chunk_size`. | One 64-token physical SSD chunk is one retrieval block; retrieval grouping parameter equals 1. | The original `chunk_size=1` is **frame grouping**, not 1-token storage. Store chunk width is 64 tokens. |
| Retrieval budget | Section 4.2: 64 frames by default, not a percentage. | `_calc_block_topk()` L430–468 selects up to `topk` blocks; if fewer are present, it returns all. Torch `topk` does not contract deterministic ties. | `k=budget_chunk_count(n_normal_chunks, 0.25)` from [`cvpr25.py` L347–353](../mmimpress/cvpr25.py): Python `round`, clamped to `[1,n]`. Exclude zero-valid chunks. Stable descending score, lower chunk ID wins ties; then sort selected IDs in source order. | Fixed ~25% normal-chunk budget matches QA/Ours. Stable tie policy is explicit local adaptation; non-tie fixture must match official Top-k. |
| Image encoding/provisioning | Section 3: independent incremental video encoding with sliding window. | [`Abstract_ReKV.encode_init_prompt/encode_video()` L22–59](https://github.com/Becomebright/ReKV/blob/1fd9a3dbf5dbff7f27069ae2f4463674c495e830/model/abstract_rekv.py#L22-L59) encodes init prompt and then video chunks; `_append_global()` offloads K. | Capture each layer's pre-RoPE visual K, V, prefix, and mappings during **the same normal pixel-based Image+Q1 forward**. Build representative metadata and SSD store afterward; no second image/prefix forward. | One static image and piggyback Turn-1 provisioning replace streaming encoding. Preserve first-answer behavior; report preparation/write/fsync costs. |
| Separator handling | Video-frame formula has no LLaVA-NeXT AnyRes newline/separator rule. | Official block code assumes uniform complete frame blocks; no separator sidecar. | Always retain required structural separator K/V outside ranking budget; remove padding; merge separator and selected spatial rows in source-relative order before compacting. | Explicit LLaVA-NeXT adaptation. Count separator bytes and actual retained tokens separately. |

## Exact execution and cache ownership

For each cache-hit question Q2–Q6, the resident K representatives contain only
Turn-1 image-derived information. Stage A tokenizes the **current question
only**, then traverses the decoder. In layer `l`, it projects current hidden
states to unrotated Q/K/V, computes `Q_rep[l]`, scores `K_rep[l,*]` in FP32,
selects the fixed normal-chunk budget, reads full-head K/V of those blocks and
required separators, and performs causal attention against the compact
context plus temporary question K/V. Residual/MLP output feeds layer `l+1`.
The representative vectors choose payload; they never stand in for selected
full K/V during attention.

The Stage-A return cache contains one layer tuple of raw initial and retrieved
visual K/V in compact order. It excludes Stage-A question K/V. Stage B performs
the normal answer-prompt prefill with that returned cache, applies the compact
RoPE/causal policy to the prompt and retrieved context, obtains the actual
first answer token, and continues greedy decoding to EOS/cap. It does not
search again, reread visual payload, or run a vision encoder. The persistent
source store is immutable. A request-local cache is discarded after the
request; the next request cannot inherit its ranking or generated text.

The pinned code has two attention regimes. In the common `len_k <= n_local`
regime, `rekv_attention_forward()` L94–129 rotates the compact raw K and Q
with `RotaryEmbeddingESM.forward()` and the init-attention branch has empty
keys. When `len_k > n_local`, the local branch truncates to the trailing
window, while the separate initial branch attends raw initial K with Q rotated
at the distance ceiling (`n_local - 1`). Any local shortcut must demonstrate
equivalence for the pilot's recorded `n_init`, `n_local`, compact lengths and
decode length. The all-block diagnostic and a small forced init/local fixture
must verify both position regimes. Original absolute image positions are kept
only for source lookup, ordering and audit mapping.

## Pilot and measurements

Use the validated GQA index's exact 40 images and six independent questions
per image. The six arms are ReComp, FullLoad, MPIC-32 (adapted), QA-Chunk25,
ReKV-Chunk25 (adapted), and Ours25. All Q1 requests are ordinary Image+Q1
pixel inference. Q2–Q6 give 200 timed cache hits per arm. The main condition
is **metadata-ready, payload-cold SSD serving**; do not describe independent
same-image questions as native dialogue. Keep existing arms, rotation,
scorer, prompt meaning, TTFT boundary, page-cache conditioning and incremental
commit/resume behavior unchanged.

Within ReKV TTFT, include request preparation, question tokenization,
retrieval forward, FP32 score/Top-k and ID transfer, SSD reads/H2D, compact
assembly/RoPE, answer prefill, first-token materialization and CUDA sync.
`retrieval_forward_wall_ms` encloses its Q scoring, selected-KV I/O,
attention and MLP; do not add nested intervals again. Track Stage-A and
Stage-B payload bytes separately, pread count/runs, selected spatial tokens,
separator bytes, compact lengths, metadata bytes, memory peaks, provisioning
costs and actual first answer token. Metadata activation and request eviction
outside the timer are separate measured costs.

## Correctness gates before any GQA main pilot

1. Representative K and Q means/head flatten, FP32 official dot scores,
   non-tie Top-k parity, deterministic ties and pre-RoPE capture relationship.
2. Per-layer selection after actual preceding-layer retrieved-KV attention;
   a text-only precomputed-Q implementation must fail the dependency test.
3. Compact source order, zero masked holes, valid separator/padding treatment,
   actual key length, causal mask and once-only RoPE under both local and
   initial-window branches.
4. Stage-A-to-B raw K/V handoff without question KV, no Stage-B visual SSD
   reread, unchanged persistent source hash, and request/method isolation.
5. All-block attention/cache diagnostic against a matching reference and
   actual quantized-model smoke with finite logits, first token, full capped
   generation, exact SSD trace and stable memory lifecycle.

Tolerance must be chosen from dtype/reference **before** observing failures.
Core-gate failure prevents the main GQA pilot and a validated verdict. A
correctness fix after a run preserves the prior artifact as superseded and
reruns affected validation and measurements.

The pinned repository root contains no `LICENSE` file at this commit. This
contract links to the original implementation but does not copy its source.
If implementation later copies substantial official code, verify permission
and preserve attribution/license information in that new file.

## Frozen numerical representation and model identity

The official `_append_global()` and `_calc_block_topk()` call `torch.mean`
without a `dtype` argument; their representative outputs have the input K/Q
dtype. The official vector-cache similarity then casts both operands to FP32
before the unnormalized matrix product. For this BF16-compute local model,
calculate K and Q representatives directly from captured BF16 pre-RoPE K and
projected BF16 Q, respectively, with `torch.mean` output in BF16. Keep K
representatives as BF16 GPU metadata and cast representatives to FP32 only for
the score. This avoids silently calculating a different mean from the FP16 SSD
payload conversion. The all-head raw K/V SSD payload is FP16. A fixture must
check the same-dtype official mean and FP32 score separately from the FP16
payload round trip.

The previous validated same-run GQA pilot fixes the model snapshot at
`c916e6cdcd760b4cecd1dd4907f84ac649f93b23`, 4-bit NF4, BF16 compute,
eager attention, greedy decoding, generation cap 16, and seed 1234. Load the
processor/tokenizer from the same frozen snapshot and record its resolved
identity in the new run config. These settings come from the local validated
pilot and `mmimpress/config.py`; they are not settings from the official
LLaVA-OV wrapper.

## Frozen attention-window and retrieval-token settings

The main adapter uses `n_local=15000`, matching the pinned repository
`README.md` evaluation command (lines 90–96) and the paper's 15K local window.
It uses `n_init=v_token_start`, the actual number of source-prefix tokens
preceding the expanded image in the current LLaVA-NeXT request, rather than
assuming the original Qwen-style system prompt length. The Stage-A current
question is tokenized with the frozen Vicuna tokenizer's default behavior,
which adds BOS ID 1. This is an explicit tokenizer/backbone adaptation of the
official LLaVA-OV wrapper call `tokenizer(question)`; the Stage-A question KV
remains temporary and is excluded from Stage-B handoff. The normal Stage-B
answer suffix uses the existing local `suffix_ids_for` contract.

Record `n_local`, `n_init`, Stage-A question token IDs and every layer's actual
compact retrieval key length. Verify all main-pilot retrieval and answer
lengths remain below 15000. For those lengths the pinned code's local branch
is a standard causal attention over consecutive compact positions and the
initial-key branch is inactive. A forced small `len_k > n_local` fixture must
still verify initial-key distance-ceiling behavior; a future workload that
crosses 15000 cannot silently use the short-context branch.

## Predeclared numerical tolerances

The actual-model all-block diagnostic compares first-token logits against the
matching full-context reference with `rtol=0.02`, `atol=0.02`, fixed before
smoke testing from the previous MPIC diagnostic precedent. The independent
same-Turn-1 pre-RoPE raw-K versus post-RoPE cache check uses
`rtol=0.015625`, `atol=0.03125` for BF16 projection/rotation; V is compared
exactly after matching dtype conversion. Pure FP32 compact-attention fixtures
use `rtol=1e-5`, `atol=1e-6`; exact shape, ordering, token IDs, cache lengths,
and read-count invariants do not use a numerical tolerance. A failed gate
requires investigation, not a wider post hoc threshold.

## Transfer and initial-context residency implementation note

For a cache-hit image, activate the small raw initial/system K/V on GPU with
the representative metadata before TTFT. This preserves the pinned code's
resident initial-key fast path while keeping the **visual image payload** on
SSD. Report initial/system bytes separately from representative metadata
bytes, and include both in peak GPU accounting; image activation cost is a
separate, timer-excluded condition cost.

Selected visual K/V is read from SSD per request into host memory, compacted
with separators in source order, copied into request-local pinned staging, and
submitted to GPU with nonblocking H2D. Keep staging alive until the
first-token synchronization proves transfer completion. These staging and H2D
operations occur inside TTFT. This is not the same as the official
`MemoryUnit`'s long-lived pinned CPU payload blocks and GPU LRU cache; the
local workload intentionally keeps unselected visual payload cold across
requests. No claim of concurrent SSD I/O and GPU execution follows merely
from `non_blocking=True` or a CUDA event. Report raw `pread` time,
read/reshape pipeline wall time, host staging/H2D submission time, and full
Stage-A enclosing wall time with their distinct meanings. Host submission
intervals are not exclusive GPU kernel time and must not be added to the
enclosing retrieval-forward wall interval.
