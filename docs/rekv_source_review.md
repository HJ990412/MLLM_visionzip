# ReKV source review for the SSD image baseline

## Provenance and scope

- Paper: Shangzhe Di et al., *Streaming Video Question-Answering with In-context Video KV-Cache Retrieval*, ICLR 2025, [official PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/67a9b444cbcd647572c88194619f72d5-Paper-Conference.pdf). The local `papers/rekv.md` and `papers/rekv.pdf` were checked; hashes are in [`rekv_baseline_contract.md`](rekv_baseline_contract.md).
- Official repository: [Becomebright/ReKV](https://github.com/Becomebright/ReKV), checked out at **`1fd9a3dbf5dbff7f27069ae2f4463674c495e830`** in a detached checkout for this review. No moving-branch source was used. This source review refers only to that commit.
- Scope: the paper's **internal** retrieval path and the LLaVA-OneVision implementation. External SigLIP retrieval is not the target algorithm. The local method is `rekv_chunk25`, publication label **ReKV-Chunk25 (adapted)**. The adaptation is a 64-token SSD image chunk and approximately 25% normal-chunk budget, not a reproduction of streaming video encoding or of reported paper numbers.

## Source trace

| Step | Direct source evidence | Behavior to preserve |
| --- | --- | --- |
| Initial and video encoding | [`Abstract_ReKV.encode_init_prompt()` and `encode_video()` L22–59](https://github.com/Becomebright/ReKV/blob/1fd9a3dbf5dbff7f27069ae2f4463674c495e830/model/abstract_rekv.py#L22-L59); [`ContextManager.append()` L616–694](https://github.com/Becomebright/ReKV/blob/1fd9a3dbf5dbff7f27069ae2f4463674c495e830/model/attention/kv_cache_manager.py#L616-L694) | An initial prompt is encoded separately, then video frames enter incrementally. `global_k/v` are unrotated attention-projection results, while the local Q/K are rotated for streaming attention. Old raw global KV is offloaded. |
| Representative key | [`ContextManager._append_global()` L569–611](https://github.com/Becomebright/ReKV/blob/1fd9a3dbf5dbff7f27069ae2f4463674c495e830/model/attention/kv_cache_manager.py#L569-L611) | Mean raw block K over the token dimension, after GQA head expansion if necessary, then flatten all heads to one vector. The representative is independent of future questions. |
| Question query | [`rekv_attention_forward()` L34–68](https://github.com/Becomebright/ReKV/blob/1fd9a3dbf5dbff7f27069ae2f4463674c495e830/model/attention/rekv_attention.py#L34-L68); [`_calc_block_topk()` L422–427](https://github.com/Becomebright/ReKV/blob/1fd9a3dbf5dbff7f27069ae2f4463674c495e830/model/attention/kv_cache_manager.py#L422-L427) | Each layer projects current hidden states to raw Q. The current question's Q is averaged over its token axis and all heads are flattened before similarity. No attention-head voting. |
| Similarity and selection | [`VectorTensor.get_cosine_similarity()` L172–180](https://github.com/Becomebright/ReKV/blob/1fd9a3dbf5dbff7f27069ae2f4463674c495e830/model/attention/kv_cache_manager.py#L172-L180); [`_calc_block_topk()` L430–468](https://github.com/Becomebright/ReKV/blob/1fd9a3dbf5dbff7f27069ae2f4463674c495e830/model/attention/kv_cache_manager.py#L430-L468) | The GPU representative-vector path casts Q and K to FP32, multiplies them, and does **not** normalize either vector. It groups frame blocks only if `chunk_size > 1`, selects Top-k, then sorts IDs in source order. If fewer than Top-k blocks exist, it returns all. |
| Selected full KV load | [`ContextManager.get_retrieved_kv()` L343–418](https://github.com/Becomebright/ReKV/blob/1fd9a3dbf5dbff7f27069ae2f4463674c495e830/model/attention/kv_cache_manager.py#L343-L418) | Initial KV and full selected block K/V are copied into a compact GPU buffer. The representative key alone is not the attention payload. The implementation has a CPU payload copy, optional pinned memory, GPU block cache and LRU behavior. |
| Question-only retrieval forward | [`LlavaOneVision_ReKV.question_answering()` L37–61](https://github.com/Becomebright/ReKV/blob/1fd9a3dbf5dbff7f27069ae2f4463674c495e830/model/llava_onevision_rekv.py#L37-L61); [`rekv_attention_forward()` L64–148](https://github.com/Becomebright/ReKV/blob/1fd9a3dbf5dbff7f27069ae2f4463674c495e830/model/attention/rekv_attention.py#L64-L148); [`patch.py` L97–114](https://github.com/Becomebright/ReKV/blob/1fd9a3dbf5dbff7f27069ae2f4463674c495e830/model/patch.py#L97-L114) | The retrieval input is question text. At each decoder layer, Q drives that layer's selection and the question actually attends selected KV; the resulting hidden states pass through later decoder layers. A text-only pass to obtain all Q would be a different algorithm. |
| Handoff to answer | `rekv_attention_forward()` L71, L77–92; `llava_onevision_rekv.py` L51–80 | Stage A concatenates question K/V for its attention calculation but returns **only retrieved past K/V** as `past_key_values`. Stage B tokenizes its answer prompt and prefills with this returned cache, then decodes. Stage-A question K/V is not persistent source or answer-prefix KV. |
| Position and causal attention | [`RotaryEmbeddingESM.forward()` L107–112](https://github.com/Becomebright/ReKV/blob/1fd9a3dbf5dbff7f27069ae2f4463674c495e830/model/attention/rope.py#L107-L112); `rekv_attention_forward()` L94–129; [`torch_impl.py` L39–96](https://github.com/Becomebright/ReKV/blob/1fd9a3dbf5dbff7f27069ae2f4463674c495e830/model/attention/dot_production_attention/torch_impl.py#L39-L96) | The selected raw K sequence is rotated at consecutive compact positions; Q is right-aligned to that key sequence. The local-window path builds causal/window mask. If key length exceeds `n_local`, a separate initial-key path uses a Q rotated at the distance ceiling. There is no preservation of gaps from original video positions in retrieved attention. |

### Paper/code differences

1. The [paper, Section 3, Eq. 2 and Fig. 2](https://proceedings.iclr.cc/paper_files/paper/2025/file/67a9b444cbcd647572c88194619f72d5-Paper-Conference.pdf) describes **cosine** similarity. It explicitly says internal retrieval uses the same similarity as external retrieval except temperature 1. The pinned `VectorTensor.get_cosine_similarity` performs an **unnormalized dot product**. The local main baseline therefore fixes `similarity_mode=official_code_dot`; cosine is neither silently substituted nor selected based on quality.
2. The paper speaks of frame or grouped-frame retrieval, with default group size one frame and 64 retrieved frames (Section 4.2). The official LLaVA-OV wrapper sets `block_size=196` tokens per frame, `chunk_size=1` as frame grouping, and `topk=64` (`llava_onevision_rekv.py` L109–126). A local 64-token SSD physical chunk and `round(0.25 × n_chunks)` budget are explicit adaptations. These two meanings of “chunk size” must not be conflated.
3. The paper says video KV can reside in RAM or disk; the pinned code primarily uses a CPU `MemoryUnit` plus an LRU GPU cache. The local payload-cold SSD hit excludes prior-request payload residency and reports actual SSD bytes/preads. This measures a different storage condition from an original warm GPU/CPU cache hit.
4. The paper's positional paragraph says selected retrieved tokens become consecutive, with original positions ignored for attention. The code confirms this in `RotaryEmbeddingESM.forward`, but it also has init/local attention branches. “Consecutive” does not justify removing those branches without showing the active lengths make a simpler path equivalent.
5. The repository's LLaVA-OV Qwen2 wrapper tokenizes `input_text['question']` and `input_text['prompt']` in two stages, without the BOS slicing used by its separate Flash-VStream wrapper. The local Vicuna LLaVA-NeXT tokenizer and normal answer prompt must use their own validated semantics. Do not claim exact wrapper token identity across backbones.

### Adaptation inventory

**Algorithmic behavior preserved:** pre-RoPE mean K and Q; all-head flattening;
per-layer independent ranking; question-only forward with selected full-KV
attention at each layer; sorted, compact selected cache; consecutive compact
RoPE with init/local handling; retrieval-to-answer KV handoff; exclusion of
Stage-A question K/V from persistent source and Stage-B starting cache.

**Local experiment adaptations:** video to static image; frame block to a
64-token canonical SSD physical chunk; default 64 frames to the shared
25%-of-normal-chunks budget; LLaVA-OV Qwen2 FP16 to the frozen local LLaVA-NeXT
Vicuna-7B 4-bit NF4/BF16 setup; incremental streaming encode to raw-K/V
capture during the normal Image+Q1 forward; official GPU/CPU payload-cache
hierarchy to metadata-ready, payload-cold SSD reads; LLaVA-NeXT structural
separator retention and partial-chunk handling. Results are interpretable
only for this stated SSD image-serving adaptation.

### Source integrity and attribution

The official checkout file SHA-256 values at the pinned commit were:

| File | SHA-256 |
| --- | --- |
| `model/attention/kv_cache_manager.py` | `f242f2148d3103821d868057582150450e77b506dc42e7d4a3282d0a00abbd12` |
| `model/attention/rekv_attention.py` | `e2571eb9c0af213b889fc8a38334d8895179467129a812b0bf3f256d3b1b8595` |
| `model/attention/rope.py` | `5b57c0dbe3d4266c04612eeb4e37c46d5f5f79fb0847a8a9e5eb97db629fe9ef` |
| `model/llava_onevision_rekv.py` | `c204c1f06aa98738e4d4e5af03e840798d5ec729a6f5bb437cfbe0b2216d644b` |
| `model/abstract_rekv.py` | `5e9664d0e5fe49406c2b3ffcd1f22a24fa10b29a91be6ef0b053e0466e62b87a` |
| `model/patch.py` | `f82612502114bdc7e3a381c1291fafc88c83c92882212461ebfceb25166a0067` |

The pinned repository root does not contain a `LICENSE` file. This document
quotes no substantial code and links to official source. If source is copied
into the implementation, preserve its attribution and verify applicable
license terms separately. No `prepare.sh` or model upgrade is part of this
source review.

### Implementation note before pilot

The local adapter activates the small initial/system raw K/V on GPU along with
representative metadata, outside the cache-hit TTFT. This mirrors the pinned
`ContextManager` fast path that leaves init KV in the GPU `global_buffer`
(`kv_cache_manager.py` L373–377, L503–514). It does **not** activate any
unselected image K/V payload. On a hit, selected image K/V is read from SSD,
compacted on the host with separator rows, copied into request-local pinned
staging, and submitted via nonblocking H2D. Pinned buffers live through the
first-token synchronization. The original `MemoryUnit` instead stores a
persistent CPU payload copy that may already be pinned and can retain an LRU
GPU copy (`kv_cache_manager.py` L31–118, L221–240). The local serving comparison
therefore does not reproduce that warm payload cache hierarchy. The current
path has not proven actual cross-stream SSD read/GPU compute overlap, so no
such overlap is attributed to it.

The main path's component timings must distinguish raw `pread` time from the
broader SSD read/reshape interval and host H2D staging/submission from
completed device transfer. CUDA event and final first-token synchronization
bound the enclosing Stage-A and Stage-B intervals. They are enclosing
intervals, not sums of nested components. The finite-logit whole-vocabulary
scan is reserved for actual-model smoke diagnostics so that it does not
selectively slow ReKV's main TTFT.
