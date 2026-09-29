# Qwen2.5-VL-7B image-only KV port contract

This port targets `Qwen/Qwen2.5-VL-7B-Instruct` only. The cached checkpoint
revision is `cc594898137f460bfe9f0759e9844b3ce807cfb5`; model, processor,
and tokenizer must all use that same revision. The installed reference runtime
is `mllm_ft` (Transformers 4.57.6, PyTorch 2.5.1+cu121, Accelerate 1.14.0,
bitsandbytes 0.49.2). The model uses 28 decoder layers, 28 query heads, four
native KV heads, head dimension 128; vision uses 32 blocks and 16 heads. Its
last vision block, index 31, is in `fullatt_block_indexes=[7,15,23,31]`.

The Qwen VisionZip reference is JIA-Lab-research/VisionZip commit
`8f86b55c6f000eb033e6912538af2dd7dcb30502`, file
`Qwen2_5_VL/qwen2_5vl_visionzip.py`, SHA256
`26f828971ac9d4058768e07d9cb5c329b792d13206c64e6bf66f5e4b8c026004`.
The score is the last ViT block's post-softmax key-received attention:
`mean_head(sum_query(attention_probability))`. It is averaged across each
`spatial_merge_size**2` patch group in window order, then indexed by
`argsort(window_index)` into original merged-token order. The stock Qwen patch
merger and all visual tokens are retained during the source forward. Score is
used only to sort SSD rows. Equal scores retain original token order.

Single-image pilot inputs use `min_pixels=256*28*28`,
`max_pixels=1024*28*28`, batch size one, seed 1234, greedy decoding, 16 new
tokens, checkpoint EOS, NF4 weights, BF16 compute, double quantization, and a
single resident RTX 4090. The same stock attention backend must serve all
arms; `sdpa` is the first choice. For exact vision scores under SDPA, a hook
captures last-block vision Q/K during the *one* source forward and performs an
additional exact row-block QK/softmax/column reduction. Its wall time and peak
memory belong to T1 if computed before the response returns, otherwise to
persistence. No decoder text query participates in ranking.

The reusable prefix ends immediately after `vision_end_token_id`. It includes
the fixed system and user header and all expanded image tokens. The current
question and all history are suffix tokens. Structural prefix KV consists of
every non-image-token row within that boundary and is budget exempt. The
source T1 forward may include the full question, but only prefix KV is saved.
Identity includes image content SHA256, checkpoint revision, processor pixel
bounds, prefix IDs, image grid, geometry, position policy, and native BF16
dtype. A request with a different prefix must miss or fail.

`get_rope_index` over the full logical prompt supplies 3-axis MRoPE positions.
Decoder K is stored after rotary embedding, at the cache update point. During
cache reuse, selected visual rows and all structural rows are assembled in
original token order into a compact DynamicCache. Suffix and generated tokens
retain the full prompt's logical MRoPE coordinates; `cache_position` is the
compact cache slot used to construct its causal mask. These two coordinates
are deliberately different. Each request gets a fresh cache, mask, and
`rope_deltas` state. In the installed Transformers source, decoder attention
applies `apply_multimodal_rotary_pos_emb` *before* `past_key_values.update`.

Qwen SSD files hold native BF16 raw bits in token-major `[visual,kv_head,dim]`
order, with one K and one V file per decoder layer. The existing LLaVA store
uses FP16 and is unchanged. Ours writes the complete visual KV once in a
global score order. Chunk size is 64 final LLM visual tokens. One K or V chunk
per layer is `64*4*128*2=65,536` bytes; all 28 layers' K+V for the same chunk
is 3.5 MiB. The nominal 25% chunk budget follows
`budget_chunk_count(n_chunks,0.25)`; last-chunk padding, actual kept tokens,
visual bytes, structural bytes, and metadata bytes are reported separately.
Each hit uses one contiguous `pread` per layer K and V file. Store activation
may verify full-file hashes; request reads may not scan full files.

Numerical gates are fixed before the GPU pilot. Structural indices, shapes,
identity hashes, BF16 round-trip, and inverse permutation require exact
equality. A float32 dense-vs-block score fixture uses `atol=1e-5,rtol=1e-5`.
For BF16 GPU logits from two paths with the *same* selected-token set, report
max absolute and relative errors, and compare using `atol=0.125,rtol=0.02`.
First-token and complete-output identity are separate strict observations;
numerical closeness never silently changes a mismatch into identity. Geometry,
mapping, causal position, and I/O gates must pass before benchmark claims.

TTFT runs from request start through prompt preparation, SSD/H2D/assembly or
pixel/vision prefill, and first-token CUDA synchronization. Request E2E ends
after decoding. JPEG decoding and dataset file loading are outside the request
timer; image preprocessing for ReComp is inside it. Model loading, common
warmup, output logging, and page-cache conditioning are outside. `DONTNEED`
conditions OS page cache only and is not an SSD hardware-cold guarantee.
Component timers may overlap and must not be summed as disjoint GPU time.

Protected pre-existing files are enumerated by
`scripts/77_protect_qwen25_artifacts.py`: existing non-Qwen source/docs,
`data/`, `kvstore_image_only_visionzip/`, `results/`, and the selected legacy
run metadata. The before and after manifests contain per-file SHA256 and are
compared exactly. The empty local `.git` directory has no HEAD or remote, so
the local source revision is recorded as unavailable; source file hashes serve
as the reproducibility fingerprint. New Qwen run/store/results paths remain
outside the protected set.
