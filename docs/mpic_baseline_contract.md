# MPIC-32 SSD baseline contract

Status: pre-implementation contract  
Baseline ID: `mpic32_ssd`  
Paper label: **MPIC-32 (SSD adaptation)**

This document separates what the MPIC paper directly specifies from the
choices required to implement an MPIC-style baseline in this repository.  The
result must be described as:

> MPIC-style selective recomputation implemented in our SSD-resident serving
> harness.

It is not a claim that the original MPIC system or its published performance
has been reproduced.

## Sources and provenance

Primary source: `papers/mpic.md`, SHA-256
`7253687b8a076fbea6e49fc8d9bffc856c3be33b1b7a372cba5fd5d00eaa503b`.
The local file contains 17 pages and 13 figures and its contents agree with the
algorithm and evaluation text exposed by arXiv revision v2 (last revised
2025-09-20).  The local markdown itself has no embedded revision metadata, so
byte-for-byte revision identity cannot be proved.

External primary sources checked on 2026-09-22:

- arXiv abstract/revision page: <https://arxiv.org/abs/2502.01960>
- author project explanation: <https://shijuzhao.github.io/pic>
- Shiju Zhao publication page: <https://shijuzhao.github.io/index_zh.html>
- Junhao Hu publication page: <https://derekhjh.com/publications/>

The paper says code and detailed results are in supplementary material, but it
does not give a repository URL.  The arXiv page, author project page, and
authors' publication entries inspected above do not expose an MPIC code link.
No author-connected public repository, commit, or software license could be
verified.  Therefore this implementation is paper-derived, not copied from or
described as the official MPIC implementation.  The arXiv manuscript license
does not establish a software license.

## Paper facts versus this adaptation

| Item | Directly stated in MPIC | This implementation | Difference / uncertainty |
|---|---|---|---|
| Goal | Reuse multimodal KV independent of exact prompt prefix while selectively recomputing enough rows to recover quality. | Reuse a single image's SSD-resident KV and selectively recompute active rows. | The pilot has one fixed-prefix image and weakly exercises position independence. |
| Backend | vLLM 0.9.0. | Transformers 4.57.6 with an isolated eager-attention selective-prefill implementation. | Different cache API, kernels, scheduler, and serving stack. |
| Models | LLaVA-1.6-vicuna-7B and LLaVA-1.6-mistral-7B. | `llava-hf/llava-v1.6-vicuna-7b-hf`. | Same model family as one paper arm, different packaging and possibly revision. |
| Precision | Not specified. | Existing validated 4-bit NF4 weights with BF16 compute; SSD KV and visual-input embeddings are FP16. | Numerics are not paper-reproducible from the description. |
| Hardware | One H800 80 GB, 20-core Xeon Platinum, 100 GB DRAM. | Existing project environment (README: RTX 4090 24 GB). | Published latency is not directly comparable. |
| Workload | Multi-image/interleaved MMDU and SparklesEval; open-answer GPT score. | Frozen GQA 40-image/240-question workload, six independent questions per image. | Single-image closed VQA accuracy is a different problem. |
| Cache creation | Multimodal files are processed when uploaded; exact source prompt and positions are unspecified. | Turn 1 is normal pixel inference. Image KV and decoder-input visual embeddings are captured from that same forward and persisted afterward. | Piggyback provisioning is a serving-harness adaptation; no extra vision or full-prefix forward is allowed. |
| Cached source context | Not specified beyond precomputed multimodal data. | Record exact expanded source prefix IDs/hash, source image logical positions, processor/AnyRes geometry, image hash, and model settings. Q1 follows the image, so causal image KV excludes Q1. | The source template is explicit here because correctness depends on it. |
| Recompute set | All text tokens and the first `k` image tokens. | All current-request text rows plus canonical original-order image rows `[0:min(k,N_image))`; default `k=32`. | No runtime saliency, query score, chunk ranking, or Ours physical order is used. |
| MPIC-k | `k` is the number of leading image tokens recomputed; evaluated variants include 4, 8, 16, 32, and 64. | `MPIC-32` means exactly 32 rows, with `k=0` and `k=N_image` exposed only for tests. | It is not a percent, chunk count, or tuned operating point. |
| Structural rows | The paper does not define AnyRes/newline handling. | `N_image` is the actual expanded LLaVA-NeXT image span. The leading rows include structural rows only if they occur in that canonical interval. Newline rows outside it remain cached and present. Record original/local/logical indices and separator indices. | Repository-specific explicit rule. |
| Selective attention | Selected tokens enter the MLLM, their new K/V replace linked-cache slots, and their Q attends other tokens. Nonselected image cache remains. | At every decoder layer, only active hidden rows undergo norm, Q/K/V, attention, residual, and MLP. Active K/V replace their original logical slots; active Q attends the full causally valid context. | The paper gives a mechanism diagram rather than executable noncontiguous-row pseudocode. |
| Dummy cache | Mutable text slots are filled with zeros because selected K/V replace them before attention, enabling one step. | Allocate dummy full-context slots, indexed-replace every active slot before attention, and mask future/padding. Sentinel tests prove dummy values never affect an allowed attention result. | Placeholder lifetime and safety are implementation details absent from the paper. |
| Image context retention | Nonselected image tokens are reused, not deleted. | All valid image rows remain in every layer's K/V context. `retained_image_context_ratio=1.0`. | No Ours25 pruning gate applies. |
| First-token path | MPIC produces the first output token in a single selective-attention step, unlike two-step full reuse. | One traversal of all decoder layers over active rows produces final-prompt hidden state and first-token logits. Later decode uses the assembled full logical cache and does not recompute image rows. | `decoder_prefill_pass_count` must be 1. A hidden FullLoad or text-prefill pass is forbidden. |
| Position/RoPE | Figure 7 explicitly omits positional embedding; pre/post-RoPE cache form and relocation are not specified. | Confirm Transformers cache stores post-RoPE K. Active Q/K receive RoPE exactly once at their target logical positions. Same-position cached K is reused unchanged. An optional shifted diagnostic phase-relocates cached K from recorded source to target positions without mutating source storage. | Phase relocation is an implementation choice, not an MPIC-paper claim, and cannot repair hidden-state changes caused by different preceding text. Shift support is therefore limited. |
| System prompt | Figures suggest common system-prefix reuse, while the selection text says all text tokens; exact policy is ambiguous. | Recompute every text row, including the short system/user prefix, to follow the explicit "all text" rule. | No system KV is treated as a free in-memory hit in the MPIC request path. |
| KV I/O | KV may reside on accelerator, CPU, or local disk. Format, dtype, chunking, and timer inclusion are unspecified. | SSD-resident raster-order token-major files; cold page-cache conditioning occurs outside TTFT. Actual returned `pread` bytes and calls are counted inside TTFT. | This deliberately stresses SSD and is not comparable to a hot original MPIC deployment. |
| Recomputed-row old KV | Not discussed. | Skip old selected-row KV only when physical chunk coverage permits it. For `k=32` and 64-row chunks, reused rows 32-63 require chunk 0, so actual KV traffic includes that whole chunk and is normally 100% of visual KV. | Logical reuse ratio and physical read ratio are reported separately. |
| Recompute inputs | Not discussed. | Persist pre-decoder-layer-0 projected/packed visual embeddings in canonical raster order. Read the first `k` rows during each hit and count bytes/time. Never use final-layer states, intermediate states, or KV as inputs. | New sidecar and persistence/read cost are explicit adaptation overhead. |
| Metadata | Not discussed. | Keep validated immutable metadata in host memory for an open image context and report its resident size. Any per-request metadata read is counted; otherwise request metadata bytes are zero by declared policy. | No hidden metadata I/O. |
| Transfer overlap | Load cache hits while computing cache-miss images, for multi-image mixed hit/miss requests. Layer-wise transfer is explicitly orthogonal. | Main pilot is single-image/all-hit, so mixed hit/miss overlap is **not applicable**. Initial correctness reference is synchronous. No overlap claim is made merely from nonblocking copies. | A serial Transformers adaptation is not presented as original MPIC's final system performance. |
| Decode cache | Not specified. | The indexed K/V tensor has exactly the target logical prompt length. Decode appends one row per generated token; selected image rows are never appended or recomputed again. | Cache-length and request-isolation tests are required. |

## Exact sequence and selection contract

For a single target prompt, let:

- `T` be the expanded logical prompt length;
- `[v_start, v_start + N_image)` be the canonical expanded image span;
- `I_k = [v_start, v_start + min(k, N_image))`;
- `X` be every target logical row outside the image span.

The active logical rows are the sorted union `A = X union I_k`.  Their
position IDs are their original target logical indices, never `0..len(A)-1`.
All active rows traverse every decoder layer.  The per-layer context has
length `T`: cached image K/V populate the image span, dummy slots initially
populate current text rows, and newly projected active K/V indexed-replace
their exact slots before attention.  The causal mask permits key `j` for query
`i` iff `j <= i` and `j` is valid.  Thus pre-image text cannot see the image,
early image rows cannot see later image/question rows, and suffix text can see
the complete prior image context.

The main GQA prompt keeps the source and target image start and preceding text
identical.  The separate shifted-prefix diagnostic may change text/image
positions, but its result is never mixed into GQA quality or latency.

## Cache and persistence contract

The MPIC Turn-1 store is run-local and no-clobber.  It contains:

1. all layers' canonical image K and V in original image-token order;
2. canonical projected/packed visual embeddings captured at decoder layer 0
   input, before any decoder layer;
3. immutable metadata: source prefix IDs/hash, source image positions/hash,
   image and processor geometry, separator indices, tensor shapes/dtypes,
   model/config identities, and provenance counters;
4. durability and byte/time accounting.

Persistence occurs after Turn-1 response timing and records materialization,
write, `fsync`, metadata, total time, and total bytes separately.  No Q1 or
generated-answer row may enter the image payload.  Store publication is atomic
and refuses overwrite.  Resume accepts only a validated existing per-image
store whose immutable hashes match the manifest.

Before every SSD hit, all MPIC payload files are passed through
`posix_fadvise(DONTNEED)` outside the request timer.  This is page-cache
conditioning, not a claim of physical NAND/controller-cache coldness.

## Timing and instrumentation contract

TTFT uses the existing schema-v2 boundary for every arm: request start before
prompt construction/tokenization/input preparation, required SSD reads and
H2D, cache assembly/position handling, prefill, first output token
materialization, and CUDA synchronization.  Page-cache conditioning and
durable result logging/fsync remain outside TTFT.  Generation continues to EOS
or the unchanged cap; request E2E is recorded separately.

Every MPIC hit records at least:

- method/image/question/request ordinal and `k`;
- image/recomputed/reused/text token counts and the four distinct ratios;
- exact selected original/local/logical rows and separator boundaries;
- source/target position hashes, context-equality flag, and position policy;
- vision-forward and decoder-prefill-pass counts;
- per-layer active queries, recomputed image/text rows, attention key length,
  and valid image key count;
- KV, embedding, separator, metadata, and total SSD bytes plus pread count;
- KV/embedding read, H2D, assembly, position, inclusive selective-prefill,
  TTFT, and E2E intervals, with overlap semantics stated;
- prediction, first/generated token IDs/count, status, and retry count.

Component intervals may overlap and are not subtracted from TTFT to invent a
"pure GPU" number.

## Correctness gates fixed before testing

Synthetic reference tests use FP32 and `rtol=1e-5`, `atol=1e-6` for attention,
hidden states, and logits unless an operation has a stricter exact invariant.
Real BF16/NF4 diagnostics use finite-value/cache-structure/token checks and a
predeclared logits comparison of `rtol=2e-2`, `atol=2e-2`; tolerances will not
be widened after observing a failure.

Required gates:

- `k=0`, same context: full image reuse plus all-text recomputation; compare
  logits and cache layout with the existing FullLoad/reference path.
- `k=N_image`: all image and text rows recomputed; compare with full prefill
  using identical embeddings, positions, and mask.
- `k=32`: exactly `min(32,N)` image rows traverse all layers; all other image
  K/V remains cached and all image rows remain addressable.
- dummy sentinel, future-token causal isolation, same-position no-double-RoPE,
  shifted phase mapping/source immutability, indexed replacement (not append),
  final cache length, and cross-request/cross-method isolation.
- actual quantized-model smoke on 3--5 images with finite logits, generation,
  exact layer counters, one decoder prefill pass, and zero extra vision pass.

CPU synthetic tests alone cannot validate the baseline.  Failure of a core
gate prevents the GQA pilot and prevents the final `VALIDATED` status.

## Evaluation scope and interpretation

After correctness and smoke pass, one balanced run evaluates exactly:

`ReComp`, `FullLoad`, `QA-Chunk25`, `Ours25`, and
`MPIC-32 (SSD adaptation)`.

It reuses the frozen 40-image/240-question GQA IDs/order and all validated
model, quantization, eager-attention, greedy-decoding, cap, and seed settings.
Q1 is normal pixels plus provisioning; Q2--Q6 are the 200 cache-hit requests
per method.  Questions remain independent and receive no invented history.

MPIC retains 100% of image context while Ours25 retains about 25%; this is not
an equal-I/O-budget contest.  Results are limited to the fixed-prefix GQA
condition and must not be generalized to the paper's multi-image interleaved
or mixed cache-hit/miss scenarios.  MT-GQA, MT-VQA-v2, ConvBench, ReKV, VPN,
and tmux work are outside this task.
