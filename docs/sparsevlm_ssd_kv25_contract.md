# SparseVLM-SSD-KV25: LLaVA Probe3 / AllHead contract

Contract version: `sparsevlm-ssd-kv25-v1`, defined 2026-09-30 UTC before the new correctness and performance experiments. This document is a normative contract, **not a PASS receipt**. Measured status belongs in the run's `validation.json`, `independent_audit.json`, and report. Missing required evidence is NOT RUN or UNRESOLVED, never an inferred PASS. Freeze this document and its SHA256 with code, config, source, and workload manifests before performance execution. After a failed gate, do not widen tolerances or replace samples.

Run: `runs/sparsevlm_ssd_kv25_20260930T041926Z/`. Results: `results/sparsevlm_ssd_kv25_20260930T041926Z/`. Starting environment, protected artifact manifest, and original source snapshot are `environment_before.json`, `protected_artifacts_before.jsonl`, `source_before.json`, and `source_before/` in that run. Git status at start returned exit 128, “not a git repository”; preserve this limitation and construct a separate source diff against the saved original files. The initial protection scan covers 89,829 files and 595,239,698,580 bytes, reported by the root protection receipt; preservation is rechecked at completion. No reset/clean, dependency upgrades, commits/pushes, prior result changes, prior store deletion, or other-job termination is authorized.

## 1. Scope and public identities

| Label | method_id | scoring_head_policy | Actual score heads |
|---|---|---|---|
| SparseVLM-SSD-KV25-Probe3 (adapted) | `sparsevlm_ssd_kv25_probe3` | `fixed_first_3` | exactly `[0,1,2]` |
| SparseVLM-SSD-KV25-AllHead (adapted) | `sparsevlm_ssd_kv25_allhead` | `all` | all actual Q heads |

Both are query-dependent token Top-k SSD cached-KV adaptations. AllHead calculates importance online from complete visual K; it is not “no importance”, and `probe=0` must not select it implicitly. Probe3 adds a head-subset approximation to this adaptation. A retained token means its K and V at **every** KV head remain visible to actual full-head attention. Neither method implements progressive hidden-token pruning, adaptive retention/layer schedules, recycling/merging, calibration, diversity, or importance repacking. Original SparseVLM published speed/accuracy results are not reproduced by this task.

Only LLaVA is in scope. Preserve Qwen, Ours-KV25, MPIC, ReKV, QA-Select/QA-Token/QA-Chunk defaults and paths. A new module/runner must use explicit policies, reuse validated storage primitives only where the contracts match, and avoid inheriting hidden QA defaults. QA-Chunk25 must not be relabeled as token KV25. The 4,061-dialogue MT-GQA main experiment and full MT-VQA are excluded.

## 2. Official source and local correspondence

Official repository: [Gumpest/SparseVLMs](https://github.com/Gumpest/SparseVLMs). Pinned main commit: `a9e71427220c4cacb6afb5a96d23cfb43142e0c3` (2026-07-30T11:23:52Z). Formula branch is **`USE_VERSION=1_0`, `V2_0=False`**; the official default is `1_0`, with `2_0` explicitly enabling SparseVLM+. This is source attribution, not a request to replace local model files. Exact downloaded source, commit response, README, paper HTML, and `official_source_pin.json` are in `official_source/`.

| Pinned file | SHA256 |
|---|---|
| `llava/model/language_model/modelling_sparse_llama.py` | `6285ea108124520bf47c6ba9f9987b1ff2437b6d4193d34f81aefea0566cb222` |
| `llava/model/language_model/score.py` | `e0a3741e23e6da4358b664b6347db8b84a91429e71b3fea5c6898fff718c06a0` |
| `llava/model/language_model/utils.py` | `f5f7d31dd8606d49266b8ec570de2dc4ad48fdffa13d0f200b5ee74a1b905c25` |

Paper: [arXiv:2410.04417v4](https://arxiv.org/html/2410.04417v4), revised 2025-06-03. Section 3.2 Eq. (6) defines the embedding rater statistic; Eq. (3) averages text-to-visual attention. Eq. (5) prints `>=`, whereas official executable code uses strict `>`; this contract follows that executable inequality and records the difference. The paper also describes adaptive sparsification and recycling, which are intentionally outside this fixed-budget SSD adaptation.

| Behavior | Official original/source | Existing local behavior | New explicit behavior / intended difference |
|---|---|---|---|
| Raters | model L179,198–204: visual/text input embeddings, text softmax, visual mean, strict `>` | `sparsevlm.select_raters` L43–65 adds FP32 and all-text fallback | preserve formula and FP32; report fallback count, reject empty suffix, record scope |
| Importance | `score.py` L27–32: per-head attention probabilities → head mean → visual/rater slice → rater mean | `rater_visual_scores_from_qk` L91–114 supports full causal context and `head_reduce=mean`; default per-head output serves other methods | explicit AllHead mean or fixed `[0,1,2]` mean only; FP32 probability/reduction |
| Attention denominator | model L459–475 / utils L95–116: causal full-key softmax before visual slice | QA selector concatenates system K + all probe visual K + suffix K | same denominator, all visual candidates at every layer, causal suffix only |
| Budget/ties | `score.py` L9–14,39–42: pruning layers 2,6,15, fixed tables, `torch.topk` tie behavior | `topk_budget` and `select_topk` retain legacy semantics; empty-candidate behavior not fail-closed | all layers exact `(N_content+3)//4`; original-index stable tie; empty/nonfinite candidates rejected |
| Application point | model L229–239,259–320 runs full layer then prunes/reconstructs output hidden rows | legacy QA prehook selects cached KV before normal layer | scoped attention adapter selects cached KV for current actual attention using that layer's live suffix Q |
| Progression | previously removed hidden rows do not remain a complete native candidate pool | native KV SSD adaptation can rescore all original positions | each layer can reselect any original token; no original progressive-pruning equivalence claim |
| Projection | official attention computes Q/K/V once | `QASelectLayerSelector` L527,588–600 projects Q/K in layer prehook, then normal attention projects again | Q/K/V once in instance-scoped wrapper, shared by scoring and answer attention |
| One-token prefill | original and legacy QA use sequence length shortcuts | legacy QA L571 skips `hidden.shape[1]==1` | explicit prefill/decode phase; one-token suffix still scores once per layer |
| Physical reads | no SSD contract in official algorithm | QA token reads probe plus touched whole K/V chunks; QA chunk averages token scores within chunks | Probe3 specified probe+chunk K/V; AllHead full K once plus selected V chunks |
| Structure/layout | original image hidden-token pruning | Ours uses repacked image-only layout; QA canonical raster | both new variants canonical original raster; structural rows outside real-token budget |
| Position | original version may shorten position arrays (L302/L314) | Ours-KV25 full logical-position dense cache/mask | preserve original logical positions; no RoPE reapplication to cached post-RoPE K |

The source references are [pinned model](https://github.com/Gumpest/SparseVLMs/blob/a9e71427220c4cacb6afb5a96d23cfb43142e0c3/llava/model/language_model/modelling_sparse_llama.py), [pinned score](https://github.com/Gumpest/SparseVLMs/blob/a9e71427220c4cacb6afb5a96d23cfb43142e0c3/llava/model/language_model/score.py), and [pinned utility](https://github.com/Gumpest/SparseVLMs/blob/a9e71427220c4cacb6afb5a96d23cfb43142e0c3/llava/model/language_model/utils.py).

SparseVLM+ is separate: model L241–251 applies RoPE gravity correction, L255–257 invokes priority-head selection, `score.py` L48–55 selects the top **14** heads by text-to-visual attention sum, L273–274 disables merge, and L311–314/L653–662 preserve original position IDs. None is the new fixed-first-3 policy. Official rater and head/rater reductions inherit tensor dtype. Custom SDPA/Flash softmax does not explicitly upcast; eager L475 upcasts softmax then casts back to Q dtype. The new FP32 rater, FP32 score softmax/reductions, fixed compute-dtype QK, and deterministic ties are intentional numerical differences, not bitwise reproduction of official low-precision execution.

## 3. Frozen environment, input policy, and source evidence

`contract_source_evidence.json` hashes the local source/read protocols, legacy raw files and validations, current Ours reports, workload source files, and cached processor/tokenizer metadata. It records configuration-only verification separately from still-required live tensor verification.

- Checkpoint, processor, tokenizer revision: `c916e6cdcd760b4cecd1dd4907f84ac649f93b23`, cached snapshot `/home/dblab/.cache/huggingface/hub/models--llava-hf--llava-v1.6-vicuna-7b-hf/snapshots/c916e6cdcd760b4cecd1dd4907f84ac649f93b23`. Use this local snapshot explicitly for all three; legacy `LlavaRunner.load()` does not itself pass a revision.
- Model config SHA256 `3ea1ed765b0b79834892d8060cb46162121ef01596e56ca6e168a48a8da08336`; preprocessor `ecef425c1d3ee91f5e05144c4d107389f9fefd04a424d5cbfc3fb6e0525b8c55`; processor `7cd8bff508772b8b4e6b0678550bf856a7f28abd0f8b237235ac83bfcb9b1c36`; tokenizer config `89833b4fed3f4010fb864ae4f70055e7f940cf9b52aba4167d42647219fe097a`; tokenizer JSON `8d13f38b3436fa9371fb2e6398dd2fac22dda34002161077311d6b712630d387`.
- Existing conda environment `mllm_ft`: torch 2.5.1+cu121, transformers 4.57.6, bitsandbytes 0.49.2; RTX 4090, driver 535.230.02. Full package list and starting GPU processes are in `environment_before.json`.
- NF4 weight quantization, double quantization enabled, BF16 compute, FP16 SSD KV, eager attention; no backend/precision/model/resolution tuning by results. Captured cache/native input dtype is measured in G1, not guessed from file dtype. FP16 SSD → BF16 compute conversion applies identically to both variants and independent references.
- `AutoConfig` from the fixed snapshot resolved 32 Q heads, 32 KV heads, 32 layers, hidden size 4096. G1 must also validate actual tensors and head dimension; derive shapes, do not hardcode them. Reject non-MHA/GQA input.
- LLaVA-NeXT processor-owned AnyRes: shortest edge 336, crop 336×336, patch 14, pinpoints `[[336,672],[672,336],[672,672],[1008,336],[336,1008]]`, existing RGB/rescale/normalize/pad rules. Do not resize independently.
- Seed 1234, batch size 1, eval/inference mode, greedy decoding, `max_new_tokens=16`. Cap reaches are recorded.
- GQA prompt is existing `LlavaRunner.prompt`: `USER: <image>\n{question} Answer the question using a single word or phrase. ASSISTANT:`. MT uses the existing generated-history `mt_prompt` in `scripts/89_eval_llava_kv25.py`, with method-local Q/A pairs and `Current question Q{turn}: ...`. `mmimpress.mt_gqa.mt_gqa_prompt` is a teacher-forced helper and must not provide history for this pilot.
- GQA/smoke scorer is `mmimpress.dataset.METRICS['gqa']` / `exact_score`: lowercase, punctuation/article removal; normalized equality or a nonempty gold token sequence matching the prediction's initial tokens. MT-GQA retains its established **strict normalized equality** scorer, `scripts/73_eval_mt_gqa_5arm_generated_shard.py:strict_gqa_score` (L172–180), reused by `scripts/89_eval_llava_kv25.py` L592. Its normalization is the same but it does **not** accept extra prediction tokens after a gold prefix. Score each dataset with its own established rule; never change it based on predictions. This pre-performance clarification corrects the initial document's incomplete scorer description; no samples, formulas, or tolerances change.

Prior evidence read: `docs/llava_kv25_budget_contract.md`; `results/llava_kv25_migration_20260929T032642Z/{REPORT.md,validation.json}`; latest spatial validation/result `results/llava_spatial_kv25_20260929T091132Z/{REPORT.md,strict_external_pilot_audit.json}`. The original Ours-KV25 is dominant/image-only `D25+C0`, distinct from legacy chunk25 and later contextual/spatial ablations. Prior KV25 matched-path logits gate fixed `atol=rtol=1e-4`; prior reports state validated 960-request migration GQA and 48-request MT smoke, and latest spatial audit states 1,920 requests. These are historical evidence, not new baseline results. Previously cleaned run-owned stores must not be assumed available.

Legacy contracts read: `results/query_aware_baseline/gqa40_240_final/{README.md,validation.json,raw.jsonl.gz}` and `results/query_aware_chunk_baseline/gqa40_240_20260919T155349Z/{README.md,validation.json,results_final.jsonl.gz}`. Both use normal T1, independent GQA hits, canonical QA layout, metadata-ready v_hidden, online probe read, outer TTFT, actual bytes/preads, and DONTNEED outside the timer. QA-Token is head-subset token ranking; QA-Chunk is `mean_valid_spatial_token_importance` followed by chunk-budget selection. Prior “Ours25” in these QA results is the legacy chunk method, not current Ours-KV25. Their old prehook projection cost/tie behavior prevents relabeling their raw rows as this new implementation. Old cross-run prediction/latency consistency allowances are not the new matched-path correctness tolerance.

## 4. T1 capture, raters, and per-layer selection

Every arm executes its own real full-image T1, with vision count 1; neither sparse variant prunes T1. Capture canonical visual K/V and layer-0 input visual embeddings from that same normal forward, with no extra vision or full-prefix LLM forward. Confirm that all prefix IDs/header/image-boundary positions before suffix are independent of question/history. Record source T1 identity, image hash, prefix IDs/hash, model/processor/precision, actual geometry and logical positions. Two new variants receive canonical KV with identical provenance. Hit vision count is 0.

The rater visual range is the complete expanded LLM image block, **including image_newline structural rows**, matching the existing QA rater range. Ranking content excludes image newline/separator/other structural rows and padding. Let `N_content` be the count of valid real rows; reject `N_content <= 0`. A suffix must contain at least one valid text position.

Each request computes raters once in FP32:

```
u = mean_visual(softmax_text(E_visual @ E_suffix.T))
rater_ids = suffix positions where u > mean(u)
```

If strict inequality yields zero raters, select all valid suffix positions and increment an explicit fallback counter. Candidates are the entire actual suffix after the image prefix: fixed template/instruction, current question, and, in MT, only this method/dialogue's actual prior generated Q/A text. Record suffix IDs, rater IDs and absolute/relative spans of current question/history/template. Never access future questions or gold as input. Same image and actual prompt must give identical rater IDs across head policies.

At each layer during suffix prefill, use Q from its current real hidden state and the matching layer's cached K. Previous layers' chosen KV must already have affected this hidden state. Do not precompute every layer Q with a teacher/text-only pass. For each score head `h` and selected rater row `i`:

```
K_context = [system/prefix K, complete original visual K, current suffix K]
A[h,i,:] = softmax_fp32(Q[h,i] @ K_context[h].T / sqrt(head_dim) + causal_mask)
s[j] = mean_heads_fp32(mean_raters_fp32(A[h,i,j]))
k = (N_content + 3) // 4
```

QK matmul uses fixed BF16 model compute dtype; score softmax/reductions are FP32. System, structural rows, and causally permitted suffix keys remain in the denominator. All original visual candidates remain present for scoring, regardless of the previous layer's selection. Rater-row blocking is allowed only with each row's exact full-key denominator. Reject NaN/Inf inputs/scores; no sampling, visual-only/block-local softmax, or Q/K/head mean before softmax.

Choose exact top-k real original token IDs per layer, ties by ascending original index. No chunk scoring. Per-layer k is constant for an image; selected IDs may differ. Keep those positions at all K/V heads and freeze the set through decoding. Stored full native KV permits reselection in other layers. No decode score or image-payload reread.

## 5. Physical storage and disjoint I/O accounting

Both use original canonical/raster token-major FP16 `layer_XX/k.bin` and `v.bin`, shape `(v_num, num_kv_heads, head_dim)`, chunk size 64. Verify metadata against binary sizes and original mapping. `config.py` contains a generic head-major comment; the actual `store.py` files are token-major and their measured layout governs this contract. Selected original IDs map to sorted unique current-layout chunks. Coalesce adjacent runs only; no gap-overread, fallback full-load, strided tiny-read policy, or importance repacking. EOF short chunks read actual remaining bytes; nonexistent padding is never counted as returned bytes.

Probe3, each layer:

1. Read all visual positions from raw `probe_k.bin` containing exactly canonical heads `[0,1,2]`, with exact FP16 bits. No mean key, quantization, different image, or representative replacement.
2. Score and select real tokens; derive chunk runs.
3. Whole-chunk read all-head K and V for those chunks; allow only selected real rows plus all structural rows in answer attention.
4. Report actual duplicate K elements/bytes shared by probe file and selected K chunks. Duplicates are real retransfers; do not redesign layout merely to remove them.

AllHead, each layer:

1. Read canonical complete visual K in one contiguous read. No complete visual V read unless the selected whole-chunk plan itself reaches every V chunk; report that realized coverage honestly.
2. Score all actual heads, select real tokens, and reuse the already-read K for selected/structural K assembly.
3. Whole-chunk read only required V runs. No probe file open/read and **zero additional selected-K SSD read**.
4. Unselected K remains excluded from actual attention even though read for scoring. This full-K read is online work, not metadata.

Structural sidecar behavior is shared and explicit. If an existing sidecar includes both K/V, record duplicated structural K against AllHead full K and overlap with selected chunks. Existing prefix/system metadata can stay activated only with its bytes/residency/activation recorded. A single physical event belongs to one summed category; classify reused logical bytes separately from retransmitted duplicate bytes.

```
Probe3 actual_total = probe_K + selected_chunk_K + selected_chunk_V + structural_or_other
AllHead actual_total = full_visual_K + selected_chunk_V + structural_or_other
AllHead additional_selected_K = 0; probe_read = 0
```

Record every `os.pread` file/range, actual returned bytes and calls, planned ranges, valid/extra real rows, structural rows, short EOF, and padding separately. Logical attention KV bytes, OS bytes, disk occupancy, and NAND/controller traffic are distinct. Neither KV25 nor Probe3 implies 25% SSD reads or an I/O/latency advantage.

Host layout metadata and v_hidden may be metadata-ready. Full visual K, probe K, and selected payload must not remain resident between requests; same-request K reuse is permitted and mandatory for AllHead. Record metadata bytes on host/GPU, activation reads/time, and per-request rater H2D. Payload acquisition is always inside TTFT.

## 6. Attention integration, cache positions, and isolation

Use the same instance/request-scoped attention wrapper for both new variants; only policy and K supply differ. It must perform each layer's normal q_proj/k_proj/v_proj once, and reuse their results for scoring and real answer attention. Count actual projection calls during suffix prefill. Score matmul/softmax is additional measured work. Explicit prefill/decode phase handles a one-token prefill correctly.

Existing `serve.py` installs `ml.eager_attention_forward = _eager_with_bias` globally at import and retains `_ORIG_EAGER`. Record this before the adapter; do not overwrite that global again or apply BIAS twice. New request-local masks must use exactly one correct causal/keep path, and restore instance forwards/hooks/state even on exception. No adapter/mask/cache tail survives into vanilla, ReComp, FullLoad, Ours, legacy QA or Qwen.

Preserve LLaVA/Ours full logical-position dense cache and the existing eager backend. Prefix/system and structural image rows are always retained; actual mask rejects future text, padding and unselected real rows. Zero K/V is not a mask. Verify captured K at the cache update point is post-RoPE; never rotate it again. Suffix and decode coordinates follow full original prompt length, not retained-token length. A full-size cache is not a 75% GPU memory reduction claim.

If projection reuse or matched-output equivalence cannot be proved, mark it unresolved and withhold paper-facing latency pilot/main readiness. A reference double-projection path cannot be presented as the optimized final baseline.

## 7. Frozen samples, tolerance, and correctness gates

CPU synthetic score/rater tolerance is **`atol=1e-5, rtol=1e-5`**. GPU same-operands score reference uses the same BF16 QK operands/path and FP32 probability reduction with the same `1e-5` tolerance; exact selected IDs remain required. GPU SSD-versus-independent in-memory **matched-shape first-step FP32 logits: `atol=1e-4, rtol=1e-4`**, inherited from the validated LLaVA KV25 contract. KV bits after the same FP16 storage round-trip, masks, positions, head IDs, counts, first/generated token IDs, and predictions must match exactly. No tolerance increase after failure. ReComp full-prefill NF4/BF16 shape differences are a separate diagnostic and cannot excuse matched-shape storage/mask failure.

Frozen GPU pairs are first ten existing GQA image entries, each `questions[4]`:

| Image | Question |
|---|---|
| n355567 | 201751701 |
| n9181 | 20929611 |
| n390187 | 201861403 |
| n133585 | 202108008 |
| n272098 | 201535625 |
| n472825 | 202101069 |
| n450919 | 2093976 |
| n37274 | 202144724 |
| n293477 | 20856909 |
| n44249 | 20935919 |

Generated-history GPU case: first existing frozen MT40 dialogue `mtgqa_002749`, image `n130464`; T1 `201233828`, T2 `201233881`, T3 `201233938`. Execute each method's generated-history chain; never seed it with gold. These are frozen identities, not evidence of execution.

CPU coverage: independent literal official FP32 rater fixture; independent full attention matrix versus rater-only score for all heads and `[0,1,2]`; identical-head score/ID equality; signal only after head 2; `N=1,63,64,65,127,128,129,255,256,257,349`; structural/padding/short chunks; ties, NaN/Inf, empty candidates, empty-rater fallback, one-token suffix, long history. Negative fixtures deliberately corrupt original-ID mapping, head slices, and future masks and must be detected. A controlled fixed-selection fixture must give equal actual keeps/outputs for both policies.

| Gate | Required independent evidence |
|---|---|
| G1 | actual model MHA/config, weight/compute/native/storage dtype, shape, head policy |
| G2 | T1 same-forward capture provenance; T1 vision=1, hit vision=0; prefix independence |
| G3 | canonical K/V round-trip; Probe3 bits exactly canonical `[0,1,2]` |
| G4 | same-operands independent score/Top-k oracle for both policies, per-layer state |
| G5 | SSD vs independent memory selector/assembly/mask: KV bits, keep/mask, logits/output; oracle cannot call production selector+loader+mask builder |
| G6 | exact ceil(N/4) real rows at every layer, retained K/V at every head |
| G7 | logical causal/RoPE/structural correctness and fixed decode selections/no reread |
| G8 | after scoring/IDs fixed, finite unselected-K/V sentinel leaves answer attention/output invariant |
| G9 | actual OS read events equal independent plan/counters; AllHead probe=0, K-reread=0 |
| G10 | actual per-layer Q/K/V projection once, scoring once in prefill, zero extra full-model/vision pass |
| G11 | request/method permutations and exceptions restore hooks/masks/cache; unpatched output recovery |
| G12 | LLaVA Ours-KV25/FullLoad/legacy QA and Qwen CPU regressions; artifact protection; Qwen GPU separately NOT RUN unless executed |

Compare independent memory reference using the same layer/suffix state, FP16 round-trip, and full cache shape. Same-operands diagnostics may snapshot operands outside timing; they must not feed a FullLoad teacher into production requests. Required FAIL/UNRESOLVED/NOT RUN blocks performance-readiness. Higher accuracy/faster TTFT/same predictions across different policies are not PASS conditions.

## 8. Storage preflight and immutable artifact handling

Starting mount was `/dev/nvme0n1p2`, ext4 `/`, available 122,123,231,232 bytes in `environment_before.json`. The subsequent frozen `storage_plan.json` measured 122,079,076,352 available bytes and reserves 21,474,836,480 bytes (20 GiB), with a 10,737,418,240-byte (10 GiB) peak working set and at most 8,589,934,592 bytes (8 GiB) of new payload. This initial plan is safe; enforce it again before allocating payload. Planned read-only roots are `runs/query_aware_baseline/gqa40_240_final_store/raster` and `runs/query_aware_baseline/gqa40_240_final_store/image_only`. Per-image compatibility and actual storage/cleanup remain required gates. Insufficient space is `BLOCKED_STORAGE`.

Prefer compatible original canonical/raster and Ours image-only stores read-only. Check image hash, checkpoint/processor/precision, prefix IDs, geometry/positions, capture provenance, binary layout, K/V dtype and v_hidden. Do not synthesize v_hidden from K. Missing Probe3 sidecar may be generated from canonical K into a new run path, with original canonical files untouched. Share canonical K/V between variants; no duplicate large stores solely for policy identity.

Read-only reuse is `CACHE-HIT REEVALUATION`, persistence/session `NOT_REMEASURED`. Making a missing sidecar does not measure whole cold-start persistence. If fresh T1 provisioning is measured, separate AllHead canonical base and Probe3-only sidecar creation; distinguish shared actual occupancy and standalone deployment footprint. Fresh build, if required, is one image at a time in run-owned scratch. Only allowlisted new temporary payload may be cleaned after raw/hash/audit durability; preserve failed-image raw and all old/external linked files.

## 9. Conditional five-arm workloads and timing

Required methods, same execution: ReComp / FullLoad / Probe3 / AllHead / Ours-KV25. MPIC, ReKV and QA-Chunk are excluded from this pilot. Legacy QA-Token versus Probe3 is a separate small diagnostic with newly measured rows, not legacy raw reuse.

- Smoke: frozen four images × three turns × five arms = 60 requests. Use the first four images of the frozen GQA manifest with its first three selected questions, preserving independent GQA suffixes. MT generated-history behavior is separately gated above.
- GQA: existing frozen 40 images × six questions × five arms = 1,200 requests; per arm 40 T1 and 200 independent hits. Source `runs/llava_spatial_kv25_20260929T091132Z/workload_manifest.json`, file SHA256 `f13964228bf7b69352e23eb844d23deac870192f74e26c27f9222394127ddd7a`; index SHA256 `514d1203d248b6f450f5e3bdacda7b931038f9c11df270b415a2e98e5c77e75a`; workload content SHA256 `97afe02f924a49cadf0c357175b50185e8f16db12b2dd4402595e2bb99d20f66`. It uses first 40 images, `questions[4:10]`; Q2–Q6 have no history.
- MT-GQA-reconstructed: existing frozen 40 dialogues × three turns × five arms = 600 requests; per arm 40 T1 and 80 hits. Data-only source `runs/qwen25_port_20260928T054537Z/mt_manifest/manifest.json`, file SHA256 `b76425302ad7de6000b9ac3079341120b367c8c5e70a1478469383709a9252b6`, source content identity `14d97f83bca9edebafd829c20d6f76fc76a3d1f6024ded1dd9a6385dcde719f1`. Reuse only dialogue/question/image membership and provenance; its Qwen model/processor/method configuration must not propagate into LLaVA. Validate against `data/mt_gqa/dialogues.json`, SHA256 `2c47cfad2a7ccbb673042b400304d7f3ca03d6fbe59d04fa83db50708c924224`.

Copy/hash the exact compatible frozen membership manifest into the new run before execution. Missing or mismatching manifests block the corresponding workload; never substitute a different sample. Every T1 is genuinely executed per method; copying another arm's T1 answer/time is forbidden. Method/dialogue history is independent even when images repeat. Differences across head policies may propagate into later layer state and later generated history.

True TTFT begins at request start before prompt construction, tokenization, image read/processor/input preparation and rater work, and includes scoring-K SSD/H2D, score/select/chunk read/assembly, actual prefill, first-token host materialization and CUDA synchronization. Full decode completion is separate request E2E. Use one identical outer timing boundary for all arms; core timers alone are not E2E. Metadata activation and file-hash validation are outside timed requests with explicit cost/residency records. Rater H2D is inside. DONTNEED conditioning happens before the timer; it does not ensure cold SSD controller/NAND.

One model/method benchmark at a time on the GPU; deterministic method-order rotation and fixed warmup. Inspect other compute load before latency; defer if active rather than terminate it. Add no unnecessary per-layer synchronize. Report overlapping host/CUDA intervals separately; do not sum overlaps into fabricated total. Controlled fixed-Q/K/rater policy comparisons run outside timing and report score correlation, Top-k overlap and chunk counts, with no teacher operands in production requests.

## 10. Raw, audit, summary, and final decision

Per-request raw must retain code/config/manifest/experiment/attempt hashes; method/image/question/dialogue/turn identity; prompt/suffix/span/rater/history provenance; actual head IDs; every layer's N/k, selected original/chunk IDs, runs, valid/extra/structural/padding rows; disjoint scoring-K/probe, selected-K, selected-V, structural/other and metadata bytes; actual pread events; reused/duplicated K elements/bytes; H2D; cache shapes/keep counts; peak allocated/reserved GPU memory and metadata residency; projection/scoring/vision counts; TTFT/E2E; first/generated IDs, answer/scorer output; source T1/provisioning mode; failure/NaN/Inf/cap/retry state.

Preserve every failed/retried/partial attempt separately. Adopt one actual executed final row per logical request by a fixed resume rule, never by favorable accuracy/timing. Independent audit must recompute from raw request counts, method identities, generated history, scoring heads, exact k, OS bytes, quality, TTFT and paired summaries; it cannot accept runner aggregate assertions as independent evidence.

Quality: report every turn, all requests, and hit-only. GQA has 200 hits and MT 80 hits per arm. Paired Probe3−AllHead and each baseline−Ours differences cover quality, TTFT and OS SSD bytes. Use image-cluster bootstrap, 10,000 resamples, analysis seed 1234; all dialogues/turns/methods for a sampled image stay together, with the same request weighting as the point estimate. Report point estimates and 95% intervals. Zero-containing CIs do not establish equivalence. More heads need not improve accuracy. MT includes method-generated-history effects and is not a pure causal head-selection contrast.

GQA and MT reports begin with these two tables (unmeasured entries explicitly NOT RUN):

| 방법 | 점수 head 수 | 전체/Hit 정답률 | T1/Hit TTFT | 실제 content KV % | SSD MB/hit |
|---|---|---|---|---|---|

| 방법 | Scoring K MB | 추가 selected-K MB | Selected-V MB | Structural/other MB | Total MB | Preads |
|---|---|---|---|---|---|---|

Also report selector/actual-attention interval scope, projection counts, peak memory, controlled head-policy overlap, paired 95% CI and measured/NOT_REMEASURED setup costs. Required outputs: new module/runner/tests, this contract and official correspondence, `source.diff`, frozen inputs/configs/hashes, `storage_plan.json`, `validation.json`, per-request JSONL, `summary.csv`, paired comparisons, I/O/timing/memory breakdown, `independent_audit.json`, protection receipts, executable reproduction/resume commands and `REPORT.md`.

Final Korean status must separately state `IMPLEMENTATION PROBE3 / ALLHEAD`, source-metric fidelity and deliberate adaptation, CPU test/GPU correctness, projection reuse/AllHead K-read reuse, GQA/MT five-arm pilot, legacy/Ours/Qwen protection, and `READY FOR LLAVA MT-GQA BASELINE INTEGRATION: YES/NO` with evidence. A required gate that has not run is not PASS. Correctness or projection reuse FAIL/UNRESOLVED means readiness NO and no paper-facing latency claim.
