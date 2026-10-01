# Prefix25 persistence ablation

## 핵심 결과

두 모델의 CPU/GPU correctness와 A/B 출력 동일성 검증을 통과했다. 모델별 smoke 48요청, GQA 960요청, MT 480요청을 새로 실행했으며, 과거 latency/persistence를 섞지 않았다. 모든 T1은 full-image inference이고 표의 content retention은 hit에서 사용하는 예산이다.

모델 | 저장/write 감소 (GQA / MT) | Persistence 감소 (GQA / MT) | MT 세션 B−A | MT 세션 B−ReComp
--- | --- | --- | --- | ---
llava | 73.98% / 74.02% | 64.42% / 65.83% | -1906.8 ms | 317.9 ms
qwen | 73.54% / 73.73% | 32.40% / 32.94% | -53.9 ms | -41.4 ms

고정 25% 정책에서 A를 B로 대체할 근거는 저장량·persistence·A 대비 세션 비용에서 확인됐다. 다만 LLaVA의 짧은 3-turn 세션은 ReComp가 더 빠르다. A/B 동등성은 선택 KV와 생성열을 직접 비교한 결과이며 ReComp 대비 품질 동등성을 뜻하지 않는다. 전체 CPU 복사와 전체 repack 비용은 남아 있다.

[MT 세션 비용 그림](session_costs.png) · [벡터 그림](session_costs.svg)

Run: `/home/dblab/hj/mllm_v2/runs/prefix25_persistence_20261001T050113Z`. All numbers below are generated from this run's raw records. MB = 10^6 bytes; GiB = 2^30 bytes.

A retains the full importance-repacked store and uses the first ceil(N/4) content rows. B serializes those same content rows only, preserving structural/system KV. Existing full CPU materialization and full repack remain in both paths.

GPU validation reuses the prior frozen samples, not an unseen holdout. LLaVA uses its full physical buffer and 1e-4/1e-4 matched-logit tolerance; Qwen uses native BF16 logical compact assembly and bitwise-exact matched first logits. Prior Qwen strict-logits v1 FAIL remains unchanged.

## llava

GPU correctness: **PASS**, 5 recorded samples.

Final format replay: **PASS**. Output token IDs, logit hashes and Qwen decode attention steps are retained in `runs/prefix25_persistence_20261001T050113Z/llava/final_gpu_regression.json`.

Image | Question | N | k | k/N | Status
--- | --- | --- | --- | --- | ---
n355567 | 201751701 | 2112 | 528 | 0.250 | PASS
n9181 | 20929611 | 2304 | 576 | 0.250 | PASS
n390187 | 201861403 | 2208 | 552 | 0.250 | PASS
n133585 | 202108008 | 2112 | 528 | 0.250 | PASS
n272098 | 201535625 | 2880 | 720 | 0.250 | PASS

### gqa

Raw audit: **PASS**; 960/960 successful requests.

Table 1. Correctness and output identity

Method | Content retention | All accuracy | Hit/followup accuracy | First agreement vs A | Sequence agreement vs A
--- | --- | --- | --- | --- | ---
ReComp | 1.000 | 0.625 | 0.630 | 0.858 | 0.858
FullLoad | 1.000 | 0.625 | 0.630 | 0.850 | 0.850
Ours-FullStore-KV25 [A] | 0.250 | 0.571 | 0.565 | 1.000 | 1.000
Ours-PrefixStore-KV25 [B] | 0.250 | 0.571 | 0.565 | 1.000 | 1.000

Table 2. Storage and first persistence (MB/image and ms)

Method | Content MB | Total MB | Allocated MB | OS write MB | D2H ms | Repack ms | Write ms | fsync ms | Persistence ms | Activation ms
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
ReComp | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000
FullLoad | 1145.045 | 1260.581 | 1260.613 | 1260.581 | 100.464 | 0.000 | 420.810 | 1011.245 | 2278.082 | 612.541
Ours-FullStore-KV25 [A] | 1145.045 | 1187.809 | 1187.816 | 1187.809 | 99.271 | 40.818 | 391.979 | 840.832 | 2043.255 | 583.421
Ours-PrefixStore-KV25 [B] | 286.261 | 309.019 | 309.025 | 309.019 | 92.086 | 40.974 | 106.514 | 276.061 | 726.898 | 160.390

Table 3. Followup requests (cached arms are hits; ReComp recomputes)

Method | Read MB | Preads | TTFT mean/p50/p95 ms | E2E mean ms | H2D KV MB | GPU peak MB
--- | --- | --- | --- | --- | --- | ---
ReComp | 0.000 | 0.000 | 519.480/525.981/580.177 | 545.299 | NOT RUN | 10150.879
FullLoad | 1165.073 | 64.000 | 722.228/687.043/1048.181 | 749.213 | 1167.694 | 8553.188
Ours-FullStore-KV25 [A] | 319.501 | 65.000 | 249.755/242.049/318.789 | 276.629 | 322.123 | 8604.947
Ours-PrefixStore-KV25 [B] | 306.289 | 65.000 | 236.115/230.752/295.542 | 262.986 | 308.910 | 8664.116

B total-store reduction: 73.98%; OS write reduction: 73.98%; persistence reduction: 64.42%. Estimated images per fixed SSD capacity: 3.844×, including metadata and structural files. Cache-hit-rate or eviction benefit was not measured.

Paired differences use 10,000 image-cluster bootstrap resamples (seed 1234). Negative B−A means lower latency. A CI spanning zero is inconclusive.

Paired metric | Mean B−A | 95% CI
--- | --- | ---
B_minus_A_persistence_ms | -1316.357 | [-1468.085, -1178.698]
B_minus_A_hit_ttft_ms | -13.639 | [-18.716, -8.380]
B_minus_A_hit_e2e_ms | -13.643 | [-18.723, -8.393]
B_minus_A_session_ms | -1809.177 | [-1973.755, -1654.615]

### mt

Raw audit: **PASS**; 480/480 successful requests.

Table 1. Correctness and output identity

Method | Content retention | All accuracy | Hit/followup accuracy | First agreement vs A | Sequence agreement vs A
--- | --- | --- | --- | --- | ---
ReComp | 1.000 | 0.758 | 0.787 | 0.917 | 0.917
FullLoad | 1.000 | 0.758 | 0.787 | 0.917 | 0.917
Ours-FullStore-KV25 [A] | 0.250 | 0.742 | 0.762 | 1.000 | 1.000
Ours-PrefixStore-KV25 [B] | 0.250 | 0.742 | 0.762 | 1.000 | 1.000

Table 2. Storage and first persistence (MB/image and ms)

Method | Content MB | Total MB | Allocated MB | OS write MB | D2H ms | Repack ms | Write ms | fsync ms | Persistence ms | Activation ms
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
ReComp | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000
FullLoad | 1175.244 | 1292.019 | 1292.054 | 1292.019 | 100.260 | 0.000 | 433.098 | 1072.196 | 2373.744 | 628.594
Ours-FullStore-KV25 [A] | 1175.244 | 1217.381 | 1217.387 | 1217.381 | 96.900 | 40.833 | 402.309 | 951.499 | 2177.919 | 595.429
Ours-PrefixStore-KV25 [B] | 293.811 | 316.256 | 316.262 | 316.256 | 93.948 | 42.318 | 109.375 | 289.388 | 744.200 | 163.836

Table 3. Followup requests (cached arms are hits; ReComp recomputes)

Method | Read MB | Preads | TTFT mean/p50/p95 ms | E2E mean ms | H2D KV MB | GPU peak MB
--- | --- | --- | --- | --- | --- | ---
ReComp | 0.000 | 0.000 | 542.887/533.410/601.366 | 570.582 | NOT RUN | 9883.281
FullLoad | 1194.957 | 64.000 | 695.518/678.904/838.959 | 724.149 | 1197.579 | 8239.642
Ours-FullStore-KV25 [A] | 327.575 | 65.000 | 264.762/262.538/312.721 | 293.222 | 330.197 | 8393.674
Ours-PrefixStore-KV25 [B] | 313.524 | 65.000 | 243.743/237.736/287.828 | 272.085 | 316.146 | 8538.878

Table 4. MT 3-turn standalone session (ms)

Method | T1 E2E | Persistence | Activation | T2 E2E | T3 E2E | Total | Δ vs ReComp | Δ vs A
--- | --- | --- | --- | --- | --- | --- | --- | ---
ReComp | 559.254 | 0.000 | 0.000 | 569.829 | 571.335 | 1700.418 | 0.000 | -2224.687
FullLoad | 567.873 | 2373.744 | 628.594 | 730.087 | 718.211 | 5018.509 | 3318.090 | 1093.404
Ours-FullStore-KV25 [A] | 565.313 | 2177.919 | 595.429 | 307.140 | 279.304 | 3925.105 | 2224.687 | 0.000
Ours-PrefixStore-KV25 [B] | 566.083 | 744.200 | 163.836 | 269.157 | 275.012 | 2018.288 | 317.870 | -1906.817

B total-store reduction: 74.02%; OS write reduction: 74.02%; persistence reduction: 65.83%. Estimated images per fixed SSD capacity: 3.849×, including metadata and structural files. Cache-hit-rate or eviction benefit was not measured.

Paired differences use 10,000 image-cluster bootstrap resamples (seed 1234). Negative B−A means lower latency. A CI spanning zero is inconclusive.

Paired metric | Mean B−A | 95% CI
--- | --- | ---
B_minus_A_persistence_ms | -1433.720 | [-1590.022, -1286.570]
B_minus_A_hit_ttft_ms | -21.019 | [-26.370, -15.792]
B_minus_A_hit_e2e_ms | -21.137 | [-26.482, -15.904]
B_minus_A_session_ms | -1906.817 | [-2067.321, -1757.046]

## qwen

GPU correctness: **PASS**, 10 recorded samples.

Final format replay: **PASS**. Output token IDs, logit hashes and Qwen decode attention steps are retained in `runs/prefix25_persistence_20261001T050113Z/qwen/final_gpu_regression.json`.

Image | Question | N | k | k/N | Status
--- | --- | --- | --- | --- | ---
n355567 | 201751701 | 345 | 87 | 0.252 | PASS
n355567 | 201751740 | 345 | 87 | 0.252 | PASS
n355567 | 201751873 | 345 | 87 | 0.252 | PASS
n9181 | 20929611 | 391 | 98 | 0.251 | PASS
n390187 | 201861403 | 368 | 92 | 0.250 | PASS
n133585 | 202108008 | 345 | 87 | 0.252 | PASS
n272098 | 201535625 | 484 | 121 | 0.250 | PASS
n472825 | 202101069 | 391 | 98 | 0.251 | PASS
n450919 | 2093976 | 266 | 67 | 0.252 | PASS
n37274 | 202144724 | 391 | 98 | 0.251 | PASS

### gqa

Raw audit: **PASS**; 960/960 successful requests.

Table 1. Correctness and output identity

Method | Content retention | All accuracy | Hit/followup accuracy | First agreement vs A | Sequence agreement vs A
--- | --- | --- | --- | --- | ---
ReComp | 1.000 | 0.567 | 0.590 | 0.858 | 0.842
FullLoad | 1.000 | 0.567 | 0.590 | 0.858 | 0.842
Ours-FullStore-KV25 [A] | 0.251 | 0.550 | 0.570 | 1.000 | 1.000
Ours-PrefixStore-KV25 [B] | 0.251 | 0.550 | 0.570 | 1.000 | 1.000

Table 2. Storage and first persistence (MB/image and ms)

Method | Content MB | Total MB | Allocated MB | OS write MB | D2H ms | Repack ms | Write ms | fsync ms | Persistence ms | Activation ms
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
ReComp | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000
FullLoad | 20.013 | 23.731 | 23.736 | 23.731 | 8.129 | 0.000 | 7.860 | 33.195 | 77.222 | 26.159
Ours-FullStore-KV25 [A] | 20.013 | 23.731 | 23.736 | 23.731 | 8.207 | 0.646 | 7.825 | 39.214 | 84.198 | 25.945
Ours-PrefixStore-KV25 [B] | 5.026 | 6.280 | 6.356 | 6.280 | 7.544 | 0.666 | 3.022 | 32.987 | 56.918 | 9.232

Table 3. Followup requests (cached arms are hits; ReComp recomputes)

Method | Read MB | Preads | TTFT mean/p50/p95 ms | E2E mean ms | H2D KV MB | GPU peak MB
--- | --- | --- | --- | --- | --- | ---
ReComp | 0.000 | 0.000 | 112.610/115.229/123.517 | 133.006 | NOT RUN | 6155.887
FullLoad | 23.683 | 57.000 | 76.405/71.628/101.234 | 96.394 | 21.217 | 6110.230
Ours-FullStore-KV25 [A] | 8.544 | 57.000 | 57.934/54.629/73.594 | 77.342 | 6.230 | 6095.233
Ours-PrefixStore-KV25 [B] | 6.230 | 57.000 | 53.710/51.833/61.261 | 73.120 | 6.230 | 6095.233

B total-store reduction: 73.54%; OS write reduction: 73.54%; persistence reduction: 32.40%. Estimated images per fixed SSD capacity: 3.779×, including metadata and structural files. Cache-hit-rate or eviction benefit was not measured.

Paired differences use 10,000 image-cluster bootstrap resamples (seed 1234). Negative B−A means lower latency. A CI spanning zero is inconclusive.

Paired metric | Mean B−A | 95% CI
--- | --- | ---
B_minus_A_persistence_ms | -27.281 | [-30.233, -24.422]
B_minus_A_hit_ttft_ms | -4.224 | [-5.492, -2.937]
B_minus_A_hit_e2e_ms | -4.222 | [-5.501, -2.923]
B_minus_A_session_ms | -64.786 | [-71.819, -57.770]

### mt

Raw audit: **PASS**; 480/480 successful requests.

Table 1. Correctness and output identity

Method | Content retention | All accuracy | Hit/followup accuracy | First agreement vs A | Sequence agreement vs A
--- | --- | --- | --- | --- | ---
ReComp | 1.000 | 0.683 | 0.700 | 0.908 | 0.908
FullLoad | 1.000 | 0.683 | 0.700 | 0.908 | 0.908
Ours-FullStore-KV25 [A] | 0.251 | 0.667 | 0.675 | 1.000 | 1.000
Ours-PrefixStore-KV25 [B] | 0.251 | 0.667 | 0.675 | 1.000 | 1.000

Table 2. Storage and first persistence (MB/image and ms)

Method | Content MB | Total MB | Allocated MB | OS write MB | D2H ms | Repack ms | Write ms | fsync ms | Persistence ms | Activation ms
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
ReComp | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000
FullLoad | 21.245 | 25.108 | 25.114 | 25.108 | 9.491 | 0.000 | 9.304 | 35.542 | 85.034 | 28.031
Ours-FullStore-KV25 [A] | 21.245 | 25.109 | 25.114 | 25.109 | 8.910 | 0.699 | 8.789 | 38.804 | 87.414 | 27.438
Ours-PrefixStore-KV25 [B] | 5.340 | 6.595 | 6.679 | 6.595 | 8.328 | 0.754 | 3.423 | 32.548 | 58.619 | 9.739

Table 3. Followup requests (cached arms are hits; ReComp recomputes)

Method | Read MB | Preads | TTFT mean/p50/p95 ms | E2E mean ms | H2D KV MB | GPU peak MB
--- | --- | --- | --- | --- | --- | ---
ReComp | 0.000 | 0.000 | 114.926/114.430/129.901 | 137.088 | NOT RUN | 6167.582
FullLoad | 25.059 | 57.000 | 81.155/76.268/106.356 | 102.765 | 22.449 | 6117.673
Ours-FullStore-KV25 [A] | 8.728 | 57.000 | 64.021/60.487/86.243 | 85.186 | 6.544 | 6101.743
Ours-PrefixStore-KV25 [B] | 6.544 | 57.000 | 60.604/57.727/88.128 | 81.728 | 6.544 | 6101.743

Table 4. MT 3-turn standalone session (ms)

Method | T1 E2E | Persistence | Activation | T2 E2E | T3 E2E | Total | Δ vs ReComp | Δ vs A
--- | --- | --- | --- | --- | --- | --- | --- | ---
ReComp | 135.030 | 0.000 | 0.000 | 136.411 | 137.765 | 409.206 | 0.000 | -12.446
FullLoad | 134.508 | 85.034 | 28.031 | 99.712 | 105.818 | 453.103 | 43.897 | 31.451
Ours-FullStore-KV25 [A] | 136.429 | 87.414 | 27.438 | 81.353 | 89.019 | 421.652 | 12.446 | 0.000
Ours-PrefixStore-KV25 [B] | 135.959 | 58.619 | 9.739 | 79.941 | 83.515 | 367.772 | -41.433 | -53.880

B total-store reduction: 73.73%; OS write reduction: 73.73%; persistence reduction: 32.94%. Estimated images per fixed SSD capacity: 3.807×, including metadata and structural files. Cache-hit-rate or eviction benefit was not measured.

Paired differences use 10,000 image-cluster bootstrap resamples (seed 1234). Negative B−A means lower latency. A CI spanning zero is inconclusive.

Paired metric | Mean B−A | 95% CI
--- | --- | ---
B_minus_A_persistence_ms | -28.795 | [-33.126, -24.504]
B_minus_A_hit_ttft_ms | -3.417 | [-6.097, -0.607]
B_minus_A_hit_e2e_ms | -3.458 | [-6.146, -0.636]
B_minus_A_session_ms | -53.880 | [-61.074, -46.534]

## Interpretation and limits

The core A/B evidence is exact selected KV and request-level generated sequences, not equal average accuracy. B retains full CPU materialization and full repack, so their costs cannot disappear with smaller writes. Short final content files remove A's chunk-boundary overread; compare raw read bytes and TTFT CIs before attributing a speed change. GPU computation structure remains unchanged. GPU peaks include coexisting arm contexts and are not isolated deployment memory measurements; no memory-reduction claim is made.

Persistence is the wall time through the equal full-checksum durability seal. It is not reconstructed by summing nested components. Qwen saliency computation in capture exit and required cloning occur after normal generation; outer T1 E2E includes them once. File hash/seal time includes full hash reads and the final envelope/parent sync. Per-component write timing follows each existing writer; total wall time is the primary comparison.

Accuracy uses the existing GQA prefix-tolerant normalized scorer for GQA, and the existing MT pilot strict normalized exact match for MT. The collection raw accuracy field uses the generic GQA scorer; MT report accuracy is independently regenerated from raw prediction and gold, without changing raw files. Allocated bytes are regular-file st_blocks × 512; directory allocation is excluded. Forced source-release GC/empty_cache is experimental isolation outside service timers.

Followup H2D is Qwen's native tensor-byte counter or LLaVA's tensor-shape-derived payload plus system KV bytes. It excludes index/input transfers and is not a bus hardware counter. Allocated/reserved peaks, full read spans, conditioning, and phase timings are in raw records. OS wchar/syscw are successful syscall write traffic, not SSD NAND traffic. DONTNEED is an OS page-cache hint and does not prove cold controller/NAND.

All pilot payload stores are cleaned only after their image's scheduled measurements and hash/mapping receipts are committed. Correctness stores remain if present; pilot reproduction must regenerate stores. No previously existing store/result/raw was deleted.

Original performance source freezes are gpu_freeze_llava.json and gpu_freeze_qwen.json. The pre-Qwen amendment includes scoring in the outer T1 boundary and rejects unsealed incomplete B stores; normal sealed-store KV/attention semantics stay unchanged. final_gpu_freeze.json binds the final GPU replay. Both original measurement source versions are retained.

Longer-session break-even values in paired_analysis.json are analytical estimates using measured request E2E and one-time costs with constant mean hit savings. They are separate from the measured MT three-turn sessions. The 4,061-dialogue full experiment was not run.

Artifact protection: PASS. Independent raw audit: PASS.

## Decisions from measured results

llava: the fixed-sample gate established A/B selected-KV, visible attention and generated-output identity under the frozen model-specific thresholds.

llava GQA: total store bytes fell 73.98%, OS write bytes fell 73.98%, and persistence changed -1316.357 ms (95% CI [-1468.085186316457, -1178.6980518329074]). Hit TTFT changed -13.639 ms (95% CI [-18.715581114927772, -8.380184020672461]).

llava MT: total store bytes fell 74.02%, OS write bytes fell 74.02%, and persistence changed -1433.720 ms (95% CI [-1590.022111836588, -1286.5703144017607]). Hit TTFT changed -21.019 ms (95% CI [-26.37016662498354, -15.792047620925587]).

llava: the measured MT three-turn standalone session improved versus A: B−A -1906.817 ms, CI [-2067.321187336056, -1757.0456568471855]. B−ReComp is 317.870 ms, CI [273.85721143480623, 370.132161033398]. This includes T1, persistence, activation and both hits.

llava: fixed-25% adoption has measured capacity/persistence support when the corresponding reductions above are positive. It remains limited to a fixed maximum budget: B rejects larger content requests. Full CPU copy and repack costs remain. Large-rerun readiness requires the independent audit and protection receipt below; bounded per-image cleanup is required for capacity.

qwen: the fixed-sample gate established A/B selected-KV, visible attention and generated-output identity under the frozen model-specific thresholds.

qwen GQA: total store bytes fell 73.54%, OS write bytes fell 73.54%, and persistence changed -27.281 ms (95% CI [-30.23275907820789, -24.422029714623932]). Hit TTFT changed -4.224 ms (95% CI [-5.492488480114844, -2.9369850264047277]).

qwen MT: total store bytes fell 73.73%, OS write bytes fell 73.73%, and persistence changed -28.795 ms (95% CI [-33.12592526723165, -24.504088691610377]). Hit TTFT changed -3.417 ms (95% CI [-6.097371435462264, -0.6066840080893614]).

qwen: the measured MT three-turn standalone session improved versus A: B−A -53.880 ms, CI [-61.07447804621188, -46.534041121194605]. B−ReComp is -41.433 ms, CI [-48.26759477669839, -34.238019724143676]. This includes T1, persistence, activation and both hits.

qwen: fixed-25% adoption has measured capacity/persistence support when the corresponding reductions above are positive. It remains limited to a fixed maximum budget: B rejects larger content requests. Full CPU copy and repack costs remain. Large-rerun readiness requires the independent audit and protection receipt below; bounded per-image cleanup is required for capacity.

## Final model status

Stage | LLaVA | Qwen
--- | --- | ---
IMPLEMENTATION | PASS | PASS
CPU REGRESSION | PASS | PASS
GPU CORRECTNESS | PASS | PASS
A/B OUTPUT IDENTITY | PASS | PASS
FINAL SOURCE GPU REPLAY | PASS | PASS
SMOKE | PASS | PASS
GQA PILOT | PASS | PASS
MT PILOT | PASS | PASS
PERSISTENCE/SESSION MEASUREMENT | PASS | PASS
ARTIFACT PROTECTION | PASS | PASS
READY FOR LARGE RERUN | PASS | PASS
