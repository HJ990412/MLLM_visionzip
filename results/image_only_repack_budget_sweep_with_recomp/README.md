# ImageOnly-Repack sequential-Prefix budget sweep with ReComp

## A. Workload

This is the frozen GQA workload: **40 images / 240 questions**, questions
`[4:10]` for every image.  Index SHA256 is `514d1203d248b6f450f5e3bdacda7b931038f9c11df270b415a2e98e5c77e75a` and ordered
workload SHA256 is `97afe02f924a49cadf0c357175b50185e8f16db12b2dd4402595e2bb99d20f66`.  All arms ran in one process with the same model, prompt, decoding, and
schema-v2 true-TTFT contract. FullLoad and Prefix arms used the same
physical SSD store; ReComp used pixels and did not access that store.

## B. Budgets

Main budgets are 20/25/30/35/40/45/50%.  Every budget uses the same immutable
ImageOnly VisionZip permutation; only `k = round(n_chunks * budget)` changes.
No calibration question, online score, diversity, or budget-specific layout is
used.

The Prefix reader was already optimized before this sweep: adjacent first-k
chunks are merged into one contiguous span per layer/K-or-V file.  Therefore
every Prefix request performs 64 normal preads plus one separator-sidecar
pread, not one syscall per logical chunk.

## C. Main table

| Method | Accuracy | Delta vs FullLoad | Delta vs ReComp | TTFT | TTFT reduction vs FullLoad | TTFT reduction vs ReComp | SSD MB | Preads |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ReComp | 62.50% | +1.25 pp | -- | 504.11 ms | +30.37% | -- | 0.00 | 0.0 |
| FullLoad | 61.25% | -- | -1.25 pp | 723.97 ms | -- | -43.61% | 1165.07 | 64.0 |
| Prefix 20% | 54.58% | -6.67 pp | -7.92 pp | 235.45 ms | +67.48% | +53.29% | 252.39 | 65.0 |
| Prefix 25% | 57.92% | -3.33 pp | -4.58 pp | 262.62 ms | +63.72% | +47.90% | 306.08 | 65.0 |
| Prefix 30% | 58.75% | -2.50 pp | -3.75 pp | 299.28 ms | +58.66% | +40.63% | 371.51 | 65.0 |
| Prefix 35% | 59.17% | -2.08 pp | -3.33 pp | 337.51 ms | +53.38% | +33.05% | 435.26 | 65.0 |
| Prefix 40% | 59.17% | -2.08 pp | -3.33 pp | 375.62 ms | +48.12% | +25.49% | 498.18 | 65.0 |
| Prefix 45% | 60.42% | -0.83 pp | -2.08 pp | 406.28 ms | +43.88% | +19.41% | 551.03 | 65.0 |
| Prefix 50% | 60.00% | -1.25 pp | -2.50 pp | 440.43 ms | +39.16% | +12.63% | 603.87 | 65.0 |

Accuracy is binary normalized GQA match.  `summary.csv` reports the
image-cluster bootstrap 95% CI for each accuracy and paired deltas against
both same-run FullLoad and same-run ReComp.  With only 40 image clusters, sub-point differences should not be
over-interpreted.

## D. Pareto frontier

The required SSD and cache-path TTFT frontiers remain restricted to
FullLoad plus Prefix. ReComp is excluded from the SSD frontier because its
zero SSD traffic comes from doing pixel recomputation, not from a better SSD
cache policy. A separate all-method TTFT frontier includes ReComp because
true TTFT and accuracy are directly comparable across the nine arms.

- Accuracy vs SSD: FullLoad, 20%, 25%, 30%, 35%, 45%
- Accuracy vs true TTFT: FullLoad, 20%, 25%, 30%, 35%, 45%
- Accuracy vs true TTFT (all methods): ReComp, 20%, 25%, 30%, 35%, 45%

Dominated points and their dominators are explicit in the three Pareto CSVs.
Pareto membership uses observed point estimates; uncertainty remains in the
paired confidence intervals.

## E. Accuracy recovery

- Aggressive (loss <=4 pp): 25%
- Balanced (loss <=2 pp): 45%
- Quality-oriented (loss <=1 pp): 45%
- Saturation by the predeclared <=0.5 pp best-future-gain rule: 45%

## F. SSD/TTFT cost

ReComp is measured in the same process and request loop. It reads pixels and
recomputes the vision tower plus multimodal prefill for every question, with
zero KV-store bytes and zero preads. Its mean true TTFT is
504.11 ms and accuracy is 62.50%.
FullLoad is -1.25 pp relative to ReComp
and its TTFT reduction relative to ReComp is
-43.61% (negative means slower).

The ReComp timer follows the existing schema-v2 implementation: image
processor work and host-to-device transfer occur before `t0`; the measured
interval includes the vision tower, multimodal prefill, and first output token.
The raw `physical_layout` CSV cell is a run-global annotation emitted by
`04_eval.py`; ReComp never accesses that layout, as verified by zero I/O and
`retrieval=recompute`.

| Step | Accuracy gain | SSD cost | TTFT cost |
|---:|---:|---:|---:|
| 20% -> 25% | +3.33 pp | +53.69 MB | +27.18 ms |
| 25% -> 30% | +0.83 pp | +65.43 MB | +36.66 ms |
| 30% -> 35% | +0.42 pp | +63.75 MB | +38.23 ms |
| 35% -> 40% | +0.00 pp | +62.91 MB | +38.12 ms |
| 40% -> 45% | +1.25 pp | +52.85 MB | +30.66 ms |
| 45% -> 50% | -0.42 pp | +52.85 MB | +34.15 ms |

Doubling 25% to 50% changes accuracy by +2.08 pp,
adds 297.80 MB/request, and adds
177.80 ms mean true TTFT.

## G. Importance coverage

| Budget | VisionZip mass | SparseVLM mass (analysis only) | Accuracy |
|---:|---:|---:|---:|
| 20% | 68.01% | 35.82% | 54.58% |
| 25% | 73.15% | 41.60% | 57.92% |
| 30% | 78.41% | 47.98% | 58.75% |
| 35% | 82.65% | 54.22% | 59.17% |
| 40% | 86.19% | 59.79% | 59.17% |
| 45% | 88.68% | 63.82% | 60.42% |
| 50% | 90.95% | 68.10% | 60.00% |

VisionZip coverage/accuracy correlation: Pearson
`0.9096593029641282` and Spearman
`0.9549937104572924`.  SparseVLM
coverage is explicitly analysis-only and never affected serving.

## H. Error analysis

`per_question_sensitivity.csv`, `per_image_sensitivity.csv`, and
`error_analysis.csv` separate 25%-wrong questions recovered at 30/35%, those
never recovered through 50%, prediction stabilization, and the existing GQA
question categories (yes/no, color, count, spatial, material/attribute,
object, other).

Restricting the attribution to questions that FullLoad answers correctly,
25% loses 19 questions; 7 recover by 30/35%, while
6 never recover at any tested budget through 50%.

## I. Recommended operating point

**Recommended operating point = 45%**.
Rule: minimum budget with point-estimate FullLoad loss <= 2 pp.  At this point the FullLoad
accuracy gap is -0.83 pp, SSD
traffic is 47.30% of FullLoad,
and true TTFT is reduced by
43.88%.

This recommendation is based on the observed point estimate.  Its paired
image-cluster bootstrap delta CI is
`[-3.75,
+2.08] pp`; therefore
this sample does **not** establish that the population accuracy loss is at
most 1 or 2 pp.

## J. Research conclusion

The ReComp arm is an additional compute baseline; cache-path operating-point
thresholds, cache-only Pareto membership, and recovery analysis remain defined
against FullLoad. The separate all-method true-TTFT Pareto includes ReComp.
`baseline_comparison.csv` provides every arm's paired quality and TTFT
comparison against both baselines.

**Claim assessment: SUPPORTED, with non-monotonic accuracy and small-sample caveats.**  Every tested Prefix budget reduces observed
SSD traffic and true TTFT versus same-run FullLoad, and higher budgets recover
some accuracy.  The curve is not monotonic: 45% is more accurate than 50% on
this 240-question sample.  No budget was selected in advance.

Direct answers: Q1=25%,
Q2=45%,
Q3=45%,
Q4=45%,
Q6=45%. Q5 is quantified in
Section F; Q7 is quantified by the coverage correlations in Section G.

Known historical warning retained: raster FullLoad vs ImageOnly-repacked
FullLoad strict prediction identity was
`238/240`, so cross-layout exactness
remains FAIL even though mapped FP16 KV structural integrity passed.  This does
not invalidate the present same-layout budget comparison, but it must not be
reported as strict raster equivalence.

## Reproduction

```bash
/home/dblab/anaconda3/envs/mllm_ft/bin/python scripts/04_eval.py --index data/index.json --store kvstore_image_only_visionzip --limit 40 --questions 6 --skip 4 --ratio 0.25 --budgets 0.20,0.25,0.30,0.35,0.40,0.45,0.50 --selectors visionzip_repack_prefix --prefix-layout visionzip_image_only --sep-policy sidecar --metric gqa --dataset gqa --expect-images 40 --expect-questions 240 --run-dir runs/image_only_repack_budget_sweep_with_recomp/main_20_50
```
