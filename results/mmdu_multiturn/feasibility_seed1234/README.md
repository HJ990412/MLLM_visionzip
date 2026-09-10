# MMDU workload and capacity feasibility

This is a read-only workload/capacity analysis, not a Static+Diverse result.

- Official dialogs / turns / image records: 110 / 1645 / 421
- Unique resolved image paths: 419 (the byte total below follows the current per-ID, non-deduplicated store semantics)
- Active images per turn: mean 3.25, max 20
- Actual visual KV per resized image: 0.617 GB
- Largest dialog visual working set: 12.331 GB
- All 421 per-ID image KVs: 259.573 GB; path-deduplicated: 258.340 GB
- Context-feasible turns (`prompt + 16 <= 4096`): 340 / 1645
- Dialogs with every turn context-feasible: 1 / 110

Bytes come from actual generated bf16 DynamicCache shapes in the progressive
GPU gate. All 1,645 workload lengths are fixed-336 tokenizer projections;
those projections match the six real processor sequences in the gate, which
is the full extent of the processor-level cross-check.
The strict cache gate failed its predeclared numeric tolerance, and the current
single-image SSD store cannot represent contextual later-image KV. Therefore no
MMDU TTFT/SSD/quality arm was run and no such score is fabricated here.
