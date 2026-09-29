# v2 gqa pilot supplement

The primary report and frozen raw data remain unchanged. This audit verifies per-image SSD geometry and disjoint session timing against the primary summary.

All image-cluster intervals use 4,000 draws and seed 1234. Intervals containing zero do not establish equivalence.

| Scope | Left minus right | Metric | Mean difference | 95% low | 95% high |
|---|---|---|---:|---:|---:|
| all | fullload_minus_recompute | accuracy | 0.000000 | 0.000000 | 0.000000 |
| all | fullload_minus_recompute | ttft_ms | -32.718542 | -35.259292 | -29.897293 |
| all | fullload_minus_recompute | request_e2e_ms | -33.865489 | -37.107835 | -30.394268 |
| all | fullload_minus_recompute | visual_read_bytes | 18732373.333333 | 17967786.666667 | 19573418.666667 |
| all | fullload_minus_recompute | total_ssd_read_bytes | 19735893.333333 | 18971306.666667 | 20576938.666667 |
| all | ours25_minus_recompute | accuracy | 0.008333 | -0.029167 | 0.045833 |
| all | ours25_minus_recompute | ttft_ms | -42.673609 | -45.481284 | -39.585684 |
| all | ours25_minus_recompute | request_e2e_ms | -42.114921 | -45.544023 | -38.388580 |
| all | ours25_minus_recompute | visual_read_bytes | 5352106.666667 | 4893354.666667 | 5734400.000000 |
| all | ours25_minus_recompute | total_ssd_read_bytes | 6355626.666667 | 5896874.666667 | 6737920.000000 |
| all | ours25_minus_fullload | accuracy | 0.008333 | -0.029167 | 0.045833 |
| all | ours25_minus_fullload | ttft_ms | -9.955067 | -11.753900 | -7.840863 |
| all | ours25_minus_fullload | request_e2e_ms | -8.249433 | -10.998791 | -5.273112 |
| all | ours25_minus_fullload | visual_read_bytes | -13380266.666667 | -13991936.000000 | -12845056.000000 |
| all | ours25_minus_fullload | total_ssd_read_bytes | -13380266.666667 | -13991936.000000 | -12845056.000000 |
| hit | fullload_minus_recompute | accuracy | 0.000000 | 0.000000 | 0.000000 |
| hit | fullload_minus_recompute | ttft_ms | -38.176450 | -41.217848 | -34.834111 |
| hit | fullload_minus_recompute | request_e2e_ms | -38.884232 | -42.675887 | -34.802284 |
| hit | fullload_minus_recompute | visual_read_bytes | 22478848.000000 | 21561344.000000 | 23488102.400000 |
| hit | fullload_minus_recompute | total_ssd_read_bytes | 23683072.000000 | 22765568.000000 | 24692326.400000 |
| hit | ours25_minus_recompute | accuracy | 0.010000 | -0.035000 | 0.055000 |
| hit | ours25_minus_recompute | ttft_ms | -50.946197 | -54.407758 | -47.248545 |
| hit | ours25_minus_recompute | request_e2e_ms | -50.493174 | -54.698269 | -45.782417 |
| hit | ours25_minus_recompute | visual_read_bytes | 6422528.000000 | 5872025.600000 | 6881280.000000 |
| hit | ours25_minus_recompute | total_ssd_read_bytes | 7626752.000000 | 7076249.600000 | 8085504.000000 |
| hit | ours25_minus_fullload | accuracy | 0.010000 | -0.035000 | 0.055000 |
| hit | ours25_minus_fullload | ttft_ms | -12.769747 | -15.028567 | -10.044233 |
| hit | ours25_minus_fullload | request_e2e_ms | -11.608941 | -14.947730 | -7.899137 |
| hit | ours25_minus_fullload | visual_read_bytes | -16056320.000000 | -16790323.200000 | -15414067.200000 |
| hit | ours25_minus_fullload | total_ssd_read_bytes | -16056320.000000 | -16790323.200000 | -15414067.200000 |

Per-image visual token counts, full and selected chunk counts, valid/padded visual bytes, actual preads/spans, store sizes, and metadata activation are in `pilot_supplement.json`.

T1 request E2E is measured before VisionScoreCapture.__exit__; its score computation and KV capture clone occur after request E2E. The store writer runs after that. First-hit activation runs outside request timing. Therefore per-image total is sum of measured request E2E + score + clone + writer + one activation, with no overlapping subcomponents added.
