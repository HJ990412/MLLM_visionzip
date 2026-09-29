# v2 mt_gqa_reconstructed pilot supplement

The primary report and frozen raw data remain unchanged. This audit verifies per-image SSD geometry and disjoint session timing against the primary summary.

All image-cluster intervals use 4,000 draws and seed 1234. Intervals containing zero do not establish equivalence.

| Scope | Left minus right | Metric | Mean difference | 95% low | 95% high |
|---|---|---|---:|---:|---:|
| all | fullload_minus_recompute | accuracy | 0.000000 | 0.000000 | 0.000000 |
| all | fullload_minus_recompute | ttft_ms | -24.585246 | -28.120705 | -20.761093 |
| all | fullload_minus_recompute | request_e2e_ms | -24.549076 | -28.329902 | -20.745949 |
| all | fullload_minus_recompute | visual_read_bytes | 15903402.666667 | 15290204.160000 | 16515072.000000 |
| all | fullload_minus_recompute | total_ssd_read_bytes | 16706218.666667 | 16093020.160000 | 17317888.000000 |
| all | ours25_minus_recompute | accuracy | -0.025000 | -0.075000 | 0.025000 |
| all | ours25_minus_recompute | ttft_ms | -33.324502 | -36.632221 | -29.553494 |
| all | ours25_minus_recompute | request_e2e_ms | -32.951660 | -37.013462 | -28.631437 |
| all | ours25_minus_recompute | visual_read_bytes | 4709853.866667 | 4465186.133333 | 4893354.666667 |
| all | ours25_minus_recompute | total_ssd_read_bytes | 5512669.866667 | 5268002.133333 | 5696170.666667 |
| all | ours25_minus_fullload | accuracy | -0.025000 | -0.075000 | 0.025000 |
| all | ours25_minus_fullload | ttft_ms | -8.739256 | -11.882563 | -5.820623 |
| all | ours25_minus_fullload | request_e2e_ms | -8.402583 | -11.622818 | -5.148329 |
| all | ours25_minus_fullload | visual_read_bytes | -11193548.800000 | -11744051.200000 | -10643046.400000 |
| all | ours25_minus_fullload | total_ssd_read_bytes | -11193548.800000 | -11744051.200000 | -10643046.400000 |
| hit | fullload_minus_recompute | accuracy | 0.000000 | 0.000000 | 0.000000 |
| hit | fullload_minus_recompute | ttft_ms | -35.528815 | -40.268931 | -30.130364 |
| hit | fullload_minus_recompute | request_e2e_ms | -34.549567 | -39.601562 | -29.049275 |
| hit | fullload_minus_recompute | visual_read_bytes | 23855104.000000 | 22935306.240000 | 24772608.000000 |
| hit | fullload_minus_recompute | total_ssd_read_bytes | 25059328.000000 | 24139530.240000 | 25976832.000000 |
| hit | ours25_minus_recompute | accuracy | -0.037500 | -0.112500 | 0.037500 |
| hit | ours25_minus_recompute | ttft_ms | -50.805402 | -55.102097 | -45.842458 |
| hit | ours25_minus_recompute | request_e2e_ms | -50.510429 | -55.862484 | -44.606617 |
| hit | ours25_minus_recompute | visual_read_bytes | 7064780.800000 | 6697779.200000 | 7340032.000000 |
| hit | ours25_minus_recompute | total_ssd_read_bytes | 8269004.800000 | 7902003.200000 | 8544256.000000 |
| hit | ours25_minus_fullload | accuracy | -0.037500 | -0.112500 | 0.037500 |
| hit | ours25_minus_fullload | ttft_ms | -15.276587 | -19.753318 | -11.149981 |
| hit | ours25_minus_fullload | request_e2e_ms | -15.960862 | -20.825047 | -11.094915 |
| hit | ours25_minus_fullload | visual_read_bytes | -16790323.200000 | -17616076.800000 | -15964569.600000 |
| hit | ours25_minus_fullload | total_ssd_read_bytes | -16790323.200000 | -17616076.800000 | -15964569.600000 |

Per-image visual token counts, full and selected chunk counts, valid/padded visual bytes, actual preads/spans, store sizes, and metadata activation are in `pilot_supplement.json`.

T1 request E2E is measured before VisionScoreCapture.__exit__; its score computation and KV capture clone occur after request E2E. The store writer runs after that. First-hit activation runs outside request timing. Therefore per-image total is sum of measured request E2E + score + clone + writer + one activation, with no overlapping subcomponents added.
