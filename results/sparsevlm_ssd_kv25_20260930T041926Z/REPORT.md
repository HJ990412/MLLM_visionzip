# SparseVLM-SSD-KV25 Probe3 / AllHead

Run: `/home/dblab/hj/mllm_v2/runs/sparsevlm_ssd_kv25_20260930T041926Z`. Independent raw audit: **PASS**.

공식 SparseVLM의 embedding rater와 causal full-context attention head/rater 평균을 SSD-resident cached KV에 적용한 fixed-budget adaptation이다. Probe3는 head [0,1,2] 근사이고 AllHead는 전체 heads를 읽고 평균한다. Progressive pruning/layer schedule/recycling 및 SparseVLM+ head-selection/위치 보정은 재현 범위에 포함되지 않는다.

## GQA: VALID

| 방법 | 점수 head 수 | 전체/Hit 정답률 | T1/Hit TTFT | 실제 content KV % | SSD MB/hit |
|---|---:|---:|---:|---:|---:|
| ReComp | 0 | 62.50 / 63.00 | 520.81 / 523.77 ms | 100.000 | 0.000 |
| FullLoad | 0 | 62.50 / 63.00 | 520.87 / 827.21 ms | 100.000 | 1165.073 |
| SparseVLM-SSD-KV25-Probe3 | 3 | 60.00 / 60.00 | 520.94 / 887.79 ms | 25.000 | 1196.502 |
| SparseVLM-SSD-KV25-AllHead | 32 | 61.67 / 62.00 | 520.67 / 901.50 ms | 25.000 | 1174.138 |
| Ours-KV25 | 0 | 57.08 / 56.50 | 521.13 / 309.93 ms | 25.000 | 319.501 |

| 방법 | Scoring K MB | 추가 selected-K MB | Selected-V MB | Structural/other MB | Total MB | Preads |
|---|---:|---:|---:|---:|---:|---:|
| ReComp | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 |
| FullLoad | 0.000 | 582.536 | 582.536 | 0.000 | 1165.073 | 64.00 |
| SparseVLM-SSD-KV25-Probe3 | 54.613 | 560.930 | 560.930 | 20.028 | 1196.502 | 157.54 |
| SparseVLM-SSD-KV25-AllHead | 582.536 | 0.000 | 571.574 | 20.028 | 1174.138 | 80.65 |
| Ours-KV25 | 0.000 | 149.737 | 149.737 | 20.028 | 319.501 | 65.00 |

단위 MB는 OS pread 반환 bytes / 1,000,000이다. FullLoad의 K는 additional-selected-K 열에 실제 읽은 canonical K로 계상하며 scoring K가 아니다. 구조 중복은 실제 이벤트에 한 번씩 포함된다. KV retention과 SSD read 비율은 다르다.

## MT-GQA-reconstructed: VALID

| 방법 | 점수 head 수 | 전체/Hit 정답률 | T1/Hit TTFT | 실제 content KV % | SSD MB/hit |
|---|---:|---:|---:|---:|---:|
| ReComp | 0 | 75.83 / 78.75 | 541.13 / 546.14 ms | 100.000 | 0.000 |
| FullLoad | 0 | 75.83 / 78.75 | 534.75 / 766.75 ms | 100.000 | 1194.957 |
| SparseVLM-SSD-KV25-Probe3 | 3 | 76.67 / 80.00 | 541.36 / 829.54 ms | 25.000 | 1230.572 |
| SparseVLM-SSD-KV25-AllHead | 32 | 75.83 / 78.75 | 534.35 / 868.66 ms | 25.000 | 1204.066 |
| Ours-KV25 | 0 | 74.17 / 76.25 | 534.50 / 301.02 ms | 25.000 | 327.575 |

| 방법 | Scoring K MB | 추가 selected-K MB | Selected-V MB | Structural/other MB | Total MB | Preads |
|---|---:|---:|---:|---:|---:|---:|
| ReComp | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 |
| FullLoad | 0.000 | 597.479 | 597.479 | 0.000 | 1194.957 | 64.00 |
| SparseVLM-SSD-KV25-Probe3 | 56.014 | 577.423 | 577.423 | 19.713 | 1230.572 | 150.60 |
| SparseVLM-SSD-KV25-AllHead | 597.479 | 0.000 | 586.874 | 19.713 | 1204.066 | 79.42 |
| Ours-KV25 | 0.000 | 153.931 | 153.931 | 19.713 | 327.575 | 65.00 |

단위 MB는 OS pread 반환 bytes / 1,000,000이다. FullLoad의 K는 additional-selected-K 열에 실제 읽은 canonical K로 계상하며 scoring K가 아니다. 구조 중복은 실제 이벤트에 한 번씩 포함된다. KV retention과 SSD read 비율은 다르다.

## 검증 및 비용 범위

| 항목 | 상태/측정 범위 |
|---|---|
| CPU TEST | PASS |
| GPU CORRECTNESS | PASS |
| PROJECTION REUSE | PASS |
| ALLHEAD K-READ REUSE | PASS |
| LEGACY/OURS/QWEN CPU | PASS |
| QWEN GPU | NOT RUN |
| Persistence/session | RO cache-hit reevaluation은 NOT_REMEASURED; fresh MT shared-bundle 비용만 per-image receipt에서 측정; AllHead base/Probe-only 생성비 분리는 NOT SEPARATELY MEASURED |
| Metadata activation | 요청 TTFT 밖; activation bytes/residency/time는 raw에서 별도 보존 |
| TTFT | 외부 request start → 첫 token materialization 및 CUDA sync; input preparation와 online scoring K read 포함 |
| Stage times | selector/actual-attention host/CUDA intervals는 겹칠 수 있어 TTFT로 합산하지 않음 |

| Phase / 방법 | Selector host ms/hit | Prefill Q/K/V calls/layer | Peak allocated MiB | Peak reserved MiB |
|---|---:|---:|---:|---:|
| smoke / ReComp | 0.000 | NOT RUN | 6912.99 | 7460.00 |
| smoke / FullLoad | 0.000 | NOT RUN | 6907.02 | 7460.00 |
| smoke / SparseVLM-SSD-KV25-Probe3 | 905.499 | 3 | 6907.02 | 7460.00 |
| smoke / SparseVLM-SSD-KV25-AllHead | 900.473 | 3 | 6907.02 | 7460.00 |
| smoke / Ours-KV25 | 0.000 | NOT RUN | 6907.02 | 7460.00 |
| gqa / ReComp | 0.000 | NOT RUN | 8179.17 | 9594.00 |
| gqa / FullLoad | 0.000 | NOT RUN | 8150.96 | 9594.00 |
| gqa / SparseVLM-SSD-KV25-Probe3 | 836.321 | 3 | 8150.96 | 9594.00 |
| gqa / SparseVLM-SSD-KV25-AllHead | 849.228 | 3 | 8150.96 | 9594.00 |
| gqa / Ours-KV25 | 0.000 | NOT RUN | 8150.96 | 9594.00 |
| mt / ReComp | 0.000 | NOT RUN | 8290.48 | 9600.00 |
| mt / FullLoad | 0.000 | NOT RUN | 8186.23 | 9600.00 |
| mt / SparseVLM-SSD-KV25-Probe3 | 777.557 | 3 | 8186.23 | 9600.00 |
| mt / SparseVLM-SSD-KV25-AllHead | 813.401 | 3 | 8186.23 | 9600.00 |
| mt / Ours-KV25 | 0.000 | NOT RUN | 8186.24 | 9600.00 |

## Paired 95% CI

| Dataset/scope | Pair A−B | Metric | Point | 95% CI | N / image clusters |
|---|---|---|---:|---|---:|
| smoke/all | SparseVLM-SSD-KV25-Probe3 − SparseVLM-SSD-KV25-AllHead | quality | 0.00000 | [0.00000, 0.00000] | 12 / 4 |
| smoke/all | SparseVLM-SSD-KV25-Probe3 − SparseVLM-SSD-KV25-AllHead | ttft_ms | -0.68491 | [-68.31990, 55.01531] | 12 / 4 |
| smoke/all | SparseVLM-SSD-KV25-Probe3 − SparseVLM-SSD-KV25-AllHead | ssd_mb | 17.60734 | [12.29892, 21.29920] | 12 / 4 |
| smoke/hit | SparseVLM-SSD-KV25-Probe3 − SparseVLM-SSD-KV25-AllHead | quality | 0.00000 | [0.00000, 0.00000] | 8 / 4 |
| smoke/hit | SparseVLM-SSD-KV25-Probe3 − SparseVLM-SSD-KV25-AllHead | ttft_ms | -1.46254 | [-101.09969, 82.66877] | 8 / 4 |
| smoke/hit | SparseVLM-SSD-KV25-Probe3 − SparseVLM-SSD-KV25-AllHead | ssd_mb | 26.41101 | [18.44838, 31.94880] | 8 / 4 |
| smoke/all | ReComp − Ours-KV25 | quality | -0.16667 | [-0.33333, 0.00000] | 12 / 4 |
| smoke/all | ReComp − Ours-KV25 | ttft_ms | 121.10071 | [100.54649, 144.69078] | 12 / 4 |
| smoke/all | ReComp − Ours-KV25 | ssd_mb | -215.30761 | [-218.10381, -212.51140] | 12 / 4 |
| smoke/hit | ReComp − Ours-KV25 | quality | -0.25000 | [-0.50000, 0.00000] | 8 / 4 |
| smoke/hit | ReComp − Ours-KV25 | ttft_ms | 180.44899 | [150.71033, 215.86779] | 8 / 4 |
| smoke/hit | ReComp − Ours-KV25 | ssd_mb | -322.96141 | [-327.15571, -318.76710] | 8 / 4 |
| smoke/all | FullLoad − Ours-KV25 | quality | -0.16667 | [-0.33333, 0.00000] | 12 / 4 |
| smoke/all | FullLoad − Ours-KV25 | ttft_ms | 394.89000 | [319.02591, 459.81650] | 12 / 4 |
| smoke/all | FullLoad − Ours-KV25 | ssd_mb | 562.03674 | [536.87091, 587.20256] | 12 / 4 |
| smoke/hit | FullLoad − Ours-KV25 | quality | -0.25000 | [-0.50000, 0.00000] | 8 / 4 |
| smoke/hit | FullLoad − Ours-KV25 | ttft_ms | 590.94231 | [476.86644, 687.28767] | 8 / 4 |
| smoke/hit | FullLoad − Ours-KV25 | ssd_mb | 843.05510 | [805.30637, 880.80384] | 8 / 4 |
| smoke/all | SparseVLM-SSD-KV25-Probe3 − Ours-KV25 | quality | -0.16667 | [-0.33333, 0.00000] | 12 / 4 |
| smoke/all | SparseVLM-SSD-KV25-Probe3 − Ours-KV25 | ttft_ms | 412.98744 | [390.96776, 431.64311] | 12 / 4 |
| smoke/all | SparseVLM-SSD-KV25-Probe3 − Ours-KV25 | ssd_mb | 587.28994 | [563.78436, 619.46812] | 12 / 4 |
| smoke/hit | SparseVLM-SSD-KV25-Probe3 − Ours-KV25 | quality | -0.25000 | [-0.50000, 0.00000] | 8 / 4 |
| smoke/hit | SparseVLM-SSD-KV25-Probe3 − Ours-KV25 | ttft_ms | 617.92898 | [586.24435, 647.55625] | 8 / 4 |
| smoke/hit | SparseVLM-SSD-KV25-Probe3 − Ours-KV25 | ssd_mb | 880.93491 | [845.67654, 929.20218] | 8 / 4 |
| smoke/all | SparseVLM-SSD-KV25-AllHead − Ours-KV25 | quality | -0.16667 | [-0.33333, 0.00000] | 12 / 4 |
| smoke/all | SparseVLM-SSD-KV25-AllHead − Ours-KV25 | ttft_ms | 413.67235 | [373.06852, 455.37687] | 12 / 4 |
| smoke/all | SparseVLM-SSD-KV25-AllHead − Ours-KV25 | ssd_mb | 569.68260 | [544.56047, 598.16892] | 12 / 4 |
| smoke/hit | SparseVLM-SSD-KV25-AllHead − Ours-KV25 | quality | -0.25000 | [-0.50000, 0.00000] | 8 / 4 |
| smoke/hit | SparseVLM-SSD-KV25-AllHead − Ours-KV25 | ttft_ms | 619.39152 | [559.57249, 679.92573] | 8 / 4 |
| smoke/hit | SparseVLM-SSD-KV25-AllHead − Ours-KV25 | ssd_mb | 854.52390 | [816.84070, 897.25338] | 8 / 4 |
| gqa/all | SparseVLM-SSD-KV25-Probe3 − SparseVLM-SSD-KV25-AllHead | quality | -0.01667 | [-0.03750, 0.00000] | 240 / 40 |
| gqa/all | SparseVLM-SSD-KV25-Probe3 − SparseVLM-SSD-KV25-AllHead | ttft_ms | -11.37650 | [-20.30960, -2.37377] | 240 / 40 |
| gqa/all | SparseVLM-SSD-KV25-Probe3 − SparseVLM-SSD-KV25-AllHead | ssd_mb | 18.63653 | [15.86425, 21.35202] | 240 / 40 |
| gqa/hit | SparseVLM-SSD-KV25-Probe3 − SparseVLM-SSD-KV25-AllHead | quality | -0.02000 | [-0.04500, 0.00000] | 200 / 40 |
| gqa/hit | SparseVLM-SSD-KV25-Probe3 − SparseVLM-SSD-KV25-AllHead | ttft_ms | -13.70567 | [-24.45774, -2.85856] | 200 / 40 |
| gqa/hit | SparseVLM-SSD-KV25-Probe3 − SparseVLM-SSD-KV25-AllHead | ssd_mb | 22.36383 | [19.03710, 25.62243] | 200 / 40 |
| gqa/all | ReComp − Ours-KV25 | quality | 0.05417 | [0.01250, 0.10000] | 240 / 40 |
| gqa/all | ReComp − Ours-KV25 | ttft_ms | 178.14566 | [162.83519, 193.17342] | 240 / 40 |
| gqa/all | ReComp − Ours-KV25 | ssd_mb | -266.25092 | [-274.66138, -257.31564] | 240 / 40 |
| gqa/hit | ReComp − Ours-KV25 | quality | 0.06500 | [0.01500, 0.12000] | 200 / 40 |
| gqa/hit | ReComp − Ours-KV25 | ttft_ms | 213.83971 | [195.44001, 231.95920] | 200 / 40 |
| gqa/hit | ReComp − Ours-KV25 | ssd_mb | -319.50111 | [-329.59365, -308.77876] | 200 / 40 |
| gqa/all | FullLoad − Ours-KV25 | quality | 0.05417 | [0.01250, 0.10000] | 240 / 40 |
| gqa/all | FullLoad − Ours-KV25 | ttft_ms | 431.02244 | [408.23102, 455.76016] | 240 / 40 |
| gqa/all | FullLoad − Ours-KV25 | ssd_mb | 704.64307 | [678.76946, 729.10985] | 240 / 40 |
| gqa/hit | FullLoad − Ours-KV25 | quality | 0.06500 | [0.01500, 0.12000] | 200 / 40 |
| gqa/hit | FullLoad − Ours-KV25 | ttft_ms | 517.27831 | [489.90472, 546.95291] | 200 / 40 |
| gqa/hit | FullLoad − Ours-KV25 | ssd_mb | 845.57169 | [814.52335, 874.93181] | 200 / 40 |
| gqa/all | SparseVLM-SSD-KV25-Probe3 − Ours-KV25 | quality | 0.02917 | [-0.01667, 0.07500] | 240 / 40 |
| gqa/all | SparseVLM-SSD-KV25-Probe3 − Ours-KV25 | ttft_ms | 481.52114 | [461.13466, 502.78159] | 240 / 40 |
| gqa/all | SparseVLM-SSD-KV25-Probe3 − Ours-KV25 | ssd_mb | 730.83372 | [704.43967, 755.88654] | 240 / 40 |
| gqa/hit | SparseVLM-SSD-KV25-Probe3 − Ours-KV25 | quality | 0.03500 | [-0.02000, 0.09000] | 200 / 40 |
| gqa/hit | SparseVLM-SSD-KV25-Probe3 − Ours-KV25 | ttft_ms | 577.86425 | [553.36213, 603.37592] | 200 / 40 |
| gqa/hit | SparseVLM-SSD-KV25-Probe3 − Ours-KV25 | ssd_mb | 877.00046 | [845.32760, 907.06385] | 200 / 40 |
| gqa/all | SparseVLM-SSD-KV25-AllHead − Ours-KV25 | quality | 0.04583 | [0.00000, 0.09583] | 240 / 40 |
| gqa/all | SparseVLM-SSD-KV25-AllHead − Ours-KV25 | ttft_ms | 492.89763 | [473.45401, 512.68717] | 240 / 40 |
| gqa/all | SparseVLM-SSD-KV25-AllHead − Ours-KV25 | ssd_mb | 712.19719 | [686.22729, 736.67849] | 240 / 40 |
| gqa/hit | SparseVLM-SSD-KV25-AllHead − Ours-KV25 | quality | 0.05500 | [0.00000, 0.11500] | 200 / 40 |
| gqa/hit | SparseVLM-SSD-KV25-AllHead − Ours-KV25 | ttft_ms | 591.56991 | [568.25761, 615.29722] | 200 / 40 |
| gqa/hit | SparseVLM-SSD-KV25-AllHead − Ours-KV25 | ssd_mb | 854.63663 | [823.47275, 884.01419] | 200 / 40 |
| mt/all | SparseVLM-SSD-KV25-Probe3 − SparseVLM-SSD-KV25-AllHead | quality | 0.00833 | [-0.01667, 0.03333] | 120 / 40 |
| mt/all | SparseVLM-SSD-KV25-Probe3 − SparseVLM-SSD-KV25-AllHead | ttft_ms | -23.74699 | [-37.87825, -11.56721] | 120 / 40 |
| mt/all | SparseVLM-SSD-KV25-Probe3 − SparseVLM-SSD-KV25-AllHead | ssd_mb | 17.67083 | [15.61556, 19.59192] | 120 / 40 |
| mt/hit | SparseVLM-SSD-KV25-Probe3 − SparseVLM-SSD-KV25-AllHead | quality | 0.01250 | [-0.02500, 0.05000] | 80 / 40 |
| mt/hit | SparseVLM-SSD-KV25-Probe3 − SparseVLM-SSD-KV25-AllHead | ttft_ms | -39.12181 | [-60.23538, -21.04658] | 80 / 40 |
| mt/hit | SparseVLM-SSD-KV25-Probe3 − SparseVLM-SSD-KV25-AllHead | ssd_mb | 26.50624 | [23.42334, 29.38787] | 80 / 40 |
| mt/all | ReComp − Ours-KV25 | quality | 0.01667 | [-0.02500, 0.06667] | 120 / 40 |
| mt/all | ReComp − Ours-KV25 | ttft_ms | 165.62576 | [158.43522, 173.50005] | 120 / 40 |
| mt/all | ReComp − Ours-KV25 | ssd_mb | -218.38343 | [-224.44769, -213.47260] | 120 / 40 |
| mt/hit | ReComp − Ours-KV25 | quality | 0.02500 | [-0.03750, 0.10000] | 80 / 40 |
| mt/hit | ReComp − Ours-KV25 | ttft_ms | 245.12540 | [234.33175, 257.07346] | 80 / 40 |
| mt/hit | ReComp − Ours-KV25 | ssd_mb | -327.57514 | [-336.67154, -320.20890] | 80 / 40 |
| mt/all | FullLoad − Ours-KV25 | quality | 0.01667 | [-0.02500, 0.06667] | 120 / 40 |
| mt/all | FullLoad − Ours-KV25 | ttft_ms | 310.56972 | [296.94666, 325.15801] | 120 / 40 |
| mt/all | FullLoad − Ours-KV25 | ssd_mb | 578.25471 | [563.15522, 595.03193] | 120 / 40 |
| mt/hit | FullLoad − Ours-KV25 | quality | 0.02500 | [-0.03750, 0.10000] | 80 / 40 |
| mt/hit | FullLoad − Ours-KV25 | ttft_ms | 465.73191 | [445.31359, 487.65047] | 80 / 40 |
| mt/hit | FullLoad − Ours-KV25 | ssd_mb | 867.38207 | [844.73283, 892.54789] | 80 / 40 |
| mt/all | SparseVLM-SSD-KV25-Probe3 − Ours-KV25 | quality | 0.02500 | [-0.01667, 0.07500] | 120 / 40 |
| mt/all | SparseVLM-SSD-KV25-Probe3 − Ours-KV25 | ttft_ms | 354.63130 | [339.73612, 370.35756] | 120 / 40 |
| mt/all | SparseVLM-SSD-KV25-Probe3 − Ours-KV25 | ssd_mb | 601.99813 | [585.32777, 620.68560] | 120 / 40 |
| mt/hit | SparseVLM-SSD-KV25-Probe3 − Ours-KV25 | quality | 0.03750 | [-0.02500, 0.11250] | 80 / 40 |
| mt/hit | SparseVLM-SSD-KV25-Probe3 − Ours-KV25 | ttft_ms | 528.52083 | [506.11573, 552.12146] | 80 / 40 |
| mt/hit | SparseVLM-SSD-KV25-Probe3 − Ours-KV25 | ssd_mb | 902.99720 | [877.99166, 931.02840] | 80 / 40 |
| mt/all | SparseVLM-SSD-KV25-AllHead − Ours-KV25 | quality | 0.01667 | [-0.02500, 0.06667] | 120 / 40 |
| mt/all | SparseVLM-SSD-KV25-AllHead − Ours-KV25 | ttft_ms | 378.37830 | [357.61518, 401.10328] | 120 / 40 |
| mt/all | SparseVLM-SSD-KV25-AllHead − Ours-KV25 | ssd_mb | 584.32730 | [568.68645, 601.87818] | 120 / 40 |
| mt/hit | SparseVLM-SSD-KV25-AllHead − Ours-KV25 | quality | 0.02500 | [-0.03750, 0.10000] | 80 / 40 |
| mt/hit | SparseVLM-SSD-KV25-AllHead − Ours-KV25 | ttft_ms | 567.64264 | [536.46584, 601.78266] | 80 / 40 |
| mt/hit | SparseVLM-SSD-KV25-AllHead − Ours-KV25 | ssd_mb | 876.49096 | [853.02967, 902.81727] | 80 / 40 |

Bootstrap은 image-cluster 10,000회, seed=1234이며 같은 image의 모든 dialogue/turn과 paired methods를 함께 resample하고 request-weighted point estimate와 동일 분모를 사용한다. CI가 0을 포함해도 동등성 증거가 아니다. MT 차이는 method별 generated history의 영향까지 포함한다.

## 최종 판정

- IMPLEMENTATION PROBE3 / ALLHEAD: PASS
- SOURCE-METRIC FIDELITY: strict-> embedding raters 및 post-softmax head/rater mean 확인; fixed KV25, SSD online retrieval, FP32 reduction, stable ties는 의도한 adaptation.
- GQA 5-ARM PILOT: VALID
- MT 5-ARM PILOT: VALID
- READY FOR LLAVA MT-GQA BASELINE INTEGRATION: YES
- Readiness 근거: all correctness, projection/I/O, protection and full pilot gates passed

구체적인 checks/failures와 source/hash/protection 상태는 `independent_audit.json`, 요청별 내역은 원 run의 phase/raw.jsonl, 통계는 `summary.csv`와 `paired_comparisons.csv`에 보존한다. 이 보고서는 미측정을 PASS로 승격하지 않는다.

## Controlled head-policy diagnostic (outside timing)

| Source policy | Layers | Mean score correlation | Mean Top-k overlap | Mean Jaccard | Probe3 / AllHead chunks |
|---|---:|---:|---:|---:|---:|
| all | 384 | 0.834 | 0.742 | 0.597 | 35.685 / 36.443 |
| fixed_first_3 | 384 | 0.835 | 0.741 | 0.596 | 35.693 / 36.453 |

H2D payload subtotal과 rater visual H2D는 별도 필드다. sysKV/IDs 등 미계측 transfer를 포함한 total H2D로 쓰지 않는다. Actual attention host/CUDA intervals, projection counts, I/O and memory는 `io_timing_memory_breakdown.csv`, setup receipt 비용은 `setup_costs.csv`를 참조한다.
