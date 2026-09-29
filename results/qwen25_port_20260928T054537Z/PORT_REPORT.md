# Qwen2.5-VL-7B-Instruct image-only Visual KV 포팅 결과

**상태: 포팅과 CPU 검증은 완료했으나 GPU 수치 정합성 검증은 실패했다. 아래 pilot은 명시적인 진단 실행 결과이며, 검증된 성능·품질 benchmark 결과로 해석할 수 없다.** 실행일: 2026-09-28. 모든 수치는 저장한 원시 기록에서 계산했다.

| 항목 | 상태 | 근거 |
|---|---|---|
| IMPLEMENTATION | PASS | Qwen 전용 adapter, 마지막 ViT attention 점수, native BF16 GQA KV store/repack, MRoPE cache reuse, 검증·pilot·집계 코드 구현 |
| CPU TEST | PASS | Qwen 19/19, 저장소 전체 351/351 |
| GPU CORRECTNESS | FAIL | SDPA 15개 gate 중 13 PASS, FullLoad 및 Prefix25 logits 근접성 2 FAIL |
| GQA PILOT | FAIL | 검증된 pilot 조건 미충족. 별도 진단 실행 40 images × 6 questions × 3 methods = 720 requests 및 80 stores 완료, raw audit PASS |
| MT PILOT | FAIL | 검증된 pilot 조건 미충족. 별도 진단 실행 40 images × 3 turns × 3 methods = 360 requests 및 80 stores 완료, raw audit PASS |
| LLAVA REGRESSION | PASS | 전체 CPU suite 통과; 보호 대상 8,883 files, 60,901,511,649 bytes SHA256 변경 없음 |

## 1. 구현 범위와 실행 환경

새 구현은 `mmimpress/qwen25/{vision,store,runner}.py`, `scripts/{77_protect_qwen25_artifacts,78_validate_qwen25,79_eval_qwen25_pilot,80_report_qwen25_pilot}.py`, Qwen CPU tests, `docs/qwen25_port_contract.md`에 있다. 기존 LLaVA 코드·store 포맷은 수정하지 않았다. GLM/Qwen3, MPIC/ReKV/QA-Chunk 포팅, 대규모 benchmark, budget sweep은 수행하지 않았다. 공식 MetaCompress 점수도 산출하지 않았다.

모델·processor·tokenizer를 모두 `Qwen/Qwen2.5-VL-7B-Instruct` revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`로 고정했다. 단일 NVIDIA RTX 4090(25,391,333,376 bytes), NF4 4-bit weight·BF16 compute·double quantization, 358개 quantized modules(언어 196, vision 162), 전부 `cuda:0`, weight offload 없음. Torch 2.5.1+cu121, Transformers 4.57.6, bitsandbytes 0.49.2, Accelerate 1.14.0이며 전체 lock은 `runs/qwen25_port_20260928T054537Z/dependency_lock.txt`에 있다. SDPA를 vision·decoder와 ReComp/FullLoad/Ours25 전부에 동일하게 적용했다. batch=1, seed=1234, greedy, 최대 16 new tokens, checkpoint EOS `[151643,151645]`, processor `min_pixels=200704`, `max_pixels=802816`이다. Pilot 중 다른 GPU compute process는 관찰되지 않았다.

[VisionZip Qwen 원본](https://github.com/JIA-Lab-research/VisionZip/blob/8f86b55c6f000eb033e6912538af2dd7dcb30502/Qwen2_5_VL/qwen2_5vl_visionzip.py)의 commit은 `8f86b55c6f000eb033e6912538af2dd7dcb30502`, 파일 SHA256은 `26f828971ac9d4058768e07d9cb5c329b792d13206c64e6bf66f5e4b8c026004`이다. 모델 공식 [checkpoint](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/tree/cc594898137f460bfe9f0759e9844b3ce807cfb5)를 참조했다. 로컬 `.git` 디렉터리에 HEAD/remote가 없어 저장소 commit ID는 기록할 수 없으며, source 파일 SHA256을 대신 보존했다. 동결된 adapter 해시는 `runner.py be63757f...`, `store.py 9dff059b...`, `vision.py de2cdf62...`이고 전체 값은 validation 및 pilot source hashes에 있다.

## 2. Image-only 점수와 Qwen cache 처리

마지막 vision block 31이 실제 full-attention block `[7,15,23,31]`에 속한다. 정상적인 T1 multimodal forward 한 번에서 그 block의 ViT Q/K만 hook으로 얻는다. `mean_head(sum_query(softmax_key(QKᵀ/√d)))`로 **각 key가 받은 attention**을 계산한다. query-row block마다 전체 유효 key에 대해 softmax하므로 query sampling이나 부분 key 정규화가 없다. 점수를 `spatial_merge_size²=4` patch group별 평균으로 만들고 `argsort(window_index)`로 LLM merged-token 순서에 되돌린다. Qwen 원래 PatchMerger와 full visual-token 입력은 유지하며, 점수는 SSD 물리적 순서에만 쓴다. 현재 질문·정답·이전 답변은 ranking에 쓰지 않는다. Score 계산은 T1 응답 이후 persistence에 포함한다. GPU dense-vs-block score 최대 절댓값 차이 `9.54e-7`, stable permutation exact 일치였다.

실제 decoder는 28 layers, 28 query heads, **4 native KV heads**, 128 head dimension이다. 저장 payload는 모든 layer의 native `[visual,4,128]` K/V BF16 raw bits이고 FP16 변환이 없다. 1 visual token의 전체 layer K+V는 `28×2×4×128×2=57,344` bytes이다. 64-token chunk의 layer별 **K 또는 V 한 파일** 크기는 65,536 bytes이며, 전 layer K+V 합계는 3,670,016 bytes(3.5 MiB)이다.

재사용 prefix는 고정 system/user header부터 `vision_end`까지이다. 검증 이미지에서 prefix 366 tokens = visual 345 + structural 21이고, 현재 질문은 suffix다. 이미지 hash, pinned checkpoint, processor 설정, prefix IDs, grid, 위치 정책, dtype, code/env revision으로 store identity를 확인한다. 모델이 MRoPE를 적용한 **뒤** cache에 쓴 K/V를 그대로 저장하므로 hit 시 rotary를 재적용하지 않는다. `get_rope_index`의 full logical 3축 MRoPE 위치와 compact cache slot을 별도로 유지한다. 검증 예시에서 suffix logical position은 `[44,44,44]`이고 25% compact 첫 slot은 149, dense 참조는 366이다. 요청마다 새 DynamicCache·mask·rope state를 만들고 GPU/CPU visual KV payload를 다음 요청에 보존하지 않는다. 원본 순서의 structural KV는 budget 밖에 두며 hit당 1,204,224 bytes를 별도 읽는다.

전 visual KV를 score 내림차순, 동점은 원래 index 순으로 한 번 재배열하여 모든 K/V layer에 같은 permutation으로 직접 저장한다. Ours25는 저장 전체 크기는 FullLoad와 같고, hit에서는 `budget_chunk_count`의 nominal 25% chunk를 각 K/V 파일의 선두 연속 span으로 읽는다. 한 hit은 visual 56 `pread` + structural 1 `pread`이다. Activation의 full-file hash 검사·metadata read는 첫 activation에 따로 기록하고 hit마다 수행하지 않는다. 25%가 보존 용량 축소라는 주장은 하지 않는다.

## 3. GPU 정합성 결과

사전 계약은 정확한 mapping·hash·BF16 round-trip에 bitwise equality, BF16 logits에 `atol=0.125`, `rtol=0.02`였다. 실패 후 tolerance나 backend를 바꾸지 않았다. Frozen SDPA `validation.json`에서 13 gate PASS, 2 gate FAIL이다.

| 검증 | 결과 | 실제 증거 |
|---|---|---|
| Geometry/score/capture/query independence | PASS | 345 score = 345 PatchMerger output = 345 expanded image IDs = 345 Visual-KV rows; 3질문의 prefix KV bitwise 동일; source vision forward 1회, 추가 prefix forward 0, hit vision forward 0 |
| Native BF16 SSD round-trip/inverse permutation | PASS | canonical·repacked file SHA 및 BF16 bits exact, 전 prefix position 및 역 mapping exact |
| FullLoad SSD vs 같은 captured prefix의 in-memory replay | PASS | 3질문의 첫 logits 최대 절댓값 차이 `0, 0, 0` |
| FullLoad SSD vs fresh pixel ReComp | **FAIL** | 차이 `0.1328125, 0.4375, 0.4375`; 후자 2개가 predeclared `allclose` FAIL |
| RepackedFull100 vs canonical FullLoad | PASS | 3질문의 첫 logits 차이 `0, 0, 0` |
| Prefix25 compact vs 같은 selected set의 dense masked 참조 | **FAIL** | 차이 `0.125, 0.15625, 0.3125`; 후자 2개가 `allclose` FAIL. Selected indices·full logical positions·cache slots의 구조 검사는 PASS |
| I/O, history, request isolation | PASS | hit first-k만 read; 잘못된 image identity 거부; 다른 method/gold history 유입 없음 |

위 비교에서 세 질문의 첫 token과 전체 생성 token sequence는 모두 같았다. 출력 일치와 logits 근접성은 별개다. SSD serialization/repack 자체는 차이가 관측되지 않았다. 수치 차이의 **원인은 아직 확정되지 않았다**. Full pixel prefill과 suffix cache replay가 서로 다른 SDPA mask/shape 경로를 쓰고, dense masked cache와 compact cache의 key 길이도 다르므로 BF16/NF4 수치 경로 차이가 가능한 설명이다. 그러나 mask·position 구현의 미발견 문제도 배제할 수 없다. Eager 별도 진단은 차이를 줄이지 못했고 추가 causal-isolation 문제를 보였으므로 pilot backend로 채택하지 않았다. 이 상태에서 `pilot_eligible=false`이며 유효 성능/품질 결론은 낼 수 없다.

## 4. 진단 pilot의 데이터와 KV 크기

GQA는 기존 `frozen-gqa40-q5to10` 40-image × 6-question의 image/question/gold/order를 사용했다. Q2~Q6은 각각 독립 질문이며 history는 비어 있다. MT-GQA reconstructed index에서 고정한 서로 다른 40 image의 완전한 3-turn 대화 40개를 사용했고, 각 방법의 생성 답변만 자기 T2/T3 history에 넣었다. GQA manifest content SHA256 `383d0b4f...`, MT `70cc3c04...`이며 full 값과 원시 ID는 각 `manifest.json`에 있다. 요청은 GQA 720, MT 360으로 빠짐없고, 각각 canonical 40 + repacked 40 stores가 있다. 두 report의 raw audit가 PASS다.

| Pilot | merged visual tokens/image (min / median / mean / max) | 원본 visual KV/image, padding 전 평균 | SSD visual payload/image, padding 포함 평균 | Ours hit kept tokens 평균 / 범위 | Ours actual kept ratio 평균 | Ours visual read/hit | FullLoad visual read/hit |
|---|---:|---:|---:|---:|---:|---:|---:|
| GQA | 266 / 345 / 349 / 484 | 20,013,056 B | 22,478,848 B | 112 / 64–128 | 31.73% | 6,422,528 B | 22,478,848 B |
| MT | 266 / 356.5 / 370.475 / 529 | 21,244,518 B | 23,855,104 B | 123.2 / 64–128 | 33.40% | 7,064,781 B | 23,855,104 B |

평균 원본 KV bytes는 token당 57,344 bytes를 평균 token 수에 곱했으며 MT는 반올림했다. GQA resized image는 예컨대 420×644일 때 grid `[1,30,46]`, merger 전 1,380 patches에서 merger 후 345 tokens였다. 가장 흔한 token 수는 GQA에서 345(12 images)·391(10), MT에서 345(17)·391(15)이다. 전체 이미지의 실제 resized shapes와 grids는 raw JSONL의 `result.geometry`에 있다. Ours visual payload read ratio는 GQA 28.27%, MT 29.75%이고, structural bytes를 포함한 total payload read ratio는 32.00%, 33.18%다. FullLoad는 padding 포함 모든 visual bytes를 읽는다. Nominal 25%와 실제 token 비율의 차이는 64-token chunk rounding/clamping 때문이다.

## 5. 진단 관측치: quality, TTFT, I/O

**다음 값은 GPU correctness 실패 때문에 검증된 benchmark 결과가 아니다.** Quality는 GQA의 기존 `exact_score`, MT의 기존 history runner와 같은 strict normalized exact 기준으로 계산했다. T1은 모든 방법의 정상 pixel inference다. 아래 hit은 GQA 방법당 200건, MT 방법당 80건이고, TTFT는 첫 출력 token CUDA 동기화까지다. 원시 생성은 최대 16 tokens이며 이 pilot에서 16-token truncation은 0건이었다.

| Pilot / method | all quality | hit quality | hit TTFT mean / p50 / p95 (ms) | hit request E2E mean (ms) | visual + structural + metadata read/hit | pread calls/hit |
|---|---:|---:|---:|---:|---:|---:|
| GQA ReComp | 56.67% | 59.00% | 110.11 / 109.88 / 127.41 | 130.02 | 0 + 0 + 0 B | 0 |
| GQA FullLoad | 56.67% | 59.00% | 71.07 / 69.39 / 84.82 | 91.18 | 22,478,848 + 1,204,224 + 0 B | 57 |
| GQA Ours25 | 57.50% | 60.00% | 57.51 / 55.26 / 69.81 | 77.57 | 6,422,528 + 1,204,224 + 0 B | 57 |
| MT ReComp | 68.33% | 70.00% | 115.26 / 114.58 / 132.17 | 136.63 | 0 + 0 + 0 B | 0 |
| MT FullLoad | 68.33% | 70.00% | 75.51 / 73.73 / 86.19 | 97.18 | 23,855,104 + 1,204,224 + 0 B | 57 |
| MT Ours25 | 65.83% | 66.25% | 61.65 / 59.88 / 69.05 | 82.26 | 7,064,781 + 1,204,224 + 0 B | 57 |

Ours25의 hit-only quality 차이는 GQA에서 FullLoad/ReComp 대비 **+1.0 percentage point**, MT에서 **−3.75 points**다. All-turn 차이는 각각 +0.83 points, −2.50 points이다. GQA hit에서 Ours 답변은 200개 중 36개 달라졌고 10개는 오답→정답, 8개는 정답→오답이었다. MT hit에서는 80개 중 12개가 달라졌고 2개가 오답→정답, 5개가 정답→오답이었다. FullLoad와 ReComp의 pilot prediction은 같은 sample끼리 모두 일치했다. MT turn별 quality는 ReComp/FullLoad `T1 65%, T2 65%, T3 75%`, Ours `T1 65%, T2 62.5%, T3 70%`이다. Hit 평균 생성 token 수는 GQA ReComp/FullLoad/Ours `2.265/2.265/2.270`, MT `2.3625/2.3625/2.3125`이다. GQA 각 Q1~Q6의 결과는 `summary.csv`에 있다.

Image-cluster paired bootstrap 4,000 draws에서 Ours−FullLoad hit TTFT는 GQA `−13.56 ms [−15.42,−11.58]`, MT `−13.86 ms [−16.23,−11.76]`였다. Ours−ReComp는 GQA `−52.60 ms [−55.08,−50.02]`, MT `−53.61 ms [−56.21,−50.92]`였다. Ours−FullLoad visual read bytes/hit은 GQA `−16,056,320 B`, MT `−16,790,323 B`; structural/metadata hit read 차이는 0이다. Hit quality 차이의 95% CI는 GQA `+1.0 pp [−3.5,+5.5]`, MT `−3.75 pp [−11.3,+2.5]`이다. 작은 subset이고 CI가 0을 포함한다는 사실은 품질 동등성 증거가 아니다. 더구나 수치 정합성 gate가 FAIL이므로 이 CI로 방법의 효과를 확정하지 않는다.

## 6. 시간 귀속과 짧은 사용 세션

Score는 T1 응답 후 계산하여 persistence에 넣었다. Ours score 추가 계산은 GQA 평균 **1.659 ms**, MT **1.847 ms**이고, score 중 추가 peak GPU allocated memory는 평균 약 43.3 MB/46.0 MB였다. T1 KV clone은 0.367/0.359 ms, repack은 0.827/0.846 ms, write는 9.60/10.14 ms, fsync는 33.08/33.90 ms, **총 persistence는 70.88/74.06 ms**다. FullLoad 총 persistence는 71.97/71.20 ms다. Ours T1 TTFT−ReComp의 image-paired 관측 차이는 GQA `−1.969 ms [95% CI −5.277,+1.376]`, MT `+0.105 ms [−3.162,+3.057]`였다. Score/clone은 응답 후 persistence에 귀속되므로 이를 capture의 인과적인 T1 TTFT 비용으로 단정하지 않는다.

GQA에서 6개 독립 요청/image의 request E2E 합 + 해당 arm의 일회 persistence·activation 평균은 ReComp **782.81 ms**, FullLoad **669.48 ms**, Ours25 **602.80 ms**였다. 이는 연결된 6-turn 대화 session이 아니다. MT의 실제 3-turn generated-history session E2E는 같은 방식으로 ReComp **409.05 ms**, FullLoad **412.39 ms**, Ours25 **388.39 ms**였다. Ours가 MT ReComp보다 관측상 20.66 ms 짧지만, GPU 수치 gate 실패 때문에 유효한 session 이득으로 주장할 수 없다. GQA/MT Ours activation 비용은 평균 13.31/13.96 ms이며, 최초 hash 검사에서 전체 visual·structural·metadata를 평균 23.72/25.10 MB 읽었다. 이는 hit TTFT 밖, session 비용 안이다. Host immutable metadata resident bytes는 평균 95,572/99,430 B다.

Hit에서 Ours/FullLoad의 raw `pread`는 GQA 11.90/21.62 ms, MT 11.73/21.75 ms이고, `store_load_inclusive`는 14.80/25.86 ms, 14.74/26.01 ms다. H2D+GPU assembly는 합쳐서 GQA 1.97/3.50 ms, MT 2.05/3.66 ms로 측정했으며 각각의 별도 시간은 측정하지 않았다. Suffix prefill은 GQA 39.91/40.86 ms, MT 43.90/44.87 ms다. 두 저장 arm 모두 hit당 반환 `pread` 57회와 읽기 span 57개(visual 56 + structural 1)를 기록했다. Fixed first-k planning은 Ours 0.029/0.031 ms, online selector와 online query-score calls는 0이다. Component들은 포함·겹침 관계가 있으므로 단순 합으로 TTFT를 재구성하지 않는다. Cache-hit peak GPU allocated 평균은 GQA Ours/FullLoad 약 6.097/6.110 GB, MT 약 6.103/6.118 GB이고, reserved는 양쪽 모두 약 7.92/7.93 GB다. 25% GPU 메모리 절감이라고 해석하지 않는다.

이미지 파일 읽기/JPEG RGB decode, model loading, 공통 warmup, 기록, `posix_fadvise(DONTNEED)`는 TTFT 밖이다. ReComp processor 이미지 전처리·vision/full visual prefill은 TTFT 안이다. `DONTNEED` 호출 실패는 0건이었으나 이것은 OS page-cache hint일 뿐 SSD controller/NAND cold나 O_DIRECT 보장은 아니다. 모든 arm은 동일 SDPA backend이고 method 순서를 이미지별 deterministic rotation했다.

## 7. 보존·한계·다음 단계

보호 전·후 manifest 비교는 `unchanged=true`, 보호 파일 8,883개/60,901,511,649 bytes, unexpected path 0이었다. 기존 source/data/store/results와 LLaVA 기본 동작을 보존했다. New Qwen raw·store는 `runs/qwen25_port_20260928T054537Z/`, report는 이 `results/qwen25_port_20260928T054537Z/` 아래 별도로 남겼다. commit/push는 하지 않았다.

우선순위는 고정된 한 실패 질문에서 (1) fresh pixel 경로와 cache 경로의 expanded input IDs 및 3축 position IDs, (2) 실제 decoder causal masks, (3) layer별 suffix hidden states/attention outputs를 비교해 처음 갈라지는 층을 찾는 것이다. Dense masked와 compact도 같은 방식으로 대조한다. 원인을 찾아 동일한 사전 tolerance를 만족한 뒤 GPU validation을 새 artifact로 다시 수행해야 한다. 그 후에만 두 pilot을 검증된 결과로 재실행·발표할 수 있다. BF16 허용치를 사후 확대하거나 기존 진단 raw를 재라벨링하지 않는다.

## 8. 재현 경로와 명령

모든 명령은 `/home/dblab/hj/mllm_v2`에서 `mllm_ft` Python으로 실행한다. 동결 검증 명령은 exit code 1을 반환하는 것이 현재 정확한 결과다. Pilot은 정상 gate에서 fail-closed이며 명시적 `--diagnostic-after-numerical-fail`만 현재 진단 실행을 허용한다. 새 실행은 새로운 run/results 경로를 사용해야 기존 raw를 보존한다.

```bash
PY=/home/dblab/anaconda3/envs/mllm_ft/bin/python
TAG=$(date -u +%Y%m%dT%H%M%SZ)
V=runs/qwen25_port_validate_gpu_${TAG}
R=runs/qwen25_port_${TAG}
O=results/qwen25_port_${TAG}
$PY -m unittest discover -s tests -q
$PY scripts/78_validate_qwen25.py --attn sdpa --out-dir "$V"
$PY scripts/79_eval_qwen25_pilot.py --dataset gqa --validation "$V/validation.json" --diagnostic-after-numerical-fail --run-dir "$R/gqa_diagnostic_sdpa"
$PY scripts/80_report_qwen25_pilot.py --run-dir "$R/gqa_diagnostic_sdpa" --results-dir "$O/gqa_diagnostic_sdpa" --diagnostic-after-numerical-fail
$PY scripts/79_eval_qwen25_pilot.py --dataset mt --validation "$V/validation.json" --diagnostic-after-numerical-fail --run-dir "$R/mt_diagnostic_sdpa"
$PY scripts/80_report_qwen25_pilot.py --run-dir "$R/mt_diagnostic_sdpa" --results-dir "$O/mt_diagnostic_sdpa" --diagnostic-after-numerical-fail
$PY scripts/77_protect_qwen25_artifacts.py verify runs/qwen25_port_20260928T054537Z/protected_before.json runs/qwen25_port_20260928T054537Z/protected_after.json
```

기존 경로에 같은 명령을 그대로 재실행하면 파일 충돌 방지가 작동할 수 있다. 새 run/results 디렉터리를 지정하라. 주요 증거는 `runs/qwen25_port_validate_gpu_20260928_0625_sdpa_frozen/validation.json`, 각 run의 `manifest.json`, `runtime.json`, `raw.jsonl`, `persistence.jsonl`, `gpu_inventory.jsonl`, `final_status.json`, 각 report의 `summary.csv`, `sessions.csv`, `persistence.csv`, `summary.json`, `validation_evidence.json`, `ANALYSIS.md`에 있다.
