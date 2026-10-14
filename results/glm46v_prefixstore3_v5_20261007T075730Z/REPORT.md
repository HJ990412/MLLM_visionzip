# GLM-4.6V-Flash PrefixStore-KV25 [B] 최종 결과

**요청·KV·raw 감사 PASS; 완전한 outer setup endpoint는 미측정.** MT-GQA-reconstructed 398 images,
4,061 dialogues × 3 turns × 3 methods의 **36,549/36,549 final requests**를
실행했다. 각 방법은 12,183 요청과 8,122 후속 요청을 완료했다. 최종 v5의
실패 attempt·재시도는 **0**이며 모든 이미지의 첫 완전한 attempt를 채택했다.
GQA 파일럿 720/720, MT 파일럿 360/360도 완료했다. 기존 LLaVA/Qwen main은
재실행하지 않았고 remote push·기존 artifact 삭제·전역 의존성 변경은 없다.

아래 session은 `write_glm_store` call-return 및 activation core 측정 범위다.
`runner.persist`의 identity 준비와 writer wrapper-return tail, activation summary/return tail은
**NOT_MEASURED**이므로 이를 완전한 outer setup/session 실측으로 해석하지 않는다.
수정된 caller outer timer를 적용한 별도 v6를 최종 계약 실행으로 사용한다.
v5 원래 보고서는 REPORT_BEFORE_OUTER_SCOPE_REVIEW.md에 byte 그대로 보존한다.

Native BF16 KV가 **40 KiB/token**으로 작은 GLM에서도 B는 효율 이득을 유지했다.
후속 TTFT는 ReComp 대비 **52.11%**, FullLoad 대비 **17.53%** 짧고,
FullLoad 대비 실제 KV read는 **71.03%**, 이미지 평균 logical store는
**70.86%** 작았다. 그 대신 후속 정답률은 각각 **3.546 pp**, **3.460 pp**
낮았다. 두 정확도 CI 모두 0을 포함하지 않으므로 정확도가 동등하다는 결과가 아니다.

## 최종 계약과 환경

모델·processor·tokenizer는 `zai-org/GLM-4.6V-Flash`,
revision `411bb4d77144a3f03accbf4b780f5acb8b7cde4e`다.
RTX4090 24GB, decoder만 NF4 4-bit/double quantization, BF16 compute/native KV,
SDPA, batch1, eval/inference, seed1234, greedy를 사용했다. Vision Encoder·downsample·
merger와 lm_head는 BF16 비양자화이고 CPU/disk weight offload는 없다.
실제 NF4 decoder Linear는 240개, vision NF4는 0개였다.

별도 `.venv-glm46v`의 Transformers 5.13.1과 기존 mllm_ft의 읽기 전용
Torch 2.5.1+cu121, bitsandbytes 0.49.2를 사용했다. Transformers 공식 tag commit은
`4626421dc6b741a329300682a6408246ee465490`이며 설치본 GLM 파일 4개가 공식 소스와
byte SHA256까지 일치했다. 자세한 pin과 소스 근거는 [PORT_REPORT.md](PORT_REPORT.md),
[환경](environment.json), [source provenance](transformers_source_provenance.json)에 있다.

Native `Glm46VProcessor`의 size는 shortest_edge=12,544,
longest_edge=1,605,632다. GLM의 temporal 반복 2를 포함한 native 면적 정책으로
최대 1,024 content tokens를 제한했고 품질 결과를 보고 해상도를 변경하지 않았다.
실제 main 이미지의 N은 126–529였다. 1,024-token/긴 질문 stress의 실제 GPU peak는
allocated 8.318 GiB, reserved 8.699 GiB로 메모리에 적합했다.

Native chat template에는 `enable_thinking=False`를 명시한다. 최종 common cap은
**32**로, 기존 Qwen/LLaVA의 max16과 다르다. GLM의 native final-answer boundary
151361/151362 안의 span을 scoring prediction으로 decode하고, strict normalized
EM/gold[0] scorer는 유지한다. History는 해당 방법의 실제 generated IDs에서 마지막
native EOS 한 개만 제외한 body다. Box/internal controls/whitespace를 보존하고
native template가 직렬화한다. Gold나 다른 방법의 history는 사용하지 않는다.

Main ordered workload SHA256은
`0287e0c57813800c781633b969c5cff336b3a3c1a1bdcdbb56d63f6ddab0ca62`로 일치했다.
이는 재구성 workload이며 공식 MetaCompress 원본 benchmark 재현을 주장하지 않는다.
계약·고정값·membership은 [CONTRACT.md](CONTRACT.md), [freeze.json](freeze.json),
[manifests/](manifests/)에 있다.

## 품질

아래 정확도는 요청 가중 strict normalized EM, 단위 %다. T1은 항상 세 arm 모두
full pixels이며 4,061개 dialogue의 T1 generated IDs가 **100% 동일**했다.
T2/T3 비교에는 각 방법의 실제 생성 history 차이가 포함된다.

| 방법 | T1 | T2 | T3 | 전체 | T2–T3 |
|---|---:|---:|---:|---:|---:|
| ReComp | 60.379 | 62.374 | 60.896 | 61.216 | 61.635 |
| FullLoad | 60.379 | 62.251 | 60.847 | 61.159 | 61.549 |
| Ours-PrefixStore-KV25 [B] | 60.379 | 58.532 | 57.646 | 58.852 | 58.089 |

원래 139,485 content rows 중 35,048개를 저장해 pooled content 보존율은
**25.1267%**, image-macro 보존율은 25.1304%다. 각 이미지의 k는 정확히
ceil(original N/4)다. 구조 19행/이미지는 별도로 전부 보존한다.
원래 N·geometry·전체 permutation·logical IDs·native 3축 positions/partial RoPE를
유지하며 선택한 post-RoPE K를 다시 회전하지 않는다. T2/T3에는 frozen first-k만
실제 SSD에서 읽고 query scoring·재선택·decode 중 재읽기는 0이다.

Native malformed/empty/cap/stray delimiter를 이유로 요청을 제외하거나 재시도하지
않았다. Main에서 thinking open/close token 및 빈 답변은 세 arm 모두 0건이다.
Cap 도달/깨진 box는 ReComp 3건, FullLoad 3건, B 2건으로 모두 분모에 포함했다.
[native_output_audit.csv](reporting_details/native_output_audit.csv)에 실제 counts가 있다.

## 요청 latency와 실제 읽기

아래는 T2–T3 요청 가중 통계다. 시간은 ms, MB=10^6 bytes다.
TTFT는 input/history 준비·pixels의 image open/decode 이전부터 첫 native
생성 token materialization/CUDA sync까지다. 대다수 첫 token은 GLM answer-boundary
control이다. 첫 lexical answer latency는 별도로 **NOT_MEASURED**다.
Source T1의 필요한 clone/importance tail은 outer request E2E에 정확히 한 번 포함했다.

| 방법 | TTFT mean/p50/p95 | E2E mean/p50/p95 | 실제 KV MB/request | pread/request |
|---|---:|---:|---:|---:|
| ReComp | 126.512 / 127.261 / 148.743 | 215.046 / 199.860 / 309.094 | 0.000 | 0 |
| FullLoad | 73.468 / 73.544 / 76.630 | 162.323 / 143.804 / 255.870 | 15.180 | 81 |
| B | 60.588 / 60.471 / 62.923 | 147.961 / 130.732 / 223.497 | 4.397 | 81 |

FullLoad/B 후속 실제 OS returned bytes 총합은 각각 **123,289,927,680**,
**35,712,204,800 bytes**다. 각 arm의 pread는 657,882 calls이고 content/structure/
padding 구간을 독립 검산했다. H2D는 실제 native tensor bytes를 기록했다.
ReComp의 KV read 0은 이미지 파일 I/O까지 0이라는 뜻이 아니다. OS read/write는
NAND 물리 I/O가 아니고 DONTNEED 성공은 cold NAND 보장이 아니다.

T1/T2/T3/all/followup mean/p50/p95와 image-macro 값은
[latency.csv](final/latency.csv)에 있다. FullLoad와 ReComp는 다른 projection/attention
shape에서 native first logits가 달라질 수 있다. GPU gate의 6 사례는 first-logit
최대 차이 0.211–0.3203이었지만 생성 IDs는 같았다. Main 정확도 차이는 -0.057 pp다.
이는 저장/복원 bits 오류 및 B의 압축 손실과 구분한다.

## 이미지 setup·저장·3-turn session

Store 및 setup는 이미지 평균, session은 dialogue/request 가중 평균이다.
Persistence는 full CPU materialization/repack/serialization/write/fsync/seal을 포함한
`write_glm_store` call-return 측정 범위다. Identity 준비/runner wrapper tail은
NOT_MEASURED다. Activation 기록은 integrity/read core를 포함하지만 summary/return tail은
NOT_MEASURED다. 아래 setup/session 값과 회계 PASS는 이 기록된 범위에 한정된다.

| 방법 | logical/allocated MB/image | Persistence | Activation | A 실제 공유-stream session | B standalone-equivalent |
|---|---:|---:|---:|---:|---:|
| ReComp | KV store 없음 | KV setup 없음 | KV activation 없음 | 641.610 | 641.610 |
| FullLoad | 15.170 / 15.504 | 63.376 | 12.274 | 543.524 | 611.924 |
| B | 4.420 / 4.739 | 52.415 | 7.252 | 513.407 | 567.432 |

A는 각 이미지/방법의 실제 build·activation을 한 번만 귀속한다. B는 같은 측정 setup을
각 dialogue에 재부과한 **DERIVED** 값으로 독립 cold-new-image session 실측이 아니다.
Non-source T1 capture 증가분은 **NOT_MEASURED**다. 실제 796 builds/796 activations의
총 setup **53,855.8786 ms**와 session 귀속 총합이 정확히 일치했다.

두 arm 모두 full CPU KV materialized 평균 15.133 MB, full content repacked 평균
14.355 MB를 유지했다. B는 serializing 단계에서만 k content rows를 기록한다.
FullLoad/B의 CPU materialization은 3.433/3.453 ms, repack은 2.958/3.043 ms,
실제 write component는 4.276/1.450 ms, fsync는 40.307/36.725 ms였다.
이 nested 값들을 outer wall에 다시 더하지 않았다. Native prefix clone은
FullLoad 0.558 ms, B 0.541 ms, B 중요도는 1.529 ms였고 source E2E에 포함했다.

All-layer/native-head BF16 token당 40,960 bytes를 실제 tensor/file bytes와 대조했다.
64-row token-major layout의 마지막 파일은 정확히 k행에서 끝나고 content/structure
payload padding은 0이다. Filesystem allocation slack은 위 allocated bytes에 포함한다.
Actual logical store 총합 FullLoad 6,037,819,615, B 1,759,348,714 bytes를 보존했다.
대용량 raw/store는 로컬이며 삭제하지 않았다. Cache saturation/hit-rate/eviction 이득은
이 image-major 실행에서 검증하지 않았다.

## Paired 분석

Seed1234, **398 이미지 단위 paired cluster bootstrap 10,000회**다. 같은 이미지의
모든 방법·dialogue·turn을 함께 resample하고 pooled sum/count로 요청 가중치를 유지했다.
아래는 B−baseline point [95% percentile CI]다.

| 비교 | T2–T3 정확도 pp | T2–T3 TTFT ms | T2–T3 E2E ms | KV read MB/request |
|---|---:|---:|---:|---:|
| B−ReComp | -3.546 [-4.416, -2.692] | -65.925 [-67.218, -64.665] | -67.085 [-68.625, -65.536] | +4.397 [4.319, 4.475] |
| B−FullLoad | -3.460 [-4.328, -2.597] | -12.880 [-13.028, -12.721] | -14.363 [-15.181, -13.504] | -10.783 [-11.018, -10.547] |

| 비교 | A 실제 공유-stream session ms | B derived standalone session ms | logical store MB/image |
|---|---:|---:|---:|
| B−ReComp | -128.203 [-131.310, -125.069] | -74.179 [-77.022, -71.325] | +4.420 [4.357, 4.483] |
| B−FullLoad | -30.117 [-31.767, -28.396] | -44.493 [-46.306, -42.620] | -10.750 [-10.938, -10.560] |

전체 정확도 차이는 B−ReComp -2.364 pp [-2.944, -1.795], B−FullLoad
-2.306 pp [-2.885, -1.731]다. 이는 생성 history 차이를 포함한 방법 수준 비교다.
CI가 0을 포함할 때도 동등성으로 해석하지 않는다. 전체 scope/setup CI는
[paired_analysis.csv](final/paired_analysis.csv)에 있다.

## Gate·이전 실패·독립 재현

최종 CPU suite **82/82 PASS**, GPU 입장 CPU gate **75/75 PASS**다.
Native GPU 6이미지의 stock/capture vision output·T1 KV/logits/IDs,
FullLoad payload/restore, B의 독립 동일-shape memory reference vs SSD bits/IDs/
positions/mask/logits/IDs, generated native history와 state isolation이 통과했다.
480개 all-layer FP64 dense/compact 비교 최대차는 7.105e-15로 사전 1e-12 기준을
통과했다. 별도 FP32 shape 진단 FAIL 2개 layer는 그대로 보존했고 main BF16/SDPA를
변경하거나 tolerance를 완화하지 않았다.

완료 후 독립 stdlib audit가 796 stores/64,476 payload hashes를 확인했다. B의
**2,803,840개 K/V token 행**을 FullLoad 원래 byte offsets와 대조해 모두 일치했다.
구조·원래 N·EOF·위치·identity·frozen 선택·모든 실제 followup pread도 PASS다.
기존 필수 source/report 25파일의 before/after full SHA256이 같았다.

이전 실행은 최종 분모에 합치지 않았다.

- v3 첫 GPU 입장은 vision 양자화 제외 경로 오류로 FAIL, 추론 전 중단했다. 수정 후
  gate를 통과하고 두 파일럿 1,080요청을 완료했으나 native box extraction 누락 때문에
  literal strict EM이 0%였다. Raw/FAIL/freeze를 유지했다.
- v3 sealed validation 디렉터리의 진단 sidecar는 12개의 reactivation FAIL을 만들었다.
  Full hash·atomic allowlist 이후 해당 새 run-owned 파일만 안전하게 이동해 12/12
  재활성화/복원 PASS를 확인했다. 원래 실패와 source-guard FAIL도 보존했다.
- v4는 native 최종 답변 parsing/cap32를 적용해 두 파일럿 1,080요청을 완료했다.
  Scoring 답변을 history로 사용하는 포팅 누락 때문에 main333요청 뒤 중단했고,
  failed image의 raw21+실행된 anomaly1을 채택하지 않았다. 공식 generate와 수동 경로의
  고정 입력 결과가 같음을 진단했다. v5의 native-body history는 새 run으로 실행했다.

[VALIDATION.md](VALIDATION.md), [독립 검토](VALIDATION_REVIEW.md),
[store audit receipt](store_audit/attempt_20261007T102733501663Z/receipt.json),
[raw evidence index](final/raw_evidence_index.json), [재현](REPRODUCE.md),
[추가 계측](reporting_details/receipt.json)에 경로·hash·실제 명령·범위가 있다.

이 결과는 **GLM의 작은 KV에서도 효율 개선이 유지되지만 KV25의 정확도 손실을
동반한다**는 근거다. 독립 cold session, cache hit-rate·eviction 또는 원 논문의
GLM VisionZip/MetaCompress 구현 재현으로 확대해서 주장하지 않는다.
