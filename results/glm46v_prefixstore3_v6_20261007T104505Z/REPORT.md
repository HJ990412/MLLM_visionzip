# GLM-4.6V-Flash PrefixStore-KV25 [B] 최종 계약 실행 v6

MT-GQA-reconstructed **398 images, 4,061 dialogues × 3 turns × 3 arms = 36,549/36,549
final requests**를 실제 RTX4090에서 완료했다. 방법당 12,183 요청/8,122 followups이며
최종 v6의 실패 attempt·재시도는 **0**이다. GQA 파일럿 720/720 및 MT 파일럿 360/360도
완료했다. Main 완료 receipt, raw 분석과 독립 store 감사는 PASS다.

기존 `qwen_mt_gqa_prefixstore3_20261002T074856Z`의 완료된 3-arm main과 이전 G7 BLOCKED
실행을 구분했다. 그 Qwen/LLaVA main 및 원래 결과는 재실행·수정하지 않았다.

Native KV가 **40 KiB/token**으로 작은 GLM에서도 B의 비용 이득을 확인했다.
후속 TTFT는 ReComp 대비 **52.12%**, FullLoad 대비 **17.56%** 감소했고,
FullLoad 대비 실제 KV read는 **71.03%**, 이미지 평균 logical store는 **70.86%** 감소했다.
후속 strict EM은 ReComp보다 **3.546 pp**, FullLoad보다 **3.460 pp** 낮았다.
두 정확도 CI가 모두 0을 제외하므로 정확도를 유지하거나 동등하다는 결과로 해석하지 않는다.

이번 실행은 **caller가 runner.persist/activate를 호출하기 직전부터 완전히 반환할 때까지**
setup wall time을 측정한다. Identity 준비·wrapper/summary 반환 tail을 포함하고 inner timer는
별도 diagnostic으로 남겼다. 이전 v5의 inner 범위 setup은 보존된 별도 실행이며 v6 분모에
합치지 않았다. 최종 inference/model/processor/history/importance/해상도는 v5와 동일하다.

## 고정 환경·워크로드·포팅

모델/processor/tokenizer는 `zai-org/GLM-4.6V-Flash`, checkpoint revision
`411bb4d77144a3f03accbf4b780f5acb8b7cde4e`다. 전용 `.venv-glm46v`에서
Torch 2.5.1+cu121, Transformers 5.13.1, bitsandbytes 0.49.2를 사용했다.
Transformers 공식 commit은 `4626421dc6b741a329300682a6408246ee465490`이며
설치된 GLM 소스 4개가 공식 tag와 byte SHA256까지 일치했다. 실제 model class는
Glm4vForConditionalGeneration, processor는 Glm46VProcessor다.

RTX4090 24GB, decoder NF4 4-bit/double quantization, BF16 compute/native KV,
SDPA, batch1, eval/inference, seed1234, greedy를 세 arm에 동일하게 적용했다.
Decoder NF4 Linear는 240개이고 vision NF4는 0개다. Vision Encoder·downsample·merger·
lm_head는 BF16 비양자화이며 CPU/disk weight offload는 없다. 실제 KV는
40 layers × 2 native KV heads × 128 head_dim × K/V × BF16 = **40,960 bytes/token**이다.
Query attention head 수 32로 KV를 확장하지 않았다.

Native processor의 `size={shortest_edge:12544,longest_edge:1605632}`를 main 이전에 고정했다.
Temporal 반복 2를 포함한 GLM native 면적 규칙의 최대 content token은 1,024다.
실제 main N은 126–529, N=1,024/긴 질문 stress의 peak allocated/reserved는
8.318/8.699 GiB였다. 기존 Qwen 옵션을 복사하거나 품질 관찰 후 해상도를 변경하지 않았다.

`enable_thinking=False`, tools 없음, 공통 **max_new_tokens=32**다. 출발점 cap16은
v3 정상 추론 점검에서 native close-box 이전 절단을 만들었으므로 main 이전에 32로 고정했다.
기존 LLaVA/Qwen cap16과 다르다. Native final-answer box span을 scoring prediction으로
추출하고 기존 strict normalized exact match와 gold[0] 기준을 유지한다. History는 각 방법의
실제 generated IDs에서 마지막 native EOS 한 개만 제외한 body다. Native box/내부 control/
whitespace를 보존하고 공식 chat template로 직렬화한다. Gold나 다른 방법의 history는 없다.

Main ordered workload SHA256은 계약의
`0287e0c57813800c781633b969c5cff336b3a3c1a1bdcdbb56d63f6ddab0ca62`와 일치했다.
고정 membership, image/source dialogue identity, original ordinal modulo3 rotation을 사용했다.
모든 dialogue의 T1은 full pixels이며 각 이미지의 고정 source T1에서만 각 cached arm의 store를
만들었다. 다른 동일 이미지 dialogue에도 T1 full inference를 실행했다.

Vision은 CLS/window attention이 없는 native 24-block/12-head 구조다. 마지막 block의
post-softmax received attention을 head 평균·query 합으로 집계하고 native 2×2 merge group을
통해 final visual raster rows에 연결했다. Query block128에서도 모든 유효 key를 softmax 분모에
포함한다. Stock vision 출력은 보존한다. 이 방식은 **GLM용 VisionZip-style adaptation**이며
원 논문의 GLM 구현을 재현했다는 주장이 아니다.

Full T1 prefill에서 native reusable prefix KV와 vision 자료를 함께 capture한다. 별도 vision/
LLM prefix forward는 없다. 구조 19행은 항상 유지한다. B는 full CPU materialization과
full importance repack 이후 직렬화 단계에서만 `k=ceil(original N/4)`를 직접 기록한다.
전체 permutation/mapping/geometry/native 3축 position/partial RoPE/offset을 보존한다.
Post-RoPE K를 재회전하지 않고 compact 물리 길이를 원래 logical 위치로 취급하지 않는다.
T2/T3에는 metadata/FD만 상주하며 frozen first-k를 실제 pread로 읽는다.

[CONTRACT](CONTRACT.md), [PORT_REPORT](PORT_REPORT.md), [환경](environment.json),
[공식 소스 provenance](official_source_provenance.json), [freeze](freeze.json)에 근거가 있다.
공식 MetaCompress 원본 benchmark나 GLM VisionZip 구현 재현을 주장하지 않는다.

## 정답률과 생성 history

아래는 요청 가중 strict EM %, T1/T2/T3 각 4,061개, 전체 12,183개, followup 8,122개다.

| 방법 | T1 | T2 | T3 | 전체 | T2–T3 |
|---|---:|---:|---:|---:|---:|
| ReComp | 60.379 | 62.374 | 60.896 | 61.216 | 61.635 |
| FullLoad | 60.379 | 62.251 | 60.847 | 61.159 | 61.549 |
| Ours-PrefixStore-KV25 [B] | 60.379 | 58.532 | 57.646 | 58.852 | 58.089 |

4,061개 dialogue의 T1 generated IDs는 세 arm 간 **4,061/4,061 동일**했다.
후속 비교는 각 방법이 만든 실제 자기 history 차이를 포함한 방법 수준 비교다.
All image-macro EM은 ReComp/FullLoad/B **61.920/61.869/59.544%**, followup image-macro는
**62.173/62.097/58.610%**다. 요청 가중 값과 image-macro를 혼합하지 않는다.

원래 content139,485행 중 B35,048행, pooled 보존율 **25.1267%**, image-macro
**25.1304%**다. 구조 19행/이미지는 이 분모에서 제외하며 전부 보존했다.
후속 선택·query scoring·decode 중 반복 store read는 재실행하지 않았다.

Empty 및 생성 think-open/think-close는 세 arm 모두 **0건**이다. Cap 도달/깨진 box는
ReComp3/FullLoad3/B2건으로 **모두 품질 분모에 포함**했고 좋은 출력까지 재시도하지 않았다.
각 방법 12,183개 중 12,176개의 첫 native token은 answer boundary였다.
TTFT는 첫 lexical answer가 아니라 첫 native generated token까지다.
First lexical answer latency는 **NOT_MEASURED**다.

파일럿은 GQA 방법당240요청의 EM ReComp/FullLoad/B **59.167/58.750/56.667%**,
MT 방법당120요청의 EM **63.333/63.333/60.000%**였다. 두 파일럿의 raw/selection/history/
I/O/setup audit는 PASS다. 품질 하락을 correctness 실패로 처리하거나 membership을 재선택하지 않았다.

## 요청 시간과 OS 읽기

아래는 followup 요청 가중 값, ms와 MB=10^6 bytes다. TTFT는 input/history/chat 준비 및
pixels image open/decode 이전부터 첫 token materialization/CUDA synchronization까지다.
Cached request의 pread/H2D/assembly/suffix prefill을 포함한다. Request E2E는 같은 outer return
끝점이며 source T1 clone/importance tail을 정확히 한 번 포함했다.

| 방법 | TTFT mean/p50/p95 | E2E mean/p50/p95 | 실제 KV MB/request | 실제 cached pread/request |
|---|---:|---:|---:|---:|
| ReComp | 126.523 / 127.200 / 149.011 | 214.964 / 199.836 / 308.486 | 0.000 | KV reader 없음 |
| FullLoad | 73.477 / 73.457 / 76.761 | 162.196 / 143.652 / 255.452 | 15.180 | 81 |
| B | 60.578 / 60.448 / 63.024 | 147.772 / 130.488 / 223.044 | 4.397 | 81 |

FullLoad/B의 실제 OS returned bytes 총합은 **123,289,927,680/35,712,204,800 bytes**,
각각657,882 pread calls다. 모든 range의 content/structure/padding 반환값을 독립 대조했다.
H2D native tensor bytes도 동일하며 H2D mean FullLoad/B **1.949/1.018 ms**, assembly
**0.577/0.559 ms**다. ReComp KV read0은 이미지 파일 I/O0을 뜻하지 않는다.
ReComp 이미지 I/O bytes와 cached H2D/assembly 등의 없는 계측값을 0으로 채우지 않는다.
미계측 필드는 CSV에서 **NOT_MEASURED**다.

FullLoad와 ReComp의 서로 다른 projection/attention shape에서 native first-logit 수치 차이는
별도로 진단했다. GPU gate 6사례의 maxabs 차이는0.211–0.3203, 생성 IDs는 같았다.
Main FullLoad−ReComp all EM 차이는−0.057 pp다. 이는 canonical 저장/복원 bits 오류와
B 압축 품질 손실을 구분해 해석한다.

T1/T2/T3/all/followup의 mean/p50/p95, image-macro, component는
[latency.csv](final/latency.csv)에 있다. OS I/O는 NAND 물리 I/O가 아니고,
DONTNEED 성공은 cold NAND 보장이 아니다. Conditioning·benchmark guards·raw/evidence journal
fsync·image adoption·post-run audit는 정의한 serving request/setup 끝점 밖이며 이를 0으로 주장하지 않는다.

## Outer setup·저장량·3-turn session

Store/setup은 image-macro 평균, session은 dialogue/request 가중 평균이다.
Persistence caller outer는 identity 준비부터 full CPU materialize/repack/serialization/actualwrite/
fsync/seal·등록·wrapper return까지다. Activation caller outer는 integrity read/metadata/FD/
summary return까지다. Nested inner/component를 outer에 다시 더하지 않았다.

| 방법 | logical/allocated MB/image | Persistence ms | Activation ms | A 실제 공유 stream session ms | B standalone-equivalent ms |
|---|---:|---:|---:|---:|---:|
| ReComp | KV store 없음 | KV setup 없음 | KV activation 없음 | 641.398 | 641.398 |
| FullLoad | 15.170 / 15.504 | 65.756 | 12.214 | 543.477 | 614.079 |
| B | 4.420 / 4.739 | 54.687 | 7.219 | 513.202 | 569.204 |

A는 이미지/방법의 build·activation을 source dialogue에 **한 번** 귀속한다.
B는 실제 측정 setup을 각 dialogue에 재부과한 **DERIVED** 값이다. 독립 cold-new-image session
실측이 아니며 non-source T1 capture 추가 비용은 **NOT_MEASURED**다.
796builds+796activations의 실제 outer setup **55,670.386903 ms**와 A 귀속 합이 같았다.
Outer build47,936.413759 ms/activation7,733.973144 ms, 각각 nested inner45,846.786648/
7,692.572124 ms를 포함한다. Inner와 outer의 차이를 추정하거나 이전 v5 값에 덧붙이지 않았다.

FullLoad/B 모두 full CPU materialized 평균 **15.133 MB**, full content repacked 평균
**14.355 MB**다. CPU peak는 이 논리 bytes와 다르며 별도 **NOT_MEASURED**다.
Materialize3.412/3.404 ms, repack2.911/2.987 ms, actualwrite4.266/1.407 ms,
fsync40.204/36.632 ms는 nested component다. Native GPU prefix clone은0.554/0.533 ms,
B importance1.478 ms로 source E2E에 포함했다. B Q/K capture8.613 MB/copy0.00949 ms,
importance sampled-live extra18.427 MB와 hook CPU0.729 ms를 기록했다. Sampled live는
모든 allocator transient의 독립 peak가 아니다.

후속 request allocated mean ReComp/FullLoad/B **8.196/8.174/8.154 GiB**, reserved mean
**8.375/8.375/8.375 GiB**다. Request마다 reset한 allocator peak이며 model/coexistingstate를
포함한다. 전체 scope/p50/p95는 추가 계측 CSV에 있다.

Token-major contiguous/native BF16 64-row chunk의 content 파일은 정확히 k행에서 끝난다.
Content/structure payload padding은0이며 filesystem slack은 allocated에 별도 포함한다.
실제 logical 총합 FullLoad/B **6,037,819,615/1,759,348,714 bytes**, OS write 및 activation
integrity returned bytes도 각 총합과 같다. 각 build의 OS write는83calls다.
Completion 이후 free disk124.284 GiB, GPU 외부 compute process0을 확인했다.
모든 새 raw/store를 보존했다. Cache saturation/hit-rate/eviction 효과를 측정했다고 주장하지 않는다.

## 이미지 단위 paired cluster bootstrap

Seed1234,398 이미지,10,000회 percentile95% CI다. 같은 이미지의 모든 방법·dialogue·turn을
함께 resample하고 pooled sum/count를 사용했다. 아래는 **B−baseline point [CI]**다.

| 비교 | Followup EM pp | TTFT ms | E2E ms | KV read MB/request |
|---|---:|---:|---:|---:|
| B−ReComp | −3.546 [−4.416, −2.692] | −65.945 [−67.250, −64.669] | −67.192 [−68.742, −65.638] | +4.397 [4.319, 4.475] |
| B−FullLoad | −3.460 [−4.328, −2.597] | −12.899 [−13.052, −12.735] | −14.423 [−15.239, −13.577] | −10.783 [−11.018, −10.547] |

| 비교 | A shared stream session ms | B derived standalone session ms | Logical store MB/image |
|---|---:|---:|---:|
| B−ReComp | −128.196 [−131.311, −125.063] | −72.194 [−75.030, −69.327] | +4.420 [4.357, 4.483] |
| B−FullLoad | −30.275 [−31.919, −28.600] | −44.875 [−46.658, −43.031] | −10.750 [−10.938, −10.560] |

All EM의 B−ReComp는−2.364 pp [−2.944,−1.795], B−FullLoad는−2.306 pp
[−2.885,−1.731]이다. Accuracy CI0 포함 여부를 동등성으로 해석하지 않는다.
T1/T2/T3/all/followup/setup/session 총54개 추정치와 CI는
[paired_analysis.csv](final/paired_analysis.csv)에 있다.

## 실제 gate·실패 보존·독립 증거

전체 CPU suite **86/86 PASS**, GPU admission CPU **75/75 PASS**다. Native GPU6이미지에서
stock/capture vision 출력·T1 all-layer/native-head KV bits·첫 logits·전체 IDs가 exact같았다.
FullLoad canonical bits 및 B 독립 동일-shape memory reference와 SSD bits/원래 ID/구조 ID/
3축 position/실제 mask visibility/firstlogits/generatedIDs, generated suffix/decode 증가,
같은 이미지의 multiple-dialogue state isolation을 통과했다. 잘못된 identity/short/incomplete
store/doublebudget/structure/position/미래 suffix/history leakage negative fixtures도 통과했다.

Operations 이전 FP64 변환을 사용하는 dense masked/compact attention **480비교**의 maxabs는
**7.105e−15**, 사전 atol/rtol1e−12를 통과했다. FP32 shape 진단의 고정2layer FAIL은 그대로
남겼으며 BF16/SDPA main precision이나 tolerance를 바꾸지 않았다. Native history regression은
문제가 드러난 n204894의 actualbody 입력에서 공식 model.generate와 수동 경로의 prompt/IDs/
firstlogitsbits/allgeneratedIDs/nativepositions/cachelen이 exact같았다.

완료 후 독립 stdlib store 감사는 **796stores/64,476payload hashes/2,803,840 selected K-or-V
token rows**를 FullLoad 원래 byte offset과 대조해 PASS했다. Budget/EOF/padding0/native KV/
position/identity/selection/실제 모든 pread와1592callerouter setup events를 검사했다.
기존 기준 자료25파일과 최종 frozen source20파일의 hash 보존 증거도 남겼다.

독립 raw 재검산은 전체36,549요청의 prediction/body/nativeprompt/inputIDs/positions/history/
selection/실제 OS read를 대조했고, 54개 paired point/CI를 별도로 재계산해 numeric residual0이었다.
[FINAL_ANALYSIS_REVIEW](FINAL_ANALYSIS_REVIEW.md)와
[독립 receipt](independent_review/attempt_20261007T130724092714Z/receipt.json)에 증거가 있다.
기존 Qwen CPU 검사는 원래 Transformers4.57.6 환경의 unittest **18/18 PASS**였다.
그 환경의 pytest 미설치 시도 및 GLM의 Transformers5.13.1 환경에서 기존 Qwen 테스트
**17PASS/1FAIL**도 로그에 보존했다. 해당 FAIL은 vision return-type 가정의 버전 차이이며
Qwen source나 원 환경을 수정하지 않았다. GLM gate86/75PASS와 구분한다.
원래 환경의 [18검사 PASS 로그](../../runs/glm46v_prefixstore3_20261007T065207Z/cpu_validation/qwen_original_environment_unittest.log),
[새 라이브러리에서의 FAIL 로그](../../runs/glm46v_prefixstore3_20261007T065207Z/cpu_validation/qwen_regression.log),
[pytest 미설치 로그](../../runs/glm46v_prefixstore3_20261007T065207Z/cpu_validation/qwen_original_environment_regression.log)를 보존했다.

이전 revision은 최종 v6 분모에 합치지 않았다.

- v3 첫 NF4 admission은 vision exclusion 경로 오류로 추론 전 FAIL했다. model.visual로 수정한
  새 admission은 PASS했다. Literal box scoring EM0 파일럿과 cap16 truncation raw를 보존했다.
- v3 sealed validation store 안 진단 sidecar의12reactivation FAIL은 fullhash/atomicallowlist
  이동 이후12/12복원 PASS로 확인했으며 원래 실패/source-guard FAIL을 보존했다.
- v4 cap32/nativeanswer parsing 파일럿은 완료했으나 scoring span을 history로 넣은 포팅 누락으로
  main333채택 요청 뒤 중단했다. Failedimage raw21+실행 anomaly1은 final에서 제외했다.
  고정 입력에서 manual/officialgenerate가 같았고 실제 boxed body history에서 정상 생성했다.
- v5는 actual generated body로36,549요청을 완료했다. 마지막 endpoint 감사에서 identity/
  wrapper tail의 complete outer setup이 NOT_MEASURED임을 확인해 QUALIFIED로 명시했다.
  Raw·결과·원 보고서 snapshot을 보존하고 같은 inference의 새 v6를 전부 실행했다.

현재 v6 output anomaly는 제외하지 않았다. 이전 smoke/validation/pilot/partial/v5 요청도
v6 main에 합치지 않았다. 기존 LLaVA/Qwen main 재실행·기존 artifact 삭제·remote push·전역
dependency 업그레이드는 없다.

[VALIDATION](VALIDATION.md), [gate 검토](VALIDATION_REVIEW.md),
[독립 분석 검토](FINAL_ANALYSIS_REVIEW.md),
[store audit receipt](store_audit/attempt_20261007T130619344841Z/receipt.json),
[raw evidence](final/raw_evidence_index.json), [selection](selection_manifest.json),
[local artifact index](local_artifact_index.json), [실패 index](failure_and_attempt_index.json),
[추가 계측](reporting_details/receipt.json), [재현](REPRODUCE.md)에 실제 경로·hash·명령이 있다.
대용량 raw/store/weights는 로컬에 있고 재생성에는 preserved pinned checkpoint와 fresh gate가 필요하다.

결론은 **GLM의 작은 KV에서도 latency·I/O·store·setup 포함 session 이득이 유지되지만,
25% KV는 명확한 정확도 손실을 동반한다**는 것이다. 독립 cold session 또는 cache 정책 이득으로
확대해 해석하지 않는다.
