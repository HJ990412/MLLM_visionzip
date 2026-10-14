# GLM-4.6V-Flash v6 포팅 및 계측 보고

이 문서는 v6의 실제 모델·구조·fresh gate와 caller outer setup 계측을 설명한다. Main 완료·품질·latency·paired CI는 이 run의 최종 raw/REPORT/CSV로 판단한다. 이전 v5 요청 또는 setup를 v6 수치에 합치지 않는다. 동결 source 수정·GPU 실행·timing 중 payload full scan 없이 작은 JSON·로그를 검토했다.

## 모델과 공식 source pin

정확한 모델은 `zai-org/GLM-4.6V-Flash@411bb4d77144a3f03accbf4b780f5acb8b7cde4e`다. [Checkpoint config](https://huggingface.co/zai-org/GLM-4.6V-Flash/blob/411bb4d77144a3f03accbf4b780f5acb8b7cde4e/config.json), [processor](https://huggingface.co/zai-org/GLM-4.6V-Flash/blob/411bb4d77144a3f03accbf4b780f5acb8b7cde4e/preprocessor_config.json), [native chat template](https://huggingface.co/zai-org/GLM-4.6V-Flash/blob/411bb4d77144a3f03accbf4b780f5acb8b7cde4e/chat_template.jinja)를 사용한다. Model type/class는 glm4v/Glm4vForConditionalGeneration, processor는 Glm46VProcessor/Glm46VImageProcessor다. GLM-4.6V106B나 다른 checkpoint를 사용하지 않았다. 공식 프로젝트는 [GLM-V](https://github.com/zai-org/GLM-V)다.

전용 `.venv-glm46v`의 Transformers5.13.1 wheel과 공식 [tag source commit4626421dc6b741a329300682a6408246ee465490](https://github.com/huggingface/transformers/commit/4626421dc6b741a329300682a6408246ee465490)의 모델/processor4파일이 full-byte SHA256으로 일치했다. 이 공식 source 대조는 이전에 취득한 primary evidence를 사용하며 v6 frozen environment의 동일한 hashes와 다시 결속했다. [official_source_provenance.json](official_source_provenance.json)에 정확한 URL/원본 evidence와 v6 비교가 있다. Installed cache/mask/SDPA 관련 파일 hash와 package versions는 environment.json에 있다.

V6 authoritative source/config/contract는 `runs/glm46v_prefixstore3_v6_20261007T104505Z/freeze.json`, SHA256 `1f6f3006a8399c8b62f60d910e422e13d4d644e63fe622bfc05913ed331bb77e`다. V6는 active pointer 없이 initial freeze를 사용한다. Native GLM package5개(__init__/runner/vision/store/analysis)는 v5와 source SHA가 정확히 같다. Controller121의 outer timers와 workflow tests·contract만 새로 동결했고 model inference/numerical tolerance/body/cap32/manifest/scorer는 바꾸지 않았다.

## 실제 실행 환경과 resolution

RTX4090 24GB, Python3.10.20, Torch2.5.1+cu121/torchvision0.20.1+cu121, Transformers5.13.1, bitsandbytes0.49.2, accelerate1.14.0, huggingface_hub1.33.0, tokenizers0.22.2, Pillow12.2.0이다. Venv는 기존 mllm_ft의 Torch/의존성을 읽기 전용으로 상속한다. 전역 dependency upgrade는 없다.

Decoder만 NF4 4-bit/double quantization/BF16 compute다. Actual Linear4bit240개는40decoder layers의 Q/K/V/O + fused gate-up/down이다. Quantized vision0, vision encoder/Conv2d downsample/merger/lm_head와 native KV는 BF16이다. 모든 weight는 CUDA, CPU/disk weight offload는 없다. Text/vision SDPA, batch1, eval/inference, seed1234, greedy cap32를 세 arm에 똑같이 적용했다.

Native `size={shortest_edge:12544,longest_edge:1605632}`는 temporal 반복2를 포함하는 면적 budget이며 공간 변을28의 배수로 맞춘다. Patch14/temporal2/spatial merge2다. 정적 image의 두 반복 frame은 temporal patch embedding 뒤 T=1이 된다. Native processor options를 실제 확인했고 Qwen min_pixels/max_pixels를 이름만 바꿔 쓰지 않았다. 실제398-image geometry의 N은126–529, 고정 synthetic896×896 fixture는N1024/k256이다. Stress GPU allocated/reserved peak는8.318/8.699GiB, Q/K capture25,165,824bytes, sampled score extra live allocation50,544,640bytes다. Request scope/model/coexisting state를 포함하며 모든 kernel workspace나 독립 deployment KV footprint가 아니다.

## Native prefix/KV/position과 중요도

Actual40decoder layers/2native KVheads/head_dim128/BF16이다. `40×2(K,V)×2heads×128×2bytes = 40,960bytes =40KiB/token`. 한 layer K또는V의 token-major `[row,2,128]` 행은512bytes다. Attention32heads로 KV를 확대해 저장하지 않는다.

Image는 첫 user의 질문 앞에 있다. Prefix는 system/role부터 native end-of-image까지이고 content N 이외의 structure19행을 모두 유지한다. 같은 image라도 model/processor/system/prefix/geometry/position/source/environment identity가 다르면 store를 거부한다. Source dialogue T1의 실제 full-pixel vision/prefill에서 prefix KV와 B importance 자료를 함께 capture한다. 별도 vision/LLM prefix forward나 T1 pre-pruning을 하지 않는다.

Decoder partial RoPE는128dims 중64dims,32frequencies의 T/H/W split[8,12,12], theta500000이며 adjacent pairs다. Vision split-half rotation과 다르다. Post-RoPE K bits를 그대로 저장·복원하고 재회전하지 않는다. Native3축 logical positions/offset은 full original-N tokenization에서 만들고 compact physical cache length는 cache/mask에만 쓴다. Request별 fresh cache와 rope_deltas reset을 적용한다.

Vision24blocks의 마지막block23,12heads/head_dim128을 사용한다. CLS/Qwen window mapping이 아닌 frame별 full patch attention이다. Native packed order의 연속4patch는2×2Conv2d receptive field로 final visual token에 연결된다. Captured native Q/K에서 전체 유효 key에 대한 softmax denominator를 유지하고 query만 blocking하여 임시 메모리를 제한한다.

수식은 구현 전에 동결한 glm46v_architecture.md를 따른다: `A_ij^h=softmax_j(q_i·k_j/sqrt(128))`, `r_j=mean_h sum_i A_ij^h`, final token score는 그 native4patch의 mean이다. 질문/답변/history/LLM query/calibration/diversity를 사용하지 않는다. Stable descending score와 원래ID tie break의 permutation 한 개를 모든 decoder layer/native heads에 적용한다. 이는 [VisionZip](https://arxiv.org/abs/2412.04467)식 received-attention을 실제 GLM 구조에 맞춘 adaptation이며 공식 논문의 GLM 구현 재현이 아니다.

B는 original N의 `k=ceil(N/4)`를 한 번 계산한다. Full CPU materialization와 full repack 후 serialization 단계에서만 상위k를 쓴다. Original geometry·full permutation·mapping을 유지하고 structure는 전부 저장한다. Full/B64-row token-major layout/partial EOF/content padding0, 동일 writer hash/fsync/seal/activation integrity 정책이다. Hit은 SSD의 frozen first-k 또는 full N과 structure만 읽고 online score/reselection/decode reread는0이다. Active reader는 metadata/FD만 보존한다.

## Native history/scoring 및 실제 outer setup

Native template의 enable_thinking=False, /nothink/빈 think prefix를 사용하고 tools는 없다. Scoring은 native answer-box IDs151361/151362의 한 쌍 내부 span, 없으면 plain, 깨진/중복이면 literal/status다. Question/gold를 extraction에 주지 않고 기존 strict normalized EM/gold[0]는 유지한다.

History는 자기 실제 IDs의 마지막 native EOS151329/151336/151338/151348 한 개만 제외한 body를 skip_special_tokens=False/cleanup=False로 decode한다. Box/internal thinking controls/whitespace를 보존한다. Native template가 과거 assistant content를 공식 규칙으로 직렬화하며 parsed/normalized prediction이나 다른 method/gold history를 넣지 않는다. Body IDs/text/lineage는 독립 tokenizer reference로 감사한다. Native malformed/empty/cap/stray marker는 분모에 유지하고 좋은 출력까지 재시도하거나 thinking token suppression/forced EOS를 하지 않는다.

V5의 model/statistics는 보존하지만 setup의 identity prep/return tail은 미측정이었다. V6는121 `timed_call()`의 caller clock으로 **runner.persist/runner.activate 진입 전부터 완전한 반환까지** 측정한다. Identity 생성·expected identity 등록·full materialization/repack/write/fsync/seal 및 activation integrity/metadata/FD/IO summary/return tail을 포함한다. Event duration_ms는 outer이고 inner_writer_call_return_ms/inner_activation_core_ms는 별도 진단이다. Source clone/importance tail은 source request E2E에 이미1회 들어가므로 setup에 다시 더하지 않는다. TTFT는 history/chat/input·image open 앞에서 native first token CUDA materialization까지, request E2E는 actual runner return까지다. Logging/audit/conditioning은 명시한 별도 scope다.

초기 v3의 공통 cap16에서는 MT B 120요청 중1개가 native end-of-box 이전에 잘렸다. 이 raw와 실패/추출 누락 기록을 보존하고 cap32를 모든 arm에 공통으로 V4 main 이전 새 계약에 고정했다. V5/V6도 cap32를 유지하며 기존 LLaVA/Qwen의 cap16과 다르다. 이후 결과를 보고 arm별 cap을 바꾸지 않는다.

V4 고정 실패 입력의 두 history 조건 모두에서 수동 경로와 공식 model.generate의 prompt·IDs·native positions·first logits bits·전체 generated IDs가 같았다. Parsed `Crown` history는 malformed closing-only 출력을 공식 경로에서도 재현했고 실제 boxed T1 body는 정상 closed box를 생성했다. 이 진단으로 scoring/history protocol 누락을 구분했으며 V4의333요청은 새 native-body 실행에 합치지 않았다.

정상 boxed gate 출력의 첫 generated ID151361은 native answer-begin control이며 첫 lexical answer나 reasoning token이 아니다. TTFT endpoint는 그 native 첫 출력 token materialization이고 first lexical answer latency는 **NOT_MEASURED**다. Native prompt의 빈 think block과 실제 generated thinking marker를 구분해 counts를 보고한다.

## Fresh admission과 archive 범위

전체 CPU86/86, GPU admission CPU75/75 PASS다. Actual6images/stress/known native history regression의 새 GPU receipt는PASS다. Stock/capture visual output·first-prefill native KV/logits/IDs, independent Full/B SSD restore·memory same-shape bits/IDs/positions, native mask visibility, all-layer FP64 dense/compact, generated-body history/state isolation을 통과했다. FP32 diagnostic FAIL2layers는 보존하며 BF16 SDPA 및 고정1e-12 semantic tolerance를 바꾸지 않았다. VALIDATION_REVIEW.md에 sample/hash/한계가 있다.

V6 GQA720/MT360 및이 run의 main36,549요청이 모두 완료됐다. V3 parser/quant/sidecar 실패, V4partial333/history protocol 실패, V5full36549 및 inner-scope setup 결과는 별도 archive이며 v6분모에0건이다. Main 완료·정답률/latency/setup/session/paired CI는 v6 최종 receipt와 raw에서만 보고한다. OSI/O를 NAND로 부르지 않고 DONTNEED를 cold NAND 보장으로 쓰지 않는다. Cache hit-rate/eviction/parallel throughput 또는 independent cold session도 검증하지 않는다.

## V6 terminal 완료 보충

Main은398 images/4,061 dialogues/36,549 final 요청, 방법당12,183 및8,122 followups로 완료됐다. Failed image attempt·retry는0이다. COMPLETED SHA256 `013a32a7423b1131211516b48d4eacbb2102bfc395739088cbd3be060eb1f609`. GQA720 및MT360도 별도 완료했다. 이전v3/v4/v5 요청은 primary 분모에0건이다.

독립 `independent_review/attempt_20261007T130724092714Z/receipt.json`는 모든 native scoring/body/lineage/prompt/inputIDs/3축positions/I/O 및54 paired point/CI를 원래 raw에서 검산해 PASS했고 numeric residual은0이다. Seed1234/10,000 image clusters/pooled sum-count를 사용했다. Actual caller outer physical setup과 session attribution 합은 모두55,670.386902522296ms다. Writer의새outer−inner2,089.627110864967ms 및activation41.401019319892ms가 실제 포함됐으며 nested 값을 중복 합산하지 않았다. Terminal 수치와 해석/한계는 FINAL_ANALYSIS_REVIEW.md 및FINAL_ANALYSIS_REVIEW_RECEIPT.json을 따른다.
