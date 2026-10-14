# GLM-4.6V-Flash V5 포팅·구조·correctness 보고

이 문서는 V5의 동결 설정, 실제 로드와 독립 gate 증거를 설명한다. 품질·latency·paired CI와 전체 workload 완료 판정은 phase별 raw audit 및 최종 REPORT/CSV를 따른다. 보고서 작성 중 GPU 명령, weight hashing, 대규모 raw scan을 수행하지 않았고 동결 소스를 변경하지 않았다.

## 모델·processor·Transformers 출처

모델·processor·tokenizer는 **`zai-org/GLM-4.6V-Flash`**, revision **`411bb4d77144a3f03accbf4b780f5acb8b7cde4e`**다. 실제 로드 클래스는 **Glm4vForConditionalGeneration**, processor는 **Glm46VProcessor / Glm46VImageProcessor**다. Checkpoint의 `model_type=glm4v`와 processor config의 `glm46v` namespace를 각각 확인했다. 106B GLM-4.6V나 다른 모델로 대체하지 않았다. 근거는 [공식 checkpoint config](https://huggingface.co/zai-org/GLM-4.6V-Flash/blob/411bb4d77144a3f03accbf4b780f5acb8b7cde4e/config.json), [processor config](https://huggingface.co/zai-org/GLM-4.6V-Flash/blob/411bb4d77144a3f03accbf4b780f5acb8b7cde4e/preprocessor_config.json), [native template](https://huggingface.co/zai-org/GLM-4.6V-Flash/blob/411bb4d77144a3f03accbf4b780f5acb8b7cde4e/chat_template.jinja), [공식 GLM-V 저장소](https://github.com/zai-org/GLM-V)다.

GLM 전용 `.venv-glm46v`는 **Transformers 5.13.1 PyPI wheel**이다. [공식 release](https://github.com/huggingface/transformers/releases/tag/v5.13.1)와 GitHub tag API가 가리키는 source commit은 **[`4626421dc6b741a329300682a6408246ee465490`](https://github.com/huggingface/transformers/commit/4626421dc6b741a329300682a6408246ee465490)**다. V5 timing 전에 공식 소스의 아래 네 작은 파일을 설치본과 byte 수준 SHA256으로 대조했다. V5의 동결 environment에도 같은 hashes가 기록되어 있으며 모두 일치한다.

| 공식 commit과 설치본 파일 | SHA256 | 일치 |
|---|---|---|
| `glm4v/modeling_glm4v.py` | `0f2a156edaa0503d70ade6a6c3f019a95230823fc1a52e2e01a20af4f8ec747d` | PASS |
| `glm46v/processing_glm46v.py` | `6d785b95e35a6c9417178042c0a428913672486433f70eba9222f1c51e640d1c` | PASS |
| `glm46v/image_processing_glm46v.py` | `41fa88e258a4b291f167d77085eee989dbebff1776c336121e2f4185b6e0eef1` | PASS |
| `glm46v/modeling_glm46v.py` | `cf9c152c412bb25e069724bdc1153b4f85c937c28ae324f50046453fc86ba3ed` | PASS |

[transformers_source_provenance.json](transformers_source_provenance.json)은 API 응답·정확한 upstream URL·설치본 hashes·V5 동결 environment 대조와 기존 조회 증거를 포함한다. 초기 architecture 조사에서 미확인으로 남겼던 git commit을 이 출처 대조로 보완했으며 기존 source/environment freeze는 수정하지 않았다.

## 실제 환경과 메모리 적합성

GPU는 **RTX 4090**이다. 실행 환경은 Python **3.10.20**, Torch **2.5.1+cu121**, torchvision **0.20.1+cu121**, Transformers **5.13.1**, bitsandbytes **0.49.2**, accelerate **1.14.0**, huggingface_hub **1.33.0**, tokenizers **0.22.2**, Pillow **12.2.0**, NumPy **2.2.6**다. 전용 venv가 기존 `mllm_ft`의 Torch와 의존성을 읽기 전용으로 사용하며 다른 환경을 전역 업그레이드하지 않았다.

Decoder NF4 4-bit / double quantization / BF16 compute가 실제 적용되었다. **Linear4bit 모듈은 240개**, 즉 40 layers 각각 Q/K/V/O 네 projection과 fused gate-up/down 두 projection이다. **Vision 양자화 모듈은 0개**다. Vision encoder·Conv2d downsample·merger·lm_head는 BF16 비양자화, native KV도 BF16이다. 모든 weight는 CUDA에 있고 CPU/disk weight offload는 없다. Text와 vision 모두 **SDPA**, batch 1, eval/inference, seed1234, greedy **max_new_tokens=32**이며 세 arm에 동일하다.

Native image processor의 고정 정책은 `size={shortest_edge:12544,longest_edge:1605632}`다. 이 native `smart_resize`의 두 budget은 반복 frame 수 2를 포함한 pixel 면적이며 공간 변을 28의 배수로 맞춘다. Qwen의 옵션 이름을 복사하지 않았다. 정적 이미지는 두 frame을 복제한 뒤 temporal patch embedding을 거쳐 `image_grid_thw`의 T=1이 된다. Patch size 14, temporal patch size 2, spatial merge size 2다.

398개 실제 main 이미지의 CPU processor preflight에서 **N=126–529**였다. 별도의 동결 seed1234 **896×896 synthetic fixture**에서는 **N=1024 / k=256**으로 full capture/importance GPU 요청을 수행했다. 이 stress는 품질 또는 final benchmark 분모에 포함하지 않는다.

해당 stress의 CUDA allocated peak는 **8,931,302,400 bytes = 8.318 GiB**, reserved peak는 **9,340,715,008 bytes = 8.699 GiB**였다. Scope는 request preparation부터 필요한 full prefix clone/importance tail까지다. Vision Q/K capture는 **25,165,824 bytes**이며 score의 sampled extra live allocation은 **50,544,640 bytes**였다. 후자는 모든 allocator/kernel workspace를 포함한 peak가 아니다. 요청 전체 CUDA peak와 별도로 기록했다. 단일 stress 결과는 이 입력의 GPU 적합성 증거이며 모든 요청의 메모리나 평균 latency를 대신하지 않는다.

## Native KV·prefix·suffix

실제 config는 **40 decoder layers, 2 native KV heads, head_dim 128**이다. 32 query heads로 KV를 확장하여 저장하지 않는다.

`40 × 2(K,V) × 2(native KV heads) × 128 × 2(BF16 bytes) = 40,960 bytes = 40 KiB / token`.

독립 raw parser와 SSD restoration이 이 기대값을 actual tensor/file bytes와 대조했다. Native token-major `[row,2,128]`에서 한 layer의 K 또는 V 행은 **512 bytes**다. 예를 들어 gate의 N=253 이미지에서 **k=ceil(253/4)=64**, 구조는 19행이다. B content payload는 **2,621,440 bytes = 64×40,960**, 구조 payload는 **778,240 bytes = 19×40,960**이며 각 layer K/V 파일은 32,768 bytes에서 끝난다. 64행 chunk와 partial 마지막 chunk를 검증했고 padding은 0이다. Metadata/seal/allocated bytes는 payload와 별도다.

Image는 첫 user의 질문보다 앞선다. Prefix는 첫 token부터 native end-of-image boundary까지이며 content N개 이외의 system/role/boundary token을 structure로 모두 보존한다. Store identity는 image hash·model/processor revision·processor/system/prefix IDs·geometry·위치 정책·source/environment를 검증한다.

Decoder의 첫 **64/128 dims**만 partial RoPE 대상이다. 32 inverse frequencies를 **[8,12,12]**로 나누어 T/H/W를 선택하며 decoder rotation은 adjacent pair다. Vision의 split-half rotation과 혼동하지 않는다. 나머지 64 dims는 그대로다. **Post-RoPE K bits를 재회전하지 않는다.**

Merged content `(t,h,w)` 위치는 `(a+t,a+h,a+w)`, image 이후 text는 `a+max(H/2,W/2)`에서 진행한다. 원래 logical IDs·3축 위치·native offset을 보존하며 suffix는 full native tokenization에서 재구성한다. Compact physical cache length는 cache/mask에만 쓰고 native position으로 취급하지 않는다. 요청별 fresh cache와 explicit positions를 쓰며 mutable `rope_deltas`를 초기화한다.

## Image-only 중요도와 PrefixStore-KV25 [B]

수식은 `docs/glm46v_architecture.md`에서 구현 전에 고정했다. [VisionZip 논문](https://arxiv.org/abs/2412.04467)과 [공식 저장소](https://github.com/JIA-Lab-research/VisionZip)의 image-only received-attention 아이디어를 GLM의 실제 full attention/Conv2d receptive field에 맞게 정의했다. 기존 Qwen 조사에서 참조한 pinned revision은 `8f86b55c6f000eb033e6912538af2dd7dcb30502`다. 이 작업은 **GLM용 VisionZip-style adaptation**이며 논문의 공식 GLM 구현을 재현했다고 주장하지 않는다.

24개 vision blocks 중 마지막 **block 23**, 12 heads, head_dim 128을 사용한다. Frame별 full patch attention이며 CLS나 Qwen window permutation이 없다. Processor의 `(frame, spatial block h/w, local h/w)` 순서에서 네 연속 patch가 native 2×2 Conv2d merge receptive field를 이룬다. Stock RMSNorm/downsample/learned merger 출력은 그대로 둔다.

\[
A^{(h)}_{ij}=\operatorname{softmax}_{j\in\mathrm{frame}}(Q_i^{(h)}K_j^{(h)\top}/\sqrt{128}),\quad
r_j={1\over12}\sum_h\sum_{i\in\mathrm{frame}}A^{(h)}_{ij},\quad
s_m={1\over4}\sum_{j=4m}^{4m+3}r_j.
\]

정상 T1의 stock final QKV projection에서 Q/K만 capture하여 native vision RoPE를 적용한다. QK matmul/scaling은 native dtype, softmax는 FP32 후 Q dtype로 cast, query 합·head/merge 평균은 FP32다. Score 계산 규칙이며 실제 forward는 SDPA 그대로다. Query만 최대128행씩 blocking하고 **softmax denominator의 key축은 frame 전체**를 유지한다. Question/gold/history/LLM query attention/calibration/diversity는 score나 rank 입력이 아니다.

전체 N개의 score로 stable descending permutation 한 개를 만들고 모든 decoder layer/native KV head에 적용한다. 동점은 원래 ID 순서다. B는 원래 **N에서 k=ceil(N/4)**를 한 번 계산하고 structure를 전부 유지한다. 정상 source T1의 full pixels, full CPU KV materialization, full repack을 유지하며 serialization만 k행으로 제한한다. 별도 vision/LLM prefix forward나 T1 사전 pruning을 하지 않는다. T2/T3는 frozen first-k를 실제 OS pread로 읽고 score/selection을 다시 계산하지 않는다. Activation 이후 metadata/FD만 유지하며 decode 중 store 재읽기가 없다.

## V5의 native assistant history와 scoring 분리

Native `enable_thinking=False`를 세 arm 모두 명시한다. Tokenized prompt에는 user의 `/nothink`와 assistant의 빈 `<think></think>` prefix가 있으며 tools는 없다. 빈 prompt marker는 generated reasoning이 아니다. 실제 generated IDs·literal decoded output 및 anomaly flags를 보존한다.

Scoring은 `native-final-answer-box-v1`이다. Native begin/end box IDs **151361/151362** 한 쌍 내부를 최종 VQA prediction으로 decode한다. Native box는 tokenizer의 ordinary added token이므로 `skip_special_tokens=True`만으로 제거되지 않는다. Boundary가 없으면 plain decode, malformed/multiple boundary면 literal output와 상태를 남긴다. Parser에는 질문/gold를 전달하지 않는다. Strict normalized EM/gold[0] scorer와 workload는 그대로다.

다음 턴 history는 scoring prediction과 분리한 **`actual-generated-body/terminal-EOS-only-v1`**이다. 자기 실제 generated IDs에서 마지막 token이 **로드한 EOS 집합 `{151329,151336,151338,151348}`**에 속하면 그 한 token만 제외한다. 남은 body IDs를 **`skip_special_tokens=False`, `clean_up_tokenization_spaces=False`**로 decode하여 `history_text`에 기록한다. **내부 registered controls, native box/thinking controls와 whitespace를 모두 보존한다.** `history_token_ids`와 `history_native_policy`도 남긴다. Native chat template가 이 actual assistant content를 공식 규칙으로 직렬화한다. Gold, 다른 arm의 답변이나 normalized/scoring prediction을 history로 쓰지 않는다.

Independent reference는 port의 body helper를 호출하지 않고 actual generated IDs에서 terminal EOS 한 개만 잘라 직접 decode한다. Body IDs/text/policy와 history lineage를 감사하며, parsed answer만 넣거나 내부 controls/whitespace를 바꾸는 negative fixtures를 거부한다. 긴 질문·same-image actual shared dialogues·suffix/decode 위치 gate도 이 native history로 수행했다.

Native malformed/empty/cap/stray thinking-delimiter 출력은 **그대로 raw와 품질 분모에 유지**한다. `enable_thinking=False` invocation과 모델이 실제 생성한 marker/이상 형식을 구분한다. 나쁜 출력을 지우거나 좋은 출력이 나올 때까지 재시도하지 않으며 닫는 delimiter만으로 내부 reasoning 상태를 단정하지 않는다. KV/cache/mask/위치/identity/파일 손상·nonfinite inference·자원 실패의 semantic gates는 계속 유지한다. 출력 anomaly 자체를 포팅 semantic 실패로 오인하여 검열하지 않는다.

## 이전 실행 보존과 cap 차이

초기 V3는 max16이었다. Native box가 ordinary token임을 놓쳐 wrapper 전체를 scoring한 포팅 누락으로 GQA720/MT360 strict EM 0% raw를 남겼고 main은 0요청이다. 또한 MT B 120요청 중 한 개가 cap16에서 end boundary 전에 잘렸다. V4에서 native final-answer extraction을 추가하고 **main 전에 모든 arm 공통 cap32**를 새 run에 동결했다. 이 cap은 기존 LLaVA/Qwen max16과 다르므로 cross-model latency 비교에서 명시한다. V5에서도 cap32·checkpoint·양자화·해상도·중요도·scorer는 바꾸지 않았다.

V4는 두 pilot을 마쳤으나 `mtgqa_000040/recompute/T2`에서 malformed native output을 output-string guard가 거부하여 main을 중단했다. **333 adopted 요청**, 실패 이미지의 **21 partial raw 요청**, 실패 output **1건**과 해당 raw/freeze/attempt를 별도로 보존한다. V4 main은 완료되지 않았으며 V5 분모와 섞지 않는다.

동일 고정 V4 입력에서 manual decode와 공식 `model.generate`는 prompt·IDs·3축 위치·첫 logits·전체 generated IDs가 정확히 같았고, parsed `Crown` history에서 같은 closing-only malformed output이 재현됐다. 실제 T1 native boxed body를 history로 사용하면 양쪽 모두 closed native box를 생성했다. 이는 수동 decode 위치 오류와 scoring/history protocol 누락을 구분한 controlled 증거다. 자세한 기록은 [V4 진단](../glm46v_prefixstore3_v4_20261007T072841Z/NON_THINKING_DIAGNOSIS.md)에 있다. V5는 수정된 protocol로 **새 freeze, 새 gate, 새 pilot, 새 main**을 수행한다. 기존 LLaVA/Qwen 결과나 이전 실패 기록을 수정하지 않았다.

## 실제 V5 correctness 결과

전체 GLM CPU suite는 **82/82 PASS**다 (`cpu_validation.log`). 실행에 직접 연결된 독립 gate의 CPU 검사는 **vision9 + store20 + runner21 + validation10 + analysis15 = 75/75 PASS**다. 나머지7개는 workflow/orchestration 검사다. GPU gate는 고정된 **6개 실제 이미지**, 최대-N stress, 알려진 native history regression fixture를 통과했다.

6개 sample `(N,k)`는 **(126,32), (529,133), (253,64), (252,63), (391,98), (414,104)**다. Min/max N, k의 64-row 경계, 긴 질문, 동일 이미지의 실제 여러 dialogue를 포함한다. Stock T1 대비 capture/scored T1의 visual pooler bits, 독립 stock first-prefill prefix KV bits, 첫 logits와 전체 generated IDs는 정확히 일치했다. Canonical/FullLoad raw payload와 independent memory/SSD B의 native all-layer/head bits, IDs, 위치, mask, 같은 shape의 logits/generated IDs도 정확히 일치했다. Question-independent score/선택 IDs 및 actual native history/state isolation/decode advancement가 통과했다.

Dense masked / compact attention은 **6 images×2 store arms×40 layers = 480 layer comparisons**다. Fixed-input Q/K/V를 연산 전에 FP64로 변환했으며 최대 absolute error는 **7.105427357601002×10⁻¹⁵**, 사전 기준 **atol=rtol=10⁻¹²** 아래다. FP32 QK/softmax/V aggregation shape 진단을 따로 남겼다. 실제 main BF16/SDPA를 FP64로 변경하지 않았다.

ReComp와 FullLoad는 native full-prompt/suffix-only projection·attention shape가 달라 6개 모두 첫 logits bits가 달랐다. Max absolute logit 차이는 **0.2109375–0.3203125**였고 이 gate의 generated IDs는 같았다. 이 수치 shape 차이를 저장/복원 오류 및 B 압축 quality loss와 구분하여 기록했다. Different-question full-prefix KV rounding도 별도 진단이다. Same-input hook transparency와 same-shape SSD/memory exact 기준은 완화하지 않았으며 B/full-KV logits 일치를 요구하지 않는다.

V5의 `n204894` regression에서는 현재 actual T1 **[151361,34,4830,151362,151336]**에서 terminal151336만 제외한 native boxed body를 사용했다. Manual과 공식 generate 모두 T2 **[151361,785,10364,151362,151336]**를 생성했다. 첫 logits bits/maxabs0, prompt/input IDs, **모든 prefill/decode 3축 위치·입력 IDs·cache lengths**가 정확히 일치했다. Think open/close는0, native box는 한 쌍이었다. Gold나 과거 실행의 정답을 넣지 않고 현재 실제 T1 output으로 재현했다. 이 fixture의 성공이 모든 workload 출력 형식의 정상성을 증명하는 것은 아니며 anomaly는 전체 raw에서 별도로 집계한다.

Authority는 `runs/glm46v_prefixstore3_v5_20261007T075730Z/validation/receipt.json`, sample evidence/attention JSON/CPU logs와 `native_history_regression.json`이다. 소비한 `freeze.json` SHA256은 **`bd19161f73a93b76f55111adb0f3a0e1c68bed70ed3cf14053263d1e68b02ca0`**다. Gate/preflight/stress/diagnostic 요청은 pilot/main final 분모에서 제외한다.

## TTFT와 출력 token의 해석

TTFT는 history/chat/input 준비 및 pixels 경로의 image open 이전부터 **첫 native generated token materialization/CUDA synchronization**까지다. Cache 요청의 pread·H2D·assembly·suffix prefill을 포함한다. Gate의 정상 boxed 출력에서는 첫 token이151361 **answer-begin control**이며 첫 답변 단어가 아니다. 이를 reasoning token으로 부르지 않는다. **첫 lexical answer-token latency는 NOT_MEASURED**다. Malformed/plain 출력에서는 실제 첫 token과 anomaly flags를 raw에 그대로 남긴다.

Request E2E는 실제 outer return과 source T1의 필요한 score/capture tail을 정확히1회 포함한다. Persistence는 full CPU materialization/repack/write/fsync/seal의 outer writer return, activation은 integrity read/metadata/FD 준비의 별도 setup이다. Nested 시간을 TTFT/E2E나 session에 중복 합산하지 않는다. OS I/O는 NAND physical I/O가 아니며 DONTNEED는 cold NAND 보장이 아니다. 측정하지 않은 값을0으로 대신하지 않는다.
