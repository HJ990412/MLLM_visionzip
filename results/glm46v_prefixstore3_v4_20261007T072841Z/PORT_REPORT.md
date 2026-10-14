# GLM-4.6V-Flash 포팅·구조·실행 정합성 보고

본 보고서는 실제 로드 설정과 v4 correctness gate의 증거를 설명한다. 품질·latency·paired CI의 최종 집계는 각 phase의 REPORT/CSV를 따른다. 소스와 설정을 재수정하지 않고 동결된 실행에서 파생한 문서다.

## 모델과 Transformers 출처

모델·processor·tokenizer는 모두 **`zai-org/GLM-4.6V-Flash`**, revision **`411bb4d77144a3f03accbf4b780f5acb8b7cde4e`**다. 106B GLM-4.6V나 다른 Flash 모델을 사용하지 않았다. [공식 모델](https://huggingface.co/zai-org/GLM-4.6V-Flash), [checkpoint config](https://huggingface.co/zai-org/GLM-4.6V-Flash/blob/411bb4d77144a3f03accbf4b780f5acb8b7cde4e/config.json), [native processor config](https://huggingface.co/zai-org/GLM-4.6V-Flash/blob/411bb4d77144a3f03accbf4b780f5acb8b7cde4e/preprocessor_config.json), [native chat template](https://huggingface.co/zai-org/GLM-4.6V-Flash/blob/411bb4d77144a3f03accbf4b780f5acb8b7cde4e/chat_template.jinja), [GLM-V 공식 저장소](https://github.com/zai-org/GLM-V)를 직접 확인했다.

실제 모델 클래스는 **Glm4vForConditionalGeneration**, processor는 **Glm46VProcessor / Glm46VImageProcessor**다. 모델 이름의 `4.6V`와 Transformers model namespace는 일대일로 대응하지 않는다. Checkpoint는 `model_type=glm4v`이고 processor config가 `glm46v` processor를 지정한다.

전용 `.venv-glm46v`는 **Transformers 5.13.1 PyPI wheel**을 사용한다. [공식 release](https://github.com/huggingface/transformers/releases/tag/v5.13.1)와 [공식 tag commit](https://github.com/huggingface/transformers/commit/4626421dc6b741a329300682a6408246ee465490)을 확인했으며 source commit은 **`4626421dc6b741a329300682a6408246ee465490`**다. GitHub tag API를 별도로 읽어 commit을 확인하고 아래 네 upstream 파일을 설치본과 byte 수준 SHA256으로 대조했다. 모두 일치한다. 작은 소스 파일만 읽었으며 GPU 명령이나 weight hash 작업을 수행하지 않았다.

| 설치본과 공식 commit 파일 | SHA256 | 일치 |
|---|---|---|
| `glm4v/modeling_glm4v.py` | `0f2a156edaa0503d70ade6a6c3f019a95230823fc1a52e2e01a20af4f8ec747d` | PASS |
| `glm46v/processing_glm46v.py` | `6d785b95e35a6c9417178042c0a428913672486433f70eba9222f1c51e640d1c` | PASS |
| `glm46v/image_processing_glm46v.py` | `41fa88e258a4b291f167d77085eee989dbebff1776c336121e2f4185b6e0eef1` | PASS |
| `glm46v/modeling_glm46v.py` | `cf9c152c412bb25e069724bdc1153b4f85c937c28ae324f50046453fc86ba3ed` | PASS |

API 응답, 정확한 upstream URL과 양쪽 hashes는 [transformers_source_provenance.json](transformers_source_provenance.json)에 있다. 초기 architecture 문서의 “source git commit 미확인”은 초기 로컬 설치 조사 당시의 상태이며, 이 사후 출처 대조가 이를 보완한다. Freeze된 environment/source 문서를 변경하지 않았다.

## 실제 환경·양자화·KV

실제 GPU gate는 NVIDIA GeForce **RTX 4090**에서 수행했다. 환경은 Python 3.10.20, Torch **2.5.1+cu121**, torchvision **0.20.1+cu121**, Transformers **5.13.1**, bitsandbytes **0.49.2**, accelerate **1.14.0**, huggingface_hub **1.33.0**, tokenizers **0.22.2**, Pillow **12.2.0**, NumPy **2.2.6**다. 전용 환경이 기존 `mllm_ft`의 Torch와 의존성을 읽기 전용으로 사용하며 기존 환경을 전역 업그레이드하지 않았다. 실행 package/source hashes는 `runs/glm46v_prefixstore3_v4_20261007T072841Z/environment.json`에 있다.

Decoder의 NF4 4-bit / double quantization / BF16 compute가 실제 적용되었다. **Linear4bit 모듈은 240개**이며, 40 layers 각각 Q/K/V/O projection 네 개와 fused gate-up/down projection 두 개다. **Vision 양자화 모듈은 0개**다. Encoder, Conv2d downsample, merger 및 lm_head는 BF16 비양자화이며 native KV도 BF16이다. 모든 weight가 CUDA에 있고 CPU/disk weight offload가 없다. Text와 vision backend 모두 **SDPA**, batch 1, eval/inference, seed 1234, greedy다. 세 arm에 같은 설정을 적용한다.

실제 config는 **40 decoder layers, 2 native KV heads, head_dim 128**이다. 32 query heads로 KV를 확장하여 저장하지 않는다.

`40 layers × 2(K,V) × 2 heads × 128 dims × 2 BF16 bytes = 40,960 bytes = 40 KiB / token`.

독립 parser와 SSD restoration이 실제 BF16 tensor bits와 파일 bytes를 대조했다. 예를 들어 gate 이미지 `n302358`은 원래 **N=253**, **k=64**, 구조 **19행**이다. B content payload는 **2,621,440 bytes = 64×40,960**, 구조 payload는 **778,240 bytes = 19×40,960**다. 각 layer의 K/V 파일은 **32,768 bytes = 64×2 heads×128×2**에서 끝나며 padding은 0이다. Metadata/seal 및 파일시스템 allocated bytes는 payload와 별도다.

## Native 해상도와 GPU 메모리 preflight

해상도는 native `size={shortest_edge:12544,longest_edge:1605632}`로 동결했다. GLM46V `smart_resize`는 반복 frame 수 2를 포함한 pixel budget을 사용하며, 공간 변을 28의 배수로 맞춘다. 정적 이미지의 temporal patch size는 2, spatial patch size는 14, merge size는 2다. Processor 입력에서 frame 두 개를 복제한 뒤 temporal embedding을 하므로 `image_grid_thw`의 T는 1이다. 원래 작은 이미지를 속도 차이를 키우기 위해 확대하지 않았다.

398개 main 이미지의 CPU processor 조사에서 실제 **N 범위는 126–529**였다. Dataset membership이나 해상도를 품질 결과로 다시 선택하지 않았다. 별도의 seed1234 **896×896 synthetic resource fixture**에서 **N=1024 / k=256**을 실제 GPU로 검증했다. 이 fixture는 품질 데이터나 pilot/main 분모에 포함하지 않는다.

이 최대-N capture/importance 요청의 실제 CUDA allocated peak는 **8,931,302,400 bytes = 8.318 GiB**, reserved peak는 **9,340,715,008 bytes = 8.699 GiB**였다. 측정 scope는 request preparation부터 필요한 full prefix clone/importance tail까지다. Q/K capture 자체는 **25,165,824 bytes**, sampled extra live allocation은 **50,544,640 bytes**다. 후자는 score의 완전한 allocator/kernel workspace peak가 아니다. Request 전체 CUDA peak와 scope를 별도로 기록했다. 이 단일 stress 측정은 4090에서 해당 입력이 적합했다는 증거이며 모든 입력의 peak나 평균 latency를 대체하지 않는다.

## Image-only VisionZip-style adaptation

중요도 수식은 `docs/glm46v_architecture.md`에서 구현 전에 고정했다. 관련 출처는 [VisionZip 논문](https://arxiv.org/abs/2412.04467)과 [공식 저장소](https://github.com/JIA-Lab-research/VisionZip)다. 기존 Qwen 조사에서 참조한 pinned 코드 revision은 `8f86b55c6f000eb033e6912538af2dd7dcb30502`이며 [해당 Qwen vision 구현](https://github.com/JIA-Lab-research/VisionZip/blob/8f86b55c6f000eb033e6912538af2dd7dcb30502/Qwen2_5_VL/qwen2_5vl_visionzip.py)의 received-attention 집계 아이디어를 GLM의 실제 full-attention/Conv2d geometry에 맞게 정의했다. 이 구현은 **GLM용 VisionZip-style adaptation**이며 원 논문의 공식 GLM 구현 재현을 주장하지 않는다.

GLM의 vision tower는 24 blocks, 12 attention heads, head_dim 128이다. 마지막 **block 23**은 frame별 full patch attention이다. CLS token이나 Qwen window permutation은 없다. Native processor patch 순서는 `(frame, spatial block h, spatial block w, local h, local w)`이며 2×2 receptive field의 네 patch가 연속이다. 마지막 block 뒤의 RMSNorm, Conv2d(kernel=stride=2), learned merger가 이 네 patch를 하나의 native visual content token으로 연결한다.

마지막 stock QKV projection에서 Q/K만 관측하고 native **vision RoPE**를 적용한다. Frame 내부의 모든 유효 key에 대해

\[
A^{(h)}_{ij}=\operatorname{softmax}_{j\in\mathrm{frame}}(Q_i^{(h)}K_j^{(h)\top}/\sqrt{128}),\quad
r_j={1\over12}\sum_h\sum_{i\in\mathrm{frame}}A^{(h)}_{ij},\quad
s_m={1\over4}\sum_{j=4m}^{4m+3}r_j.
\]

QK matmul/scaling은 native dtype, softmax는 FP32 후 Q dtype로 cast, query 합과 head/merge 평균은 FP32다. 이는 별도의 score 계산 규칙이며 실제 모델 forward는 SDPA 그대로다. Query를 최대 128행씩 blocking하고 **key축과 softmax denominator는 항상 frame 전체**를 유지한다. 별도 vision/LLM prefix forward, 질문·gold·history·LLM query attention·diversity·calibration 입력은 없다. Stock N개 visual 출력과 full T1 prediction을 유지한다.

전체 score의 stable descending rank 한 개를 모든 decoder layer와 native KV head에 적용한다. 동점은 원래 native token ID 순서다. B의 예산은 원래 **N에서 k=ceil(N/4)**로 한 번 계산하고 구조 토큰을 전부 보존한다. T2/T3는 frozen first-k만 실제 OS pread로 읽으며 재선택/재점수화나 decode 중 SSD 재읽기가 없다. Full CPU KV materialization와 전체 importance repack을 유지하고 serialization에서만 k행으로 제한한다.

## Prefix·partial RoPE·suffix

Native image는 첫 user message의 질문보다 앞선다. Prefix는 첫 token부터 `end_of_image`까지이며 image content 외 system/role/boundary token을 structure로 모두 보존한다. Store identity는 image hash, checkpoint/processor revision, processor/system/prefix IDs, geometry, 위치 정책과 source/environment를 검증한다.

Decoder는 첫 **64/128 dims**에만 partial RoPE를 적용한다. Frequency sections **[8,12,12]**가 T/H/W를 선택하며 decoder는 adjacent-pair rotation, vision은 split-half rotation이다. 마지막 64 dims는 그대로다. 선택한 **post-RoPE K bits를 재회전하지 않는다**.

Merged image content `(t,h,w)` 위치는 `(a+t,a+h,a+w)`다. 이미지 이후 text는 `a+max(H/2,W/2)`에서 진행한다. 원래 token logical IDs, native 3축 positions, native offset과 full tokenization suffix를 보존한다. Compact physical cache length는 cache/mask assembly에만 쓰며 native position으로 취급하지 않는다. 요청별 fresh cache와 명시적 suffix/decode positions를 사용하고 mutable `rope_deltas`를 초기화한다. 각 방법의 실제 최종 답변만 해당 방법의 다음 history에 넣는다.

## Non-thinking, native answer boundary와 cap 변경

모든 요청에 native chat template **`enable_thinking=False`**를 명시한다. 실제 tokenized prompt에는 user의 `/nothink`와 assistant generation prefix의 빈 `<think></think>`가 있다. 빈 prompt marker는 reasoning 생성이 아니다. Raw generated IDs/전체 decoded text 및 reasoning-leak/cap 상태를 보존한다. Tool calling은 사용하지 않는다.

초기 실행은 공통 max16이었다. GLM의 native answer boundary **151361 / 151362**는 pinned tokenizer에서 special token으로 등록되지 않아 `skip_special_tokens=True`만으로 제거되지 않는다. 초기 port가 이 포장을 그대로 prediction으로 채점한 문제를 발견했다. 초기 v3의 GQA 720 / MT 360 raw, strict EM 0%, gate/failure/freeze를 보존했으며 v3 main은 0요청이다. 이를 v4 완료 결과와 합치지 않는다.

v4는 **main 전에 별도 run과 공통 max_new_tokens=32**를 고정했다. 초기 MT B의 120요청 중 한 개가 cap16에서 end boundary 이전에 잘려 cap을 바꿨다. Native 경계 한 쌍 안의 token span만 final answer로 decode하는 `native-final-answer-box-v1`을 모든 arm에 적용한다. 질문이나 gold를 parser에 전달하지 않는다. 경계가 없으면 plain decode, 중복·깨진 경계면 literal output와 malformed 상태를 보존한다. Strict normalized EM / gold[0] scorer, 질문, gold, workload membership, 중요도, 해상도와 backend는 그대로다. 기존 LLaVA/Qwen max16과 달라 cross-model latency 비교에 이 차이를 명시해야 한다.

**TTFT는 첫 생성 token materialization/CUDA sync까지**다. GLM의 첫 생성 token은 실제 gate에서 151361 native answer-begin control token이다. Reasoning token으로 간주하지 않으며 첫 답변 내용 단어까지의 latency라고 부르지 않는다. **첫 lexical answer-token latency는 NOT_MEASURED**다. Request E2E는 실제 전체 generated output과 필요한 source capture/importance tail을 포함하며, persistence/activation의 nested component를 TTFT/E2E에 중복 가산하지 않는다.

## 실제 correctness 결과와 수치 차이 구분

v4 독립 CPU gate는 **vision 9, store 20, runner 14, validation 6, analysis 8 = 57/57 PASS**다. GPU gate는 동결 manifest의 **6개 실제 이미지 + 최대-N stress PASS**다. 원래 N/k는 `(126,32), (529,133), (253,64), (252,63), (391,98), (414,104)`로 최소/최대 N, chunk 64 경계, 긴 질문과 동일 이미지의 실제 여러 dialogue를 포함한다.

6개 모두 stock T1 대비 capture/scored T1의 native visual pooler bits, 독립적으로 관측한 stock first-prefill prefix KV bits, 첫 logits와 전체 generated IDs가 정확히 일치한다. Canonical/FullLoad raw payload와 memory/SSD B의 all-layer/native-head bits, IDs, positions, mask, 같은-shape 첫 logits/generated IDs도 정확히 일치한다. 서로 다른 질문에서도 image-only scores와 선택 IDs가 동일하며 actual generated history, suffix/decode advancement, dialogue isolation 검증이 통과했다.

각 sample의 FullLoad와 B에 대해 40 layers 전체의 fixed-input Q/K/V를 **연산 전에 FP64**로 바꾸어 dense masked / compact attention을 비교했다. 최대 absolute error는 **7.105427357601002×10⁻¹⁵**, 사전 기준 atol=rtol **10⁻¹²** 아래다. FP32 QK/softmax/V aggregation의 shape 차이는 별도 진단으로 남겼다. Main inference의 BF16/SDPA를 gate 통과용 FP64로 바꾸지 않았다.

ReComp와 FullLoad는 full-prompt와 suffix-only native projection/attention shape가 달라 **6개 모두 첫 logits는 bitwise 일치하지 않았다**. Max absolute logit 차이는 **0.2109375–0.3203125**이며 6개 gate generated IDs는 모두 같았다. 이는 SSD bits restoration 또는 B 압축 정확도 손실과 구분한 진단이다. 서로 다른 길이 질문의 native prefix KV 반올림 차이도 진단으로 보존하며, 같은 입력/shape의 hook transparency와 memory/SSD exact gate는 그대로 유지했다. B와 full-KV logits 일치를 요구하지 않는다.

검증의 출처는 `runs/glm46v_prefixstore3_v4_20261007T072841Z/validation/receipt.json`과 그 sample evidence/attention JSON/CPU logs다. 소비한 source freeze는 `freezes/pre_pilot_v2/freeze.json`, SHA256 **`e8d5868ae33c2a77e19eceaa09349421febc9320de8876bf54e6a5a8772a9b50`**다. Gate와 preflight/stress 요청은 pilot/main final 분모에서 제외한다. 이 포팅 보고서가 benchmark 전체의 correctness나 품질 동등성을 추가로 증명하는 것은 아니며, 실제 raw audits와 final 분석이 이어진다.
