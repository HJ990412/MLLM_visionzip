# GLM-4.6V-Flash 포팅 구조와 구현 전 중요도 계약

작성 기준: 2026-10-07 KST. 아래 중요도 수식은 `mmimpress/glm46v/vision.py` 작성 **전에** 고정했다. 이 문서는 구조 조사와 CPU helper의 계약이며, checkpoint를 실제 GPU에서 로드하거나 GPU gate를 통과했다는 증거가 아니다. 최종 로드 환경과 gate 결과는 새 GLM run의 기록을 따른다.

## 공식 checkpoint와 processor

모델 및 processor revision을 모두 `411bb4d77144a3f03accbf4b780f5acb8b7cde4e`로 고정한다. [공식 commit](https://huggingface.co/zai-org/GLM-4.6V-Flash/commit/411bb4d77144a3f03accbf4b780f5acb8b7cde4e)과 [공식 모델](https://huggingface.co/zai-org/GLM-4.6V-Flash)을 확인했다. 구현 조사에는 [GLM-V 공식 저장소](https://github.com/zai-org/GLM-V), [Transformers glm4v](https://github.com/huggingface/transformers/tree/main/src/transformers/models/glm4v), [Transformers glm46v](https://github.com/huggingface/transformers/tree/main/src/transformers/models/glm46v)를 사용했다.

[checkpoint config](https://huggingface.co/zai-org/GLM-4.6V-Flash/raw/411bb4d77144a3f03accbf4b780f5acb8b7cde4e/config.json)은 `model_type=glm4v`, `architectures=[Glm4vForConditionalGeneration]`이다. 이름만 보고 `Glm46VForConditionalGeneration`으로 대체하지 않는다. [preprocessor config](https://huggingface.co/zai-org/GLM-4.6V-Flash/raw/411bb4d77144a3f03accbf4b780f5acb8b7cde4e/preprocessor_config.json)는 `Glm46VProcessor`와 `Glm46VImageProcessor`를 지정한다. 두 구현 namespace를 함께 조사해야 하는 이유다.

Decoder 설정은 40 layers, 32 query heads, **2 native KV heads**, hidden size 4096, head dimension 128이다. 저장 토큰당 BF16 payload 기대값은 `40 × 2(K,V) × 2(KV heads) × 128 × 2(bytes) = 40,960 bytes = 40 KiB`다. 실제 로드 config/tensor/file과의 대조는 GPU preflight에서 별도로 수행한다. 이 값에는 metadata/구조 토큰/file allocation이 포함되지 않는다.

로컬 조사본은 `/home/dblab/anaconda3/envs/mllm/lib/python3.10/site-packages/transformers`의 **5.13.1** 배포본이다. `direct_url.json`이 없어서 소스 git commit은 확인되지 않았다. 실행 환경은 부모 작업에서 GLM 전용으로 구성하고 해당 배포본 파일 SHA256 또는 설치 wheel/source commit을 freeze해야 한다. 기존 `mllm_ft`의 Transformers 4.57.6을 전역 변경하지 않는다.

## native 이미지 geometry와 mapping

Vision config는 depth 24, hidden size 1536, 12 heads, head dimension 128, patch size 14, temporal patch size 2, spatial merge size 2, output hidden size 4096이다. 정적 이미지는 같은 frame을 두 번 복제한 후 Conv3d patch embedding에 넣으므로 `image_grid_thw=[1,H,W]`다. Native processor는 패치 입력을 `(T, H/2, W/2, local_h, local_w, C, temporal_patch, pixel_h, pixel_w)` 순서로 flatten한다. 따라서 한 spatial 2×2 merge 그룹의 네 patch가 연속이고, 그룹 순서는 최종 visual token의 raster 순서다.

각 vision block은 frame별 전체 patch attention을 수행한다. CLS token 및 Qwen window reorder가 없다. 마지막 block index는 **23**이다. 마지막 block 이후 RMSNorm을 거친 hidden state를 `[-1,2,2,1536]`으로 묶어 Conv2d(kernel=stride=2, out channels 4096)에 넣고, learned projection/normalization/gated merger를 적용한다. 최종 visual content token 수는 `N=T×H×W/4`이고, processor가 `<|image|>` placeholder를 정확히 N개 확장한다. Begin/end image, system, role/boundary token은 content N에 포함하지 않는다. 중요도 계산은 이 learned downsampling이나 stock 출력값을 변경하지 않는다.

해상도 제어는 native **`size={shortest_edge: ..., longest_edge: ...}`**다. 이름과 달리 두 값은 공간 변 길이가 아니라 `smart_resize`에 전달되는 최소/최대 **temporal×spatial pixel budget**이다. GLM46V helper는 `num_frames=temporal_patch_size=2`, factor=`14×2=28`을 사용한다. checkpoint 기본값은 `{shortest_edge:12544,longest_edge:9633792}`이며 큰 이미지의 spatial area 상한은 대략 4,816,896 pixels, content N 상한은 6,144다. 28의 배수 rounding과 aspect ratio에 따라 실제 N은 달라진다. Qwen의 `min_pixels/max_pixels` 키를 복사하지 않는다. 본실험 resolution은 quality를 보기 전에 전용 preflight 결과를 바탕으로 모든 arm에 동일하게 freeze해야 한다.

## 구현 전 고정한 image-only 중요도

이 방식은 **GLM용 VisionZip-style adaptation**이다. VisionZip 논문의 공식 GLM 구현을 재현했다고 주장하지 않는다. 한 frame의 마지막 vision attention에서 stock QKV projection의 Q/K만 capture한다. Native vision RoPE를 적용한 Q와 K에 대해, vision head h와 patch query i/key j의 received attention을 다음과 같이 정의한다.

\[
A^{(h)}_{ij}=\operatorname{softmax}_{j\in\text{same frame}}\left(Q^{(h)}_i K^{(h)\top}_j / \sqrt{128}\right),
\qquad r_j={1\over 12}\sum_{h=1}^{12}\sum_{i\in\text{same frame}}A^{(h)}_{ij}.
\]

수치 규칙은 native dtype QK matmul/scaling, FP32 softmax, Q dtype로 probability cast, FP32 query 합산 및 head 평균이다. 이는 GLM의 eager post-softmax 경로를 따른다. 실제 요청은 frozen **SDPA** 그대로 실행하며, score 계산을 위해 eager 모델 forward로 바꾸지 않는다. Float32 softmax denominator에는 해당 frame의 **모든 유효 key**가 들어간다. Query만 최대 128행씩 blocking한다. Key pruning, query별 선택, 질문/정답/LLM attention 입력은 없다.

Native merge group `G_m={4m,4m+1,4m+2,4m+3}`에 대해 `s_m=(1/4)∑_{j∈G_m}r_j`로 최종 content token score를 만든다. 그룹 평균은 native Conv2d의 receptive field에 따른 score aggregation이며 Conv2d weight를 중요도에 추가로 사용하지 않는다. 모든 N개의 `s_m`를 한 번 stable descending sort하며 동점은 원래 token order를 유지한다. 이후 `k=ceil(N/4)`를 선택하고 모든 decoder layer/native KV head에 같은 ID를 적용한다. T1 visual forward와 N개 stock 출력은 전부 유지한다.

Hook은 normal T1의 `Glm4vVisionModel` 한 번에만 붙는다. `blocks[-1].attn` pre-hook으로 실제 `cu_seqlens`와 `(cos,sin)`을 읽고 `attn.qkv` output hook으로 Q/K를 copy한다. Stock tensors를 in-place 변경하지 않고 V는 보관하지 않는다. Context 종료 시 image-only score를 계산하고 Q/K와 hook을 해제한다. 추가 계산 wall time, QK copy bytes, score가 사용하는 live tensor allocation 표본, request 전체 GPU peak를 구분한다. score 전용 global peak reset은 request peak를 훼손하므로 하지 않는다. 표본 최고값을 완전한 allocator peak나 kernel workspace 측정이라고 부르지 않는다.

## native prefix, 3축 partial RoPE와 suffix

[공식 template](https://huggingface.co/zai-org/GLM-4.6V-Flash/raw/411bb4d77144a3f03accbf4b780f5acb8b7cde4e/chat_template.jinja)는 `[gMASK]<sop>`와 native role/image boundary token을 쓴다. User content를 image, text 순서로 전달하면 image가 질문보다 앞선다. Store prefix는 tokenized native prompt의 첫 토큰부터 **image end boundary까지**의 모든 KV다. Content는 `image_token_id=151363`의 N개이고 나머지 prefix token은 structure로 전부 보존한다. 질문, `/nothink`, assistant boundary 및 generated history는 suffix에서 매 요청 native template로 재구성한다. Prefix identity는 image hash뿐 아니라 model/processor revision, processor settings, system/prefix IDs, geometry와 position policy를 포함해야 한다.

`enable_thinking=False`를 template 인수로 명시한다. Template는 각 user message에 `/nothink`를 붙이고 assistant generation prefix에 빈 `<think></think>`를 넣는다. 따라서 이 두 문자열의 prompt 포함은 reasoning 생성의 증거가 아니다. 실제 tokenizer IDs와 output에서 nonempty reasoning, empty answer, cap truncation 여부를 별도 GPU gate로 확인한다. 초기 greedy cap은 16이다.

Text head의 첫 **64 dimensions**만 RoPE 대상이고 나머지 64는 그대로다. 32 inverse frequencies는 `[8,12,12]`로 나누어 T/H/W 위치를 사용한다. Decoder rotation은 **adjacent pair** `(x0,x1)→(-x1,x0)`이며 vision의 split-half rotation과 다르다. Post-RoPE K는 원래 bits로 저장/복원하고 재회전하지 않는다.

Image span이 native position a에서 시작하면 merged raster content `(t,h,w)`의 위치는 `(a+t,a+h,a+w)`다. Image content span 이후 text position은 `a+max(H,W)/2`에서 진행한다. Native `rope_delta=max(all native positions)+1-full prompt token length`를 보존한다. Physical compact cache length는 logical token ID나 native position으로 사용하지 않는다. Suffix 위치는 **full native prompt**에서 계산한 위치를 slice하고 decode는 마지막 native text 위치부터 각 축에서 1씩 증가시킨다. Compact cache는 retained prefix와 suffix의 causal attention을 유지해야 하며 미래 suffix는 보이지 않아야 한다. Model의 mutable `rope_deltas`를 image/dialogue 사이에 재사용하면 state leakage가 생기므로 매 요청 explicit positions 또는 검증한 per-request state를 전달해야 한다.

## 검증 범위

CPU tests는 blocking attention denominator, per-frame isolation, patch→merge mapping, stable ties, invalid geometry와 실제 작은 native vision model의 hook transparency를 검증한다. 이 tests의 작은 random weights는 checkpoint 추론/generation 정합성 gate를 대신하지 않는다. Full T1 logits/KV/IDs, SSD canonical/selected 복원, suffix/decode advancement, same-image different questions에 대한 checkpoint GPU gate가 완료되어야 pilot/main을 허용한다.
