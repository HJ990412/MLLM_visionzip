# GLM-4.6V-Flash PrefixStore-KV25 [B] 실험 계약 v6

이 문서는 결과 관찰 전 계약이다. 완료 판정은 실행 receipt로만 한다.
사용자 원문 REQUEST.md가 우선한다. 기존 LLaVA/Qwen 결과와 실패를 변경하지 않는다.

## 고정 설정

모델/processor/tokenizer: zai-org/GLM-4.6V-Flash,
revision 411bb4d77144a3f03accbf4b780f5acb8b7cde4e.
checkpoint의 model_type은 glm4v이고 processor는 Glm46VProcessor이다.
GLM 전용 .venv-glm46v: Transformers 5.13.1, 읽기 전용으로 기존 mllm_ft의
Torch 2.5.1+cu121을 사용한다. 전역 의존성을 변경하지 않는다.
RTX4090, decoder NF4 4-bit/double quant, BF16 compute/native KV, SDPA,
vision 전체(encoder/downsample/merger)와 lm_head는 BF16 비양자화,
모든 weight CUDA0, batch1, eval/inference, seed1234, greedy max_new_tokens32,
CPU/Torch/OMP/MKL threads2. 세 방법 모두 동일하다.

native image processor size={shortest_edge:12544,longest_edge:1605632}.
이는 GLM smart_resize가 반복 프레임 t=2를 포함한 면적을 제한하는 native
정책이다(14 patch, 2x2 merge). 1024 visual token 상한을 출발점으로 하되
실제 processor의 grid/메모리 검증이 필수다. 작은 입력을 속도를 위해 확대하지 않는다.
v3의 max16 출발 설정에서 두 파일럿을 수행했다. Native answer-boundary IDs
151361/151362가 tokenizer의 special IDs가 아니어서 skip_special_tokens만으로는
box 포장 전체가 prediction이 되는 포팅 누락을 확인했다. v3 raw/0% strict EM/
gate/실패/freeze를 별도 실행으로 보존하고 main은 실행하지 않았다.
또한 MT v3 B 120요청 중1개에서 16-token cap이 끝 boundary 전에 답을 잘랐다.
이 규칙은 gold나 EM 개선에 맞춰 answer 단어를 변경하지 않는다.
v4는 main 전에 공통 max_new_tokens=32를 새 run에서 고정한다. 기존 Qwen/LLaVA
max16과 다르며 latency 차이 해석에 명시한다. 이후 cap/질문/manifest는 변경하지 않는다.
Native generated IDs/전체 decoded text를 보존하고, 정확히 한 쌍의 native
151361/151362 안의 token span만 최종 VQA answer로 decode한다. 경계가 없으면
plain decode, 중복·깨진 경계면 literal decode와 INVALID 상태를 남긴다.
질문/gold는 extraction에 전달하지 않는다. Strict normalized EM/gold[0] scorer는
그대로다.

v4는 정상 파일럿 GQA720/MT360을 마친 뒤 main333요청에서 native 출력의
닫는 think delimiter를 gate로 잘못 처리해 중단했다. 모든 raw/freeze/partial
attempt를 별도 실행에 보존한다. 고정 실패 입력에서 수동 decode와 공식
model.generate의 위치/첫 logits/전체 IDs가 정확히 같았고, 실제 native
T1 body를 history에 넣으면 정상 출력이었다. Scoring용 추출 답변과 native
assistant body를 같은 field로 사용하는 포팅 누락을 확인했다.

v5는 scoring prediction과 history를 분리한다. history_text는 자기 실제
generated IDs에서 마지막 native EOS(로드한 generation_config의 실제 집합)에
해당하는 token 한 개만 제외하고 skip_special_tokens=False로 decode한 body다.
box/thinking control 및 whitespace를 보존하고 native chat template가 과거
assistant content를 공식 규칙으로 직렬화한다. 독립 tokenizer decode로
body IDs/text/lineage를 감사한다. Gold/다른 방법/정규화 prediction은 history에
넣지 않는다. 이 변경으로 입력 IDs가 달라져 v4 main333을 v5에 합치지 않는다.
v5 새 run에서 gate/pilot/main 모두 수행하고 native settings/cap32/scorer를 고정한다.

Native malformed/empty/cap/stray thinking delimiter 출력은 raw에 모두 기록한다.
enable_thinking=False의 실제 prompt를 유지하고 출력 anomaly를 숨기거나 좋은
출력을 얻을 때까지 재시도하지 않는다. 출력 형식/품질 저하 자체는 포팅
correctness 실패로 처리하지 않는다. 실제 KV/위치/mask/identity/파일 손상,
nonfinite inference 또는 자원 오류는 계속 실행을 차단한다. Thinking token
관찰 건수 및 TTFT의 native 첫 출력 token 해석을 별도로 보고한다.

v5는 요청36549 및 전체 KV/raw 감사를 완료했다. 독립 endpoint 검토에서
runner.persist의 identity 준비/wrapper-return tail과 activation summary/return
tail이 setup clock 밖에 있음을 확인했다. v5 원시 결과·hash·보고서 snapshot은
보존하며 완전한 outer setup 비용을 주장하지 않는다. v6는 **추론/model/processor/
해상도/history/cap32/importance/선택/질문/gold/membership을 변경하지 않고**
새 run에서 caller의 persist/activate 진입 직전부터 함수 반환 완료 직후까지
wall time을 측정한다. Identity 준비와 wrapper tail을 포함하며 기존 inner
측정값은 nested 진단값으로 따로 남긴다. Source E2E의 clone/score tail은
여전히 한 번만 귀속한다. v5 요청을 v6 분모에 섞지 않는다.

GPU guard, DONTNEED conditioning, raw/evidence journal fsync, post-image
독립 rehash와 adoption commit은 외부 benchmark/audit protocol overhead다.
3-turn session은 실제 serving request E2E와 필수 KV build/activation setup의
합이며 전체 실험 executable wall time으로 부르지 않는다.
native chat template enable_thinking=False, tools 없음.
system='You are a helpful assistant. Give concise factual answers.';
question 뒤 'Answer using a single word or short phrase.'를 붙인다.

## 범위와 저장

첫 user의 image를 질문보다 먼저 넣는다. prefix는 첫 expanded image 뒤의
native end_of_image까지이며 system/user/image boundary 모두 구조로 보존한다.
content는 image_token_id(151363)의 native expanded 행, N은 원래 개수,
k=ceil(N/4). 모든 native decoder layer/KV head에 stable descending
image-only rank 한 개를 적용한다. 중요도 정의는 glm46v_architecture.md에서
구현 전에 고정한다. 이는 VisionZip-style GLM adaptation이며 논문의 GLM 재현이 아니다.

FullLoad/B 모두 각 이미지 고정 source dialogue T1의 정상 full pixel forward에서
KV를 capture한다. B만 마지막 vision attention의 Q/K를 같은 pass에서 capture한다.
별도 vision/prefix forward는 없다. Full CPU KV materialization와 full repack 후
직렬화만 B의 k개 content로 제한한다. 구조/전체 permutation/원래 IDs/geometry/
native 3축 positions/partial RoPE 설정과 offset을 보존한다. Native KV heads를
확장하지 않는다. 64-row token-major contiguous layout, exact short EOF, padding0.
두 writer에 동일 fsync, payload hash, atomic completion seal, activation full hash를 적용한다.
활성화 후 metadata/FD만 보유하고 요청마다 실제 pread한다. Decode 재읽기/재선택0.

post-RoPE K는 그대로 읽는다. compact physical length는 cache 및 mask에만 사용한다.
suffix/decode native positions는 원래 full tokenization의 get_rope_index로 정한다.
새 dialogue마다 fresh cache와 method-local generated history를 사용하고 rope_deltas를
초기화한다. store는 image/model/processor/system/prefix/code/environment identity를 검증한다.

## 사전 gate와 실행 순서

CPU: independent rank/payload bits/budget/chunk boundary/patch mapping/native position/
suffix causal visibility/identity/중복/누락/미래 mask/짧은 파일/미완 seal negative fixtures.
GPU: stock vs capture T1 출력/KV/첫 logits/전체 생성 ID exact; canonical vs FullLoad
payload exact; independent same-shape memory vs 실제 B SSD의 KV/IDs/positions/mask/
첫 logits/생성 ID exact; all-layer dense masked vs compact FP64 attention atol=rtol1e-12
(Q/K/V를 연산 전에 FP64로 변환). BF16 main SDPA를 바꾸지 않는다.
FP32 dense/compact atol=rtol1e-5는 진단이며 failure도 보존한다.
ReComp/FullLoad native shape 차이는 별도 진단하고 B/full logit 동등성을 요구하지 않는다.
min/max N, k%64 양측, longest question, same-image 여러 dialogue를 CPU preflight로
고른 manifest를 결과 관찰 전에 동결한다. unresolved semantics이면 pilot/main 차단한다.

기존 pilot membership을 그대로 사용: GQA40x6x3=720, MT40x3x3=360.
통과 후 main398 images/4061 dialogues/3 turns/3 methods=36549 final 요청;
방법당12183, followups8122. workload SHA256
0287e0c57813800c781633b969c5cff336b3a3c1a1bdcdbb56d63f6ddab0ca62.
image-major, canonical ordinal modulo3 method rotation, turn마다 같은 rotation.
모든 dialogue/방법 T1 full pixels, image별 source T1에서만 store build.
GQA pilot 6질문은 각 독립 요청(history 없음), 첫 source 질문만 capture.
raw에 IDs/history lineage/source/selection/I/O/attempt를 기록한다.
재시도는 전체 image 새 attempt, first complete success만 final 채택.

## 계측과 분석

TTFT는 input/history 준비 및 image open 이전부터 첫 token materialization/CUDA sync까지.
request E2E는 실제 outer return까지이며 capture/score tail을 정확히1회 포함한다.
persistence는 CPU materialization/repack/write/fsync/seal 포함 outer writer return;
activation 별도. nested 시간을 중복 가산하지 않는다.
실제 pread bytes/calls/ranges, content/structure/padding, H2D bytes/time, assembly,
logical/allocated/write bytes, GPU allocated/reserved peak scope를 기록한다.
NOT_MEASURED를0으로 채우지 않는다. OS I/O는 NAND physical I/O가 아니며
DONTNEED는 cold NAND 보장이 아니다. foreign GPU 작업이 있으면 대기/중단한다.
30GiB reserve와 최대 store/진단 공간을 점검하고 기존 artifact를 삭제하지 않는다.

raw를 독립 strict normalized EM(gold[0]) 재계산한다. T1/T2/T3/all/followup,
mean/p50/p95, request-weighted와 image-macro 구분. image-paired cluster bootstrap
10000회 seed1234, pooled sum/count로 B-ReComp/B-FullLoad accuracy/TTFT/E2E/SSD/
session CI. 생성 history 차이를 포함한 방법 비교이며 CI0포함은 동등성이 아니다.
shared stream setup은 image별1회, standalone-equivalent는 setup을 dialogue마다
재부과한 DERIVED; 독립 cold session 실측 아님. 비-source capture 추가비용 NOT_MEASURED.

환경/연결/gate 불충족으로 중단하면 실제 완료 수와 이유를 기록하고 성능을 만들지 않는다.
