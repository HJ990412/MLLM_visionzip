# Qwen2.5-VL MT-GQA Generated-History 본실험: ReComp / FullLoad / Ours-KV25

마지막의 `=== END OF QWEN MTGQA KV25 MAIN PROMPT ===`까지 읽고 실행하라.
도구 출력이 잘리면 다음 줄 범위부터 이어서 읽어라. 계획 작성이나 실행 시작을 완료로
보고하지 않는다. 검증·공간 조건을 충족하면 전체 본실험, 독립 감사, 분석을 실제 수행하라.

## 1. 목표와 범위

저장소: /home/dblab/hj/mllm_v2
모델: Qwen/Qwen2.5-VL-7B-Instruct
데이터: 이미 LLaVA 본실험에 사용한 frozen MT-GQA-reconstructed 전체.
History: model/method/dialogue별 실제 이전 생성 답변을 사용하는 generated history.

Main methods는 정확히 세 개다.
- ReComp: 모든 turn에서 pixels부터 정상 재계산.
- FullLoad: 후속 turn에서 SSD의 전체 visual prefix KV를 재사용.
- Ours-KV25: 검증된 image-only repack + 약 25% content KV + sequential whole-chunk read.

규모:
- 398개 이미지, 4,061개 대화, 대화당 3턴.
- 방법당 12,183요청 = T1 4,061 + T2/T3 8,122.
- 3방법 총 36,549 final logical requests, T2/T3 비교 모집단 총 24,366요청.
- Validation/smoke/retry는 main final coverage와 별도로 센다.
ReComp의 T2/T3는 hit 비교 모집단이지만 실제 SSD-KV cache hit는 아니다.

Qwen MPIC, SparseVLM AllHead/Probe3, ReKV, QA-Chunk 및 legacy Ours-Chunk25는
main에서 제외한다. MT-VQA, 추가 모델, 해상도·budget·chunk sweep은 실행하지 않는다.
완료된 LLaVA 5방법 main은 재실행하거나 변경하지 않는다.
이번 목적은 검증된 Qwen KV25를 전체 MT-GQA로 확대하는 것이지 새 방법 설계가 아니다.

## 2. 로컬 근거와 실행 코드 확인

우선 다음 실제 로컬 문서와 연결된 runner/config/raw/validation을 읽어라.
- docs/qwen25_correctness_contract_v2.md
- docs/qwen25_kv25_budget_contract.md
- docs/qwen25_kv25_pilot_timing_addendum.md
- docs/qwen25_kv25_pilot_timing_erratum.md
- mmimpress/qwen25/runner.py, store.py, vision.py 및 현재 Qwen MT pilot runner.
- scripts/91_validate_qwen25_kv25.py 및 연결된 독립 참조·회귀 테스트.
- results/qwen25_correctness_v2_20260928T081111Z/REPORT.md
- results/qwen25_kv25_migration_20260929T042949Z/REPORT.md와 독립 감사.
- data/mt_gqa/dialogues.json 및 config/stats/provenance.
- 2026-10-01 오전 09:10 KST 완료로 보고된 최신 LLaVA main의 실제 run/results,
  frozen manifest, timing/storage 계약, 최종 independent audit.

최신 LLaVA run은 ReComp/FullLoad/MPIC-32/SparseVLM-SSD-KV25-AllHead/Ours-KV25의
60,915요청 run이다. 과거 ReKV 포함 run 또는 Chunk25 run과 혼동하지 않는다.
해당 경로는 로컬에서 method registry와 완료 audit로 찾아 고정한다.
찾지 못하면 과거 결과로 대체하지 말고 cross-model 결과 추출만 BLOCKED로 표시한다.
Qwen 자체의 frozen 데이터와 필수 검증이 확인되면 Qwen 실행은 별도로 진행 가능하다.

GitHub에 아직 없는 로컬 변경을 과거 공개 코드로 되돌리지 않는다.
기존 Qwen KV25가 없으면 DEPENDENCY_MISSING이며 Chunk25를 이름만 바꾸지 않는다.
기존 LLaVA main의 scheduling/checkpoint/audit 구조는 재사용 가능하지만,
LLaVA 모델 클래스, prompt, separator, FP16 writer, cache/position 로직을 복사하지 않는다.
별도 Qwen 3-arm runner/schema/result 경로로 연결하고 기존 runner의 기본 의미는 보존한다.

새 docs/qwen_mt_gqa_kv25_main_contract.md에 실제 entry point, method ID 대응표,
config, environment, source, timing, dataset, storage plan과 중단 기준을 고정한다.
실제 core model/store revision과 새 orchestration revision은 따로 기록한다.
Git 상태를 확인하되 HEAD가 없는 로컬 사본이면 재초기화하지 말고 source snapshot/hash/diff로
추적한다. 사용자 미커밋 변경과 기존 코드·data·store·results는 보존한다.
reset/clean, 무단 upgrade/commit/push, 기존 결과 수정은 금지한다.
Main 시작 후 코드나 실험 조건이 바뀌면 새 experiment version으로 분리한다.

## 3. 동일 데이터와 inference 설정 고정

기존 전체 MT-GQA 기준:
index SHA-256:
2c47cfad2a7ccbb673042b400304d7f3ca03d6fbe59d04fa83db50708c924224
workload SHA-256:
0287e0c57813800c781633b969c5cff336b3a3c1a1bdcdbb56d63f6ddab0ca62

실제 398/4,061/12,183 counts, 파일 hash와 ordered image/dialog/question IDs를 확인한다.
Hash 차이가 JSON 직렬화나 경로만의 차이라면 ordered Q1/Q2/Q3 IDs·text·gold·이미지
identity를 독립 비교해 증명하고 새 raw hash와 차이를 남긴다.
실제 membership/grouping/turn order가 다르면 DATASET_MISMATCH로 중단한다.
40-dialogue pilot을 full main으로 사용하거나 누락 질문을 임의 대체하지 않는다.
실행은 이미지 단위로 분할해도 각 대화 구성과 turn 순서는 바꾸지 않는다.
이 데이터는 MT-GQA-reconstructed다. MetaCompress의 exact released artifact 재현이라고
쓰지 않으며 논문에 보고된 대화 수 일치만으로 identity를 주장하지 않는다.

Qwen KV25 계약의 기준값을 실제 로컬 검증 설정과 대조해 동결한다.
- checkpoint/processor/tokenizer revision: cc594898137f460bfe9f0759e9844b3ce807cfb5
- NF4 weights, BF16 compute, native BF16 SSD KV payload, production SDPA.
- min_pixels=200704, max_pixels=802816.
- batch_size=1, eval/inference mode, seed=1234, greedy, max_new_tokens=16.
- chunk_size=64, ratio=0.25, Ours budget_unit=visual_kv.
- 실제 config 기준 28 decoder layers, 4 native KV heads, head_dim=128을 확인한다.
값이 다르면 검증 이력과 차이를 먼저 조사한다. 결과를 보고 유리한 설정을 고르지 않는다.
LLaVA의 eager/FP16/AnyRes 설정에 Qwen을 억지로 맞추지 않는다.
CPU/disk model-weight offloading, 다른 checkpoint, 해상도 확대, backend 교체는 금지한다.

## 4. Qwen Ours-KV25 경로를 명시적으로 연결

기존 Qwen의 default budget_unit은 chunk일 수 있다.
신규 Ours는 실제 request/load 호출에 budget_unit=visual_kv를 반드시 명시하고,
registry 이름과 별개로 per-request actual keep count를 검증한다.

N = 실제 PatchMerger 이후 LLM prefix의 non-structural image-content token 수.
k = ceil(0.25*N) = (N+3)//4.
m = ceil(k/64) = (k+63)//64.
N<=0은 fail-closed한다. 구조 토큰·system/user prefix·padding은 N에서 제외한다.
선택 단위는 한 token 위치의 모든 native KV heads의 K와 V다. 모든 layer에 같은
image-only permutation과 selected original-token 집합을 적용한다.

Ours의 정상 hit:
1. 이미 T1에서 정해진 importance 순서의 stored real rows [0,k)를 선택.
2. 이를 담은 [0,m)개의 whole chunks를 layer별 K/V의 연속 span으로 읽음.
3. 원래 padded final chunk/EOF 규칙대로 실제 bytes를 계상.
4. 초과 real rows와 padding을 제거한 뒤 H2D, original logical order로 복원.
5. 필요한 모든 structural prefix와 결합해 기존 P2 compact cache로 inference.

기존 vision received-attention, merge/window mapping과 stable tie 정책은 변경하지 않는다.
FullLoad는 canonical full-content cache를 그대로 재사용한다.
Ours가 읽은 extra rows를 attention에 보이게 하거나 last chunk를 partial byte-read로
바꾸지 않는다. 전체 Visual KV 저장본은 보존하며 25%를 storage compression이라 부르지 않는다.
Ceil 때문에 실제 k/N은 25%를 약간 넘는다. logical retention과 실제 SSD read ratio는 다르다.

Qwen에 LLaVA newline separator를 가정하지 않는다. 실제 prefix IDs의 구조 토큰을 따른다.
BF16 raw-bit 저장을 유지하고 Q-head 수만큼 repeat한 K/V를 SSD에 저장하지 않는다.
물리 stored index, compact cache slot, original sequence position, 3축 MRoPE를 구분한다.
Post-MRoPE key에 rotary를 다시 적용하지 않으며 suffix/decoding의 logical MRoPE를
남은 token 수로 당기지 않는다. cache_position/causal mask는 기존 v2 방식으로 처리한다.
각 hit에는 vision forward=0, online image importance/query scoring=0이어야 한다.

## 5. Generated history와 T1/source-store 처리

각 method×dialogue는 history를 새로 시작한다.
T1: pixels+Q1 → 해당 방법의 실제 A1.
T2: 같은 image context+Q1/A1+Q2 → 해당 방법의 실제 A2.
T3: 같은 image context+Q1/A1+Q2/A2+Q3 → 해당 방법의 실제 A3.

기존 검증된 Qwen chat-template/prompt renderer와 short-answer instruction을 유지한다.
이미지는 첫 message에 한 번만 나타나고 text/history는 재사용 prefix 뒤의 suffix다.
모든 방법·모든 dialogue의 T1은 full-image inference를 실제 수행한다.
ReComp의 T1 답변이나 시간을 다른 방법에 복사하지 않는다.
Gold는 scorer에만 주고 prompt/selection/capture membership으로 전달하지 않는다.
History에는 해당 방법의 실제 decoded prediction을 넣으며 scorer normalization,
정답 치환, 답변 보정은 하지 않는다. 정상 EOS/빈 답변/cap 도달도 그대로 기록한다.
Qwen 다른 방법이나 LLaVA의 답변을 가져오지 않는다.
각 history 항목에 source model/method/dialogue/turn/logical/physical ID를 연결한다.

Fresh 실행은 이미지별 첫 source dialogue의 정상 T1 forward에서 해당 store에 필요한
KV/score를 piggyback capture한다. 별도 vision/full-prefix forward를 추가하지 않는다.
이후 같은 이미지의 대화들은 store를 재사용하되 dialogue history는 공유하지 않는다.
재사용 prefix에는 질문·답변 KV를 넣지 않고 prefix IDs/geometry/positions 호환성을 검사한다.
다른 T1 길이의 NF4 실행에서 prefix bits가 무조건 같다고 가정하지 않는다.
하나의 명확한 source capture를 고정하고 그 동일 캡처에서 만든 in-memory reference와 비교한다.
Model 내부 rope_deltas 및 mutable state는 각 request의 올바른 값으로 처리한다.
Persistent text/history KV 재사용이라는 추가 최적화는 하지 않는다.

## 6. SSD 공간과 이미지 단위 실행

실제 SSD mount/device/free space, 가장 큰 입력 geometry, store+임시 버퍼+raw 크기를
측정/산정해 storage_plan.json을 main 전에 고정한다. 과거 35 GiB 등은 현재 값으로 쓰지 않는다.
가능하면 LLaVA main과 같은 SSD/hardware에서 실행하고 다른 경우 명시한다.

기본은 이미지별 fresh-streaming이다.
- 한 이미지에 필요한 canonical FullLoad와 importance-repacked Ours store를 만든다.
- 그 이미지의 모든 대화를 3방법으로 평가한다.
- raw/history/선택/시간/I/O와 source capture·store 생성 정보·payload hash를 보존한다.
- 이미지별 completeness/audit 및 durable atomic commit 뒤 이번 run의 scratch payload만 정리한다.
- 다음 이미지로 진행한다. Shard는 관리 단위이지 여러 이미지 store 동시 보유 조건이 아니다.

정리 가능한 것은 새 run이 생성하고 disposable로 등록한 scratch뿐이다.
기존 store/source/data/results나 외부 symlink/hardlink 대상은 건드리지 않는다.
Resolved path, 생성 기록, allowlist를 검사하고 cleanup receipt와 재생성 recipe를 남긴다.
실패 이미지의 미완료 evidence를 자동 삭제하거나 raw를 지워 공간을 확보하지 않는다.
기존 대형 store를 snapshot 명목으로 통째로 복제하지 않는다.

FullLoad/Ours는 physical layout이 다르므로 같은 파일이라고 가정하지 않는다.
Source capture나 불변 structural sidecar를 공유하려면 실제 호환성·hash·비용 귀속을
검증하고 공유 내역을 남긴다. 요청 간 visual payload의 CPU/GPU residency는 허용하지 않는다.

안전 여유는 기존 검증 정책을 바탕으로 prospectively 고정하고, largest-image peak,
build 중복과 raw 증가량을 포함한다. 실행 중 여유 기준을 낮추거나 safety check를 끄지 않는다.
호환 기존 store의 read-only 재사용은 대안이나 provisioning mode를 시작 전에 명시한다.
RO에서는 persistence=NOT_REMEASURED이며 cold-start session 속도 향상을 새로 주장하지 않는다.
Fresh와 RO가 섞이면 image별 범위와 별도 집계를 공개하고 setup 측정으로 합치지 않는다.
최소 작업 공간도 부족하면 BLOCKED_STORAGE로 둔다. 기존 파일을 지우지 않는다.

이 방식은 payload-cold SSD serving 평가다. 동시에 수백 이미지 cache를 유지하는
online scheduler나 CPU/GPU 메모리 포화·concurrent throughput을 직접 검증했다고 쓰지 않는다.

## 7. v2 통합 검증 → 36요청 smoke → full main

기존 Qwen KV25 CPU/GPU 검증 및 v2 허용치/독립 reference를 재사용한다.
이전 v1 strict-logit FAIL, 최초 v2 UNRESOLVED, 후속 v2 PASS artifact는 모두 보존한다.
새 wrapper 통합으로 생긴 차이를 임의로 정상 수치 차이라고 면제하거나 tolerance를 바꾸지 않는다.

기존 고정 10쌍과 geometry/길이를 고려해 사전 선정한 complete-dialogue smoke로 검사한다.
기존 10쌍을 새로운 holdout이라고 쓰지 않는다. main 전체의 입력 geometry/길이 분포는
사전에 점검하며 OOM sample을 결과에서 조용히 빼지 않는다.

필수 integration gates:
G1. 전체 데이터 identity, 3-arm registry, 모델/환경/visual_kv budget 명시.
G2. Native BF16 round-trip, canonical↔repacked 100% inverse identity.
G3. FullLoad SSD 대 동일 captured-prefix in-memory reuse의 v2 정합성.
G4. Ours SSD 대 독립적으로 score를 stable-rank/gather한 canonical KV25 reference.
    Production selector/loader/assembler를 그대로 reference에도 사용하지 않는다.
    동일 compact shape/order/backend에서 KV bits, mask, first logits, generated IDs를 비교한다.
G5. 모든 layer/native head의 content rows=k, 구조 토큰 보존, extra/padding 비노출.
    읽힌 extra rows의 유한 sentinel 교체가 compact KV/출력에 영향을 주지 않아야 한다.
G6. Stock full-logical MRoPE·causal visibility·decode cache positions 및 state isolation.
G7. 동일 Q/K/V 기반 독립 dense-mask/compact FP32 attention oracle와 negative fixtures.
G8. 실제 source T1 full forward=1, hit vision/scoring=0, 동일 image selected IDs 불변.
G9. Model/method/dialogue별 generated-history lineage, future/gold leakage 없음.
G10. 실제 pread trace와 planned ranges/returned bytes, 최소 ceil(k/64) whole chunks 일치.
G11. 기존 Qwen legacy 회귀와 완료 LLaVA/main/Qwen artifact 보호 확인.
     LLaVA GPU를 재실행하지 않았다면 별도로 NOT RUN; 이를 실행한 PASS로 표기하지 않는다.
G12. Timer 경계, callback/adapter 예외 복구, unique IDs/resume/atomic commit/scratch 안전성.

Matched-path exact invariant는 기존 기준으로 요구한다. Full-vs-split NF4/BF16 및
서로 다른 attention shape의 수치 진단은 별도 보존한다. 실제 structural error나
설명되지 않은 새 divergence는 UNRESOLVED이며 해당 main을 보류한다.
GPU 검증 자체가 미실행이면 통과라고 표시하지 않는다. 정확도·속도 향상은 gate가 아니다.

검증 통과 후 frozen main에서 4개 서로 다른 이미지의 대화 1개씩으로 smoke를 한다.
4 images × 3 turns × 3 methods = 36요청.
Boundary extra rows, 다른 image geometry 및 history lengths를 포함한다.
Smoke 통과 후 4,061개 대화 전체를 새 main에서 실행한다.
Smoke/과거 pilot 결과를 main으로 복사하지 않는다. 해당 dialogue도 main에서는 다시 실행한다.

## 8. True TTFT, I/O 및 runtime 공정성

최신 Qwen timing addendum/erratum을 따르고 오래된 decode-excluded 문구를 우선하지 않는다.
모든 방법의 T1과 ReComp T2/T3는 요청마다 이미지 파일을 실제로 열고 RGB decode한다.
이 시간과 Qwen processor/vision/full prefill은 pixel-request TTFT/E2E 안에 포함한다.
FullLoad/Ours hit에서는 이미지 파일을 decode하지 않는다. decode 필드는 null/not-applicable다.
Image SHA 검증은 요청 전에 수행하며 request image decode와 혼동하지 않는다.

Paper-facing TTFT는 request 시작(입력 준비 전)부터 첫 token materialization+CUDA sync까지다.
Hit에서는 prompt/chat-template/tokenization, 필요한 SSD pread, H2D, compact assembly,
suffix prefill과 첫-token 처리가 모두 포함된다. Request E2E는 전체 생성 종료까지다.
가능하면 절대 request-start/first-token/end timestamp로 계산한다.
기존 runner의 decode_ms 등 구성값을 더할 때 이미 포함된 image decode를 두 번 더하지 않는다.
Core TTFT와 end-to-end TTFT, postprocess·logging 시간을 구분한다.

모델 로딩/공통 warmup/결과 logging/전체 hash 검사는 TTFT 밖이다.
Store activation/full payload integrity 검사는 activation 비용으로 측정하고,
그 작업이 page cache를 데웠으면 실제 hit 전에 동일 page-cache conditioning을 다시 한다.
POSIX_FADV_DONTNEED 등의 conditioning은 timer 밖에서 수행하고 성공/실패를 기록한다.
이것이 SSD controller/NAND cold 보장이나 O_DIRECT라고 주장하지 않는다.
매 hit 안에서 전체 payload hash scan을 실행하지 않는다.

Metadata-ready 조건으로 작은 불변 metadata를 상주시킬 수 있으나 bytes/위치/activation을
공개한다. 이전 요청의 visual KV payload를 RAM/GPU에 남겨 읽기를 생략하지 않는다.
ReComp의 SSD KV 읽기는 0이지만 raw image I/O도 0이라는 뜻은 아니다.
Normal K/V, structural, metadata 및 raw image read 범위를 각각 구분한다.
Actual OS-returned bytes/preads와 valid content/padding/H2D bytes를 따로 계상한다.
Actual NAND traffic을 측정했다고 쓰지 않는다.

Method 실행 순서는 dialogue ordinal에 따른 deterministic 3-way rotation으로 균형화한다.
한 GPU에서 Qwen과 LLaVA 또는 여러 방법을 동시에 benchmark하지 않는다.
다른 작업을 강제 종료하지 말고 동시 GPU/SSD 부하가 있으면 측정을 보류·기록한다.
Heavy audit/hash/backup은 timed request와 겹치지 않게 한다.
Peak GPU allocated/reserved, active metadata/compact-cache bytes를 보고한다.
Overlapping I/O/H2D/prefill component를 합산해 total이나 순수 GPU 시간을 만들어내지 않는다.

## 9. Persistence와 session 비용

Fresh 실행에서는 이미지별 source T1 뒤 실제 score 후처리/permutation/repack,
CPU clone/transfer, write/fsync/publication/activation의 시간과 bytes를 기록한다.
Core runner가 E2E 뒤 수행한 KV clone도 setup 비용에서 누락하지 않는다.
정상 T1에 이미 포함된 capture 비용을 또 더하지 않는다. 추가 forward가 없다는 이유로
score extraction/QK 계산 비용을 0이라고 쓰지 않는다.

Actual shared-image stream:
실제 T1+T2+T3 request E2E와 그 source dialogue에 귀속된 한 번의 persistence,
실제로 발생한 activation을 중복 없이 합친다. 다른 dialogue에 write를 반복 청구하지 않는다.

Standalone-equivalent:
각 3턴 대화가 별도 provisioning한다고 가정한 환산은 DERIVED다.
Image별 실측 setup을 어떤 식으로 재부과했고 capture/activation을 어떻게 처리했는지
공개한다. 직접 모든 dialogue를 fresh로 반복 측정한 값이라고 쓰지 않는다.

RO 재사용이면 setup/cold-start는 NOT_REMEASURED다. 과거 persistence를 새 measured 표에
붙이지 않는다. 실제로 구현·측정하지 않은 background overlap을 가정하지 않는다.
Hit TTFT 감소를 짧은 cold-start session 전체의 개선이라고 확대 해석하지 않는다.

## 10. Raw, restart, 독립 감사

Logical ID에는 experiment version/model/method/dialogue/turn을 포함하고,
실제 attempt마다 별도 physical execution ID를 부여한다.
Final logical row는 하나만 채택하며 failures/retries/interrupted attempts는 따로 보존한다.
정답이나 latency에 유리한 attempt를 골라 채택하지 않는다.

이미지별 checkpoint는 모든 dialogue×turn×method coverage와 source capture, config/code
hash, history lineage, request raw를 포함해 fsync 및 atomic publish한다.
동일 config/code/manifest의 완전한 image artifact만 resume에서 건너뛴다.
Partial image는 검증된 continuation state가 없으면 해당 이미지 전체를 새 attempt로
재실행하고 이전 partial을 남긴다. 서로 다른 attempt의 history를 임의 혼합하지 않는다.

Raw 필수 내용:
- image/dialog/question/turn/model/method, logical/physical IDs, provisioning/source-store IDs.
- 실제 history text와 source lineage, input IDs 또는 재검증 가능한 보존 표현/hash.
- first/generated token IDs, prediction, gold/score는 inference 입력과 별도 경로.
- N_content/S_structural/k/m, selected stored/original IDs, permutation provenance.
- layer별 compact lengths/keep counts, padding/extra rows, actual read bytes/preads, H2D bytes.
- absolute timing endpoints, request/core TTFT/E2E, setup/capture/activation, status/retries.

성능을 오염시키는 모든 hidden/logit full dump를 main에 추가하지 않는다.
필요 tensor dump는 validation fixture에 제한한다. Raw를 hash만 남기고 버리지 않는다.
선택 ID가 image별 동일하면 불변 selection artifact를 참조해 중복 저장을 줄일 수 있으나
independent auditor가 raw에서 원래 값을 복원·대조할 수 있어야 한다.
새 파일의 무손실 압축은 허용하되 기존 원본을 무단 삭제하지 않는다. 대형 raw 무단 Git 추가 금지.

Independent auditor는 runner의 summary를 복사하지 말고 raw에서 재계산한다.
- 398 images/4,061 dialogues/3 turns, ordered membership과 원본 identity.
- 방법당 final 12,183, T1 4,061, T2/T3 8,122; 전체 36,549.
- unique finals, 실제 execution ID 및 누락/중복/실패/retry/attempt lineage.
- 모든 history가 같은 method/dialogue의 앞선 실제 출력에 연결됨.
- Ours의 실제 visual_kv 예산, zero hit vision/scoring, 과잉 rows 비노출 및 I/O 계상.
- 정확도·시간·retention·read ratio·setup 표와 원시자료의 일치.
- 기존 보호 대상 무변경과 의도한 source diff, 허용된 새 scratch만 정리했음.

## 11. 결과 표, 통계 및 LLaVA와의 비교

기존 프로젝트 strict normalized exact-match scorer를 유지한다.
모델별 prediction은 그대로 보존하고 normalization은 채점에만 적용한다.
주 품질 지표는 T2/T3 hit accuracy, 주 성능 지표는 pooled 8,122건 hit TTFT다.
T1/T2/T3·all-turn 정확도와 정답 수/모집단을 함께 제시한다.
T1이 full-image라 all-turn 평균에서 손실이 희석될 수 있음을 명시한다.

Ours 대 ReComp/FullLoad의 paired hit accuracy 차이(%p), hit TTFT 차이(ms),
감소율/speedup과 95% CI를 계산한다.
Image-cluster bootstrap 10,000회, analysis seed=1234.
각 resample은 한 이미지의 모든 대화·turn·paired methods를 함께 포함하고,
point estimate와 같은 request-weighted estimator를 사용한다.
같은 이미지 내 질문을 독립 표본으로 취급하지 않는다. Image-balanced 요약은 별도 diagnostic이다.
CI가 0을 포함해도 equivalence나 무손실을 주장하지 않는다.
Generated-history가 달라진 뒤의 품질 차이는 method-level 누적 효과다.
동일 텍스트 조건의 순수 KV 선택 효과만 측정한 것처럼 해석하지 않는다.

[Qwen main]
| Method | Acc T1 | Acc T2 | Acc T3 | All Acc | Hit Acc | T1 TTFT | Hit TTFT mean/p50/p95 | SSD MB/hit |

[Budget/I/O]
| Method | Budget unit | Mean N | Content kept % | Structural-included kept % | K/V/structural bytes | FullLoad 대비 total read ratio | Preads/hit |
Image별 ratio 평균과 합계 token/byte 비율을 구분한다. MB=10^6 bytes, GiB=2^30 bytes.

[Paired comparison]
| Ours vs | Δ Hit Acc (%p), CI | Δ Hit TTFT (ms), CI | TTFT 감소율 | Speedup | SSD 감소율 |
Baseline SSD=0인 ReComp에 대한 SSD 감소율은 N/A다. 0으로 나누거나 100% 절감이라고 쓰지 않는다.

[Setup/session]
| Method | Provisioning mode | Measured setup/persistence | Actual-stream session E2E | Derived standalone E2E |
미측정 값은 N/A/NOT_REMEASURED로 둔다.

추가로 visual N/retention/실제 읽기 비율 분포, generated lengths/cap 도달, peak GPU memory,
T2↔T3 Ours selected-set invariance, ReComp/FullLoad prediction agreement를 보고한다.

완료된 최신 LLaVA main 원본을 읽기 전용으로 확인할 수 있을 때만 cross-model 표를 만든다.
LLaVA의 5개 방법 main 표는 그대로 보존하고 공통 3방법만 secondary 표에 추출한다.
| Backbone | Method | Hit Acc | Hit TTFT | SSD MB/hit | Actual content retention |
해당 run IDs, raw/audit hashes, 동일 데이터 확인 및 timing 경계 차이까지 명시한다.
사용자가 요약해 준 숫자를 raw 대신 하드코딩하거나 과거 Chunk25/ReKV run을 섞지 않는다.

비교의 핵심은 각 모델 내부에서 Ours가 ReComp/FullLoad 대비 갖는 quality-latency-I/O
trade-off다. 서로 다른 backbone/precision/backend/visual-token 수의 절대 속도 차이를
저장 기법 하나의 인과 효과로 해석하지 않는다. Qwen에는 AllHead를 실행하지 않았으므로
Qwen에서도 query-dependent 방식보다 우월하다고 일반화하지 않는다.

## 12. 산출물과 최종 상태

새 경로 예시(기존 파일과 충돌하면 일관된 새 이름 사용):
- docs/qwen_mt_gqa_kv25_main_contract.md
- 별도 Qwen 3-arm orchestrator/runner/report/audit 및 필요한 integration tests.
- runs/qwen_mt_gqa_kv25_main_<timestamp>/
- results/qwen_mt_gqa_kv25_main_<timestamp>/
- frozen config/environment/code/model/manifest hashes, storage_plan.json,
  validation.json, per-image raw/attempts/checkpoints, summary.csv,
  paired_quality.csv, paired_latency.csv, budget_io.csv, persistence/session 표,
  independent_audit.json, 한국어 REPORT.md, REPRODUCE.md.

최종 한국어 보고서에 다음을 실제 근거와 함께 명시한다.
DATASET IDENTITY / METHOD-CONFIG CONTRACT: PASS / FAIL.
GPU INTEGRATION UNDER v2: PASS / FAIL / UNRESOLVED / NOT RUN.
SMOKE: actual/36, PASS / FAIL / NOT RUN.
QWEN 3-ARM MAIN: VALID UNDER v2 / INVALID / PARTIAL / NOT RUN; actual/36,549.
INDEPENDENT AUDIT / PROTECTED ARTIFACTS: 실제 판정.
PERSISTENCE/SESSION: MEASURED / DERIVED / NOT_REMEASURED, 적용 범위.
CROSS-MODEL TABLE READY: YES / NO 및 identity/timing 비교의 제한.

대기·실행 중·일부 완료·전체 완료·감사 완료를 구별한다.
환경/GPU/공간/검증 문제로 막히면 완료 범위, 실패 원인과 정확한 재현·resume 명령을 남긴다.
조건을 바꿔 성공한 척하거나 장기 실행을 시작만 해놓고 최종 VALID로 보고하지 않는다.
기존 LLaVA 결과는 이번 실행으로 다시 생성하거나 덮어쓰지 않는다.

=== END OF QWEN MTGQA KV25 MAIN PROMPT ===
