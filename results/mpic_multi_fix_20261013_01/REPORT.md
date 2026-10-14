# FullLoad 수정 · MPIC 다중 이미지 · MMDU 소규모 검증

2026-10-13 KST. **코드 수정과 가능한 CPU 검증은 완료했다. pretrained GPU 검증과 MMDU 생성은 환경상 완료하지 못했다.** GPU 실행을 명시적으로 시도했으며, NVIDIA driver 통신 실패와 torch CUDA device_count=0 때문에 weights를 로드하지 않고 exit3/NOT VERIFIED를 남겼다. 준비한 MMDU는5대화/15턴, 실제 생성 완료는 **0대화/0턴**이다. TTFT·품질 값을 만들어내지 않았다.

새 산출물은 이 디렉터리에만 저장했다. 기존 결과·캐시·이전 감사 디렉터리는 수정하지 않았다. 이전에 감사한249개 기존 source/doc/test 중 의도적으로 변경한 것은 `mmimpress/serve.py`, `scripts/108_eval_llava_mt_gqa_B.py` 두 파일이다. 나머지247개가 보존됐으며 기존 `mmimpress/mpic.py`도 그대로다. Git metadata가 비어 HEAD/status를 얻을 수 없어 source SHA와 수정 전 snapshot을 사용했다.

## 완료 상태

| 요청 항목 | 판정 | 근거와 제한 |
|---|---|---|
| FullLoad 한 토큰 suffix 오류 수정 | PASS (CPU) | 보존한 실제 수정 전 source와 수정 후 source를 같은 tiny Llama fixture에서 실행. 이전0 bytes/0 K·V → 이후10,240 bytes/4 pread, 각 layer K640/V640 nonzero elements. 별도 test에서는 모든 설치값과 native dense logits까지 비교. |
| 최초 prefill/decode 구분 | PASS (CPU) | 길이 판별 삭제, request-local layer done 상태 사용. suffix1/2 모두 로딩하고 이후 decode에서는 재로딩하지 않음. |
| prefix/cache ID/suffix 계약 | PASS (CPU) | 전체 prompt의 pre-image token IDs, image placeholder count, context/reader/cache metadata, 명시적 cache ID 및 제공 suffix의 prompt 일치를 검사. 잘못된 경우 I/O 전에 ValueError. |
| 요청 간 이미지 캐시 분리 | PASS (CPU) | image0→image1→image0의 설치 K/V 및 첫 logits 검증. request별 cache/selector 신규 생성. |
| 실제 GPU FullLoad 설치 | NOT VERIFIED | GPU 없음. 재현 script의 GPU gate는 native attention에 제공되는 cache 설치와 source payload 비교를 수행하도록 준비했지만 실행하지 못함. |
| MPIC 다중 이미지/interleaving | PASS (CPU) | TextA/I1/TextB/I2/TextC를 그대로 연결. 독립 이미지 cache, 서로 다른 N40/N47, 순서 교환, 위치 이동을 실제 decoder에서 검사. |
| 원 논문 multi-image k 규칙 확정 | NOT VERIFIED | 원문§4.2와 부록B.1은 선두 토큰/두 이미지의 sink를 설명하지만 global/per-image selector를 확정할 공식 source를 확인 못함. 두 정책을 구현하고 caller가 명시하게 함. |
| selective attention/dummy/position | PASS (CPU) | 8조건×2 layers=16 layer tensor evidence. 독립 SDPA/직접 linking/독립 RoPE inverse 참조와 hidden/K/V/logits 일치. 최대 logits 오차5.96e-8. |
| k0/k32/k전체 경계 | PASS (CPU) | 모든 text 계산; 전체 image context 유지; 전체 선택은 동일 visual inputs의 native dense와 strict 대조. partial k32를 dense와 같아야 한다고 요구하지 않음. |
| 실 pretrained GPU6조건 | NOT VERIFIED | code gate에 단일, 변경 prefix, interleaving, 재배치, 고정3턴 history, FullLoad suffix1/2 검사가 있음. 이 GPU branch 자체도 현재 환경에서 미실행. |
| MMDU 공식 benchmark 준비 | PASS (로컬 provenance/preprocessing) |110대화/1645질문 benchmark.json 사용, training45k 제외. 원본 파일 hash·선정/제외 이유·선택 이미지 hash/크기·Native processor token bound 보존. upstream byte identity는 별도 미검증. |
| MMDU 실제 생성/TTFT/quality | NOT VERIFIED | 준비5대화/15턴; 실행0대화/0턴; judge 미실행. empty answers.jsonl/header-only ttft_io.csv와 명시적 status 저장. |
| hit read / miss compute overlap | PASS (CPU 작업 구간만) | 실제 hit pread worker와 실제 miss tiny decoder forward 및 새 raw KV 저장을 실행. serial/parallel logits bitwise 일치. thread/interval 보존. GPU kernel overlap은 미검증. |
| GPU H2D와 miss compute overlap | NOT VERIFIED / 미구현 | 현재 prepared hit payload는 CPU resident. query engine에서 H2D한다. §3.2 전체 전송 병렬화와 같다고 주장하지 않음. |
| 모든 cache hit인 MMDU의 mixed 검증 | NOT APPLICABLE | benchmark용 독립 이미지 caches를 미리 준비함. 이 실행만으로 mixed 병렬화를 인증하지 않음. 별도 작은 GPU profiler gate를 준비함. |

## 출처와 원 논문 대응

기준 논문은 [MPIC arXiv v2](https://arxiv.org/abs/2502.01960v2), 2025-09-20 개정이다. 로컬 `mpic.pdf`는 없고 `papers/mpic.md` SHA `7253687b8a076fbea6e49fc8d9bffc856c3be33b1b7a372cba5fd5d00eaa503b`와 [v2 HTML](https://arxiv.org/html/2502.01960v2) §3.2/§4.1/§4.2/§5 및 부록B/C/D를 대조했다. 저자 논문 목록/설명도 재조회했으나 공식 MPIC source·repository commit은 확인하지 못했다. **논문 기반 재구현**이다. 공식 구현이 없다고 단정하지 않는다. browser 근거는 [browser_sources.json](evidence/browser_sources.json)에 있다.

| 논문 근거 | 구현 | 차이/판정 |
|---|---|---|
| §4.2 모든 text와 leading k image tokens | `mpic_multi.build_multi_plan` | global/per_image 둘 다 지원. MMDU용 설정은 **per_image32**, 두 이미지에서64 image rows. 원 논문 규칙의 확정이라는 주장이 아님. |
| §4.1 선택 토큰만 decoder 입력 | `MultiSelectivePrefill.run` | active hidden만 norm/QKV/attention/MLP에 전달. 전체 image forward 후 KV 일부 교체가 아님. |
| §4.1 dummy K/V와 attention 전 교체 | `assemble_multi_kv` | 모든 image spans 설치 후 active K와V를 원래 위치에 index_copy. coverage assertion으로 text dummy 노출 방지. |
| §4.1 selected Q가 linked full context 참조 | full K/V length=T, Q length=A | causal mask는 original positions 기준. 모든 과거 image keys 접근, 미래 probability0. leading image Q가 나중 image에 접근하지 않는 것은 올바른 causality. |
| 위치 인코딩: Fig7 상세 생략 | post-RoPE K phase relocation | Llama 1-D RoPE만 지원. current active Q/K는 target position에서1회 적용. cached hidden의 앞 문맥 변화는 완전히 복원하지 않음. Qwen/GLM/Mistral 지원을 주장하지 않음. |
| §3.2 hit load/miss compute 병렬화 | `prepare_mixed` | 실제 SSD→CPU hit read worker와 main miss computation을 겹치게 함. GPU transfer/kernel overlap 인증은 남아 있음. load 실패는 전파하며 paper의 자동 fallback 정책과 다름. |
| §5 vLLM0.9.0/H800/두 LLaVA 모델 | 현재 HF Vicuna7B 한 모델, Transformers4.57.6 | NF4/BF16/FP16 SSD adaptation, batch1/eager. original paper precision/cache source/생성 cap/정확한 judge revision은 충분히 명시되지 않음. |
| §5.2 Full reuse는2단계 | `mmdu_mpic.full_reuse_prefill` | text-only native prefill(original RoPE positions, compact causal indices) → 독립 image KV linking → 마지막 text token re-forward로 첫 logits. 이 마지막 query 구성은 명시한 Transformers adaptation. 기존 FullLoad와 동일하지 않음. |
| prefix caching | `prefix_prefill`, `logical_labels` | 이전 native prompt cache와 현재 text IDs/image identity/local index의 실제 공통 prefix만 재사용. decode-output KV는 다음 요청 state에 보존하지 않는 prompt-cache 정책. 완전히 matching prefix에 포함된 이미지는 pixel I/O/vision 계산도 생략. |
| 부록D GPT-assisted7차원 scoring | 실행하지 않음 | generated responses가 없고 judge를 실행하지 않았다. EM/임의 자동 지표로 대체하지 않음. |

### 정확성 검증의 근거

새 CPU fixture는 random local Llama2 layers, hidden32, Q heads4/KV heads2/head_dim8, FP32, seed1234다. 실제 pretrained weights나 vision encoder가 아니다. source image spans는 각각40/47 rows로 독립 normal decoder pass에서 생성한 raw KV다. 구조 row는각 image 마지막 row라는 synthetic annotation이다.

대상 prompt는 text3 + image40 + text2 + image47 + text2, total94이다. per_image32에서 각 layer Q `[1,4,71,8]`, active K/V `[1,2,71,8]`, full K/V `[1,2,94,8]`, next hidden `[1,71,32]`. global32는 Q39행, k0는Q7행, 전체는Q94행이다. 이미지들을 입력 앞쪽으로 옮기거나 합성하지 않았다.

독립 참조는 production plan/mask/assembly/relocation을 호출하지 않는다. image source를 raw 읽고 old rotation을 직접 제거한 뒤 target rotation을 적용하고, original index에 직접 K/V를 채우며 SDPA로 attention을 계산한다. 실제 production Q/K/V/hidden/mask tensor를 파일로 보존했다. 기준은 실행 전에 **rtol1e-5/atol1e-6**로 고정했고 완화하지 않았다. partial k32의 dense 차이는 별도 현상이며 PASS 조건으로 dense equality를 요구하지 않는다. 전체 재계산은 동일 visual inputs의 native dense logits와 일치한다.

[cpu_evidence/layer_traces.json](evidence/cpu_evidence/layer_traces.json)와 같은 디렉터리의 tensor files에 실제 positions, selected local rows, per-image source/target offsets, mask, K/V replacement/보존 및 hidden shape가 있다. dummy0/12345 불변성, image/cache ID 오류, sequential request isolation 및3턴 generated-history/prefix 검증은 [9개 새 tests](evidence/multi_tests_complete.log)에 있다. 기존 MPIC17개 및 main safety8개도 새로 통과했다. 총 **34 tests**다.

## FullLoad 수정과 기존 결과 영향

`LayerSelector._hook`가 `h.shape[1]==1`이면 반환해 initial prefill도 decode로 오인했다. 새 요청의 fresh prefix cache에 image K/V는 zeros였고, first query가 이를 사용했다. `li in self.done`만으로 후속 decode를 구분하도록 수정했다. 이 done 집합은 request별 selector에 속하며 최초 layer load 후 설정된다.

`validate_full_load_contract`는 full prompt가 있을 때 pre-image IDs/position을 source prefix와 비교한다. cache ID가 있으면 image metadata와 일치해야 하고, suffix를 별도로 제공하면 full prompt suffix와도 일치해야 한다. suffix-only API는 명시적으로 context 소유 prefix를 사용하는 계약이며, caller가 제공하지 않은 원래 prompt/외부 image ID를 검증했다고 주장하지 않는다. 현재108 main FullLoad wrapper는 full prompt와 image ID를 둘 다 전달하도록 수정했다.

같은 fixture에서 archived faulty source와 fixed source를 실행한 [before/after](evidence/fullload_before_after_final.json)는 실제 zero K/V에서 nonzero image K/V 설치로 바뀐 것을 보여준다. 출력 token 우연 일치만으로 통과하지 않았다. unit test는 설치 K/V 전체가 raw payload와 같고 first logits가 native full prefill과 일치함을 검사한다.

[기존 결과 영향 검사](evidence/fullload_impact.json)는2478개 raw file 후보/24개 run family를 식별하고, family별 첫 file의 최대40 records를 bounded read했다. 별도 Qwen/GLM backend는 이 legacy 수정 대상에서 제외했다. **463 records 중 legacy stored FullLoad62개**, 그중 suffix 길이를 확인/복원한26개는 최소20 tokens로 suffix1 trigger가 없었다. **36개는 suffix 길이를 복원하지 못해 NOT VERIFIED**로 request ID/record ordinal/run 경로를 따로 남겼다. 전체2478개를 모두 분석했다는 뜻이 아니다.

이전 감사에서 최신 LLaVA main의 별도9개 요청도 native tokenizer/history/timestamp를 재검사했으며 suffix34/56/74였다. fixed footer/prefix 계약은 이 결함 조건을 배제하는 근거지만 전체 historical numerical correctness를 다시 인증하지 않는다. 검사된 과거 요청에 확인된 suffix1 trigger는0개이며, 이번 controlled regression에는 명확히 trigger가 있다. 다른 old suffix-only caller의 changed-prefix 영향은 별도 미검증이다.

전체 main을 자동 재실행하지 않았다. 재실험 권고 범위는 실제 suffix1 stored request가 확인된 세션, 또는 source prefix 불일치가 확인된 세션이다. unknown36개와 미검사 corpus는 추가 증거 확보가 먼저다. 기존 frozen config의 source SHA를 새 코드에 맞춰 덮어쓰지 않았다. 기존 frozen run resume는 source mismatch로 거부될 수 있으며, 새 코드로 실행하려면 새 run/config를 만들어야 한다.

## MMDU 준비와 실제 실행 상태

공식 [MMDU repository](https://github.com/Liuziyu77/MMDU)가 연결하는 [laolao77/MMDU](https://huggingface.co/datasets/laolao77/MMDU)의 benchmark를 로컬에서 사용했다. training `mmdu-45k.json`는 사용하지 않았다. 로컬 benchmark110 dialogues/1645 user turns, SHA **31dd8b7c15416bf22be86a4077e6b755b8adc48226f6ca6796129333385af080**. upstream benchmark file commit은 browser link에서 `269d0d6a20edd1774a05388ca7cc25df9631d63c`로 확인했지만 원격 byte 다운로드/직접 hash 대조는 못했으므로 로컬 bytes를 그 commit과 동일하다고 단정하지 않았다.

[공식 LLaVA-NeXT generation script](https://github.com/Liuziyu77/MMDU/blob/main/model_generation/LLaVa_next_gen_ans.py)는 전체 이미지를336×336 Lanczos로 resize하고, marker 누적 개수에 따라 image list를 늘리며 generated history를 사용한다. cap2048/[INST] formatting이다. 이번 준비에서도 **전체 이미지336×336 resize**를 모든 방법에 똑같이 사용한다. 이미지 crop/삭제/재배치나 질문 텍스트 절단이 아니다. native Vicuna USER/ASSISTANT format 및 upfront smoke cap256은 adaptation이다. MPIC 논문의 생성 cap 자체를 확인하지 못했으므로2048을 paper setting이라고 부르지 않았다.

선정 규칙은 seed1234의 SHA256(seed:source_id) 순서, 처음3턴까지 **정확히 두 이미지를 도입하는 대화**, native processor expanded length + 이전/현재 응답 cap reserve + boundary64≤4096, 처음5개다. quality/latency에 따른 선정은 없다. 포함/제외 이유와 이미지 원래 크기/해시를 전부 [subset manifest](evidence/mmdu_run/subset_manifest.json)에 저장했다.

| dialogue ID | 준비한 turns | 최대 reserved context bound | 실제 생성 turns |
|---|---:|---:|---:|
|31|3|3274|0|
|40|3|3278|0|
|25|3|3280|0|
|69|3|3290|0|
|35|3|3296|0|

이 subset은 전체 MMDU의 대표 성능 표본이 아니다. 이후 GPU runtime은 실제 generated history가 bound를 넘으면 해당 request를 명시적으로 실패시키며 임의 truncation을 하지 않는다. history는 method별 실제 생성 답변만 누적한다. fixed controlled3턴 GPU 기능 검사와 MMDU generated-history 평가는 별도다.

실행 요청 결과는 [status.json](evidence/mmdu_run/status.json), [실행 로그](evidence/mmdu_run.log)다. `nvidia-smi` driver 통신 실패, torch2.5.1+cu121 CUDA availability=False/device_count0. torch CUDA library가 있다는 사실을 GPU 사용 가능으로 해석하지 않았다. weights load=False, exit3. 기존 mllm_ft Python3.10.20/Transformers4.57.6/NumPy2.2.6 환경을 사용했고 설치·업그레이드·offload/precision fallback은 하지 않았다.

| 방법 | pretrained 완료 요청 | TTFT | KV SSD bytes/preads | 응답 품질 | truncation |
|---|---:|---|---|---|---|
|Prefix caching|0|NOT VERIFIED|NOT VERIFIED|NOT VERIFIED|NOT VERIFIED|
|Full reuse (adapted)|0|NOT VERIFIED|NOT VERIFIED|NOT VERIFIED|NOT VERIFIED|
|MPIC-32 (adapted, per_image)|0|NOT VERIFIED|NOT VERIFIED|NOT VERIFIED|NOT VERIFIED|

[answers.jsonl](evidence/mmdu_run/answers.jsonl)는 empty이고 [ttft_io.csv](evidence/mmdu_run/ttft_io.csv)는 header only다. zero TTFT/zero 품질 점수를 의미하지 않는다. 생성이 없어 judge 실행도 없으며 EM로 대체하지 않았다.

## 측정 계약과 병렬화

GPU 재실행용 command는 `gpu_resident`와 `ssd_cold` 두 조건을 별도 output directory에서 선택한다. 모두 같은 snapshot/NF4/BF16/eager/greedy/cap256/input geometry다.

- GPU resident: 독립 image KV/input sidecar를 request 밖에서 GPU에 준비하고 setup read bytes를 별도로 기록한다. paper-style residency 조건이지만 original vLLM/hardware까지 재현한 설정은 아니다.
- SSD cold: MPIC/Full reuse 요청마다 새 KV pread, DONTNEED conditioning은 timer 밖. syscall 성공은 NAND/controller cold 증거가 아니다. metadata activation/provision 비용도 setup으로 분리한다.
- Prefix caching: 동일 text IDs뿐 아니라 image identity/local token index가 일치하는 실제 prefix만 재사용한다. 해당 prefix에 완전히 들어간 image는 다시 pixel read/vision을 하지 않는다. 첫 요청은 previous cache가 없어 재계산하며 warm-up의 cache를 측정 요청에 몰래 이월하지 않는다.
- TTFT는 outer start(prompt 생성/토큰화/필요한 pixel 전처리 이전)→첫 argmax를 host에서 materialize하고 GPU sync한 시점. 총 생성 시간을 토큰 수로 나누지 않는다. LM head/attention/I/O의 inclusive interval들을 중복 합산하지 않는다.
- KV SSD bytes는 os.pread로 반환된 KV/visual-input bytes다. Prefix pixel image-file I/O와 일반 OS readahead/NAND bytes까지 포함한 total storage traffic이라는 주장이 아니다. prefix KV read0과 image I/O0을 혼동하면 안 된다.
- Native last-token decode cache는 prompt length+실제로 forward한 output tokens만큼 증가하는지 검사한다. 실패한 loading/identity/functional gate는 그대로 전파하고 ReComp로 숨겨서 바꾸지 않는다.

CPU mixed timeline은 [mixed_timeline.json](evidence/cpu_evidence/mixed_timeline.json)에 실제 worker/main thread IDs와 start/end를 보존했다. 새 miss callback이 decoder를 실행하고 **그 결과를 새 raw store에 저장**한 뒤 해당 context를 attention에 연결한다. serial/parallel logits는 bitwise 동일하다. 작은 CPU 작업에서는 thread overhead로 parallel wall이 더 길 수 있으며 이 결과로 GPU speedup을 주장하지 않는다. H2D는 engine 단계에서 수행하므로 hit CPU→GPU copy까지 miss compute와 겹친다는 구현은 아직 아니다.

GPU profiler/mixed native miss provisioning 경로도 `125`에 준비했지만 실제 GPU에서 실행하지 못했다. host interval이 겹친다는 사실만으로 GPU kernel overlap을 PASS로 바꾸지 않는다. CUDA trace에서 실제 memcpy/compute를 확인해야 한다.

## 사용 표기와 확대 조건

적절한 이름은 **MPIC-32 (adapted, per-image k)**다. 독립 source cache, 원래 interleaving, all-text selective attention 및 full image KV 유지라는 핵심 기능을 구현했지만 원 논문과의 수치/성능 동등성은 미검증이다. old single-image API/main MPIC arm을 자동으로 multi-image 경로로 바꾼 것은 아니다. multi-image는 새 `MPICMultiServer`/MMDU entry를 사용한다.

**현재 전체 MMDU 평가로 확대하면 안 된다.** 먼저 실제 GPU에서6조건·k boundaries·native installed tensors·quantized numeric differences를 검증하고, 동일 조건의 bounded MMDU45 method requests를 완료해야 한다. 그 뒤 pinned judge/protocol, source dataset byte identity, long-context policy(현재4096), 많은 이미지의 memory bounds, full hit-transfer overlap/SSD residency와 성능 계측을 확정해야 한다. 이 작업에서는 전체 MMDU/online load/MT-GQA main 재실행을 하지 않았다.

## 변경 파일

| 파일 | 변경 |
|---|---|
|`mmimpress/serve.py`|initial one-row prefill fix와 fixed-prefix/cache ID/suffix validation|
|`scripts/108_eval_llava_mt_gqa_B.py`|FullLoad wrapper가 prompt와 image ID 전달|
|`mmimpress/mpic_multi.py`|multi spans/선택/linked K/V/positions/active decoder/독립 decode 및 mixed preparation|
|`mmimpress/mmdu_mpic.py`|strict generated-history renderer, exact-prefix cache, distinct two-pass Full reuse|
|`scripts/124_validate_mpic_multi.py`|CPU layer tensors/FullLoad/mixed evidence|
|`scripts/125_eval_mpic_mmdu_smoke.py`|5-dialogue subset/CPU preflight/optional pretrained GPU gates 및 두 residency 조건|
|`scripts/126_audit_fullload_impact.py`|bounded read-only old-result trigger inventory|
|`scripts/127_check_fullload_before_after.py`|archived faulty vs fixed actual source regression|
|`tests/test_mpic_multi.py`|9개의 independent decoder/identity/history/prefix/mixed tests|

[REPRODUCE.md](REPRODUCE.md)에 실행 명령과 실패 해석을 적었다. source snapshot/hash 및 기존 산출물 보존 확인은 [preservation.json](evidence/preservation.json)에 있다.
