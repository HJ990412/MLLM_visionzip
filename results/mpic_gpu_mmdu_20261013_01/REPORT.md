# MPIC GPU 기능 검증과 MMDU 5대화·15턴 파일럿

2026-10-13 KST. **LLaVA pretrained GPU 기능 검증과 실제 생성·측정을 완료했다. 현재 새 다중 이미지 경로는 `MPIC-32 (adapted, global k)` 또는 `MPIC-32 (adapted, per-image k)`로 사용할 수 있다. 원 논문 시스템·성능·judge 재현 완료를 의미하지 않는다.**

보존된 5대화의 첫 3턴을 네 방법으로 실행했다. SSD 조건과 GPU 상주 조건에서 각각 60건, 총 **120건의 측정 요청**을 완료했다. 생성 실패·OOM·재시도·ReComp fallback은 모두 0건이다. 모든 답변·입력·history가 두 조건 사이에서 동일하다. 품질은 별도의 실제 로컬 visual judge로 평가했으며, 60건 중 59건에서 유효 점수를 얻었다. 1건의 누락을 명시하고 네 방법 공통 14턴으로 비교했다.

이 보고서는 새 실행 증거를 기준으로 이전 보고서의 GPU NOT VERIFIED를 갱신한다. 기존 보고서·실험·캐시는 덮어쓰지 않았다. 전체 MMDU, 온라인 부하, MT-GQA 본실험은 실행하지 않았다.

## 1. GPU 접근과 출처

| 항목 | 관측 결과 / 판정 |
|---|---|
| sandbox `nvidia-smi`, CUDA tensor | FAIL: NVIDIA character device가 노출되지 않고 driver communication / `No CUDA GPUs are available` 오류 |
| host GPU 실행 | PASS: 자동 검토가 허용한 `require_escalated` 실행에서 실제 GPU 접근 |
| GPU / driver | NVIDIA GeForce RTX 4090 / 535.230.02 |
| CUDA | nvidia-smi 표시 12.2; PyTorch 빌드 12.1 |
| 실제 연산 | `arange(16, device='cuda:0')`의 제곱합 + synchronize = **1240.0** |
| 초기 메모리 | free 24,882,774,016 / total 25,391,333,376 bytes |
| 다른 작업 | compute 작업 없음; Xorg/Gnome graphics 존재. 다른 작업 종료 없음 |
| Python / packages | 기존 mllm_ft Python 3.10.20; torch 2.5.1+cu121, Transformers 4.57.6, bitsandbytes 0.49.2, NumPy 2.2.6 |
| generator | `llava-hf/llava-v1.6-vicuna-7b-hf` |
| checkpoint revision | `c916e6cdcd760b4cecd1dd4907f84ac649f93b23` |
| precision / backend | NF4 double quant, BF16 compute, FP16 raw image cache; eager; batch 1; CUDA:0 |

이전 “드라이버 접근 불가”는 **sandbox의 GPU 미노출**이었다. host driver나 선택한 Python 환경의 고장으로 판정할 근거는 없으며, 이번에는 host 경로에서 pretrained weights를 로드하고 실제 실행했다. 드라이버 재설치·패키지 교체·offload·정밀도 fallback을 하지 않았다. [진단](evidence/gpu_access_diagnosis.json), 각 실행의 `environment.json`과 [source hash](source_sha256.json)를 보존했다.

기준은 [MPIC arXiv v2](https://arxiv.org/abs/2502.01960v2), **2025-09-20 개정**이다. 첨부 mpic.pdf는 확인되지 않았고, 로컬 `papers/mpic.md` SHA `7253687b8a076fbea6e49fc8d9bffc856c3be33b1b7a372cba5fd5d00eaa503b` 및 [v2 HTML](https://arxiv.org/html/2502.01960v2)을 사용했다. §2/§3.2/§4.1/§4.2/§5/부록D에 따른 논문 기반 검증이다. 저자 공식 MPIC 소스·repository commit과 다중 이미지 k 규칙은 **NOT VERIFIED**다. arXiv의 일반 code-finder 링크를 공식 구현으로 간주하지 않았다. [출처 기록](evidence/web_sources_final.json)과 [이전 출처 감사](../../audits/mpic_20261013_cpu_01/AUDIT_REPORT.md)를 참조한다.

Git HEAD는 sandbox `.git`에 HEAD/config가 없어 **NOT VERIFIED**다. 실행 config에 실제 소스 SHA를 저장했으며, [source_final](source_final/)에 실행 소스를 복사했다. 두 측정 실행 및 그 안에서 재실행한 완전한 기능 gate는 동일한 최종125 source SHA `80b2e40964262b0c44eace79035d466f1e30d46b23478f03f0c255471eaa4634`와 일치한다. 최초 gate-only는 이전 중간125 source SHA `e69c0cb3f16a9805e6de2ef177f1babd959f5b093785ab27c2d9d4b5420208c3`를 기록했지만 그 transient 전체 snapshot은 남아 있지 않아 이 한 snapshot의 보존은 NOT VERIFIED다. 모든 gate의 core module SHA는 동일하다. 최종 소스의 실제 재실행 gate를 검증 기준으로 연결했다. [execution lineage](evidence/execution_source_lineage.json).

기존 단일 이미지 `mmimpress/mpic.py` SHA `43c4c6a17c05a18eaf1040602398543e8733691fe0f840fa1a4051585b99f1cb`는 보존됐다. 새 검증 대상은 `scripts/125_eval_mpic_mmdu_smoke.py → MultiSelectivePrefill`이며, 과거 single-image main arm을 이 결과로 자동 인증하지 않는다.

## 2. 입력·캐시·비교 계약

기존 [subset_manifest.json](subset_manifest.json)을 **byte 단위로 그대로** 사용했다. seed 1234, 순서 **31 → 40 → 25 → 69 → 35**, 각 첫 3턴이다. 학습 MMDU-45k가 아닌 로컬 평가 `data/mmdu/raw/benchmark.json` 110대화·1645턴을 사용했다. benchmark SHA는 `31dd8b7c15416bf22be86a4077e6b755b8adc48226f6ca6796129333385af080`, manifest SHA는 `70203885eee1cd402c248d364be19ba3b867be26737c3a69f0d2461c4ca0c28f`다. 원격 dataset 파일의 byte identity는 여전히 NOT VERIFIED이며, 로컬 해시가 이번 입력의 기준이다.

기존 선정 규칙은 SHA256(seed:source_id) 순서, 첫 3턴에 정확히 두 이미지 도입, 4096 context 내 cap reserve 충족이다. 실행 결과를 보고 재선정하지 않았다. 이미지 ID·순서·질문·삽입 위치·원본 이미지 해시를 검증했다. 최대 reserved context bound는 대화별 3274/3278/3280/3290/3296이다.

모든 방법은 동일 모델/revision/양자화/eager/greedy/EOS/cap **256**을 사용했다. 전체 이미지 336×336 Lanczos resize를 공유했고 이미지 결합·삭제·crop·입력 truncation이 없다. USER/ASSISTANT Vicuna 형식은 현재 모델에 대한 adaptation이다. 공식 MMDU LLaVA-next generator의 cap2048과 다르며 MPIC 논문의 정확한 생성 cap은 확인되지 않았다. 각 방법은 **자기 생성 답변만 다음 턴 history에 사용**한다. 정답은 generation history에 넣지 않고 judge reference에만 사용한다.

| 실행 | 실제 호출 경로와 캐시 계약 | 논문과의 차이 |
|---|---|---|
| Prefix caching | `prefix_prefill` + native vision/decoder. text IDs 및 image ID/local row의 exact common prefix만 재사용. 이전 **prompt** KV는 GPU에 보존; 생성 decode KV는 다음 요청 state에 보존하지 않음 | §5 offline의 system-prompt-only reuse보다 3턴에서 많은 prefix를 재사용하는 대화 정책. 첫 요청에는 warm-up cache를 이월하지 않음 |
| Full reuse | `full_reuse_prefill`: 원래 RoPE position의 text-only native prefill, compact causal index → independent image KV linking → 마지막 text token re-forward | 논문 §2/§5의 two-step 성격을 반영하지만 마지막 query 구성은 Transformers adaptation. 기존 fixed-prefix **FullLoad와 다른 계약** |
| MPIC-global32 | `MultiSelectivePrefill(selection_policy='global')`: 모든 text와 입력 순서상 전체 이미지 시퀀스 선두 32개만 active | 두 이미지에서는 첫 occurrence에서32, 두 번째0; 공식 multi-image 규칙 확정 아님 |
| MPIC-per-image32 | 같은 경로, `selection_policy='per_image'`: 이미지 occurrence별 선두32개 active | 두 이미지에서는32+32=64; 공식 규칙 확정 아님 |

이미지 KV는 각 이미지에 대해 독립 native multimodal forward, source prompt `USER: <image>\nDescribe this image. ASSISTANT:`에서 생성했다. 이미지 source start=5, N=1176, 32 decoder layers, 32 KV heads, head_dim128이다. source text는 현재 MMDU 문맥과 다르다. cached image hidden의 문맥 의존성은 재사용 근사에 남아 있고, RoPE relocation만으로 모든 문맥 변화가 복원된다고 주장하지 않는다. source generation·raw persistence·metadata context setup 비용은 TTFT 밖이며 별도 `image_*_persistence.json`에 남긴다.

## 3. 논문–코드–GPU 대응 및 판정

| 검사 | 논문 근거 | 실제 코드 / 증거 | 판정과 제한 |
|---|---|---|---|
| 모든 변경 text 계산 | §4.1/§4.2 | `build_multi_plan`; text positions = image spans의 여집합, active에 전부 포함 | PASS. BOS/USER/ASSISTANT, 이미지 앞/사이/뒤 text와 history도 포함. 별도 system prompt 추가 없음 |
| image32 선택 기준 | §4.2 leading tokens | `build_multi_plan`; canonical raster image input 순서, local `[0,32)` | PASS 두 명시 정책. importance/repacked 순서가 아님. 공식 global/per-image 판정은 NOT VERIFIED |
| 실제 forward 범위 | §4.1 selective attention | active hidden만 norm/QKV/attention/MLP; actual module projection hook과 layer tensors의 A 일치 | PASS. 전체 image forward 후 일부 KV 교체가 아님 |
| dummy text K/V | §4.1 | `assemble_multi_kv`: full dummy 생성 → image K/V 설치 → active K와V index_copy → coverage assertion → QK/V 연산 | PASS. 계산 전 text dummy는 임시값이며 attention에는 노출되지 않음 |
| active Q의 retained image 접근 | §4.1 | Q length=A, K/V length=T; original-position causal mask | PASS. 모든 필요한 과거 image rows 유지, 미래 probability=0. 후행 이미지에 leading Q가 접근하지 않는 것은 causal 조건 |
| 새 K/V 설치와 hidden 전달 | §4.1 | 매 layer active K/V와 full cache 해당 위치 exact equality, residual/attention/MLP의 active hidden `[1,A,4096]` | PASS. K와V 모두 attention 전에 교체, 다음 layer에 active hidden 전달 |
| image 위치·offset·RoPE | Fig7/position independence | `expand_multi_prompt`, `relocate_post_rope_keys`; cached post-RoPE K를 source→target phase 변환; current Q/K는 target position 1회 RoPE | PASS 이 Llama 1-D RoPE 계약. 변경 pretext, image 순서 교환, interleaving 포함 |
| mask/position | §4.1 | mask `[1,1,A,T]`, active original IDs; future0; all-selected native dense strict 대조 | PASS. Qwen mRoPE/GLM/Mistral 경로는 NOT VERIFIED |
| full image KV 보존 | §3/§4 partial reuse | `MPICRawReader`, raw canonical store; N1176의 전체 K/V를 설치하고 selected만 교체 | PASS. Ours 25% budget/importance repacking 미적용; 전체 image context 유지 |
| SSD attribution | §3 disk cache | `os.pread` bytes/calls 및 설치 수, 별도 input sidecar | PASS syscall 및 설치 계약. 물리 NAND traffic / SSD controller cold 여부는 NOT VERIFIED |
| hit SSD→CPU / miss preparation | §3.2 | `prepare_mixed` actual worker/main thread + native miss provision, serial/parallel logits strict 일치 | PASS host 작업 기능. 아래 timeline과 profiler 보존 |
| hit CPU→GPU 전송과 miss compute overlap 구현 | §3.2 전체 전송 병렬화 | prepared hit payload는 CPU에 머물며 H2D는 후속 engine에서 수행 | FAIL 원 논문의 전체 GPU 전송 병렬화 기준: 이 부분은 미구현. 기능 정확성 오류와 구분 |
| actual pread / CUDA kernel 교차 overlap 정량 | §3.2 | `gpu_mixed_profiler.json`, host task timeline | NOT VERIFIED: 개별 syscall과 CUDA 이벤트의 clock 정렬/교차를 분석하지 않음; host interval만으로 인증하지 않음 |
| 측정 MMDU mixed hit/miss | §3.2 | 측정용 독립 image cache 전부 사전 생성 | NOT APPLICABLE. all-hit 평가로 병렬화를 인증하지 않음 |
| paper engine/환경/성능 재현 | §5 | RTX4090, HF eager, NF4/BF16 한 모델; 원문 vLLM0.9.0/H800 환경과 차이 | NOT VERIFIED. latency 방향만으로 algorithm 오류라고 판정하지 않음 |

구조 image rows(newline)는 source metadata `newline_idx`로 표시되고 image span에 포함된다. 이미지 경계의 별도 text 토큰은 text로 active다. 실제 입력/position/token type/선택 여부는 [gpu_token_layout.jsonl](evidence/gpu_token_layout.jsonl), source/target spans와 각 layer Q/K/V/hidden/mask shape·교체/보존 결과는 최종 소스로 재실행한 [gpu_functional.json](mmdu_ssd_attempt1/dialog_31/gpu_functional.json)에 있다. layout 파일은 최초 gate의 actual IDs/bindings로부터 파생한 annotation이며 독립 GPU tensor dump라고 부르지 않는다.

### 고정 입력 기능 증거

최초 독립 gate는 대화31의 이미지 cache를 사용했다. single image, 변경 pretext, TextA/image1/TextB/image2/TextC, 이미지 재배치·순서 교환, controlled fixed history t2/t3, 실제 frozen dialogue t1의 **7 fixture**에 각각 k0/global32/per-image32/all-selected를 실행했다. **28 cases × 32 layers = 896 layer observations**이다. 수치 기준은 실행 전에 `rtol=atol=1e-4`로 지정했고 변경하지 않았다.

all-selected는 **동일 persisted visual inputs**의 native dense decoder와 모든 K/V 및 마지막 logits를 strict 비교했다. 7 fixture에서 logits max abs 차이는 모두 **0.0**이고 모든 K/V assert가 통과했다. 이는 동일 cache/input 계약에서의 ReComp 관계이며, 다른 vision preprocessing이나 원 논문 FP16 engine까지 동일하다는 근거가 아니다. k0는 image forward0, all text compute와 full image reuse; k32는 위 선택 범위만 계산한다. partial k0/k32의 dense logits 차이는 예상되는 재사용 근사이며 equality를 PASS 조건으로 두지 않았다.

interleaved 실제 입력 T=2373, text21, 이미지2×1176, source start `[5,5]`, target start `[7,1186]`다.

| 설정 | selected local rows 개수 | actual Q / active K,V | full K,V | next hidden |
|---|---|---|---|---|
| global32 | `[32,0]` | `[1,32,53,128]` | `[1,32,2373,128]` | `[1,53,4096]` |
| per-image32 | `[32,32]` | `[1,32,85,128]` | `[1,32,2373,128]` | `[1,85,4096]` |

추가로 대화31의 3턴을 **네 방법 모두 동일한 fixed history**로 실행한12건을 보존했다. history는 이미 저장된 Prefix caching 생성 답변이며 gold가 아니다. 같은 turn의 prompt/token IDs 동일, native Q/K/V projection module의 실제 입력/output shape를 hook으로 관측했다. MPIC의 첫 projection이 A행만 받는 것을 별도로 검사했다. [fixed_history_gpu](fixed_history_gpu/validation.json), [records](fixed_history_gpu/records.jsonl). 이 진단의 cap3 답변/TTFT는 파일럿 요약과 품질에 포함하지 않는다.

기존 CPU의 production plan/mask/assembly/relocation을 호출하지 않는 독립 SDPA/RoPE 참조는 [이전 layer evidence](../mpic_multi_fix_20261013_01/evidence/cpu_evidence/layer_traces.json)에 보존돼 있다. 이번에도 관련9개 regression tests를 재실행해 통과했다. [log](evidence/regression_tests.log). GPU에서는 native all-selected 참조와 실제 layer/projection/install 증거를 추가한 것이다. 모든 GPU K/V tensor의 full raw dump 또는 독립 partial32 GPU numerical oracle를 만들었다고 주장하지 않는다.

## 4. FullLoad 수정과 과거36건

기존 `LayerSelector._hook`의 `h.shape[1]==1` early return은 새 요청의 첫 suffix1 prefill도 decode로 오인했다. 이전에 수정된 `li in self.done` request-local guard를 실제 pretrained GPU에서 검증했다. [primary gate](gpu_gate_attempt1/dialog_31/gpu_functional.json)는 suffix1/2 모두32layers×K,V=64 pread, **616,562,688 bytes**, 이미지 **1176 K/V rows/layer** 설치를 확인했다. raw payload exact equality와 native attention에 제공되는 실제 CUDA cache를 검사했다. 이후 decode에는 새 SSD 로딩이 없다.

독립 [request-isolation GPU 실행](fullload_gpu_isolation/validation.json)은 서로 다른 image A/B/A/B, suffix 길이1/1/1/2를 사용했다. source pre-image IDs/positions 및 system KV 계약을 공유하고 image payload는 독립으로 구성했다. 매 request64pread/616,562,688bytes, native attention K/V와 raw 이미지 KV exact equality, 첫/세 번째 A 결과 동일을 검사했다. wrong cache ID/wrong full prefix는 I/O 전에 ValueError로 거부됐다. **PASS**. suffix-only API는 context-owned prefix 계약이므로 caller가 제출하지 않은 외부 prompt를 검증했다는 뜻이 아니다.

미검증36건은 원본 raw line ordinal, prompt hash, **dtype + shape + binary tensor bytes**의 원래 suffix hash를 맞춰 복원했다. 과거 JSON-list hash와 binary hash를 혼동했던 검사를 바로잡았다. GQA는 당시 current-question-only prompt, VisDial9건은 hash-verified historical gold_teacher_forced index로 당시 prompt만 복원했다. 이 과거 정책은 새 MMDU 생성에 사용하지 않는다.

| 과거36건 분류 | 건수 |
|---|---:|
| 오류 조건에 해당하지 않음 | **36** |
| 오류 조건에 해당하며 영향 확인 | 0 |
| 오류 조건에 해당하나 영향 미확인 | 0 |
| 자료 부족 NOT VERIFIED | 0 |

최소 suffix23토큰이고 모든 prompt/suffix hash가 일치했다. [최종 분류](fullload36_final/classification.json), [보존 원본36건](fullload36_final/original_requests.jsonl). 기존에 확인한26건과 합쳐 해당 bounded legacy FullLoad62건에서는 suffix1 trigger가 확인되지 않았다. 이 판단은 **특정 오류 조건 부재**이며 과거 모든 logits/품질/position 계약의 정상성을 인증하지 않는다. 이번 정상 실행으로 과거 정상을 추론하지 않았다. 이36건은 그 오류 때문에 재실험할 필요가 없다. 미검사 corpus 또는 실제 suffix1/prefix mismatch가 확인되는 세션은 별도 영향 검사 대상이다.

## 5. 실제 MMDU 측정

두 조건 모두5대화·15턴·4방법=60건을 완료했다. 각 method의 history가 자신의 직전 prediction과 정확히 일치함을 독립 helper로 확인했다. [SSD accounting](ssd_final_verified/independent_request_checks.json), [GPU-resident accounting](gpu_resident_verified/independent_request_checks.json). 실패·OOM·retry·fallback0건이며 실패 요청을 제외하거나 입력을 잘라 통과시킨 사례가 없다. 길이 cap에 도달한 응답은 숨기지 않고 표시했다.

| 방법 | mean TTFT SSD ms | mean TTFT GPU상주 ms | SSD KV MB/request | SSD preads/request | peak allocated SSD / GPU상주 GiB |
|---|---:|---:|---:|---:|---:|
| Prefix caching | 233.14 | 233.19 | 0.00 | 0.00 | 7.99 / 9.16 |
| Full reuse | 738.77 | 107.21 | 1027.60 | 106.67 | 8.53 / 9.69 |
| MPIC-global32 | 818.75 | 159.94 | 1027.87 | 107.67 | 8.33 / 9.51 |
| MPIC-per-image32 | 831.40 | 165.26 | 1028.04 | 108.33 | 8.33 / 9.51 |

MB=10^6 bytes, GiB=2^30 bytes. 방법별 n15이며 통계적 유의성을 확정하지 않는다. 두 조건의 요청별 생성 tokens·prediction·input IDs·prompt·history·termination은 **60/60 완전히 일치**한다. [condition_equivalence.json](condition_equivalence.json). GPU상주 조건의 요청 내 KV read는 전부0이며 setup reads는 `resident_setup.json`에 별도 기록한다.

GPU peak는 각 request 전체 생성 동안의 allocator peak다. 모델, 이미 준비한 다른 이미지 source cache와 shared-process prefix state도 포함한다. method별 독립 프로세스/증분 메모리 요구량이 아니다. 모든 방법의 maximum reserved는 **17.494 GiB**이고 이전 warm-up/다른 method allocator history도 영향을 준다. 위 peak로 독립 method memory 우열을 주장하지 않는다.

| 방법 | 평균 output tokens | EOS 종료 /15 | cap256 종료 /15 |
|---|---:|---:|---:|
| Prefix caching | 187.20 | 9 | 6 |
| Full reuse | 185.20 | 11 | 4 |
| MPIC-global32 | 212.80 | 7 | 8 |
| MPIC-per-image32 | 210.27 | 8 | 7 |

cap256은 파일럿 제한이며 이 응답들도 품질 평가에 포함했다. 입력 truncation과 구별한다. 방법마다 생성 history가 다르므로 후속 input 길이·종료 시점도 다르다. fixed-history 기능 검증과 혼합해서 해석하지 않는다.

원시 결과는 [SSD answers](mmdu_ssd_attempt1/answers.jsonl) / [TTFT CSV](mmdu_ssd_attempt1/ttft_io.csv), [GPU상주 answers](mmdu_gpu_resident/answers.jsonl) / [TTFT CSV](mmdu_gpu_resident/ttft_io.csv)에 있다. p50/p95/출력 수/메모리는 [SSD summary](ssd_final_verified/summary.csv), [GPU상주 summary](gpu_resident_verified/summary.csv)에 모두 저장했다.

### TTFT와 I/O 해석

`run_one`은 필요한 page-cache conditioning 후 CUDA sync, peak reset, `perf_counter` start를 수행한다. **prompt 생성/토큰화/필요한 pixel 파일 읽기·전처리/cache request 준비/SSD pread/H2D/선택 계산/attention/LM head/첫 argmax host materialization/sync**를 포함하고, 첫 토큰이 host에 확정된 timestamp에서 끝난다. decoding 전체 시간을 token 수로 나누지 않는다. model load, image KV 생성·persist, metadata context activation, resident setup, DONTNEED conditioning은 제외했다. 실패 시 ReComp로 바꾸는 분기는 없다.

warm-up은 대화별 네 method의 동일 첫 turn fixture를 별도로 수행하고 그 cache/history를 측정 요청에 이월하지 않았다. 파일럿은 method 순서가 고정돼 있으므로 순서·thermal·allocator 효과를 통제한 성능 확정 실험은 아니다. warm-up/diagnostic/실측 구분은 `warmup.json`과 request logs에 있다.

`ssd_cold`는 POSIX_FADV_DONTNEED **best-effort**와 actual buffered `os.pread` 조건이다. CPU/GPU source payload를 measured MPIC/Full reuse 요청 사이에 resident로 보존하지 않는다. OS page eviction 성공, physical NAND/controller cache cold, 전체 storage traffic은 NOT VERIFIED다. Prefix는 exact-prefix GPU cache를 보존하고 필요한 image pixel 파일을 읽어 vision 계산한다. 그래서 Prefix의 KV SSD bytes0은 pixel I/O0 또는 모든 cache 조건 동일이라는 뜻이 아니다. 이 residency 차이는 보고해야 한다.

FP16 이미지 cache는1176×32heads×128dims×K,V×32layers×2bytes=**616,562,688bytes/image**다. chunk64 정렬 때문에 k32도 full range를 읽고 selected32 KV를 덮어쓴다. MPIC의 Full reuse와 비슷한 SSD bytes는 **불필요한 alignment I/O를 포함한 성능 차이**이며 image context 누락/정확성 오류가 아니다. sidecar 추가는 global32 262,144bytes/1pread, per-image32는 이미지별 같은 값이다. 읽기와 GPU 설치 개수를 따로 검증했다.

[phase_observations.json](phase_observations.json)은 actual pread syscall elapsed와 MPIC prefill inclusive interval을 보존한다. SSD mean pread ms는 Full reuse351.08/global346.28/per-image353.91, MPIC inclusive prefill mean은811.25/824.06이다. prefill은 읽기·H2D·attention·projection·설치·진단 assertion·sync를 포함한다. **서로 포함된 시간을 더하지 않는다.** cache lookup/H2D/attention-only/첫 argmax-only 비용의 독립 GPU event 분해는 NOT VERIFIED다. 검증 assertion의 overhead가 TTFT에 포함되므로 optimized serving engine 성능으로 해석하지 않는다.

GPU상주에서 Full reuse가 MPIC보다 빠른 결과는 실제 측정값이다. two-pass 호출 횟수가 single-pass보다 많다는 사실만으로 속도를 보장하지 않는다. 이 eager/quantized 실행의 연산 형태와 계측 overhead가 다른 조건이며 원문의 speedup이 재현되지 않았다는 이유로 기능 FAIL을 선언하지 않는다. 더 빠른/높은 점수의 정책을 공식 k 규칙으로 정하지 않는다.

### 응답 품질

실제 pretrained [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`를 로컬 GPU judge로 사용했다. NF4 double quant/BF16/SDPA/greedy/cap1024, seed1234. [config](judge_qwen/config.json), [runtime](judge_qwen/runtime.json), [rubric](judge_qwen/rubric.txt), [raw responses](judge_qwen/raw.jsonl). 원문의 ChatGPT judge exact model/version은 확인되지 않았고 같은 평가기로 재현했다는 주장은 **NOT VERIFIED**다.

논문 부록D의 7차원·integer1..10 rubric을 사용했고 page marker/Markdown brace만 정리했다. method name은 judge prompt에 숨겼다. 현재 question, 해당 reference answer, candidate, 현재까지 도입한 원래 순서의 이미지들을 제공했다. judge prompt에 전체 대화 history를 추가하지 않았다. 멀티턴의 이전 지시·지시대상 해석에 영향을 줄 수 있는 평가 한계다. generation history에 정답을 주입하지 않았으며 EM/임의 점수/차원 평균으로 overall을 대체하지 않았다.

초기 strict dict parser는52건 유효/8건 형식 실패였다. 원본 judge/status는 보존했다. 사후 파싱에서 **이미 출력된 six-dimension dict와 명시된 유일한 numeric Overall Score**를 읽어7건을 회복했다. 점수 범위/7차원 요구는 변경하지 않았고 점수를 추론·보충하지 않았다. [parse audit](ssd_final_verified/judge_parse_audit.json).

남은 실패는 **대화35 turn2 Full reuse**의 Image Relationship Understanding=`N/A`다. 유효7차원 점수가 없어 **FAIL (judge evaluation)**로 남겼고 재시도/0점 대체를 하지 않았다. 생성 실패가 아니다. 모든 유효 건 평균과 각 차원은 [quality.json](ssd_final_verified/quality.json), 비교용 공통14턴은 [quality_paired.json](ssd_final_verified/quality_paired.json)에 있다. 동일한60답변이므로 GPU상주에도 같은 점수가 해당하며 judge를 독립적으로 두 번 돌렸다고 주장하지 않는다.

| 방법 | 유효 평가 수 /15 | 공통14턴 overall mean /10 |
|---|---:|---:|
| Prefix caching | 15 | **6.86** |
| Full reuse | 14 | **5.93** |
| MPIC-global32 | 15 | **6.07** |
| MPIC-per-image32 | 15 | **6.43** |

이는 작은 quantized local visual judge의 파일럿 점수다. 전체 MMDU의 품질·논문 score loss·유의한 우열·공식 k 규칙을 확정하는 결과가 아니다.

## 6. 병렬화 진단의 범위

첫 gate에서 hit image1을 worker thread로 실제 pread하고 miss image2를 native multimodal forward → capture → FP16 raw persistence로 준비했다. serial/parallel logits strict 비교는 PASS다. [mixed timeline](gpu_gate_attempt1/dialog_31/gpu_mixed.json), [CUDA profiler trace](gpu_gate_attempt1/dialog_31/gpu_mixed_profiler.json). serial wall1245.05ms/host overlap0; parallel wall736.52ms/host task overlap626.25ms를 관측했다. profiling 및 page state가 섞인 소수 작업으로 speedup을 확정하지 않는다.

이 interval에는 CPU 작업과 SSD write도 포함된다. **개별 hit pread와 miss CUDA kernel의 실제 겹침, 특히 hit H2D overlap을 PASS로 표시하지 않는다.** 현재 hit H2D는 worker가 아니라 후속 selective engine에서 이루어진다. 측정 MMDU는 all-hit이며 §3.2의 mixed workload 성능 검증은 별도 남아 있다. layer-wise synchronous reads를 논문 mixed scheduling으로 부르지 않는다.

## 7. 사용 가능 범위와 다음 재검증

**현재 연구에서 새로운 다중 이미지 Llama 경로를 adapted baseline으로 사용할 수 있다.** 표기는 `MPIC-32 (adapted, global k)`와 `MPIC-32 (adapted, per-image k)`다. text+선택 image만 layer computation, full retained image KV, attention 전 K/V 교체, original-position mask 및 1-D RoPE라는 핵심 동작은 pretrained GPU 파일럿 범위에서 PASS다. 단일 명칭 `MPIC-32`만 쓰거나 원 논문 시스템 재현 완료라고 쓰면 근거를 넘는다.

새로 발견된 generation correctness FAIL은 없다. 다음은 구현/평가 차이와 미검증이며 별도 조치가 필요하다.

| 항목 / 위치 | 영향 | 조치와 재검증 |
|---|---|---|
| `mpic_multi.prepare_mixed`, GPU H2D overlap 미구현 | §3.2 전송 효율 재현 불가 | hit pinned-memory/H2D와 miss CUDA compute의 실제 scheduling 구현 후 event/stream/sync 포함 serial 참조·실제 overlap 확인 |
| `MPICRawReader.read_reused_layer`, chunk64 | k32도 selected KV까지 읽는 추가 I/O | 최적화가 필요하면 alignment/read plan 변경 후 bytes/calls/full KV 설치/모든 k boundaries 재검증. 이번에는 core reader를 수정하지 않음 |
| `prefix_prefill`, prompt-only prefix 정책 | 원문 system-only offline 및 generated-output-cache 재사용 정책과 차이 | 전체평가 전 요청별 reuse contract 명시; 변경하면 동일-history logits와 generated-history3턴 비교 재검증 |
| fixed method order / shared memory / timed assertions | latency·memory 추정에 제한 | full experiment 전에 순서 교차·반복·독립 memory 측정, diagnostic 없는 동등 경로와 검증 overhead를 분리 |
| cap256 / context4096 / 최대2이미지 | 응답 cap 종료25/60; 전체 MMDU 긴 입력 지원 미검증 | 긴 대화·다수 이미지에 대한 명시적 context/OOM 정책; input을 조용히 truncation하지 않고 소규모 capacity/termination 검증 |
| `judge_qwen`, N/A1건 및 local NF4 judge | 공통14턴만 비교 가능; 논문 judge 재현 아님 | 전체평가 전 judge 모델/revision/이미지·history protocol/누락 정책 고정, 실패를 포함한 coverage 보고; 변경 judge로 재평가할 때 새 결과 생성 |
| 공식 source / multi-image k / upstream byte identity | exact reproduction 명칭 불가 | 확인 가능한 source 또는 저자 근거 확보, 둘 중 좋은 점수를 보고 원문 정책을 추론하지 않음 |
| 다른 모델/backbone | Llama 외 정확성 인증 없음 | Mistral/Qwen/GLM에 대해 자체 positional/attention/cache 계약과 실제 GPU gate를 별도로 수행 |

전체 MMDU 확대의 전제는 **기본 GPU gate와 이 파일럿 실행 완료**다. 이번에 충족했다. 추가로 context/이미지 capacity, 종료 cap, judge protocol, 실제 사용 경로의 cache residency와 측정 overhead를 고정해야 한다. 논문 병렬화 성능을 주장하려면 H2D/miss overlap 구현·검증이 필요하다. 이번에는 승인 범위를 넘어 전체평가로 확대하지 않았다.

과거36건의 특정 FullLoad 오류 때문에 본실험 전체를 재실행할 이유는 확인되지 않았다. 새로운 baseline/정책/측정 contract를 도입하는 본실험은 기존 결과와 구별되는 새 config/run으로 수행해야 한다.

## 8. 변경과 산출물

이번에 기존 core model/cache 구현을 다시 만들거나 수정하지 않았다. prior implementation source hash9개 중 변경은 `scripts/125_eval_mpic_mmdu_smoke.py` 하나다. frozen-manifest 입력, 두 k policy comparator, actual functional/FullLoad hooks, gate-only, per-request memory/termination/retry evidence를 추가했다.

추가 scripts는 `128_resolve_fullload36.py`(과거36건), `129_judge_mpic_mmdu.py`(실제 local judge), `130_validate_fullload_gpu_isolation.py`(legacy GPU request isolation), `131_report_mpic_gpu_pilot.py`(독립 집계·strict score parsing), `132_validate_mpic_fixed_history_gpu.py`(네 방법 고정 history GPU projection 검증)다. source_before/source_execution/source_final 및 SHA를 남겼다. [보존 검사](evidence/preservation.json). 기존 main representative raw hash와 이전 subset bytes도 그대로다.

재현은 [REPRODUCE.md](REPRODUCE.md)를 따른다. generator root `status.json`의 quality NOT VERIFIED는 **해당 generator 자체는 judge를 실행하지 않는다**는 당시 상태로 보존했다. 이후 실제 judge 및 parse audit의 최종 coverage는 [FINAL_STATUS.json](FINAL_STATUS.json)과 이 보고서가 연결한다. parser 오류와 generation 오류를 합치거나 전체 PASS로 덮어쓰지 않았다.
