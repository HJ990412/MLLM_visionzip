# MPIC-32 독립 감사

감사일: **2026-10-13, Asia/Seoul**. 기존 구현·결과·store를 수정하지 않았다. 이 디렉터리의 진단 코드와 작은 CPU fixture만 새로 만들었다.

## 전체 판정

**분류: 차이를 명시하면 사용할 수 있는 `MPIC-32 (adapted)` baseline.** 사용 범위는 현재 LLaVA 단일 이미지·고정 pre-image prefix·batch 1·전체 이미지 KV SSD 재사용 계약이다. 이 범위에서 핵심 selective attention에 대한 CPU 실행 증거가 있다. **원 논문 시스템 전체를 충실히 구현한 MPIC-32로 표기하거나, 이번 감사가 실제 NF4/BF16 GPU 정확성·TTFT를 재인증했다고 주장할 수 없다.** 다중 이미지 또는 mixed hit/miss workload에 현재 코드를 사용하는 것은 불가하다.

- **PASS:** 실제 production selective decoder, FP16 persistence/reader, all-text + leading-32 선택, 레이어별 K/V 설치와 hidden 전달, causal mask, 원래 logical position, dummy sentinel 불변성, 원본 payload 불변성. 판정 범위는 CPU fixture다.
- **FAIL:** 두 이미지 입력을 거부하며 §3.2 mixed hit/miss 지원이 없다. 비교 대상 FullLoad에는 한 토큰 suffix에서 이미지 KV 로딩을 건너뛰는 실제 정확성 오류가 있다.
- **NOT VERIFIED:** 현재 pretrained quantized MLLM의 GPU tensor·수치·품질·TTFT, GPU timeline, 물리적으로 cold인 SSD, 공식 MPIC source, 다중 이미지에서 논문의 k가 전체 기준인지 이미지별 기준인지.
- **NOT APPLICABLE:** 현재 단일 이미지 all-hit followup에서 mixed hit/miss 병렬화; Qwen/GLM 본실험에서의 MPIC 감사(해당 baseline이 등록되지 않음).

기존 PASS 파일을 검증 oracle로 사용하지 않았다. 과거 raw의 한 대화와 현재 소스의 일치 여부를 직접 검사했으며, 새 CPU 실행은 독립 수식 참조와 실제 tensor를 대조했다.

## 논문과 구현의 출처

읽은 논문은 [arXiv v2](https://arxiv.org/abs/2502.01960v2), **2025-09-20 13:54:37 UTC 개정**, 17쪽/13개 figure다. v1은 2025-02-04 제출이다. [v2 HTML](https://arxiv.org/html/2502.01960v2) 및 [PDF browser representation](https://arxiv.org/pdf/2502.01960v2)에서 §2, §3.2, §4.1, §4.2, §5를 확인했다. workspace에서 첨부 `mpic.pdf`는 발견되지 않았다. 로컬 `papers/mpic.md`의 SHA256은 `7253687b8a076fbea6e49fc8d9bffc856c3be33b1b7a372cba5fd5d00eaa503b`다. 이 파일 자체에는 개정일 메타데이터가 없어 PDF와 byte-level 버전 동일성은 증명하지 않았다. 직접 PDF 다운로드는 실행 환경의 DNS 실패로 불가능했다. browser 조회 근거는 [browser_sources.json](evidence/browser_sources.json)에 보존했다.

공식 구현은 **확인하지 못했다**. arXiv와 [저자 설명](https://shijuzhao.github.io/pic), [Shiju Zhao 논문 목록](https://shijuzhao.github.io/index_zh.html), [Junhao Hu 논문 목록](https://derekhjh.com/publications/)의 MPIC 항목에 연결된 실행 가능한 MPIC repository를 검증할 수 없었다. 저자 GitHub 목록의 일반 vLLM fork는 보이지만 해당 source fetch가 실패했고, MPIC 공식 구현이라는 근거도 없다. 다른 논문 COMB/EPIC의 code를 MPIC code로 취급하지 않았다. **공식 코드 부재를 단정하지 않으며, 이번 감사는 논문 기반 감사다.**

§4.2는 “all text tokens and k tokens at the beginning of image tokens”를 선택한다고 설명한다. 그러나 multi-image에서 global k인지 per-image k인지 실행 가능한 selector로 명시하지 않는다. 동기 실험은 한 이미지의 위치를 바꾸는 실험이다. 공식 selector를 확인하지 못했으므로 이 모호성은 **NOT VERIFIED**다. 현재 selector는 하나의 연속 image span만 받기 때문에 이를 해결하지 않는다.

현재 `.git`에는 사용 가능한 HEAD가 없으며 `git rev-parse HEAD`가 실패한다. commit을 추측하지 않았다. 파일 SHA256과 본실험에 기록된 source SHA256으로 revision을 고정했다.

| 경로/실험 | 실제 실행/역할 | 출처 확인 |
|---|---|---|
| 과거 `scripts/65_validate_mpic.py` | 3–5개 이미지의 quantized-model smoke를 정의하는 과거 validator. 단일 이미지별 실행이며 interleaved multi-image 지원이 아니다. | 기존 PASS를 이번 인증으로 사용하지 않음. tolerance는 기존 코드에서 0.02/0.02이며 이번 CPU tolerance와 다름. |
| 과거 `scripts/67_eval_mpic_gqa.py`, 2026-09-22 pilot | `mpic32` → `MPICServer`/`MPICContext`. 40이미지·240질문, generated history 없이 독립 질문. | 당시 mpic.py SHA `82ffb5617c60c293e7eac0f797b4f78973ef0719f800aec839ff01fbc70b54a4`. 현재 SHA와 다름. 과거 구현 source 전체를 복원한 감사는 아니다. |
| `scripts/73_eval_mt_gqa_5arm_generated_shard.py` | 이전 5-arm generated-history runner에서 같은 MPIC module 호출. | 현재 최종 main의 entry로 혼동하지 않음. |
| 현재 `scripts/108_eval_llava_mt_gqa_B.py:281,628` | `mpic32_ssd` → `MPICServer(k_recompute=32, max_new_tokens=16)` → `MPICSelectivePrefill.run`. | 최신 3-model 집계의 LLaVA 원본 run은 `runs/mt_gqa_B_llava5_qwen3_20261001T093921Z/llava`. 관련 current source가 main config의 해시와 일치. |
| 현재 `mmimpress/mpic.py` | 전용 MPIC canonical store/loader/manual active-row decoder. | SHA `43c4c6a17c05a18eaf1040602398543e8733691fe0f840fa1a4051585b99f1cb`. |
| `scripts/110_eval_qwen_mt_gqa_B.py`, `mmimpress/glm46v/runner.py:25` | Qwen/GLM main은 ReComp/FullLoad/Ours 3-arm. | MPIC 등록 없음. Llama 1-D RoPE 검사 결과를 Qwen MRoPE/GLM에 적용할 수 없음. |

현재 main model은 `llava-hf/llava-v1.6-vicuna-7b-hf`, snapshot `c916e6cdcd760b4cecd1dd4907f84ac649f93b23`. NF4 double quant weights, BF16 compute, FP16 SSD KV, eager attention, Transformers 4.57.6, torch 2.5.1+cu121, seed 1234, batch 1, greedy, cap 16. 원 논문은 vLLM 0.9.0, H800 80GB 및 MMDU/SparklesEval 평가다. 원 논문의 정밀도·정확한 cache source prompt·I/O timer 계약은 충분히 명시되지 않아 일치를 주장하지 않았다.

본실험 환경을 기록한 config는 [main_config.json](evidence/main_config.json), source 비교는 [main_source_comparison.json](evidence/main_source_comparison.json), 과거 revision은 [legacy_source_revision.json](evidence/legacy_source_revision.json)이다.

## 논문–코드 대응표

`PASS`는 명시한 범위의 기능 검사 통과이고 원 논문 전체 재현을 뜻하지 않는다. `FAIL`은 해당 요구를 만족하지 못한다는 뜻이며 미지원과 잘못된 수치 연산을 구분해 적었다. 이하 논문 근거는 모두 위 v2 기준이다. 위치는 수정하지 않은 현재 source의 line number다.

### A. 재계산 토큰 선택

| 항목 | 논문 근거 | 현재 코드 위치·동작 | 차이·실행 증거 | 판정 |
|---|---|---|---|---|
| 모든 변경 가능한 text | §4.1/§4.2 | `mpic.py:120–143,886–905`: image span 외 모든 행 선택. | system/user prefix, template, question, method-local history 모두 계산. 15 CPU case에서 실제 projection row 수와 active index 일치. | PASS (CPU) |
| 어떤 32 image 행인가 | §4.2 | `mpic.py:130–136,338–352`: canonical local `[0,32)`의 layer-0 visual inputs. | 32%/chunk 32개/importance top32가 아님. actual Q/K/V 입력은 이 행들과 모든 text뿐. | PASS (CPU) |
| global k / per-image k | §4.2의 서술 및 single-image 동기 실험 | `ActiveRowPlan`에 visual_start/count 하나만 있음. | 원문은 이 구분을 실행 가능한 규칙으로 확정하지 않음; 공식 source 미검증. 어느 쪽이라고도 단정하지 않음. | NOT VERIFIED |
| 다중 이미지 selection | §2 interleaving, §5 multi-image | `mpic.py:832–847` | `<image>` 2개에서 ValueError. 이미지2의 32개를 선택하거나 global 선택하는 forward가 존재하지 않음. 두 입력을 실제 거부; 0 pread, 0 projection. | FAIL (미지원) |
| system prompt/구조 행 | §4.1 text; 구조 정책은 미명시 | `mpic.py:120,697–701`; `model.py` AnyRes layout | BOS 및 pre-image text를 다시 계산. image newline은 ordinary image row로 전체 유지, local32 안에 있으면 계산. 실제 sample newline 첫 local624로 선택32와 겹치지 않음. synthetic 구조 행63/79/95도 보존. AnyRes pixel packing 자체는 GPU 미검증. | PASS (선언된 정책·CPU) |
| 선택 순서 | §4.2 leading image tokens | `mpic.py:427–487,650–701` | store가 canonical raster이고 reordered=False이어야 open 가능. physical order=original image order. Ours importance permutation 사용 없음. | PASS |
| 계획과 실제 forward 일치 | §4 selected-only input | `mpic.py:962–1019` | 3-layer 실제 projection inputs/Q와 MLP 모두 A행. counter 문자열만으로 판정하지 않음. | PASS (CPU); GPU NOT VERIFIED |

### B. Selective attention과 dummy cache

| 항목 | 논문 근거 | 현재 코드 위치·동작 | 차이·실행 증거 | 판정 |
|---|---|---|---|---|
| text dummy slots | §4.1 | `mpic.py:185–238` full logical K/V를 임시 값으로 할당 | 모든 image slot과 active slot으로 coverage 충족 후 반환. 0과12345 dummy 실행의 logits/KV가 bitwise 같음. | PASS (CPU) |
| K와 V 교체·시점 | §4.1/Fig7 | `mpic.py:223–227,987–1005` | 둘 다 index_copy로 원래 slot에 설치한 뒤 QK. 각 레이어 actual active K/V가 해당 full slot과 같음. | PASS (CPU) |
| 재사용 image K/V 접근 | §4 selected Q + linked K/V | `mpic.py:993–1007` | Q 길이 A, full K/V 길이 T. suffix text Q는 전체 image KV에 접근. leading image Q는 미래인 unselected image에 접근하지 않으며 이것은 올바른 causal 동작. | PASS (CPU) |
| hidden state 다음 레이어 | §4 selective propagation | `mpic.py:1008–1016` | residual+attention+MLP의 A행만 전달. 모든 layer hidden이 독립 selective oracle와 일치. | PASS (CPU) |
| 별도 text prefill 여부 | §2.2 full reuse와 §4.1 single step 차이 | `mpic.py:861–1035,1121–1139` | selective traversal 1회 뒤 LM head. text-only prefill/FullLoad 호출 없음. 첫 token 뒤에만 일반 decode 호출. | PASS |
| 전체 image forward 후 부분 교체인가 | §4 selected-only 연산 절감 | `mpic.py:877–905,962–1016` | N=96/k32에서 decoder는62행, full126행이 아님. sidecar도 첫32행만 읽음. KV attention context는 full126 유지. | PASS (CPU) |
| dummy가 일반 attention에 노출되는가 | §4.1 | `mpic.py:229–238` coverage assertion + causal mask | 빈 text slot 없이 K/V replacement 완료. sentinel invariance·future attention probability0 확인. | PASS (CPU) |

### C. 위치와 causal mask

| 항목 | 논문 근거 | 현재 코드 위치·동작 | 차이·실행 증거 | 판정 |
|---|---|---|---|---|
| cache source/현재 배치 | §2/§4 position-independent reuse | `mpic.py:675–701,912–927,1183–1195`; main prompt factory `108:116–136` | source USER/BOS 5행 → image → question. main 모든 턴도 동일 5행/image 배치; history는 image 뒤. sample source/target 동일. 이 workload는 위치 독립성을 강하게 검증하지 못함. | PASS (계약 일관성), 일반화 NOT VERIFIED |
| reuse K의 RoPE 형식 | Fig7은 position embedding details 생략 | `mpic.py:246–277,946–960`; installed `modeling_llama.py:241–246` | native cache.update 이전에 RoPE 적용하므로 post-RoPE K 저장. same position은 그대로 사용, active Q/K는 target position으로 1회 적용. | PASS (source+CPU) |
| shifted K 처리 | 원문 구체 공식 미명시 | `mpic.py:246,946–960` | old phase inverse/new phase 적용. source5→target16에서 독립 inverse/rotate oracle와 일치. **paper-specific 규칙으로 확인된 것은 아니다.** V나 old hidden content를 고치는 연산이 아니다. | PASS (phase mechanics), paper equivalence NOT VERIFIED |
| 앞 text 내용 변화 | §2/§4 partial reuse는 근사 | `mpic.py:1183–1195` stats only | 같은 start5에서 USER→NOTE, start16 shift 각각 실제 실행. 모든 current text와32 image만 새로 계산; reused hidden 의존성은 source 문맥에 남음. dense 차이를 오류로 단정하지 않음. 실제 모델 품질 회복 여부 미검증. | PASS (mechanics); 품질 NOT VERIFIED |
| 원래 위치 causal mask | §4 linked attention; autoregression | `mpic.py:146–170,925–927` | query의 original index ≥ key logical index일 때만 허용. 각 layer 미래 허용0·필요 과거 누락0·미래 probability0. | PASS (CPU) |
| 다중 이미지 offset/index | §2/§5 | `mpic.py:832,120` | 두 번째 span에 대한 offset 계산/캐시 linking 없음. forward 자체 거부. | FAIL (미지원) |
| model별 위치 방식 | §5 LLaVA backbone | Llama imports `mpic.py:919`; Qwen/GLM method lists | fixed Llama 1-D RoPE만 감사. GQA repeat_kv는 기존 tiny test로 검사되지만 Qwen MRoPE 호환성을 의미하지 않음. Qwen/GLM baseline 자체가 없음. | NOT APPLICABLE |

### D. SSD 저장과 로딩

| 항목 | 논문 근거 | 현재 코드 위치·동작 | 차이·실행 증거 | 판정 |
|---|---|---|---|---|
| 전체 재사용 image KV 준비 | §3 storage, §4 partial reuse | `mpic.py:582–819` | T1 cache의 전체 canonical image K/V를 모든 layer에서 FP16 저장; selected inputs sidecar도 저장. source prefix provenance 포함. CPU에서 실제 writer→validated context 실행. | PASS (CPU) |
| Ours25/repacking 유입 | §4 partial reuse는 context 유지 | `mpic.py:49,427,688`; `108:329–338,628` | MPIC 독립 builder/reader. 25% 예산은 다른 method branch에만 있음. noncanonical store 거부. | PASS |
| selected 제외로 나머지 누락 | §4 linked full context | `mpic.py:354–382,935–946,987–999` | read reused `[k,N)`의 covering chunk 후 모든 active slots를 교체. image context N개 유지. CPU k0/32/N에서 installed image count는96. GPU 설치는 이번에 확인 못함. | PASS (CPU); GPU NOT VERIFIED |
| bytes/읽기 수 | §3 disk transfer, 형식 미명시 | `mpic.py:330–382`; `108:281–297` | os.pread 반환 range와 counter를 독립 비교. FP16 N96/L3/H4/D16에서 k0=73,728B/6회, k32=77,824B/7회, kN=12,288B/1회. | PASS (CPU·recorded sample) |
| 재계산 old KV도 읽는가 | paper 미명시 | `mpic.py:370` row_start=floor(k/64)*64 | k32는 첫 chunk의32–63 reused 행 때문에 old0–31도 읽음. 이후 교체되므로 정확성 문제 없음; 불필요한 I/O임. FullLoad와 비슷한 bytes를 오류로 판정하지 않음. | PASS (정확성); 성능 차이 명시 |
| 요청별 payload residency | §3 GPU/CPU/disk 모두 가능 | `mpic.py:407,935`; `108:362–384,246` | main activation은 metadata만 MPIC context에 유지; request별 KV 새 read. FullLoad는 pre-image sys KV host resident. payload session pooling과 같은 계약이 아님. CPU reader와 full payload hash 불변 확인. | PASS (source·CPU) |
| OS page cache·진짜 SSD cold | §5 세부 조건 미명시 | `mpic.py:384–391,1110–1115`; `108:143–157` | DONTNEED는 timer 밖 advisory. syscall 성공은 물리 NAND/controller cache cold 증거가 아님. CPU cold diagnostic 실제 file read 성공. | NOT VERIFIED (물리 cold) |
| 원본 KV 보존/무결성 | paper 미명시 | `mpic.py:458–498,1249–1264`; main full completion hash | request 결과 before/after hash는 검증된 과거 값을 echo하므로 per-request live checksum으로 믿지 않음. 이번에는 모든 fixture payload 전체 SHA를 전후 직접 재계산해 동일함. main에는 activation와 image-end full hash 감사가 별도 존재. | PASS (CPU), 기존 live GPU payload NOT VERIFIED |

### E. 병렬화

| 항목 | 논문 근거 | 현재 코드 위치·동작 | 차이·실행 증거 | 판정 |
|---|---|---|---|---|
| mixed hit load / miss compute | §3.2/Fig6 | 단일 `MPICContext`; `mpic.py:832,935` | multi-image 및 miss branch가 없다. mixed input은0 I/O/0 forward로 거부. thread/stream/async miss orchestration 없음. | FAIL (미지원) |
| 현재 all-hit workload | §3.2의 mixed 조건 | main T1 provision, T2/T3 single-image hit | 다른 miss 이미지 계산과 겹칠 작업이 없다. baseline latency를 §3.2 overlap 성능이라고 설명할 수 없음. | NOT APPLICABLE |
| layer-wise loading과 구분 | §3.2는 이를 orthogonal optimization으로 구분 | `mpic.py:931–1019` | 레이어별 blocking pread→to(device)→indexed assembly. 명시적 prefetch worker/stream은 없음. CPU timeline은 main-thread serial. GPU에서는 이전 layer 비동기 compute와 다음 CPU read가 우연히 겹칠 가능성까지 부정하지 않음. | PASS (관측한 CPU 순서); GPU overlap NOT VERIFIED |
| GPU timeline/동기화 | §3.2 | `_sync` at transfer/assembly 및 first token | source에 barrier가 많음. nonblocking API 호출 근거로 overlap claim하지 않음. GPU가 없어 실제 stream timeline 없음. | NOT VERIFIED |

## 새 실행 증거와 수치 기준

`nvidia-smi`는 driver 통신 실패, 선택한 Python torch CUDA availability=False/device_count=0. host RAM125GiB 중 available 약121GiB, 파일시스템 `/dev/nvme0n1p2`, free 약127GiB. GPU 이름·VRAM은 확인 불가이며 과거 README의 RTX4090을 현재 장치로 단정하지 않았다. 기본 shell Python은 torch1.11/NumPy2 호환 문제가 있으므로 **기존 `mllm_ft` 환경의 Python3.10.20/torch2.5.1/Transformers4.57.6**을 사용했다. 설치·업그레이드를 하지 않았다.

새 CPU fixture: seed1234, random Llama 3 layers, hidden64, Q/KV heads4, head_dim16, eager FP32 decoder; pinned native LLaVA tokenizer; synthetic visual96행. vision encoder·실제 이미지 전처리·pretrained weights는 실행하지 않았다. synthetic 구조 행 `[63,79,95]`는 실제 AnyRes geometry 재현이 아니다. production FP16 writer/context를 그대로 실행하고, 별도의 FP32 diagnostic store는 양자화/저장 오차를 제거한 수학적 계약 대조에만 사용했다.

독립 selective oracle는 production plan/mask/assembly/relocation을 호출하지 않는다. 원래 index로 full K/V를 직접 설치하고, einsum attention, 독립 mask와 RoPE inverse/rotate를 사용한다. production 관측은 projection hooks, 실제 `torch.matmul` 인자, assembly 반환, post-attention residual/MLP를 읽는다. wrapper는 같은 tensor/result를 반환하고 종료 시 원상 복구된다.

**사전 고정 tolerance:** `rtol=1e-5, atol=1e-6`. 독립 selective oracle와는 모든 case의 logits, layer K/V, active hidden을 비교한다. 동일 source prefix·FP32 store인 k0/32/N은 dense와 비교할 근거가 있으므로 엄격 대조한다. kN은 동일 sidecar inputs로 dense와 엄격 대조한다. FP16 reuse 및 바뀐 prefix의 k32를 dense와 같아야 한다고 요구하지 않는다. **dense allclose=False는 그대로 로그에 남겼으며 threshold를 완화하지 않았다.**

| 새 입력/경계 | 실제 실행 | 결과·증거 |
|---|---|---|
| 1. 단일 image+question | FP32/production FP16 각각 k0/32/96 | 6 case PASS (selective oracle). FP32 dense logits max abs1.79e-7. FP16 k0/32 dense diff2.49e-5/2.06e-5, allclose=False; 저장 오차를 포함한 oracle는 PASS. k96 dense PASS. |
| 2. 동일 image3턴 | 각 방법 T1 native full decoder, T2/T3 ReComp/FullLoad/MPIC로 각각 답변 생성 | 총9 request. 실제 각 method 답변을 production prompt_factory에 다음 history로 전달. MPIC T2/T3 레이어 traces 포함. CPU cap3은 진단용; main cap16과 다름. semantic answer quality 검사는 아님. |
| 3. TextA/I1/TextB/I2/TextC | MPICServer 실제 요청 | ValueError before forward, FAIL(미지원). hypothetical expanded offset 및 global/per-image 선택 후보를 로그에 구분; 어떤 후보도 실제 selector라고 주장하지 않음. |
| 4. image 앞 text 변화 | 같은 index5 USER→NOTE, index5→16 prefix 길이 변화; 각각 k32/k96 | 4 case selective oracle PASS. k32 dense diff7.57e-4/1.32e-3. k96 dense max abs≤2.39e-7 PASS. 문맥 stale hidden의 근사 효과와 위치 회전의 수치 동작을 구분. |
| 5. hit+miss 두 image | MPICServer 실제 요청 | single-image guard로 거부, FAIL(미지원). miss compute/I/O overlap timeline이 존재하지 않음. |
| dummy0→12345 | 동일 FP32 k32 request | logits와 full layer K/V bitwise identical, PASS. |
| FullLoad와 k0 계약 | 같은 FP32 cached prefix·image·suffix, legacy Server CPU shim | 첫 logits strict allclose PASS. 이 조건에서 수학적으로 동등. full sys KV가 stale하거나 저장 dtype이 다르면 무조건 같다고 요구할 수 없음. |
| FullLoad suffix1 | legacy Server CPU shim 실제 실행 | 0 pread, 모든 layer의 image K/V nonzero count0, logits max abs diff0.13176 vs 올바른 cache. FAIL. |
| KV short read | 새 audit fault_store를 context open 후 truncate | `short pread` 예외 전파. ReComp fallback 없음, PASS(failure visibility). |

15개 정상 selective case/45개 layer trace가 완료됐다. image96, total126, text30인 대표 k32 case는 각 layer에서 Q/active K/V `[1,4,62,16]`, full K/V `[1,4,126,16]`, QK `[1,4,62,126]`, MLP/next hidden `[1,62,64]`. active logical rows는0–36과101–125, selected image는5–36, reused image는37–100이다. full126행을 forward한 뒤32행만 바꾼 결과가 아니다.

actual position IDs·전체 token 유형/ID·선택 인덱스·source/target positions·모든 layer shape·설치/보존 검사·read range·timeline은 [layer_traces.jsonl](evidence/cpu_probe_v4/layer_traces.jsonl)에 있다. forward가 거부된 입력의 layer shape는 만들어내지 않고 [multi_image_rejections.json](evidence/cpu_probe_v4/multi_image_rejections.json)에 NOT APPLICABLE로 적었다.

기존 CPU unit17개 및 main history safety8개도 새로 통과했다. pytest package가 없어 첫 pytest 실행은 실패했고 기존 unittest runner로 실행했다. 최초 probe 두 번은 **감사 fixture 오류**(ASSISTANT prefix 길이가 USER와 다름; diagnostic metadata의 n_spatial 누락)로 완료되지 않았으며 로그를 보존했다. 이를 production 실패 또는 PASS로 집계하지 않았다. v3가 완료된 뒤 FullLoad 실제0 KV/잘못된 logits 관측과 changed-prefix control을 보강한 **v4가 최종 근거**다.

## TTFT 비교의 공정성

| 조건 | ReComp | FullLoad | MPIC-32 | 판정·한계 |
|---|---|---|---|---|
| 모델·정밀도·backend | 같은 pinned LLaVA/NF4/BF16/eager | 동일; SSD KV FP16→BF16 | 동일; SSD KV/input FP16→BF16; manual eager attention | PASS(source/config). 실제 GPU kernel shape에 따른 rounding은 NOT VERIFIED. |
| prompt/history | full native processor prompt | 동일 prompt의 image 뒤 suffix; pre-image+image cache | 동일 full prompt에서 active rows 선택 | main에서는 fixed pre-image 계약 일관, method-local generated history PASS(CPU+raw sample). 바뀐 pre-image에서는 FullLoad 계약 검증이 필요. |
| cache 생성 | 모든 request pixels/vision | 각 method 자신의 source dialogue T1 정상 pixel forward capture | 자신의 T1 KV와 layer0 inputs capture | 별도 full/vision forward 없음(source 확인). 동일 source는 CPU 동등성 조건에만 사용; main 독립 source tensor가 bitwise 같다고 가정하지 않음. |
| SSD/host residency | KV read0, image file/preprocessing은 비용 | image full KV fresh read; sys KV host resident | image reused KV fresh read; metadata resident; sys text recompute | resource 계약 차이 공개 필요. metadata activation/validation/store lookup 준비는 session setup, TTFT 밖. 일반 serving upload/lookup 비용을 모두 포함한 재현 아님. |
| decoding | generate do_sample=False,16; first token marker | manual argmax/EOS,16 | manual argmax/EOS,16 | pinned generation_config는 BOS1/EOS2/PAD0 외 특별한 logits processor 설정 없음. PASS(source), fresh GPU outputs 미검증. |
| outer TTFT 시작 | main108:203, prompt factory/image open/preprocessing 이전 | main108:232, suffix tokenize/H2D 이전 | main108:285, prompt factory/full tokenize 이전 | main wrapper의 outer boundary는 input preparation 포함. **legacy serve.py의 core ttft는 준비 제외**이므로 그대로 섞으면 불공정. |
| TTFT 끝 | serve.py:48 첫 token host materialization+CUDA sync | serve.py:1773–1776 argmax+sync | mpic.py:1139–1141 argmax+sync | source상 동일 첫 token 결정 boundary; raw9개 timestamp 식 독립 재계산 PASS. GPU sync의 실제 실행은 NOT VERIFIED. |
| warm-up | QA._warmup normal full request | 별도 FullLoad hit warm-up 없음 | 별도 MPIC hit warm-up 없음 | warm-up sample은 main과 구분되지만 hit path별 warm-up 대칭성은 부족. 작은 latency 비교에서 개선 필요. 이번 CPU warmup artifact는 측정 summary와 분리됨. |
| 실패/fallback | 정상 pixels 경로 | load 예외 전파 | load 예외 전파 | main108:577 PARTIAL 후 raise. 이미지 재attempt는 별도 기록하며 failed request를 MPIC 성공으로 세탁하지 않음. 새 fault probe도 fallback 없음. 논문§3.2 failure fallback와는 다른 정책. |

현재 `108` main row의 authoritative TTFT는 `end_to_end_ttft_ms`다. `MPICServer.ttft_ms`, `Server.request.ttft`, `Server.recompute.ttft` 같은 내부 timer는 outer boundary가 달라 단독으로 비교하지 않는다. `113` 보고용 입력이 사용하는 main row는 outer값으로 생성된다(`108:418`).

기존 main의 deterministic image `n100552`/dialogue `mtgqa_002992`만 read-only로 뽑아 3방법×3턴 raw9개를 독립 재검사했다. prompt·각 방법 답변의 history lineage·native tokenizer suffix·TTFT timestamp identity·pread 실제 recorded 범위가 일치했다. suffix token 수는 T1=34/T2=56/T3=74라 FullLoad suffix1 결함 조건에 해당하지 않는다. main 고정 footer는 일반적으로 여러 suffix token을 만든다. 이 작은 raw 검사로 전체 corpus correctness를 다시 인증하지 않는다.

이 sample image N=2438(visual2400+newline38), L32/H32/D128. hit FullLoad는 **1,278,214,144B/64pread**, MPIC는 같은 KV 전량+input262,144B, **1,278,476,288B/65pread**. MPIC의 image context는2438행, selected32/reused2406행이다. 비용 증가262,144B는 sidecar overhead이며 25% budget 오염 증거가 아니다. sample MPIC T2의 recorded active count는93(5 pretext+56 suffix+32), attention key count2499다. **이는 historical counter consistency이며 당시 actual Q/K/V tensor 관측은 NOT VERIFIED**다.

순수 cache lookup/setup·GPU attention·CPU→GPU transfer·argmax 각각의 현재 GPU 시간은 측정하지 못했다. MPIC의 기존 stage fields는 다음처럼 해석해야 한다.

| 비용 | 이번 증거 | 계측 한계 |
|---|---|---|
| context open/compatibility/integrity 준비 | activation source·saved historical store records | request timer 밖. 일반 serving lookup end-to-end 비용으로 주장 불가. |
| SSD read | 실제 CPU pread 및 historical recorded range; `kv_read_ms`, `embedding_read_ms` | OS 반환 bytes. NAND 실트래픽·cold physical media는 미검증. |
| H2D | source의 synchronized `.to` 구간; CPU에서는 device copy/preparation만 실행 | CPU `h2d_ms`를 GPU transfer 수치로 사용하지 않음. |
| KV allocation/assembly | `cache_assembly_ms`, decode cache assembly | GPU kernel-only 비용 아님. |
| selective compute+attention | `selective_prefill_interval_ms` 및 actual tensor shapes | 이 interval은 read/H2D/assembly/norm/MLP/LM head를 **이미 포함**. 다른 stage와 더하지 않음. pure GPU compute NOT VERIFIED. |
| 첫 output token | LM head→cache seed→argmax→sync의 boundary | 별도 argmax kernel 시간 미계측. TTFT outer start/end가 권위 값. |

CPU cold diagnostic 단일 시간은 [cpu_request_timing.json](evidence/cpu_probe_v4/cpu_request_timing.json)에 기록했다. microsecond/millisecond 값은 CPU/file page 상태 진단이며 GPU latency ranking용이 아니다. CPU 실제 read/compute timeline은 main thread에서 순차지만 GPU compute/I/O의 겹침을 CPU 로그로 추정하지 않았다. 원 논문의 latency 수치를 재현하지 못했다는 이유로 correctness FAIL을 부여하지 않았다.

## 발견 문제와 수정 후 재검증

| ID | 위치 | 종류·영향 | 제안 | 수정 후 최소 재검증 |
|---|---|---|---|---|
| F1 | `serve.py:371–376` | **정확성 오류:** h.shape1==1이면 initial suffix prefill도 decode로 오인, full image KV 미설치. logical_kv_ratio=1.0이라고 표시돼도 실제0 KV. | 단순 길이로 phase를 판별하지 말고 request별 first-prefill 상태/캐시 상태를 사용. 최초1행은 반드시 load, 이후 decode는 재load하지 않음. | suffix1/2/긴 text 각각 installed K와V·byte수·logits·decode append; 실제 모델1개 GPU sample. 현재 fixed-template main을 전부 재실행할 필요는 이 증거만으로 없음. |
| F2 | `serve.py:1708–1722`; `108:230–251` | **입력 계약 검증 누락:** incoming pre-image text를 버리고 source prefix를 그대로 사용. CPU USER→NOTE에서 FullLoad logits이 완전히 동일하게 나옴. | FullLoad prefix IDs/position을 비교하고 mismatch를 explicit reject하거나 재계산 경로로 별도 표기. 변경된 prefix workload의 FullLoad를 같은 재사용 계약으로 비교하지 않음. | 동일/동길이변경/길이변경 pre-image 입력; silent reuse가 아닌 명시적 계약 동작과 MPIC k0 comparison의 조건 검사. |
| F3 | `mpic.py:832–847,120–143` | **범위 미지원:** multi-image/interleaved 및 mixed hit/miss 불가. | 여러 span/context의 logical mapping을 구현하고 k/global-per-image 정책을 명시. 공식 규칙이 검증되기 전에는 adapted 표기 유지. | 요청3/5, 서로 다른 N/offset, 모든 text·구조 행, 전layer K/V·mask·원본 보존, k0/32/N. |
| F4 | `mpic.py:935–960` | **성능 시스템 차이:** §3.2 miss compute/hit transfer orchestration 없음. | multi-image 지원 후 thread/stream 및 dependency 설계; loading 실패 정책도 명시. | mixed timeline에서 실제 겹침·sync race·fallback 기록. all-hit는 overlap 효과 분모로 쓰지 않음. |
| F5 | `scripts/113_report_mt_gqa_B.py:10`; `results/mt_gqa_three_models_summary_20261012/aggregate.py:15` | **표기 문제:** 앞선 adapted label이 최신 표에서 `MPIC-32`로 축약됨. | 신규 발표/표는 `MPIC-32 (adapted)`와 single-image/Transformers/SSD 조건 명시. 기존 산출물은 이번에 변경하지 않음. | 동일 raw에서 표기만 재검사; 본실험 재실행 불필요. |
| F6 | `mpic.py:935–1019`; `49_eval_query_aware_baseline.py:521`; `108:638` | **성능 구현 차이:** chunk-aligned oldselected read, layer allocation/복사/barrier, normal-only warm-up. | 정확성을 보존한 채 불필요한 read/할당을 최적화할 수 있음. 우선 method별 동일 warm-up·GPU profiler로 stage scope 고정. | 소수 동일 request의 before/after logits/KV 및 fresh GPU TTFT, 실제 returned bytes; overlap interval 합산 금지. |
| F7 | current GPU 없음; official source 미검증 | **증거 부족:** actual NF4/BF16 tensor 수치·품질·timeline, global/per-image k 확정 불가. | GPU 확보 후 이 감사의 fixed 기준을 먼저 선언하고3개 내외 실제 image로 검증. 공식 소스가 나오면 pinned commit과 selector 확인. | single image·method-local3turn·변경 prefix, k0/32/N, 실제 pixel/AnyRes와 cache sidecar identity. 지원 추가 후 multi-image/mixed. |

F1/F2는 FullLoad 비교자의 문제이고 MPIC selective core의 오류로 혼동하지 않는다. F3/F4는 미지원/성능 충실성 문제다. RoPE phase relocation 후 reused hidden이 문맥에 의존해 dense와 달라지는 것은 partial reuse의 근사이며 그 자체를 FAIL로 판정하지 않았다. 현재 GPU validation 없이 품질 손실 허용치를 결정하지 않는다.

## 증거 위치

- [REPRODUCE.md](REPRODUCE.md): 안전한 새 디렉터리에서 동일 CPU 감사 재실행.
- [summary.json](evidence/cpu_probe_v4/summary.json), [layer_traces.jsonl](evidence/cpu_probe_v4/layer_traces.jsonl), [generated_history.json](evidence/cpu_probe_v4/generated_history.json).
- [fullload_contract.json](evidence/cpu_probe_v4/fullload_contract.json), [failure_propagation.json](evidence/cpu_probe_v4/failure_propagation.json), [source_payload_integrity.json](evidence/cpu_probe_v4/source_payload_integrity.json).
- [historical_sample_checks.json](evidence/historical_sample_checks.json) 및 원본9개 [historical_raw_9_requests.jsonl](evidence/historical_raw_9_requests.jsonl): 새 GPU 실행으로 취급하지 않음.
- [environment.json](evidence/environment.json), current source snapshot/installed Transformers source snapshot, [source_preservation.json](evidence/source_preservation.json).
