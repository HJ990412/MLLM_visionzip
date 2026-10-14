# GLM-4.6V-Flash v5 validation 독립 검토

v5 admission 상태는 **PASS**다. GPU를 다시 실행하지 않고 작은 JSON·로그·동결 source를 읽고 hash를 확인했다. Main은398 images/36,549 final 요청으로 완료됐다. 이 문서의 admission PASS와 terminal raw coverage·통계 PASS는 구분하며, full outer setup timer 누락은 아래 완료 보충 및 FINAL_ANALYSIS_REVIEW.md에 명시한다.

## Freeze 및 fresh gate evidence

Authoritative freeze는 이 run의 `freeze.json`이다. SHA256 `bd19161f73a93b76f55111adb0f3a0e1c68bed70ed3cf14053263d1e68b02ca0`이며 active pointer는 없다. 이 파일에 source19개, config/environment/contract와 frozen manifests의 full SHA256이 있다. Source19파일의 실제 현재 hash 및 manifests hash가 모두 일치했다. GPU admission의 source freeze hash와 validation manifest hash도 정확히 일치했다.

| Evidence | SHA256 |
|---|---|
| `validation/receipt.json` | `e8881c3d557c7130334fcd11cec72a70d9c72191fb87798160861fe8d173f430` |
| `validation_manifest.json` | `1d79a90ddd1c84901c0d382e6436bfa744d652295884bbedf67f76e735cf442a` |
| `validation/attempt_20261007T080624757356Z/native_history_regression.json` | `799a3c000ef84baf9ddcc3443c742b6f7e05b7a0d21f6df825bd4ceec16f474b` |
| `cpu_validation.log` 전체 suite | `79acad4974eca4bfd440015bf219b99b07012e8ab5d94f121ff4412f5ea82b93` |

Admission CPU는 vision9/store20/runner21/validation10/analysis15, 총75개 PASS다. 별도의 frozen source 전체 pytest suite는82 passed/13 warnings를 기록했다. GPU sample evidence6파일 및 CPU log5파일의 hash와 returncode도 admission 기록과 일치했다. 이전 버전의 PASS를 복사한 gate가 아니다.

## Fixed sample coverage

| 이미지 | 원래 N | k=ceil(N/4) | k%64 | 구조 S |
|---|---:|---:|---:|---:|
| n264887 | 126 | 32 | 32 | 19 |
| n153293 | 529 | 133 | 5 | 19 |
| n302358 | 253 | 64 | 0 | 19 |
| n150962 | 252 | 63 | 63 | 19 |
| n170941 | 391 | 98 | 34 | 19 |
| n100552 | 414 | 104 | 40 | 19 |

실제 processor의 min/max N=126/529, k=63/64 경계, 가장 긴 native 질문, 같은 이미지의 실제 두 dialogue를 포함한다. 별도 synthetic seed1234/896×896 resource fixture는 N=1024/k=256이며 품질 표본이 아니다. 그 request GPU allocated/reserved peak는8.318/8.699 GiB다. Model과 coexisting state를 포함하는 request allocator scope이며 독립 배포 KV footprint가 아니다.

## Exact implementation 및 semantic 비교

Stock/scored-capture T1의 native visual pooler output, 첫 prefill prefix의 모든 layer/native-head KV bits, 첫 logits 및 전체 generated IDs가 일치했다. 같은 이미지의 다른 질문에서도 native vision bits/image-only scores/선택 IDs가 같았다.

FullLoad/B의 independent stable-rank canonical memory와 실제 SSD에 대해 모든40 layers×native2 KV heads의 payload bits, original logical IDs, native positions, 첫 logits·전체 generated IDs를 비교했다. Native SDPA causal **visibility**는 원래 prefix와 suffix index로 독립 구성한 bool reference와 exact 비교했다. 원래 3축 위치/partial RoPE/offset과 compact physical cache length를 구분하고 decode 증가, loader 요청당1회/decode 재읽기·재선택0, metadata/FD-only residency를 검사했다.

Dense masked와 compact 선택 KV의 FP64 attention은480개 layer 비교(6samples×2cached arms×40layers)에서 atol=rtol=1e-12를 만족했다. Q/K/V를 연산 전에 FP64로 변환했으며 최대 절대차는 `7.1054273576010019e-15`다. BF16 SDPA main 정밀도는 변경하지 않았다. FP32 1e-5/1e-5 진단의 아래2개 실패를 그대로 보존했고 tolerance를 완화하지 않았다.

| 이미지 | arm | layer | FP32 V aggregation 최대 절대차 |
|---|---|---:|---:|
| n153293 | prefix25 | 25 | 1.8596649169921875e-05 |
| n100552 | prefix25 | 22 | 3.8504600524902344e-05 |

Same-image mtgqa_002992/002993를 두 cached arms에서3 turns씩 수행한12개 실제 integration 요청과 native generated-body history를 evidence에 남겼다. Empty initial history/state isolation/long-question suffix/memory-SSD exact 비교를 통과했다. CPU fixtures는 wrong image/model/processor/prefix/KV/position, head expansion, 중복·누락 structure, double budget, future visibility, corrupt/short/unsealed store와 history/state leakage를 거부한다.

Native fingerprint는40layers/2KVheads/head_dim128/BF16, 토큰당40,960 bytes다. Decoder NF4 modules240, quantized vision0, vision·merger·lm_head BF16, SDPA, RTX4090/no offload를 확인했다. Native KV heads를 attention32heads로 확장해 저장하지 않는다. Full/B는64-row token-major contiguous layout/partial EOF/content padding0이며 B는 full CPU materialization·full repack 후 original-N의 상위 k만 직렬화한다.

## Native history와 공식 generate regression

Known v4 failure fixture n204894/mtgqa_000040에서 **실제** T1 generated IDs `[151361,34,4830,151362,151336]`로 body `<|begin_of_box|>Crown<|end_of_box|>`를 구성했다. 마지막 EOS151336만 제외했다. Question/gold로 history 단어를 만들거나 정규화 prediction을 history에 넣지 않았다.

Native body history에서 수동 ReComp와 공식 `model.generate`의 prompt/input IDs, 첫 logits bits(max abs0), 전체 generated IDs 및 모든 prefill/decode position IDs·cache lengths가 정확히 같았다. First logits SHA256은 양쪽 모두 `2ee40ffe0865e3448ea47f983dea408238b475363a7eb9c81bda38688be1a7d9`다. 이 regression의 recorded comparisons는 masks의 raw bits 직접 비교를 주장하지 않는다. Cached sample native masks의 독립 visibility 비교와 구분한다.

Scoring prediction은 기존 native answer-box span 처리이고 strict normalized EM/gold[0] scorer는 같다. History는 `actual-generated-body/terminal-EOS-only-v1`: loaded generation_config의 EOS집합151329/151336/151338/151348에 속하는 마지막 token 한 개만 제외하고 skip_special_tokens=False로 decode한다. Box/thinking controls·whitespace를 보존한 actual body를 native chat template가 과거 assistant의 공식 규칙으로 직렬화한다. Analyzer는 pinned tokenizer로 body/text/IDs/lineage를 독립 재구성한다.

Thinking token을 suppression하지 않고 forced EOS를 넣지 않는다. Native malformed/empty/cap/stray marker는 raw에 모두 기록하고 그 결과 자체를 gate FAIL 또는 좋은 출력까지 재시도 조건으로 삼지 않는다. KV/position/mask/identity/file integrity/nonfinite/resource 오류는 계속 실행을 차단한다. TTFT는 native 첫 생성 token materialization까지이며, marker가 관찰되면 해당 요청이 최종 답변 첫 token까지의 시간이라는 해석을 구분해야 한다.

## 이전 실행과 현재 scope

v3는 decoder quantization exclusion 실패와 이후 gate/pilots의 parser omission에 따른0% strict EM/cap16 truncation 기록을 보존했다. v4는 parser와cap32를 고정한 별도 실행이나 native body 대신 scoring prediction을 history에 넣어 native protocol이 달라졌다. Main333채택/실패 image raw21·STARTED22를 보존했으며 공식 generate 비교가 해당 출력의 native behavior와 history representation 영향을 확인했다. **v4의333은 v5에 합치지 않는다.** v5는 새 gate·720 GQA·360 MT pilot을 모두 실행했다.

v3 sealed validation store에 추가됐던 diagnostic JSON의 reactivation FAIL, exact allowlist relocation/rehash/reactivation 보완 및 source-guard FAIL 기록은 기존 STORE_REVIEW/STORE_SUPPLEMENT와 receipt에 남아 있다. V4/v5 diagnostic JSON은 처음부터 store 밖 `sample_*/diagnostics/`에 저장한다. 이번 v5 검토에서도 store내 진단 부재와 외부 JSON 경로를 확인했다.

현재 GQA720/720 및 MT360/360 COMPLETED receipts는 PASS이고 같은 freeze hash다. Main COMPLETED는PASS(398images/36,549requests), failedattempt/retry0이다. GPU timing 종료 후 store audit와 독립 raw/통계 재계산 및25-file baseline SHA 검산을 수행했다. Payload full scan은 별도 store audit receipt에서 확인한다. 성공 semantic Q/K/V 전체 `.pt`가 없어 CPU-only fixed-input replay를 주장하지 않으며 동결 모델로 fixture를 재생성해야 한다.

OS pread/write는 NAND 물리I/O가 아니며 DONTNEED는 cold NAND 보장이 아니다. 이 실행은 cache saturation/hit-rate/eviction 효과를 검증하지 않는다.

## Terminal completion 보충

`main/COMPLETED.json` SHA256 `d80ec3060b1c0fce9e2b4898242c578dada60cce70c4150ee39eda499496df77`. 독립 `independent_review/receipt.json`은 native body/scoring/lineage/prompt/inputIDs/positions/I/O/54 paired CI를 원래 raw에서 다시 검산하여 PASS했다. Recorded setup physical sum과 attribution sum은 모두53,855.878580827266ms다. 다만 v5 duration은 write_glm_store call/activation-core scope이며 identity 준비·wrapper-return tail 전체가 NOT_MEASURED다. 완전한 outer session 계약을 충족했다고 주장하지 않으며 별도 v6가 call-site wall을 다시 측정한다. 자세한 실제 수치와 제한은 FINAL_ANALYSIS_REVIEW.md를 따른다.
