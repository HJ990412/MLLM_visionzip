# GLM-4.6V-Flash v4 validation 독립 검토

v4 admission의 상태는 **PASS**다. 이 문서는 GPU를 다시 실행하지 않고 작은 JSON·로그·동결 source를 읽고 hash를 확인한 파생 검토다. Main 완료 판정은 별도 `main/COMPLETED.json`과 최종 raw 분석이 필요하다.

## Freeze와 원본 evidence

- Active freeze: `freezes/pre_pilot_v2/freeze.json`, SHA256 `e8d5868ae33c2a77e19eceaa09349421febc9320de8876bf54e6a5a8772a9b50`.
- Admission: `validation/receipt.json`, SHA256 `b9bbd8042eb6561187f8dbd90c40d3433e2ffe143b7630517def17f457bcbaa0`.
- Validation membership: `validation_manifest.json`, SHA256 `9a73f3c8513482955958bec3c1462211326ca29e1c5c98c5ace43813e2edab7e`.
- Active pointer/admission/동결 manifest의 hash가 일치하고 source19파일의 현재 full SHA256도 동결 값과 일치했다. 초기 `freeze.json`이나 run root `config.json` 대신 active pointer가 가리키는 freeze/config/contract를 인용해야 한다.
- Admission에 연결된 6개 sample evidence와 CPU log5개의 실제 hash 및 returncode가 일치했다. CPU vision9/store20/runner14/validation6/analysis8, 총57개 PASS다. 별도 workflow tests는 admission log 목록 밖이다.

## 샘플 coverage와 gate

| 이미지 | 원래 N | k | k%64 | 구조 S |
|---|---:|---:|---:|---:|
| n264887 | 126 | 32 | 32 | 19 |
| n153293 | 529 | 133 | 5 | 19 |
| n302358 | 253 | 64 | 0 | 19 |
| n150962 | 252 | 63 | 63 | 19 |
| n170941 | 391 | 98 | 34 | 19 |
| n100552 | 414 | 104 | 40 | 19 |

실제 processor min/max N=126/529, k=63/64 경계, 가장 긴 native 질문, 같은 이미지의 실제 두 dialogue를 포함한다. 별도 frozen synthetic896×896 resource fixture는 seed1234, N=1024/k=256이며 데이터 품질 표본이 아니다.

Stock/scored-capture T1의 native visual output과 첫 prefill prefix KV bits, 첫 logits 및 전체 생성 ID가 같았다. 두 질문의 vision bits/image-only scores/선택 IDs가 같았다. FullLoad/B는 독립 canonical memory reference와 실제 SSD의 모든40 decoder layers×native2 KV heads의 payload bits/원래 logical IDs/native positions/첫 logits/생성 ID를 비교했다. Causal mask, 원래 위치와 compact physical length 구분, decode 증가, 요청당 loader1회 및 decode 재읽기/재선택0을 검사했다.

같은 이미지 n100552의 사전 고정 mtgqa_002992/002993를 두 cached arm에서 각3 turns 실행한12개 integration 요청과 방법 자체의 generated history를 evidence에 기록했다. Dialogue별 empty history에서 시작했고 memory/SSD·repeat와 가장 긴 질문 suffix 비교를 통과했다. CPU negative suite는 wrong image/model/processor/prefix/KV/position, 중복·누락 structure, head expansion, double budget, future visibility, corrupt/short/unsealed store와 state/history leakage를 거부한다.

Native40 layers/2heads/128/BF16의 토큰당40,960 bytes는 실제 loaded fingerprint에서 확인했다. Store payload의 native2heads를 attention32heads로 늘려 저장하지 않는다. Full/B 모두64-row token-major layout, exact partial EOF, content padding0이다. 전체 CPU materialization/repack을 유지하고 B의 serialization만 ceil(original N/4)로 제한한다.

480개 layer semantic 비교(6samples×2cached arms×40layers)는 입력을 연산 **전에** FP64로 바꾸고 atol=rtol=1e-12를 만족했다. 최대 절대차는 `7.1054273576010019e-15`다. Main은 BF16 SDPA를 유지한다. FP32 1e-5/1e-5 진단의 아래 실패2개는 그대로 보존됐으며 tolerance를 완화하거나 PASS로 바꾸지 않았다.

| 이미지 | arm | layer | FP32 V aggregation 최대차 |
|---|---|---:|---:|
| n153293 | prefix25 | 25 | 1.8596649169921875e-05 |
| n100552 | prefix25 | 22 | 3.8504600524902344e-05 |

ReComp/FullLoad first logits의 native shape 차이는 별도 진단이다. 6sample에서 first logits bits는 다르지만 전체 generated IDs는 같았다. 이를 SSD 복원 오류나 25% 압축 품질 손실로 합치지 않는다. 성공 semantic Q/K/V 전체 `.pt`는 저장하지 않아 CPU-only fixed-input replay를 주장하지 않는다. 원래 동결 모델과 source로 fixture를 다시 생성한다.

## v3 실패와 v4 변경의 귀속

기존 `glm46v_prefixstore3_20261007T065207Z`는 별도 역사적 실행이다. 첫 GPU attempt는 vision/merger/lm_head exclusion 검사에서 forward 전에 FAIL했다. 이후 gate와 두 pilot의 raw/freeze/0% strict EM 및 cap16 truncation을 보존했고 main은 실행하지 않았다. v4는 해당 final requests와 섞지 않는다.

v3 validator가 sealed store 안에 진단JSON을 추가하여 재활성화가 거부된 기록은 `results/glm46v_prefixstore3_20261007T065207Z/STORE_REVIEW.md`에 남아 있다. `STORE_SUPPLEMENT.md`와 repair receipt는 payload/meta/seal을 유지한 채12개 새 run-owned 진단JSON만 정확 allowlist로 store 밖으로 이동하고12개 store 재활성화를 통과한 보완을 기록한다. 원래 실패·중간 source-guard FAIL도 보존했다. v4 validator는 처음부터 `sample_*/diagnostics/{full,prefix25}/`에 진단을 저장하여 sealed store를 바꾸지 않는다. 이 검토는 해당 경로의 JSON 존재와 store내 진단 부재를 확인했고 timing 중 payload full scan을 수행하지 않았다.

v4는 native answer boundary IDs151361/151362의 한 쌍 안 token span을 최종 VQA answer로 decode한다. Question/gold를 extraction에 전달하지 않고 raw generated IDs/전체 decoded text/추출 상태를 남긴다. 경계가 없으면 plain decode, 중복·깨진 경계면 literal decode와 MALFORMED 상태를 남긴다. 동일한 shared runner를 세 arm에 적용하고 strict normalized EM/gold[0] scorer는 그대로다. History에는 그 방법의 실제 최종 추출 답변을 넣는다.

v3 B MT pilot의16-token cap truncation1건을 보존하고 v4 공통 cap32를 main 전에 새 run에서 고정했다. 기존 Qwen/LLaVA cap16과 다르다. v4는 v3 결과가 관찰되기 전 고정됐다고 주장하지 않으며, v4의 GPU/pilot/main 결과 전에 변경 이유와 최종 설정을 고정한 실행이다.

## 자원과 현재 scope

RTX4090/decoder NF4 double quantization/BF16 compute/native KV/SDPA/batch1/greedy/seed1234/no offload를 기록한다. Decoder NF4 modules240, quantized vision0, vision·merger·lm_head BF16이다. N=1024 stress의 request allocator allocated/reserved peak는8.318/8.699 GiB다. Independent deployment KV footprint나 main latency가 아니다. Normal non-thinking output, empty/reasoning/cap checks도 통과했다.

검토 시점 GQA pilot720/720과 MT pilot360/360의 COMPLETED receipts는 PASS이고 active freeze hash가 같다. Main 완료 수·품질·latency·paired CI는 최종보고서에서 판단한다. Validation 요청은 pilot/main 분모에서 제외한다. OS read/write는 NAND 물리I/O가 아니며 DONTNEED 성공도 cold NAND 보장이 아니다.
