# GLM-4.6V-Flash validation 독립 검토

결론: 기록된 v3 GPU 정합성 gate는 **PASS**다. 이 검토는 receipt·샘플 evidence·동결 source를 읽기 전용으로 검토한 것이다. GPU inference를 다시 실행하지 않았고 pilot/main 완료나 품질 동등성을 승인하지 않는다.

## Evidence와 freeze 귀속

- 최종 admission: `validation/receipt.json`, SHA256 `f0c7f7b7388a5e231b312034798cd7563061a90a508815121708f9db45a0f1cb`.
- 사용한 source freeze: `freezes/pre_pilot_v3/freeze.json`, SHA256 `6d4fea3b2581ed15d2bfdcecabe20e018e35aa609d4e95357012f245d656a5aa`. active pointer와 admission의 freeze hash가 정확히 일치하고 동결 source 19파일의 현재 전체 hash도 일치했다.
- 사전 validation manifest SHA256 `9a73f3c8513482955958bec3c1462211326ca29e1c5c98c5ace43813e2edab7e`. 초기 `run/freeze.json`은 이전 freeze이며 v3 GPU/main의 최종 freeze로 인용하면 안 된다.
- 최종 샘플 evidence 6파일의 실제 전체 SHA256은 admission에 기록된 값과 모두 일치한다. CPU log 5파일 hash/returncode도 일치한다.

## 고정된 샘플 coverage

| 이미지 | 원래 N | k=ceil(N/4) | k%64 | 구조 행 S |
|---|---:|---:|---:|---:|
| n264887 | 126 | 32 | 32 | 19 |
| n153293 | 529 | 133 | 5 | 19 |
| n302358 | 253 | 64 | 0 | 19 |
| n150962 | 252 | 63 | 63 | 19 |
| n170941 | 391 | 98 | 34 | 19 |
| n100552 | 414 | 104 | 40 | 19 |

실제 398-image processor geometry의 min N=126, max N=529, chunk 경계 k=63/64와 가장 긴 native 질문을 포함한다. n100552는 사전에 고정한 같은 이미지의 실제 두 dialogue를 추가한 샘플이다. GPU resource stress는 manifest에서 고정한 seed1234 synthetic 896×896 이미지이며 N=1024/k=256을 확인했다. 실제 데이터 품질 표본이 아니다.

## Mandatory gate에서 확인한 것

- Stock/scored capture T1의 native visual pooler bits, 첫 prefill의 모든 layer/native-head prefix KV bits, 첫 logits와 전체 생성 ID 비교가 통과했다. 질문을 바꾼 동일 이미지의 native vision 출력·image-only scores·선택 IDs도 같았다.
- FullLoad와 PrefixStore25 모두 독립 stable rank/canonical memory reference와 실제 SSD의 모든 layer/head KV bits, logical IDs, 원래 native positions, 첫 logits 및 전체 generated ID 비교를 통과했다. Position reference는 포팅 helper 대신 geometry로 따로 구성했다.
- Cached suffix의 causal visibility와 decode cache 증가를 검사했다. SSD loader는 요청당 한 번 호출되고 decode 재읽기/재선택은 없었다. 활성화 reader의 metadata/FD 보존과 payload residency 금지를 검사했다.
- 같은 image의 mtgqa_002992/002993 실제 두 dialogue를 각 cached arm에서 3 turns씩 실행한 12개 integration request를 evidence에 보존했다. 각 dialogue는 empty history에서 시작하고 해당 arm의 실제 이전 prediction만 이어 갔다. State isolation repeat와 가장 긴 질문 suffix memory/SSD 비교도 validator의 필수 assertion으로 통과했다.
- Dense masked 선택 KV와 compact 선택 KV는 모든 40 layers×두 cached arms×6 samples=480 layer 비교에서 FP64 atol=rtol=1e-12를 만족했다. Q/K/V를 연산 전에 FP64로 변환했으며 최대 절대차는 `7.1054273576010019e-15`였다. Main의 BF16 SDPA precision/backend는 바꾸지 않았다.
- Native heads=2, layers=40, head_dim=128, BF16 40,960 bytes/token을 실제 fingerprint와 store metadata에서 확인했다. 각 sample의 content/structure payload bytes 및 실제 file size가 이 식과 일치했다. 두 정책 모두 64-row layout/partial EOF/zero padding이며 probe sidecar는 없었다.
- CPU suite는 vision9/store20/runner13/validation6/analysis8, 총56개 PASS다. Wrong image/model/processor/prefix/position/state, KV/shape/head expansion, 중복·누락 structure, permutation/double budget, 미래 suffix visibility, corrupt/short/oversized/missing/unsealed store 등의 negative fixture 이름과 성공 로그를 확인했다. 별도 workflow suite는 이번 admission CPU log 목록에 포함되지 않으며 앞선 CPU evidence와 구분한다.

## 수치 진단과 보존한 실패

FP32 dense/compact 진단의 1e-5/1e-5 실패는 아래 2 layer에 그대로 남아 있다. Native same-shape exact gate와 FP64 semantic gate를 이 실패로 대체하지 않으며 진단 tolerance를 완화하지 않았다.

| 이미지 | arm | layer | FP32 V aggregation 최대 절대차 |
|---|---|---:|---:|
| n153293 | prefix25 | 25 | 1.8596649169921875e-05 |
| n100552 | prefix25 | 22 | 3.8504600524902344e-05 |

FP32 QK/softmax/V stage 최대차가 evidence에 따로 기록돼 있다. 성공 gate의 Q/K/V 전체 tensor `.pt`는 보존되지 않았으므로 이 진단을 CPU-only 고정 입력으로 재생할 수 있다고 주장하지 않는다. 동일 동결 모델·processor·source로 sample을 재생성해야 한다.
ReComp/FullLoad의 다른 native projection/attention shape에서는 6/6 sample의 first logits가 달랐고 최대차는 약0.211–0.3203이었다. 전체 generated IDs는 6/6 같았다. 이는 SSD 복원 오류/압축 품질 손실과 구분한 진단이다.
v2 첫 GPU attempt `attempt_20261007T071330714753Z`는 vision/merger/lm_head BF16 exclusion 검사에서 FAIL했고 GPU sample forward 이전에 중단했다. receipt·traceback·v2 source freeze는 남아 있다. v3는 Transformers5의 실제 `model.visual` module prefix를 exclusion에 명시해 원래 decoder-only NF4 계약을 지켰다. 실패를 PASS로 바꾸지 않았다.

## 자원과 품질 보고의 제한

실제 RTX4090 fingerprint는 NF4 decoder modules240/quantized vision0, BF16 vision, SDPA, no offload를 기록한다. N=1024 resource stress의 allocated/reserved peak는 `8.317924` / `8.699219` GiB다. 이는 request allocator scope와 coexisting model 상태의 관찰값이며 독립 KV footprint나 main latency가 아니다.
정상 inference의 empty/reasoning/cap checks는 통과했다. 다만 native prediction에 `<|begin_of_box|>fence<|end_of_box|>` 같은 box marker가 literal로 남는다. 기존 strict scorer는 이를 제거하지 않으므로 의미상 답이 맞아도 gold[0]와 strict EM이 실패할 수 있다. 이 frozen 실험의 최종 품질은 이 형식 제약을 포함해 보고해야 하며 사후 marker 제거를 기존 strict 결과로 바꾸면 안 된다.
Validation 호출은 pilot/main final 분모에서 제외한다. 이번 검토 중 진행 중인 pilot/main 요청·실패·latency·session/paired CI는 별도 최종 raw 감사가 필요하다.
독립 검토는 metadata/file sizes를 확인했으며 timing 중인 pilot에 추가 I/O를 주지 않도록 validation payload 전체 bytes를 재읽는 별도 full hash scan은 수행하지 않았다. Native payload bits/full hashes는 현재 validator와 durable writer/activation evidence가 뒷받침한다.

기존 필수 source/report/details25파일의 전후 full SHA256은 모두 일치했다. receipt: `baseline_protection_after.json`, SHA256 `ae7a676f7b8d7790179096c80b916e96a321321ffd469f5fee088a99880efce1`. 이 보호 범위는 전체 작업 트리나 모든 기존 대용량 artifact의 full-hash 보호를 뜻하지 않는다.
