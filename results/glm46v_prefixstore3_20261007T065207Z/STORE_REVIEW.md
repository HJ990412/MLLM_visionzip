# GLM PrefixStore-KV25 독립 저장·읽기 검토

이 문서는 v3 source freeze 이후 구현과 기존 artifact를 수정하지 않고 수행한 파생 검토이다. **6개 GPU validation 이미지의 FullLoad/B 총 12개 store에서 payload·seal·원래-N 예산·선택 KV bits·구조 행·native 위치·기록된 실제 OS pread가 일치했다.** 다만 validation 진단 파일이 sealed store 안에 추가되어, 현재 validation 디렉터리 그대로의 재활성화는 거부된다. 이 제한은 아래에 명시한다. 파일럿·본실험의 품질이나 latency 결과를 평가하는 문서는 아니다.

## 사용한 고정 증거

- 성공 receipt: `runs/glm46v_prefixstore3_20261007T065207Z/validation/receipt.json`, SHA256 `f0c7f7b7388a5e231b312034798cd7563061a90a508815121708f9db45a0f1cb`.
- 성공 GPU attempt: `validation/attempt_20261007T071615022603Z`, 상태 PASS, sample 6개, 실패 목록 없음.
- 해당 receipt가 참조하는 freeze: `freezes/pre_pilot_v3/freeze.json`, SHA256 `6d4fea3b2581ed15d2bfdcecabe20e018e35aa609d4e95357012f245d656a5aa`.
- validation manifest SHA256 `9a73f3c8513482955958bec3c1462211326ca29e1c5c98c5ace43813e2edab7e`.
- 이 검토 시점의 모든 freeze 대상 source SHA256 및 6개 `evidence.json` SHA256이 receipt/freeze와 일치했다.
- 이전 실패 `attempt_20261007T071330714753Z`는 그대로 남아 있다. 실패 원인은 `vision/merger/lm_head must remain native BF16`이며 성공 attempt와 구분한다.

## 독립 확인 방법과 실제 일치 범위

포팅의 serializer·permutation·reader helper를 사용하지 않고 Python 표준 라이브러리로 payload를 읽었다. metadata의 hash를 seal과 비교하고, manifest에 있는 모든 파일의 실제 길이와 SHA256을 다시 계산했다. 그 다음 FullLoad canonical 파일에서 `original_index * 512`의 byte 구간을 직접 잘라 B의 각 물리 행과 비교했다. FullLoad/B 구조 파일 전체 bytes도 직접 비교했다. 점수 정렬 자체의 재계산 범위는 아래의 제한과 구분한다.

실제 shape는 40 decoder layers, 2 native KV heads, head dimension 128, BF16이었다. K 또는 V 한 content 행은 `2 heads × 128 dimensions × 2 bytes = 512 bytes`, 모든 layer의 K/V 합은 **40,960 bytes = 40 KiB/content token**이었다. attention head로 확장된 파일은 없었다.

- 총 972개 payload 파일의 길이·SHA256이 일치했다. 각 store는 80개 content K/V 파일과 1개 structural 파일을 가진다.
- B의 K/V token 행 **39,520개**, 즉 20,234,240 bytes가 해당 FullLoad 원래 행과 정확히 일치했다. 모든 decoder layer/native KV head를 포함한다.
- 모든 store의 structure는 원래 prefix에서 content logical IDs를 제외한 정확한 complement이며 중복·누락이 없었다. 이미지별 **19행, 778,240 bytes**를 FullLoad/B 모두 보존했다. 구조 행은 25% 분모에 들어가지 않았다.
- FullLoad/B의 prefix input IDs, content logical IDs, 구조 IDs, 3축 위치, RoPE offset, 모델·processor·이미지·prefix identity가 일치했다. 원래 논리 순서로 구성한 선택+구조 IDs가 GPU evidence의 `logical_indices`와 일치했다.
- raster 공간 좌표를 scalar loop로 따로 구성한 위치가 실제 metadata와 일치했다. image content는 `[start, start+h, start+w]`, image 이후는 `start+max(merged_height, merged_width)`부터 진행하며 compact cache 길이를 사용하지 않았다. metadata의 K state는 `post_rope`이고 partial RoPE 설정은 native `[8,12,12]`, factor 0.5였다.
- 12개 store의 sealed logical bytes 합은 **110,475,944 bytes**였다. 이 수치는 payload+metadata+seal이며, 검사 시점의 진단 sidecar와 디렉터리 inode allocation을 제외한다. 같은 범위의 실제 `st_blocks × 512` 합은 **111,742,976 bytes**였다. 검토를 위한 full payload hash 읽기는 **110,059,520 bytes**였다.

이 검토에서 파일 크기를 실제 OS write 반환량으로 바꾸어 주장하지 않았다. 성공 GPU validation evidence에는 writer의 개별 `os_write`/activation metric이 저장되지 않았으므로 그 실행의 해당 비용을 이 문서에서 재구성할 수 없다. writer의 실제 returned-write 기록과 short-write 동작은 고정 CPU test가 검증하며, 파일럿/본실험 비용은 해당 setup raw를 사용해야 한다.

## 원래-N 예산과 실제 읽기

아래 값은 실제 파일의 byte 수와 GPU evidence의 `call_details.actual_bytes` 합을 각각 확인한 결과이다. MB 환산은 사용하지 않고 정확한 bytes를 표시했다.

| 이미지 | 원래 N | B k=ceil(N/4) | 실제 content 보존율 | B chunk 수 | 마지막 유효 행 | Full content bytes | B content bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| n264887 | 126 | 32 | 25.396825% | 1 | 32 | 5,160,960 | 1,310,720 |
| n153293 | 529 | 133 | 25.141777% | 3 | 5 | 21,667,840 | 5,447,680 |
| n302358 | 253 | 64 | 25.296443% | 1 | 64 | 10,362,880 | 2,621,440 |
| n150962 | 252 | 63 | 25.000000% | 1 | 63 | 10,321,920 | 2,580,480 |
| n170941 | 391 | 98 | 25.063939% | 2 | 34 | 16,015,360 | 4,014,080 |
| n100552 | 414 | 104 | 25.120773% | 2 | 40 | 16,957,440 | 4,259,840 |

| 이미지 | Full 요청 returned pread bytes | B 요청 returned pread bytes | Full sealed logical bytes | B sealed logical bytes |
|---|---:|---:|---:|---:|
| n264887 | 5,939,200 | 2,088,960 | 5,960,978 | 2,110,338 |
| n153293 | 22,446,080 | 6,225,920 | 22,495,347 | 6,272,272 |
| n302358 | 11,141,120 | 3,399,680 | 11,171,616 | 3,428,926 |
| n150962 | 11,100,160 | 3,358,720 | 11,130,588 | 3,387,890 |
| n170941 | 16,793,600 | 4,792,320 | 16,833,479 | 4,830,090 |
| n100552 | 17,735,680 | 5,038,080 | 17,777,127 | 5,077,293 |

각 GPU evidence 요청은 **81 spans / 81 actual pread calls**였다. 80개의 content K/V 파일은 offset 0부터 각 파일의 정확한 저장 행 수까지 한 연속 span으로 읽었고, structural 파일은 한 span으로 읽었다. 모든 call의 actual bytes가 requested bytes와 일치했고 합계가 위 표 및 `ssd_read_bytes`와 일치했다. content는 64행 logical chunk를 사용하지만, 연속 chunk 여러 개를 요청할 때는 파일당 하나의 연속 pread span이다. chunk 수와 pread call 수를 혼동하지 않는다.

FullLoad는 N행, B는 원래 N에서 한 번 계산한 ceil(N/4)행에서 정확히 끝났다. 마지막 파일의 padding 및 extra content 행은 모두 **0**이었다. 전체 permutation은 N개의 원래 content ID를 정확히 한 번 포함하고 inverse가 일치했으며, B 저장 행 mapping은 그 permutation의 첫 k개였다. 선택 이후 길이나 chunk 수로 예산을 다시 계산한 흔적은 없었다. 기록된 요청 읽기는 OS 반환량이고 NAND 물리 읽기나 cold 상태 보장을 뜻하지 않는다.

## 위치·bits·attention 증거와 경계 coverage

성공 GPU evidence는 stock native vision pooler bits, 첫 native prefill prefix KV bits, score capture T1 logits/생성 ID, canonical/선택 memory reference와 SSD의 native bits 및 동일 shape logits/생성 ID 일치를 기록했다. 직접 payload 검토는 이 runtime gate와 독립적으로 B가 FullLoad 원래 행을 정확히 저장했음을 확인했다. 여섯 이미지의 두 arm 각각 40 attention layers, 총 480개 layer evidence가 native suffix visibility와 dense/compact FP64 의미 검증을 통과했다. 최대 FP64 오차는 **7.105427357601002e-15**였다. 별도 FP32 shape 진단의 최대 V aggregation 차이는 **3.8504600524902344e-05**였으며, 이 값을 main 정밀도 변경이나 사후 허용오차 완화로 사용하지 않았다.

398개 `geometry_manifest`의 최소/최대 원래 N은 126/529이며 두 값 모두 실제 GPU store sample에 포함되었다. 실제 GPU store sample은 k=63과 k=64 경계 및 133행의 partial 마지막 chunk를 포함한다. frozen longest question과 같은 이미지의 여러 실제 dialogue/generated history evidence도 포함한다.

실제 최대 해상도 synthetic GPU stress는 **N=1024, k=256**의 정상 source capture를 수행했다. allocated peak는 8,931,302,400 bytes (**8.317924 GiB**), reserved peak는 9,340,715,008 bytes (**8.699219 GiB**)였다. 이 stress는 full/B store를 직렬화·복원한 별도 GPU fixture가 아니므로 GPU store coverage를 N=1024까지라고 부르지 않는다.

성공 receipt의 CPU store 20개 test 중 예산 경계 test가 실제 실행한 N 목록은 `[1,2,3,4,5,63,64,65,251,252,253,255,256,257,511,512,513,1024]`이다. k=63/64/65 및 128/129, 극소 N, partial chunk와 padding 0을 CPU에서 확인했다. 이 일반 경계 test는 작은 native tensor fixture이며, 별도 CPU test가 실제 기대 40×2×128의 N=1 bytes를 확인했다.

**manifest의 `synthetic_chunk_boundaries_N` 목록 `[1,4,5,249,252,253,256,257,508,509,512,513]` 전체가 validation120에서 별도 실행된 것은 아니다.** N=249/508/509의 개별 실행 증거가 없으므로 이를 모두 통과했다고 해석하지 않는다. k=65는 CPU N=257에서 검증되었으며 해당 GPU store sample은 없다. 이것은 선언 목록과 실제 증거 범위의 차이로 기록한다.

## 재활성화와 ranking 재검산의 제한

`NativeAttentionObserver(diagnostic_dir=path)`가 hash activation 이후 각 sealed store 안에 `attention_layers.json`을 작성했다. 따라서 현재 12개 validation store를 그대로 새 `open_glm_store(..., expected_identity=...)`로 열면 모두 `unexpected/incomplete store files`가 발생한다. 직접 재활성화를 시도하여 이 실패를 확인했다. 이 sidecar는 KV content를 추가하거나 payload bits/hash를 바꾸지는 않았으므로 위 직접 byte·seal 검산과 이미 수행된 GPU memory/SSD gate를 무효화하지 않는다. 하지만 **현재 validation 디렉터리 자체가 그대로 재활성화 가능한 store라고 보고해서는 안 된다.** 파일럿/본실험 store에 같은 진단 sidecar를 작성하지 않는 경로는 이 문제와 구분한다.

기존 validation artifact를 삭제·이동하거나 frozen 소스를 바꾸지 않았다. 재사용이 필요하면 새 audit 소유 디렉터리에 meta/seal과 manifest에 명시된 payload만 복사하거나 frozen 설정으로 store를 새로 생성해야 한다. attention 진단 evidence는 원래 위치에 보존한다. 그러한 복사·재생성은 이 읽기 전용 검토에서 실행하지 않았다.

전체 score vector는 store/evidence에 저장되지 않고 SHA256과 full permutation만 남는다. 이번 검토는 score hash의 일치, permutation의 완전성, 첫-k mapping과 runtime 독립 score-rank reference PASS를 확인했다. **surviving artifact만으로 전체 score descending 순위를 다시 계산한 것은 아니다.** 전체 순위의 독립 재현에는 frozen processor/vision capture를 다시 실행하여 같은 image-only scores를 생성해야 한다.

이 문서는 성공 GPU gate를 근거로 latency 이득, 품질 보존, cache hit-rate 또는 NAND 이득을 추론하지 않는다. 해당 결론은 파일럿/본실험의 완료 raw와 독립 분석이 필요하다.
