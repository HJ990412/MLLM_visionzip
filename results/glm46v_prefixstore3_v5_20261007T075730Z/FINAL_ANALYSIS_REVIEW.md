# GLM-4.6V-Flash v5 최종 분석 독립 검토

**Raw coverage·native history·strict EM·I/O·통계 재계산 PASS.** v5는398 images/4,061 dialogues/36,549 final 요청을 완료했다. 방법당12,183 및8,122 followups이고 실패 image attempt·재시도는0이다. 기존 v3/v4 결과·실패·v4 partial333은 이 분모에 합치지 않았다.

다만 v5의 setup timer는 native writer/read service 범위다. `runner.persist()` 전체 호출 이전의 identity 준비와 호출 후 receipt-return tail을 모두 포함한 outer wall은 **NOT_MEASURED**다. 따라서 아래 session 값의 정합성 PASS를 전체 outer setup 계약의 완전한 이행으로 해석하지 않는다. 이 누락을 추정값으로 메우지 않으며 별도 v6가 올바른 call-site outer timer를 측정한다.

## 독립 검산의 범위와 evidence

- Production runner·store·analysis helper를 import하지 않은 `independent_review/analysis_recheck.py`가 adopted raw398파일/setup398파일의 전체 SHA와 소유 attempt/count를 재확인했다.
- 원래 dialogue/turn/question/gold[0]/ordinal/method rotation을 그대로 확인했다. Pinned tokenizer와 공식 chat template를 사용해 generated IDs→scoring answer와 native body를 각각 재구성했고, 모든36,549 요청의 자기 method/dialogue history·lineage, prompt·expanded image token IDs와 geometry로 직접 만든3축 positions/offset이 일치했다.
- 모든4,061 dialogue의 T1 generated IDs는 세 arm 사이에 exact 같았다. 모든 T1과 ReComp followup은 실제 full-pixel input/vision1회와 KV pread0이었다. Ours raw의 `k`는 T1에서도 저장 예산을 나타내므로 이를 T1 attended content 개수로 읽으면 안 된다.
- Cached 요청의 native2heads/40layers/128/BF16 payload ranges81개, 반환 bytes/calls, H2D bytes, source T1 provenance, original-N budget/선택 IDs/compact length/native positions/decode 증가를 독립 검산했다. 모든 같은 이미지 B followup의 frozen 선택 집합이 같았다.
- Root final CSV54개 paired point/CI를 seed1234·10,000회·398개 이미지 cluster의 pooled sum/count로 독립 재계산했고 numeric residual은 정확0이었다. Request-weighted mean/p50/p95와 session 통계도 원본 raw에서 다시 계산해 일치했다.

독립 receipt SHA256: `c940ee6b662a113af352b11e9deb38638eac9031a46d241ea446ff31831cf2e8`. Bootstrap 재검산: `independent_review/paired_analysis.csv`. 원본 evidence와 source hash는 receipt/script 및 최종 artifact index에서 확인한다.

## 검산한 품질·latency·I/O

| 방법 | 전체 EM % | Followup EM % | Followup TTFT mean/p50/p95 ms | Followup KV read MB/request |
|---|---:|---:|---:|---:|
| recompute | 61.216 | 61.635 | 126.512/127.261/148.743 | 0.000000 |
| fullload | 61.159 | 61.549 | 73.468/73.544/76.630 | 15.179750 |
| ours_prefixstore_kv25 | 58.852 | 58.089 | 60.588/60.471/62.923 | 4.396972 |

Followup correct counts는 R5006/F4999/B4718, 분모는 모두8122다. Strict normalization은 기존 소문자화·punctuation 제거·공백 collapse·a/an/the 제거 규칙과 gold[0] exact match를 그대로 적용했다. Native body/control 토큰을 scoring answer와 혼동하지 않았다.

실제 OS KV returned bytes 총합은 FullLoad123,289,927,680, B35,712,204,800이다. 두 cached arms는 각각657,882 pread calls(요청당81)이며 structure19행을 모두 읽고 padding0이다. ReComp KV read0은 image-file I/O가0이라는 뜻이 아니다. Image-file bytes 자체는 NOT_MEASURED다.

| B−baseline | Followup EM Δpp [95% CI] | TTFT Δms [95% CI] | E2E Δms [95% CI] | KV read ΔMB [95% CI] |
|---|---:|---:|---:|---:|
| recompute | -3.546 [-4.416, -2.692] | -65.925 [-67.218, -64.665] | -67.085 [-68.625, -65.536] | +4.397 [+4.319, +4.475] |
| fullload | -3.460 [-4.328, -2.597] | -12.880 [-13.028, -12.721] | -14.363 [-15.181, -13.504] | -10.783 [-11.018, -10.547] |

GLM의 작은 native40KiB/token KV에서도 B의 요청 TTFT와 실제 읽기량 개선은 관찰됐다. 정확도 CI는 모두 음수이므로 KV25는 measured cohort의 품질 손실을 동반한다. Method-local generated history가 달라지는 방법 수준 비교이며 순수 한 turn KV 삭제 효과만으로 환원하지 않는다. Mean CI를 p95 CI로 바꾸어 해석하지 않으며 0을 포함하는 CI를 동등성 증거로 삼지 않는다.

## Setup 회계와 endpoint 제한

796 build/796 activation events의 **기록된** physical sum과 source-dialogue attribution sum은 모두 `53855.878580827266` ms였다. 같은 setup을 여러 dialogue에 실제 비용으로 중복 청구하지 않았다. Allocation/metadata/file IO 및 source clone/score tail을 request 비용에 다시 더하지 않았다.

| 방법 | A 기록된 shared-stream session mean ms | B derived standalone-equivalent mean ms |
|---|---:|---:|---:|
| recompute | 641.610 | 641.610 |
| fullload | 543.524 | 611.924 |
| ours_prefixstore_kv25 | 513.407 | 567.432 |

| B−baseline | A 기록된 session Δms [95% CI] | B derived Δms [95% CI] |
|---|---:|---:|
| recompute | -128.203 [-131.310, -125.069] | -74.179 [-77.022, -71.325] |
| fullload | -30.117 [-31.767, -28.396] | -44.493 [-46.306, -42.620] |

B standalone-equivalent는 동일 image별 측정 setup을 각 dialogue에 다시 귀속한 DERIVED 값이며 독립 cold-new-image session 실측이 아니다. Non-source T1 capture 추가 비용은 NOT_MEASURED다. V5에는 다음 timing 경계도 추가로 빠져 있다.

`runner.persist()`는 `self.identity(capture)`를 native timer 시작 전에 실행한다. Native timer는 `write_glm_store` call부터 반환 직후까지이며 expected-identity 등록/receipt dict 반환이 뒤에 있다. `runner.activate()`도 duration 평가 뒤 activation IO summary/receipt 반환 tail이 있다. V5 121의 event duration은 이 내부 값을 사용했다. Materialization/repack/serialization/write/fsync/seal과 activation integrity-read/metadata/FD 준비의 주요 core 비용은 기록됐지만 complete outer call wall이라고 부를 수 없다.

그 때문에 A/B 위 수치와 CI는 **기록된 scope의 일치**다. 빠진 identity prep/tail의 값이나 그 영향을0으로 가정하지 않는다. 모든 필요한 setup을 포함하는 deployment session 결론은 별도 v6의 실제 outer timers에서 판단해야 한다. Frozen v5 source/raw/receipts는 수정하지 않았다.

## Native 출력·보호·재현 한계

- Native thinking open/close 및 empty 출력은 세 방법 모두0. Cap/malformed는 R3/F3/B2이며 모두 final 분모와 실제 자기 native-body history에 포함했다. Output 때문에 retry/filter를 하지 않았다.
- 대부분 첫 생성 token은 native answer boundary다. TTFT는 native 첫 출력 token materialization이며 first lexical answer latency는 NOT_MEASURED다. Output-controls를 reasoning token이라고 잘못 세지 않았다.
- Image/request 가중 통계를 구분한다. Bootstrap은 동일 이미지의 모든 방법·대화·turn을 함께 resample하며 request estimand는 pooled sum/count다. Bootstrap은 독립 반복 실행이나 모집단 일반화의 보장이 아니다.
- Logical/allocated store는 file bytes/file st_blocks scope다. Native2heads와 구조/metadata overhead 때문에 content25.1267% 저장이 total bytes75% 감소를 뜻하지 않는다. Directory/inode/kernel FD memory·cold NAND·hit-rate/eviction·병렬 throughput은 이 비교에서 별도로 측정하지 않았다.
- Root REPORT의 품질/TTFT/I/O/paired 해석은 원본 데이터와 일치한다. Full outer setup이라는 원래 표현은 위에 명시한 recorded core scope로 한정해야 한다. ReComp/FullLoad 다른 native shape에 따른 logits 차이와 compression loss를 구분하는 해석도 유지한다.

기존 필수 source/report/details **25파일**의 before/after full SHA256은 모두 일치했다. `independent_review/baseline_protection_after.json` SHA256 `ce1b2956b354c1b78a3fc667c3cd8e77831814375c78c227fed0d874d3180b20`. 이 범위는 모든 기존 대용량 artifact의 global full-hash 보호를 뜻하지 않는다.

추가 payload 전체 검산은 별도 `store_audit/attempt_20261007T102733501663Z/receipt.json`의 PASS와 구분한다. 이번 독립 분석은 그 payload scan을 중복 수행하지 않고 adopted raw·setup·native token/prompt/position·통계와 old-baseline hashes를 검토했다. Reviewer의 첫 실행은 Ours T1의 k를 attended budget으로 가정한 assertion에서 실패했고 `independent_review/attempt_01_failure.json`에 보존했다. Storage budget metadata 해석을 바로잡았으며 production raw와 gate/tolerance는 바꾸지 않았다.
