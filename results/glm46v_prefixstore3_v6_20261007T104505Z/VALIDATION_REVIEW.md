# GLM-4.6V-Flash v6 validation 독립 검토

Fresh v6 admission은 **PASS**다. Frozen source나 GPU inference를 수정·실행하지 않고 작은 JSON·로그와 hashes를 읽었다. Main이 완료됐으며 terminal raw/54 paired 재검산과 caller outer session 회계를 아래 완료 보충에서 확인한다.

## 실제 freeze와 gate

- `freeze.json`: SHA256 `1f6f3006a8399c8b62f60d910e422e13d4d644e63fe622bfc05913ed331bb77e`; active pointer 없음.
- `validation/receipt.json`: SHA256 `4e0ddebb7cac5a7359db0c26ce51be3257f616cc52ff785c49e591671f672a82`.
- `validation_manifest.json`: SHA256 `1d79a90ddd1c84901c0d382e6436bfa744d652295884bbedf67f76e735cf442a`.
- Native history regression: SHA256 `799a3c000ef84baf9ddcc3443c742b6f7e05b7a0d21f6df825bd4ceec16f474b`.
- Entire CPU log: SHA256 `e9c32101f1c105ad45758175ad77b229c3f879f4c92a28f449edb1dd33faf8c0`,86passed/13warnings.

Source20파일 및 frozen manifests의 현재 hash가 일치했다. V5와 native package5파일의 exactSHA가 같다. Model/checkpoint/processor/dtype/backend/resolution/seed/batch/greedy/cap32/body/scorer가 같다. 새121caller outer timer와4workflow checks를 동결했고 기존 main/gate/tolerance/quality selection을 복사·변경하지 않았다. GPU admission CPU는vision9/store20/runner21/validation10/analysis15=75PASS, 별도 workflow11을 합친 전체는86PASS다. Six evidence와5CPU log의 실제hash/returncode도 일치했다.

| 이미지 | N | k | k%64 | structureS |
|---|---:|---:|---:|---:|
| n264887 | 126 | 32 | 32 | 19 |
| n153293 | 529 | 133 | 5 | 19 |
| n302358 | 253 | 64 | 0 | 19 |
| n150962 | 252 | 63 | 63 | 19 |
| n170941 | 391 | 98 | 34 | 19 |
| n100552 | 414 | 104 | 40 | 19 |

Actualmin/max126/529, chunkboundary63/64, 가장긴 질문, n100552의 실제two dialogues를 포함한다. Independent image-only score/rank invariance, stock native vision output/full T1first-prefill all40layer/native2head KV bits/firstlogits/allIDs exact, canonicalFullLoad 및 독립Bmemory/actualSSD payload/IDs/nativepositions/firstlogits/generatedIDs/state isolation을 통과했다.

ActualSDPA **causal visibility**는 independent original logical prefix/suffix relation과 exact맞았다. FP64 inputs를 연산 전에 변환한 dense masked/compact480layer comparisons는 preset1e-12/1e-12를 만족했고 maximum `7.1054273576010019e-15`다. BF16SDPA main backend/precision은 그대로다. FP32 shape diagnostic의 n153293/B/layer25 및n100552/B/layer22 FAIL을 유지한다. NativeFP64는 exact arithmetic이나 main inference가 아니다.

Same-image2dialogues×3turns×2cachedarms12integration requests 및 longest-question suffix는 자기actual nativebody history로 실행했다. Native official generate regression은n204894/mtgqa_000040의 actualT1body를 사용해 prompt/inputIDs, firstlogitsbits(maxabs0), entiregeneratedIDs와 모든prefill/decode positionIDs/cache lengths가 수동 경로와exact같았다. Gold/invented history를 쓰지 않았다. 이 regression은 actualmask rawbits 직접저장·비교를 주장하지 않으며 cachedsample maskvisibility comparison과 구분한다.

Loaded40layers/2KVheads/128/BF16(40KiB/token), NF4decoder240/vision0, vision-merger-lm_headBF16/SDPA/nooffload를 실제fingerprint에서 확인했다. Native1024-token stress의 peakallocated/reserved는8.318/8.699GiB다. Scope는model/coexistingstate 포함request allocator다. Model이 실제support되고 해당stress가메모리에맞았다는 증거이며 모든요청/independentGPUfootprint를 대표하지 않는다.

V6 diagnostics는sealedstore 외부sample_*/diagnostics/에있다. Timing중payload전체scan은하지않았고 성공Q/K/V.pt CPU-onlyreplay를 주장하지않는다. Frozenmodel/source로fixture를 재생성한다. 기존v3sealedsidecarFAIL와allowlistrepair, v4partial333, v5complete36549/recordedcore-setup은불변archive다. Priorrequests를v6final에합치지않았다.

## Caller outer setup boundary

Controller의 timed_call은 caller start부터 runner.persist/activate의 완전한 반환까지 측정한다. Identity preparation, expected identity/FD 등록, native writer의 CPU materialization·repack·write·fsync·seal, activation IO summary와 receipt return tail을 포함한다. Inner writer/activation-core duration은 구분해 기록한다. Source clone/importance tail은 request E2E에 이미 포함되므로 중복 청구하지 않는다. 추가 workflow timer fixture 4개가 이 boundary를 검사했다. 이 변경은 native KV·position·decoder·scorer를 변경하지 않았다.

GQA 720/720 및 MT 360/360 COMPLETED는 PASS이고 같은 freeze를 참조한다. Main의 최종 coverage, 실제 physical setup 합과 source attribution 합, paired bootstrap, 기존 baseline SHA는 해당 run의 raw로 재검산해 PASS했다. Output malformed/empty/cap/think marker를 품질 gate로 제외하거나 재시도하지 않으며 semantic/file/identity/nonfinite/resource 오류는 차단한다. OS returned bytes는 NAND 물리 I/O가 아니고 DONTNEED는 cold NAND 보장이 아니다. Native first-token TTFT와 first lexical answer latency도 구분한다.

## V6 terminal 완료 보충

Main은398 images/4,061 dialogues/36,549 final 요청, 방법당12,183 및8,122 followups로 완료됐다. Failed image attempt·retry는0이다. COMPLETED SHA256 `013a32a7423b1131211516b48d4eacbb2102bfc395739088cbd3be060eb1f609`. GQA720 및MT360도 별도 완료했다. 이전v3/v4/v5 요청은 primary 분모에0건이다.

독립 `independent_review/attempt_20261007T130724092714Z/receipt.json`는 모든 native scoring/body/lineage/prompt/inputIDs/3축positions/I/O 및54 paired point/CI를 원래 raw에서 검산해 PASS했고 numeric residual은0이다. Seed1234/10,000 image clusters/pooled sum-count를 사용했다. Actual caller outer physical setup과 session attribution 합은 모두55,670.386902522296ms다. Writer의새outer−inner2,089.627110864967ms 및activation41.401019319892ms가 실제 포함됐으며 nested 값을 중복 합산하지 않았다. Terminal 수치와 해석/한계는 FINAL_ANALYSIS_REVIEW.md 및FINAL_ANALYSIS_REVIEW_RECEIPT.json을 따른다.
