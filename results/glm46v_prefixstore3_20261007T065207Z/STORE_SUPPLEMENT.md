# Store 독립 검토 보완 기록

**고정 v3 store 모듈 범위에서 CPU 경계 12/12, FullLoad/B 24개 synthetic store, validation 진단 파일 이동 12/12, 이동 후 CPU 재활성화 12/12가 통과했다.** 이전 [STORE_REVIEW.md](STORE_REVIEW.md)는 수정하지 않았다. 이 보완 작업은 GPU 추론·품질·본실험 요청을 실행하지 않았고 frozen store 코드를 변경하지 않았다.

최종 범위 receipt는 [final_scope_receipt.json](../../runs/glm46v_prefixstore3_20261007T065207Z/cpu_validation/store_supplement_20261007T072723Z/final_scope_receipt.json)이며 상태는 `CONDITIONAL_PASS_STORE_SCOPE`이다. SHA256은 `01b990a8647a189df4eb0e7c467bc0b1b538402361cb60685398be80387b273e`이다. 여기서 conditional은 보완 도중 root가 별도 v4 실험을 시작하여 전체 v3 source guard가 실패했다는 범위 구분이다. **전체 v3 source가 여전히 동일하다는 PASS는 주장하지 않는다.** 실제 검증에 사용한 `mmimpress/glm46v/store.py`는 v3 hash `7064adc822702613d6a235aa62447045ba6a5678cec0fd92457896e4f515a8e1`와 계속 일치했고 독립 CPU fixture source도 v3와 일치했다.

## 선언된 CPU 경계 전체 실행

frozen `validation_manifest.json`에 선언된 N 12개를 그대로 실행했다. 모두 실제 **40 layers × 2 native KV heads × 128 dimensions × BF16**, batch 1의 synthetic CPU tensor를 사용했다. 각 N에 FullLoad/B를 각각 직렬화하고, scalar byte-offset reference로 모든 K/V 물리 행과 native 복원 bits를 비교했다. 전체 native token당 40,960 bytes, 구조 행의 예산 제외, full CPU materialization/repack, 마지막 partial chunk, padding 0, 실제 returned pread를 확인했다. score 정렬은 포팅 helper와 독립적인 `sorted` reference를 사용했다.

| 원래 N | B k | B chunk 수 | 마지막 유효 행 | FullLoad/B CPU 검증 |
|---:|---:|---:|---:|---|
| 1 | 1 | 1 | 1 | PASS / PASS |
| 4 | 1 | 1 | 1 | PASS / PASS |
| 5 | 2 | 1 | 2 | PASS / PASS |
| 249 | 63 | 1 | 63 | PASS / PASS |
| 252 | 63 | 1 | 63 | PASS / PASS |
| 253 | 64 | 1 | 64 | PASS / PASS |
| 256 | 64 | 1 | 64 | PASS / PASS |
| 257 | 65 | 2 | 1 | PASS / PASS |
| 508 | 127 | 2 | 63 | PASS / PASS |
| 509 | 128 | 2 | 64 | PASS / PASS |
| 512 | 128 | 2 | 64 | PASS / PASS |
| 513 | 129 | 3 | 1 | PASS / PASS |

이에 따라 이전 검토에서 개별 실행 증거가 없던 N=249/508/509도 CPU store 검증되었다. **이것은 CPU synthetic storage 보완이며 해당 N의 GPU 모델 logits나 실제 native pixels forward를 새로 수행한 결과가 아니다.** CPU geometry는 coherent one-row merged grid로 구성했고 prefix/content IDs는 명시적 synthetic fixture IDs이다.

24개 store의 payload는 총 **174,080,000 bytes**이며 새 run 소유 `cpu_validation/store_supplement_20261007T072723Z/synthetic_stores/`에 보존했다. 각 arm에서 전체 payload hash activation과 요청당 81개 actual pread를 기록했다.

- [CPU receipt](../../runs/glm46v_prefixstore3_20261007T065207Z/cpu_validation/store_supplement_20261007T072723Z/cpu_boundaries_receipt.json): SHA256 `f9aa2420351572d7d9a9ea4c07a63566936522ef9aac766a1a5136f274d7013b`.
- [실행 로그](../../runs/glm46v_prefixstore3_20261007T065207Z/cpu_validation/store_supplement_20261007T072723Z/execution.log): 각 frozen N의 두 arm PASS와 12개 store 재활성화 PASS를 보존한다.
- [실행한 run 소유 audit script](../../runs/glm46v_prefixstore3_20261007T065207Z/cpu_validation/store_supplement_20261007T072723Z/audit_store_supplement.py): 기존 frozen source와 별개의 파생 audit artifact이다. 기존 결과를 덮어쓰지 않는 설계이며, 이미 수행된 이동을 같은 디렉터리에서 재실행하는 명령으로 사용하지 않는다.

## 새 run 소유 진단 파일만 이동

이동 전 12개 validation store의 모든 payload를 다시 full hash 검산하고 meta/seal/진단 파일의 길이·SHA256을 기록했다. 각 store의 재활성화가 기존과 동일한 `unexpected/incomplete store files`로 실패함을 확인하고 **아무 이동도 하기 전에 실패 receipt를 보존했다.**

그 후 source/destination 12쌍, 진단 파일의 SHA256·inode·device·nlink, 변경하지 않을 전체 payload hashes를 포함한 allowlist를 atomic commit하고 디렉터리를 fsync했다. source의 symlink·hardlink, run 밖 경로, 예상 외 파일, 기존 destination을 거부했다. 실제 이동에는 Linux `renameat2(RENAME_NOREPLACE)`를 사용하여 overwrite를 허용하지 않았고 source/destination 디렉터리를 fsync했다.

정확히 아래 형태의 **12개 `attention_layers.json`만** 이동했다. payload, metadata, seal, GPU evidence 및 실패 기록은 삭제·이동·변경하지 않았다.

`validation/attempt_20261007T071615022603Z/sample_NNN_IMAGE/{full,prefix25}/attention_layers.json`

→ `validation/attempt_20261007T071615022603Z/sample_NNN_IMAGE/diagnostics/{full,prefix25}/attention_layers.json`

대상 이미지는 `n264887`, `n153293`, `n302358`, `n150962`, `n170941`, `n100552`의 두 arm이다. 이동 후 진단 파일 hash/inode/device/bytes가 그대로였고 metadata/seal 및 972개 payload hash가 이동 전과 일치했다.

- [이동 전 실패 receipt](../../runs/glm46v_prefixstore3_20261007T065207Z/cpu_validation/store_supplement_20261007T072723Z/before_reactivation_failures.json): SHA256 `b3b2f04b09822537dea9dd3d0f42347a7a541a4ce150c7d809ac0038d172c79a`.
- [이동 전 atomic allowlist](../../runs/glm46v_prefixstore3_20261007T065207Z/cpu_validation/store_supplement_20261007T072723Z/mutation_allowlist.json): SHA256 `ac35d4aae283ec1925c8b7bcfd295aa40051cb2a2e6b90c25c64d8e2f26360df`.
- [이동·재활성화 receipt](../../runs/glm46v_prefixstore3_20261007T065207Z/cpu_validation/store_supplement_20261007T072723Z/repair_receipt.json): 상태 PASS, SHA256 `7a1f4df348f351970b0cbe1e40a353f3dd30ffd0d64d899019f4ff23bc01d342`.

## 이동 후 실제 검증과 source guard 실패의 구분

12개 store를 저장된 완전한 native prefix identity로 새로 활성화했다. 모두 strict file allowlist를 통과하고 전체 payload hash 검증을 수행했다. 실제 activation returned OS bytes 합은 **110,475,944 bytes**로 payload+metadata+seal 합과 일치했다. 활성화 후 prefix load returned OS bytes 합은 **110,059,520 bytes**였다. 모든 layer/native head의 복원 bits를 FullLoad canonical payload의 직접 byte-offset reference와 비교하여 exact 일치를 확인했다. CPU audit의 activation 시간·I/O는 기존 GPU latency/setup 실측을 대체하거나 추가하는 비용이 아니다.

처음 실행의 마지막 전체 source guard는 실패했고, [실패 receipt](../../runs/glm46v_prefixstore3_20261007T065207Z/cpu_validation/store_supplement_20261007T072723Z/supplement_failure.json)와 `execution.exit_code=1`을 그대로 보존했다. 이 실패는 CPU 경계 또는 diagnostic repair 실패가 아니다. 보완 실행 시작 시 전체 v3 source guard는 통과했지만 실행 중 root가 별도 `glm46v_prefixstore3_v4_20261007T072841Z` 실험을 위한 source 변경을 시작했다. root는 v3 raw/freeze/실패를 보존하며 native box answer 처리 보완과 main 이전 공통 cap 결정을 별도 v4로 다룬다고 알렸다. 최종 receipt는 해당 변경 목록을 기록하고 **변경되지 않은 v3 store/CPU fixture 범위에서만 PASS**로 마감한다.

`repair_receipt.json`의 `frozen_source_changed=false`는 repair 코드가 frozen source를 변경하지 않았다는 뜻이며, 전체 작업 공간이 계속 v3 source와 같다는 인증으로 해석하지 않는다. 최종 인증 범위는 `final_scope_receipt.json`과 이 문서의 source guard FAIL 기록을 따른다.

현재 validation store의 replay artifact 문제는 해결되었다. 원래 [STORE_REVIEW.md](STORE_REVIEW.md)의 당시 실패·coverage 제한은 역사적 기록으로 유지하고 이 보완을 함께 읽는다. v4 본실험은 v4의 별도 freeze·GPU correctness gate·파일럿 조건을 충족한 뒤 진행해야 한다.
