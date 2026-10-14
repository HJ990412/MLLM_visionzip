# 완료된 v6 main의 독립 store·setup audit

GPU main 완료 후 실행 지시를 받아 독립 stdlib audit를 수행했다. **398개 adopted 이미지, 796개 store, 64,476개 payload full hash와 2,803,840개 선택 K/V token 행 비교가 모두 PASS**이며 실패 gate는 없었다. Audit wall time은 25.688초이고 GPU inference latency에 포함하지 않는다. [성공 receipt](../store_audit/attempt_20261007T130619344841Z/receipt.json)의 SHA256은 `d9eb354e6da0cbb54d44f6ec0c077029faee478b78aa4d98d8d15e071b1a5fdc`이다. Payload와 frozen source는 수정하지 않았다.

```bash
python results/glm46v_prefixstore3_v6_20261007T104505Z/audit_tools/audit_adopted_main_stores.py \
  --run-dir /home/dblab/hj/mllm_v2/runs/glm46v_prefixstore3_v6_20261007T104505Z \
  --execute-after-main-completion
```

v5 independent audit와 동일하게 adopted COMMITTED owner만 읽고 full hash, 모든 layer/native-head FullLoad→B K/V 및 구조 byte offsets, 40×2×128 BF16·40 KiB/token, 원래-N quarter budget, exact EOF, partial64-row chunk, padding0, native positions/identity, followup selection 및 실제 pread를 검사한다. 기존 v5 script/결과/receipt를 변경하지 않았다. GPU나 포팅 helpers를 호출하거나 payload를 삭제·수정하지 않는다.

v6 추가 검사는 이미지별 두 store method의 **build 1회+activation 1회, 총4개 event**와 source T1 capture 각각1회를 검증한다. caller outer build/activation scope 선언을 고정 계약과 비교하고, outer duration과 nested inner duration을 따로 기록한다. outer가 inner보다 짧거나 scope·source·ownership·event count가 틀리면 실패한다. build 실제 write bytes와 activation integrity returned bytes도 sealed store와 대조한다.

새 audit attempt에 저장하는 파일:

- `receipt.json`
- `store_geometry_hashes.csv`
- `FullLoad_B_bits_selection.csv`
- `caller_outer_setup_scopes.csv`

`caller_outer_setup_scopes.csv`는 caller outer build, inner writer return, caller outer activation, inner activation core, 각각의 preparation/wrapper 차이, 실제 OS write와 activation integrity bytes, 이벤트 ID와 endpoint/scope를 별도 열로 남긴다. 실제 setup 합에는 outer duration만 사용하고 nested inner duration을 다시 더하지 않는다.

Serving E2E+setup 합은 실행 파일 전체 wall time이 아니다. benchmark guard, DONTNEED conditioning, raw/evidence journal write/fsync, post-image full hash/adoption 및 이 독립 audit 비용은 정의된 serving endpoint 밖이다. 이 script는 그 비용을 0이나 GPU 실측으로 채우지 않는다.

실제 audit는 이미지별 두 method의 build/activation 각1회, 총 **796 build +796 activation =1,592 events**를 확인했다. 모든 caller outer duration과 nested inner duration을 별도로 남겼다. 합계는 다음과 같으며 단위는 ms이다.

| Method | caller outer build | inner writer | caller outer activation | inner activation core | 물리 setup 합계(outer만) |
|---|---:|---:|---:|---:|---:|
| FullLoad | 26,170.968 | 25,126.668 | 4,860.990 | 4,840.389 | 31,031.958 |
| B | 21,765.446 | 20,720.118 | 2,872.983 | 2,852.183 | 24,638.429 |
| 합계 | 47,936.414 | 45,846.787 | 7,733.973 | 7,692.572 | 55,670.387 |

실제 caller preparation/wrapper 차이는 build 합계 2,089.627ms, activation 합계 41.401ms였다. 이를 inner duration에 다시 더하지 않고 실제 setup 비용에 caller outer 값만 사용한다.

FullLoad/B sealed logical bytes는 각각 **6,037,819,615 /1,759,348,714 bytes**였다. 실제 main followup 반환 OS bytes는 **123,289,927,680 /35,712,204,800 bytes**, 각 method의 actual pread는 **657,882 calls**였다. 각 method의 source setup OS write 합과 activation integrity returned bytes 합도 sealed logical bytes와 일치했다.

성공 attempt 디렉터리에 receipt와 세 CSV가 보존되어 있다. 실행 로그는 `execution_main_payload_audit.log`, exit code는 0이다. 준비 중 compile과 파일 I/O가 없는 synthetic in-memory setup scope/count/negative 및 scalar 위치 sanity 검사도 통과했다. 전체 score vector가 저장되지 않으므로 full descending rank 재계산은 vision capture 재생성이 필요하고, 이 script는 완전 permutation·첫-k mapping·실제 저장 bits를 검증한다.
