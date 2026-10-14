# 완료된 v5 main의 독립 store audit

`audit_adopted_main_stores.py`는 Python 표준 라이브러리만 사용한다. 포팅 helper 또는 GPU를 호출하지 않고 adopted `COMMITTED.json` 소유 attempt의 파일만 읽는다. 삭제·이동·store 수정은 수행하지 않는다.

**현재는 실행 대기 상태다.** GPU main timing이 끝나고 실행 허가를 받은 뒤 아래 명령을 실행한다. 스크립트도 `main/COMPLETED.json`의 PASS와 398개 이미지 / 36,549개 요청 coverage를 요구하며, 미완료 main에서는 payload scan을 거부한다.

```bash
python results/glm46v_prefixstore3_v5_20261007T075730Z/audit_tools/audit_adopted_main_stores.py \
  --run-dir /home/dblab/hj/mllm_v2/runs/glm46v_prefixstore3_v5_20261007T075730Z \
  --execute-after-main-completion
```

새 output attempt에 `receipt.json`, `store_geometry_hashes.csv`, `FullLoad_B_bits_selection.csv`를 쓴다. 기본 경로는 이 result 디렉터리의 `store_audit/attempt_UTC_TIMESTAMP/`이며 기존 output을 덮어쓰지 않는다. 실패 receipt와 그때까지의 CSV도 남긴다.

검사 범위:

- 완전한 adopted raw/setup/completion/payload-evidence ownership과 hashes.
- strict store seal·metadata·payload allowlist, 전체 파일 SHA256, native 40×2×128 BF16와 40,960 bytes/token.
- 원래-N 기준 ceil(N/4), full permutation/inverse, 정확한 k행 EOF, 64행 partial chunk, padding 0, 모든 구조 행 보존.
- FullLoad canonical byte offsets와 B의 모든 layer/native-head 선택 K/V bits 및 구조 bytes 비교.
- scalar merged-raster 3축 위치·원래 offset·post/partial RoPE identity, frozen image/model/processor/code identity.
- 모든 adopted followup의 선택 집합/원래 IDs/위치/cache 길이/identity 불변성과 실제 returned OS pread 범위·합계.

전체 score vector가 저장되지 않으므로 descending score rank 재계산은 frozen image-only vision capture를 재생성해야 한다. 이 audit은 저장된 완전 permutation과 선택 mapping 및 선택 KV bits를 검증한다. GPU logits·attention 재실행, NAND 읽기, cold 상태나 latency 이득을 검증했다고 주장하지 않는다.

준비 단계에서 Python compile 및 파일을 읽지 않는 synthetic in-memory 위치/선택 bytes/손상 negative sanity check만 수행했다. main payload scan은 아직 실행하지 않았다.
