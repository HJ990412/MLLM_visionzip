# Qwen MT-GQA generated-history KV25 본실험 — 검증 단계 중단

**QWEN 3-ARM MAIN: NOT RUN, 0/36,549.** 전체 VALID가 아니다. 필수 통합 gate G7이 기존 FP32 허용치를 통과하지 않아 smoke와 main을 실행하지 않았다.

| 항목 | 실제 상태 |
|---|---|
| DATASET IDENTITY / METHOD-CONFIG CONTRACT | PASS; 398 images·4,061 dialogues·12,183 turns/method, 원본 SHA와 ordered identity 일치 |
| 기존 Qwen KV25 core GPU 검증 | PASS 10/10 fixed pairs, 기존 G1–G12 모두 PASS |
| CPU regression | 기존 427 + 새 안전성 9 = 436 tests PASS |
| GPU wrapper integration | FAIL at G7; original aggregate UNRESOLVED 보존 |
| Integration 일반 생성 요청 | 13/36 executed raw; 9 requests image-committed; 나머지 4개는 실패 이미지의 미채택 evidence |
| 36요청 smoke | NOT RUN, 0/36 |
| Qwen 3방법 main | NOT RUN, 0/36,549 |
| 독립 감사 | PASS, 실행된 evidence/중단 준수 범위; full-main PASS를 뜻하지 않음 |
| 기존 artifact·source 보존 | PASS; 101,576 files, 기존 변경 0건 |
| LLaVA GPU / MT-VQA / Qwen MPIC·SparseVLM·ReKV | NOT RUN |
| 최신 LLaVA 원본 확인 | PASS, 398 committed images·60,915 raw requests 재검증 |
| CROSS-MODEL TABLE READY | NO; Qwen main 관측치 없음 |

실패 요청은 image `n313060`, dialogue `mtgqa_000285`, Ours-KV25 T2다. N=529, k=133, m=3 whole chunks, structural S=21, compact prefix154, suffix60. 동일 captured-prefix의 독립 rank/gather reference와 production SSD wrapper의 KV bits·first logits·generated token IDs 일치는 assertion을 통과한 뒤 FP32 oracle에서 중단됐다. 이를 main 품질/성능의 검증으로 확대하지 않는다.

기존 고정 criterion은 elementwise `atol=1e-5, rtol=1e-5`다. Saved FP32 dense/compact 출력 215,040개 원소 중 **16개 위반**, 최대 절대차 **2.16215848923e-05**. 별도 실제 GPU 재현에서 Q/K/V 입력은 모두 bitwise exact였다. 독립 FP64 대조 최대 차이는 **1.7763568394e-15**, `1e-12/1e-12` 통과다. 이 근거는 FP32 연산 shape/반올림 영향과 양립하지만 frozen G7 면제나 threshold 완화 근거로 사용하지 않았다. G7 FAIL을 유지한다.

실패 run의 raw, config/manifest/source snapshots, 정상 source T1 store와 미완료 scratch, 원래 UNRESOLVED 검증 JSON을 보존했다. 통과한 첫 이미지의 scratch만 독립 감사+atomic commit+hash allowlist 후 정리했다. 실패 이미지는 4개 일반 요청을 생성했고 전체 이미지 완료가 아니므로 final logical 결과로 채택하지 않았다. 별도 참조 forward·sentinel·예외 주입·재현 진단은 validation controls이며 main 요청 수에 포함하지 않는다. 첫 별도 진단의 JSON 직렬화 오류도 로그로 보존하고 retry1에서 수치 evidence를 저장했다.

모델은 Qwen2.5-VL-7B-Instruct pinned revision, NF4/BF16/SDPA, native4 KV heads·28 layers·head_dim128, chunk64, ratio.25, Ours `budget_unit=visual_kv`다. 기존 core runner/store/vision은 수정하지 않았다. 각 방법이 자신의 실제 decoded 이전 답변만 사용하는 history를 관측 raw에서 독립 재구성했다. Full-image T1은 각 방법별로 실제 실행했다.

본실험 관측치가 없으므로 turn/all/hit 정답률, true TTFT mean/p50/p95, retention, SSD bytes/preads, paired CI 및 session 비용의 **main 통계는 모두 N/A**다. Validation 값을 main 숫자로 대체하지 않는다. 실행된 요청의 true outer TTFT·실제 OS bytes·선택 IDs·layer shapes·absolute endpoints는 integration raw에만 있다.

Persistence/session은 validation 범위에서만 fresh score 후처리·KV clone을 포함한 generation-end→core-return tail, writer materialize/repack/write/fsync/publication 및 activation 전체 payload hash를 측정했다. Main persistence/session은 NOT RUN. Source T1 hooks는 request에 이미 포함되어 중복 청구하지 않는다. 계획한 standalone 환산은 DERIVED이며 hypothetical non-source capture-hook 증분은 별도 미측정이다. POSIX_FADV_DONTNEED는 OS hint이며 NAND cold 보장이 아니다.

최신 LLaVA `llava_mt_gqa_allhead_kv25_main_20260930T093251Z/analysis_v2`의 audit/manifest와 main raw SHA를 읽기 전용으로 확인했다. 공통 3방법 수치는 `llava_common_3_reference_only.csv`에 원본 raw에서 재계산한 참고값으로만 보존했다. Qwen 값이 없으므로 cross-model 결과 표로 발표하지 않는다.

보호 검사는 기존 전체 data/run/result/store와 source 및 외부398개 이미지에 대해 통과했다. Source 및 8MiB 이하 파일은 full SHA256, 큰 기존 파일은 9×64KiB framed fingerprint+inode/size/mtime/ctime이며 full-byte 증명은 아니다. 새 raw/diagnostic/store evidence에는 full SHA256을 사용했다.

정확한 main 재개 진입점:
```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 /home/dblab/anaconda3/envs/mllm_ft/bin/python scripts/103_eval_qwen_mt_gqa_kv25_main.py --run-dir /home/dblab/hj/mllm_v2/runs/qwen_mt_gqa_kv25_main_20261001T031459Z --phase all --resume
```
**현재 이 명령은 integration gate 때문에 중단된다.** G7을 기존 허용치로 해결·재검증하기 전 main을 강제로 진행하지 않는다. 코드/조건 수정이 필요하면 실패 evidence를 유지한 새 prospective experiment version을 만들어 core/통합검증→별도36요청 smoke→main 순서로 실행해야 한다. 동일 조건의 partial image는 새 attempt에서 전체 재생성한다.

실패 진단 재현(기존 디렉터리를 덮어쓰지 않는 새 경로):
```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 /home/dblab/anaconda3/envs/mllm_ft/bin/python /home/dblab/hj/mllm_v2/runs/qwen_mt_gqa_kv25_main_20261001T031459Z/reproduce_oracle_failure.py --output-dir /home/dblab/hj/mllm_v2/runs/qwen_mt_gqa_kv25_main_20261001T031459Z/oracle_reproduction_02
```
