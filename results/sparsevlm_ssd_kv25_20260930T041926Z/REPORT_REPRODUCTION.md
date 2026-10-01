# 검토본과 완료 후 재현 명령

최종 수치와 projection 표시 정정은 [REPORT_REVIEWED.md](REPORT_REVIEWED.md), 추가 측정 범위는 [SUPPLEMENT.md](SUPPLEMENT.md)에 있다. 원 REPORT와 기존 산출물은 보존했다.

완료된 run에 맞춘 실행·재검산 명령은 [reproduction_commands_after_completion.json](reproduction_commands_after_completion.json)에 보존했다. 기존 output을 지정하는 이전 audit 명령은 덮어쓰기 방지로 거부되므로, 재검산에는 아래 새 output 경로를 사용한다. 이 명령들은 추가 요청을 실행한 기록이 아니다.

```bash
cd /home/dblab/hj/mllm_v2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
# resume_completed_run
/home/dblab/anaconda3/envs/mllm_ft/bin/python scripts/98_eval_sparsevlm_ssd_kv25.py --run-dir /home/dblab/hj/mllm_v2/runs/sparsevlm_ssd_kv25_20260930T041926Z --results-dir /home/dblab/hj/mllm_v2/results/sparsevlm_ssd_kv25_20260930T041926Z --gpu-gate runs/sparsevlm_ssd_kv25_20260930T041926Z/gpu_full_01/gpu_validation.json --phase all --resume
# independent_reaudit_new_output
/home/dblab/anaconda3/envs/mllm_ft/bin/python scripts/99_audit_sparsevlm_ssd_kv25.py --run-dir /home/dblab/hj/mllm_v2/runs/sparsevlm_ssd_kv25_20260930T041926Z --output-dir /home/dblab/hj/mllm_v2/results/sparsevlm_ssd_kv25_20260930T041926Z_reaudit_01
# fresh_setup_replay_new_output
/home/dblab/anaconda3/envs/mllm_ft/bin/python /home/dblab/hj/mllm_v2/runs/sparsevlm_ssd_kv25_20260930T041926Z/setup_split_diagnostic.py --output-dir /home/dblab/hj/mllm_v2/runs/sparsevlm_ssd_kv25_20260930T041926Z/setup_split_replay_01 --cleanup
```

재검산·fresh setup 진단을 반복할 때는 다시 사용하지 않은 output 경로를 선택한다. 완료 run의 resume는 이미 채택된 이미지/대화를 건너뛰고 raw를 교체하지 않는다. Fresh setup replay는 별도 한 이미지 진단이며 5-arm pilot을 대체하지 않는다.

명령 JSON SHA256: `ffe9ca8c2c5578de986bd2e0998f214091030355026e9cd450e3dfeb6133cd9c`.
