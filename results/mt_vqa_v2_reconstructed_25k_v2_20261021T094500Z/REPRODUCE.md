# 재현과 재개

작업 경로 `/home/dblab/hj/mllm_v2`, 현재 run `/home/dblab/hj/mllm_v2/runs/mt_vqa_v2_reconstructed_25k_v2_20261021T094500Z`.
기존 결과·원본 이미지·검증 자료를 삭제/수정하지 않는다. remote push 없음.
원본 확보/데이터 구축은133→134, 공식 채점 검증135, native geometry136,
membership137, freeze140, GPU gate139, 실제 실행138, 독립 분석142 순서다.

```bash
cd /home/dblab/hj/mllm_v2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=.:scripts:tests
/home/dblab/anaconda3/envs/mllm_ft/bin/python scripts/139_validate_mtvqa25k_gpu.py --run /home/dblab/hj/mllm_v2/runs/mt_vqa_v2_reconstructed_25k_v2_20261021T094500Z --model llava
/home/dblab/anaconda3/envs/mllm_ft/bin/python scripts/138_run_mtvqa25k.py --run /home/dblab/hj/mllm_v2/runs/mt_vqa_v2_reconstructed_25k_v2_20261021T094500Z --model llava --phase smoke
```

완료한 phase는 위 명령으로 재실행하지 않는다. 미완료 phase만 `--resume`로 재개한다.
LLaVA/Qwen은 mllm_ft Python, GLM은 `.venv-glm46v/bin/python`을 사용한다.
Source/config/environment/model/workload 변경이면 같은 main 재개를 거부한다.

```bash
/home/dblab/anaconda3/envs/mllm_ft/bin/python scripts/138_run_mtvqa25k.py --run /home/dblab/hj/mllm_v2/runs/mt_vqa_v2_reconstructed_25k_v2_20261021T094500Z --model llava --phase main --resume
```

Gate→smoke→pilot250→pilot40→main admission을 우회하지 않는다. 각 GPU process는
타 compute PID가 있으면 중단하며 타 작업을 종료하지 않는다. Main은 pilot 실측 자원
계획 PASS도 필요하다. 현재 stage/PID/heartbeat/명령은 run/*_STATUS.json에 있다.
Full hashes와 atomic COMMITTED가 있는 첫 성공 image만 채택한다. Partial image는
새 attempt에서 전체 실행한다. Raw는 lossless gzip으로 보존하며 원본/압축SHA가 commit에 있다.
정리된 새 payload 재생성은 동일 frozen image/source/config의 full T1부터 수행한다.
독립 분석은 `142_analyze_mtvqa25k.py --run ... --model ... --phase ... --output <새 경로>`다.
