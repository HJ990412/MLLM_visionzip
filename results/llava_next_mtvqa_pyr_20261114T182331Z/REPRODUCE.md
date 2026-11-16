# 재현 절차 — LLaVA-NeXT PYR-beta-raw packed MT-VQA 25k 단일 arm

환경: conda env `mllm_ft` (`/home/dblab/anaconda3/envs/mllm_ft/bin/python`), RTX 4090, 저장소 `/home/dblab/hj/mllm_v2`. 다른 GPU compute 프로세스가 없어야 한다.
필수 입력(읽기 전용): v3 run `runs/mt_vqa_v2_reconstructed_25k_v3_20261021T100500Z`(이미지별 raw gzip journal 25,000개)와 그 results, `results/layer_budget_kv25_pyrbeta_20261110T073604Z`의 벡터 json,
`data/mt_vqa_v2_reconstructed_25k`. 계약 `docs/llava_next_mtvqa_pyr_contract.md`.

```
PY=/home/dblab/anaconda3/envs/mllm_ft/bin/python
RUN=runs/llava_next_mtvqa_pyr_<UTC timestamp>          # 새 run root (기존 것을 덮어쓰지 않는다)
# 0) 새 파일을 만들기 전의 소스 SHA256 inventory(source_before.json)를 준비한 뒤
$PY scripts/181_freeze_mtvqa_pyr.py --run $RUN --stage prepare --source-before <source_before.json>   # v3 raw index, 데이터·채점·벡터·k_l·모델/환경 검사
$PY scripts/184_analyze_mtvqa_pyr.py --run $RUN --stage selfcheck                                      # v3 paired 표 재현(분석 코드 자기 검사)
$PY scripts/181_freeze_mtvqa_pyr.py --run $RUN --stage freeze
# 1) gate
for s in vg1 vg2 vg3 vg6 vg4; do $PY scripts/182_validate_mtvqa_pyr.py --run $RUN --stage $s; done
$PY scripts/183_run_mtvqa_pyr.py --run $RUN --phase gate_u_pilot40 && $PY scripts/182_validate_mtvqa_pyr.py --run $RUN --stage vg5
$PY scripts/182_validate_mtvqa_pyr.py --run $RUN --stage summary
# 2) smoke -> pilot40 -> admission -> main
$PY scripts/183_run_mtvqa_pyr.py --run $RUN --phase smoke   && $PY scripts/182_validate_mtvqa_pyr.py --run $RUN --stage t1 --phase smoke
$PY scripts/183_run_mtvqa_pyr.py --run $RUN --phase pilot40 && $PY scripts/182_validate_mtvqa_pyr.py --run $RUN --stage t1 --phase pilot40
$PY scripts/183_run_mtvqa_pyr.py --run $RUN --phase admit
$PY scripts/183_run_mtvqa_pyr.py --run $RUN --phase main            # 중단되면 같은 명령에 --resume (freeze 불변일 때만). 2,500장에서 INTERIM_2500.md를 한 번 쓴다
# 3) 독립 감사·분석·판정·보고
$PY scripts/184_analyze_mtvqa_pyr.py --run $RUN --stage final
```

gate가 실패해 코드를 고치면 `--stage refreeze --amendment <json>`으로 수정 기록을 남기고 모든 gate와 smoke·pilot40을 다시 실행한다.
정확도는 greedy·seed 고정이라 같은 모델 snapshot·패키지 버전에서 같은 token이 나와야 한다(G-det-T1, G-det-U가 그 확인이다). TTFT는 실행 환경에 따라 달라진다.
