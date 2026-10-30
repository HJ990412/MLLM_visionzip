# 실행 중인 실험

Run: `/home/dblab/hj/mllm_v2/runs/mt_vqa_v2_reconstructed_25k_v3_20261021T100500Z`

GPU correctness: LLaVA 7/7, Qwen 6/6, GLM 7/7 PASS. Smoke 및 raw 독립 분석: 60/36/36 요청 COMPLETE. 기존250 파일럿→새40 파일럿→실측 자원 admission→main→독립 분석을 관리 런처가 순차 실행한다.

**전체 825,000-request main은 아직 완료되지 않았다. 아래는 4-image smoke만의 측정이다.** 기존250 파일럿·새40 파일럿·25k main 결과를 대신하지 않는다.

| 모델 | FullLoad All VQA % | Ours All VQA % | Ours−FullLoad pp [95% CI] | Followup TTFT 감소 % | OS read bytes 감소 % | FullLoad→Ours setup 포함 session ms |
|---|---:|---:|---:|---:|---:|---:|
| llava | 71.67 | 57.50 | -14.17 [-28.33, 0.00] | 63.01 | 73.80 | 3833.58 → 1671.54 |
| qwen | 80.00 | 85.00 | +5.00 [0.00, 15.00] | 23.04 | 74.20 | 415.96 → 344.88 |
| glm | 58.33 | 50.00 | -8.33 [-25.00, 0.00] | 16.29 | 70.09 | 607.15 → 610.41 |

LLaVA smoke에서 Ours의 setup 포함 session은 ReComp보다 느렸다(1671.54 vs1356.72ms). GLM에서는 FullLoad 대비 session 차이 CI가 0을 포함한다. 어느 CI도 동등성/비열등성으로 해석하지 않는다. 모든 baseline 비교는 각 모델 smoke/paired_analysis.csv에 있다.

ReComp KV read=0은 image file I/O=0을 뜻하지 않는다. SSD 수치는 OS 반환 bytes이며 NAND 측정이 아니다. 이미지당 dialogue 하나에 persistence/activation을 한 번 더했다. AllHead 공유 build의 독립 배포 환산은 별도 CSV에 DERIVED로 표시했다.

실시간 기준 파일: run/MANAGER_STATUS.json, run/*_STATUS.json, run/WATCHDOG_STATUS.json. 본 문서는 생성 시점 스냅샷이다.

관리 PID594025, watchdog PID601656으로 시작했다. 재시작되면 상태 JSON의 PID가 권위 있는 값이다. Watchdog는 세션과 분리되어 기존 실행을 관찰하며 unexpected exit만 같은 frozen 설정으로 재개한다. 선언된 gate/resource BLOCKED는 반복 재시도하지 않는다.

재개 명령(기존 manager가 실행 중이면 lock으로 중복 실행 거부):

```bash
cd /home/dblab/hj/mllm_v2
/home/dblab/anaconda3/envs/mllm_ft/bin/python scripts/144_manage_mtvqa25k.py --run /home/dblab/hj/mllm_v2/runs/mt_vqa_v2_reconstructed_25k_v3_20261021T100500Z
```

초기 v1/v2 진단 실패와 원본 evidence는 보존했다. 현재 main 재개 대상은 v3다. 원본 데이터, 기존 MT-GQA/250-image 결과 및 모델 환경을 변경하거나 삭제하지 않았고 remote push는 하지 않았다.

남은 한계: main 성능과 pilot 기반 최종 시간/용량 예측은 아직 미측정이다. Gate의 추가 독립 reference 생성 전체 물리 횟수는 NOT_MEASURED이며 production runner journal의 physical/final 횟수와 구분한다. pilot250 raw의 일반 dataset 태그는 DATASET_SCOPE_NOTE.md 및 manifest hash로 해석해야 한다.
