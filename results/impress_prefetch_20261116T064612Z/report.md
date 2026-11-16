# IMPRESS 다음 layer probe key 선행 로딩 측정

측정값만 적는다. 측정하지 않은 값은 "미실행"이다.

## 설정

- 워크로드: GQA 40장 / 240문항 (`--skip 4 --questions 6`), 비율 0.25, α 0.6, probe head 3개, cold page cache
- store: `runs/impress_fidelity_20261116T052410Z/kvstore_h2o` (재배치 지표 ['h2o'])
- 두 IMPRESS arm 공통: `importance_metric=h2o`, `fallback_v=selected`
- IMPRESS (동기): `prefetch=none` / IMPRESS (선행 로딩): `prefetch=next_layer`
- 측정 순서: 요청마다 ReComp → FullLoad → 두 IMPRESS arm. 두 IMPRESS arm의 순서는 요청마다 번갈아 바꿨다(`--alternate-selector-order`=True). 요청별 순서는 `results.json`의 `selector_order`와 각 arm의 `measure_index`에 기록돼 있다. 동기 먼저 120건, 선행 로딩 먼저 120건.
- 명령: `/home/dblab/hj/mllm_v2/.venv-glm46v/bin/python scripts/04_eval.py --skip 4 --questions 6 --limit 40 --ratio 0.25 --selectors impress_sync,impress_prefetch --importance-metric h2o --fallback-v selected --alternate-selector-order --store runs/impress_fidelity_20261116T052410Z/kvstore_h2o --expect-images 40 --expect-questions 240 --run-dir results/impress_prefetch_20261116T064612Z/main`

## 결과 표

FullLoad·ReComp 대비는 같은 실행 안의 TTFT 평균 비율이다.

| arm | 정확도 | TTFT 평균(ms) | p50 | p95 | FullLoad 대비 | ReComp 대비 | SSD MB | fallback 비율 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ReComp | 62.50% | 508.3 | 510.2 | 595.5 | 0.747× | 1.000× | 0.0 | - |
| FullLoad | 62.08% | 680.9 | 679.5 | 807.1 | 1.000× | 1.339× | 1165.1 | - |
| IMPRESS (동기) | 61.25% | 781.9 | 796.8 | 895.0 | 1.148× | 1.538× | 1009.8 | 33.3% |
| IMPRESS (선행 로딩) | 61.25% | 731.4 | 740.0 | 850.0 | 1.074× | 1.439× | 1009.8 | 33.3% |

## 통과 기준 1 — IMPRESS (동기)가 이전 결과를 요청 단위로 재현하는가

- 판정: **PASS** (기준: `results/impress_fidelity_20261116T052410Z/h2o`의 같은 설정 arm, 공통 요청 240개)
- 요청 단위 일치: `answer` 240/240, `generated_token_ids` 240/240, `ssd_read_bytes` 240/240, `preads` 240/240, `fallback_rate` 240/240. 하나라도 다른 요청 0개
- 집계: 정확도 61.25% → 61.25%, SSD MB 1009.8 → 1009.8, fallback 33.3% → 33.3%

## 통과 기준 2 — 두 IMPRESS arm의 출력과 읽기량이 같은가

- 판정: **PASS** (240개 요청, 한쪽 누락 0)
- 일치한 요청 수: `generated_token_ids` 240/240, `ssd_read_bytes` 240/240, `preads` 240/240, `k_read_bytes` 240/240, `v_read_bytes` 240/240, `probe_read_bytes` 240/240, `fallback_rate` 240/240, `answer` 240/240

## paired TTFT 차이 (IMPRESS (선행 로딩) − IMPRESS (동기))

음수면 선행 로딩이 빠르다. 신뢰구간은 요청 단위 차이의 Student t 95% 구간이다.

- 전체: -50.45 ms (95% CI -62.65 ~ -38.26, n=240), 중앙값 -43.81 ms. 선행 로딩이 빠른 요청 175개, 느린 요청 65개
- 동기를 먼저 잰 요청: -45.26 ms (95% CI -62.29 ~ -28.22, n=120)
- 선행 로딩을 먼저 잰 요청: -55.65 ms (95% CI -73.29 ~ -38.01, n=120)
- 순서 효과 (나중에 잰 arm − 먼저 잰 arm, arm 구분 없이): +5.20 ms (95% CI -8.57 ~ +18.97, n=240)
- 두 순서 그룹의 평균을 같은 가중치로 합친 차이: -50.45 ms

## probe key 읽기 시간과 hook 시간 (요청당 평균, ms)

선행 로딩에서는 읽기와 연산이 겹친다. 따라서 **hook 시간과 디스크 시간의 합을 TTFT로 해석하면 안 된다.** 아래 값들은 서로 겹치는 구간을 재므로 더해서 쓰지 않는다.

- probe 읽기: probe key 읽기 한 번의 벽시계 시간(pread + 버퍼 변환) 합. 어느 스레드에서 읽었든 포함한다.
- probe pread: 그중 `os.pread` 시간만.
- hook이 probe 때문에 멈춘 시간: hook 안의 동기 읽기 시간 + 선행 읽기를 기다린 시간.
- 선행 읽기 대기: hook이 선행 읽기 결과를 기다린 시간.
- 도착 시 완료: hook이 결과를 요청했을 때 이미 읽기가 끝나 있던 layer 수(선행 읽기 layer 수 중).

| arm | probe 읽기 | probe pread | hook이 probe 때문에 멈춘 시간 | 선행 읽기 대기 | 선행 읽기 layer 수 | 도착 시 완료 | hook 시간 | 디스크(pread) 시간 | prefill 시간 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IMPRESS (동기) | 45.34 | 39.32 | 45.34 | 0.00 | 0.0 | 0.00 | 710.1 | 421.5 | 776.9 |
| IMPRESS (선행 로딩) | 40.90 | 35.97 | 3.34 | 0.28 | 31.0 | 30.78 | 657.1 | 411.6 | 726.5 |

## 해석 메모

### 예상과 다른 점
- 예상은 "probe key가 전체 읽기의 약 5%라 줄어드는 시간은 크지 않다"였다. 측정된 감소는
  평균 50.5 ms(TTFT의 6.5%)로, probe key 읽기 시간 거의 전부가 사라진 크기다.
- 바이트 비중(5.4%)보다 시간 비중이 큰 이유: probe key는 layer당 1.7 MB짜리 작은 읽기라
  pread 시간이 0.72 ms/MB로, 전체 평균 0.42 ms/MB보다 비싸다. 동기 arm에서 hook이 probe
  때문에 멈춘 시간은 요청당 45.3 ms(TTFT의 5.8%)였다.
- 선행 로딩에서는 그 시간이 3.3 ms로 줄었다(차이 42.0 ms). 선행 읽기 31개 중 평균 30.8개가
  hook이 요청했을 때 이미 끝나 있었고, 기다린 시간은 요청당 0.28 ms였다. 즉 layer i 연산
  구간(hook 밖 prefill 시간: 요청당 약 67 ms, layer당 약 2 ms)이 layer i+1의 probe 읽기
  한 번(약 1.3 ms)을 덮기에 충분했다.
- 그래도 IMPRESS (선행 로딩)은 같은 실행의 FullLoad보다 1.074×, ReComp보다 1.439× 느리다.
- 측정 순서 효과는 이번 실행에서 +5.2 ms(95% CI −8.6 ~ +19.0)로 0과 구분되지 않는다.
  순서를 번갈아 바꿨으므로 두 arm의 차이에는 영향이 상쇄돼 있다.

### 시간 값 읽는 법
- 선행 로딩에서는 probe 읽기가 연산과 겹친다. hook 시간(657.1 ms)과 디스크 pread 시간
  (411.6 ms)은 서로 포함·중첩 관계이고, 워커의 pread 시간은 hook 시간 밖에서도 흐른다.
  **두 값을 더해 TTFT로 해석하지 않는다.** TTFT는 표의 TTFT 열만 쓴다.
- hook 시간 감소(53.1 ms)가 hook이 probe 때문에 멈춘 시간 감소(42.0 ms)보다 큰 차이
  약 11 ms의 원인은 따로 측정하지 않았다(미실행).

### 구현상 확인한 것
- 선행 읽기는 layer hook 끝에서만 걸리므로 TTFT 타이머 시작 뒤에만 일어난다. 워커 스레드는
  타이머 전에 만들고 시작해 둔다. 요청 전 page cache 비우기와 TTFT 경계는 그대로다.
- 워커는 요청의 `IOCounter`를 직접 건드리지 않는다. 개별 카운터에 기록한 뒤, 메인 스레드가
  결과를 쓸 때 합친다. 그래서 바이트·pread 횟수·종류별 집계가 동기 버전과 같다.
- `logs/equivalence_h2o_selected.log`, `logs/equivalence_default_sparsevlm_full.log`:
  변경 전 `serve.py`(`logs/pre_change_backup/`), 동기, 선행 로딩을 같은 프로세스에서 실제 요청
  15건씩에 돌려 답, token id, 바이트, pread 횟수, 로그, layer별 mask가 모두 같음을 확인했다.
  FullLoad는 선행 로딩을 요청해도 동기로 동작한다.
- `smoke_2images/`: 본 실행 전 2장(12문항) 점검. 표의 수치에는 쓰지 않았다.
- `kvstore/`, `runs/impress_fidelity_20261116T052410Z/`, `results/impress_fidelity_20261116T052410Z/`,
  `results/eval_b25.json`의 파일 크기·수정 시각은 전후로 같다(`logs/protected_manifest_before.txt`).

### 하지 않은 것
- MT-GQA 실행, 재배치 방식 변경, 캐시 계층: 미실행(범위 밖).
- 반복 실행(실행 간 변동), warm cache, 0.25 외 비율, 다른 store(sparsevlm 재배치 store)에서의
  선행 로딩 TTFT 측정: 미실행.
- 선택된 K·V chunk 읽기나 fallback K 읽기의 비동기화, 두 layer 이상 앞선 선행 읽기: 미실행(범위 밖).
