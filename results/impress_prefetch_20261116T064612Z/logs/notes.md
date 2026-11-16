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
