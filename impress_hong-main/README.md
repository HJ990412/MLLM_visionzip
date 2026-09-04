# IMPRESS on FlexGen — FAST'25 논문 재현

FAST'25 논문 **"IMPRESS: An Importance-Informed Multi-Tier Prefix KV Storage
System for Large Language Model Inference"** (Chen et al.)를 FlexGen 위에
재현한 저장소입니다. 원 논문은 비공개 시스템이므로, 공개된 FlexGen
(OPT-6.7B)을 베이스로 논문의 모든 핵심 메커니즘을 단계별로 구현하고
§6.1 실험 조건을 그대로 따라 검증했습니다.

- 베이스: [FMInference/FlexGen](https://github.com/FMInference/FlexGen)
  commit `004ffef` — 원본 README는 [FLEXGEN_README.md](FLEXGEN_README.md)
- **FlexGen 원본 코드는 한 줄도 수정하지 않았습니다.** 모든 IMPRESS 코드는
  `flexllmgen/impress/` 패키지 안에 있고, attention 훅 등은 전부
  monkey-patch로 주입됩니다 (diff가 곧 구현 전체).

## 구현 범위

| 논문 | 모듈 | 내용 |
|---|---|---|
| §4.1 | `importance.py` | H2O importance (post-softmax attention column 합) |
| §4.2 | `prefix_kv.py`, `radix_tree.py` | 64-token chunk 디스크 저장, radix tree R/NR 매칭 |
| §4.3 | `similarity_guided.py` | 유사도 기반 ITF — probe head 3개, Jaccard, 임계값 t=j^α (α=0.6) |
| §4.4.1 | `reordering.py` | 주기적 KV reordering + per-layer mapping list |
| §4.4.2 | `token_cache.py` | TokenCache (score=freq×importance, dual min-heap) |
| §5 | `impress_serving.py` | 통합 서빙 모드 (GPU/CPU/디스크 3-tier) |
| §6.5 | `selective_loading.py` | probe-key sidecar 기반 선택적 chunk 로딩 (디스크 read 스킵) |
| — | `pos_extend.py` | OPT 학습형 위치 임베딩 Position Interpolation (2048→10240) |

각 단계마다 `test_*.py`(논문 Figure 9/13/14 예제를 assert하는 단위 테스트)와
`verify_*.py`(OPT-6.7B 실측 검증)가 있습니다.

## 대표 결과 (논문 §6.1 조건, RTX 4090 24GB)

논문 기대 대역: prefix KV I/O 시간 **1.5–3.8× 감소**, TTFT **1.2–2.8× 개선**,
accuracy 하락 **1%p 미만**.

| 지표 (vs FullLoad) | RTE | COPA |
|---|---|---|
| IMPRESS TTFT | 336.1 ms | 521.1 ms |
| FullLoad TTFT | 981.5 ms | 963.7 ms |
| ReComp TTFT | 1468.4 ms | 1632.7 ms |
| **I/O 시간 감소** | 8.46× (대역 초과) | **3.80× (대역 상단 ✓)** |
| **TTFT 개선** | 2.92× | **1.85× (대역 내 ✓)** |
| **accuracy 차 (vs ReComp)** | +3.3%p | **+0.0%p ✓** |

COPA는 세 지표 모두 논문과 정합. RTE의 대역 초과(원인: FullLoad baseline
차이, fallback rate 2.7%)를 포함한 이탈 항목별 분석은
[IMPRESS_paper_condition.md](IMPRESS_paper_condition.md)의 deviation 표 참조.

`analysis/`에는 Figure 7(b)/8 (head 간 important-token Jaccard 유사도) 재현과
query-조건부 importance 분석이 있습니다 — 전 레이어 평균은 논문 30B 수치
바로 아래(10%: 0.41 vs 0.48, 40%: 0.57 vs 0.68)로 정합하지만, 레이어별
프로파일은 논문의 "초반 최고" 형태가 아닌 중간층(L9–20) 봉우리형으로
관측되었고 이를 그대로 보고합니다 (자세한 내용:
[analysis/fig7b/README.md](analysis/fig7b/README.md)).

## 설치

```bash
conda create -n impress python=3.10 -y && conda activate impress
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements-impress.txt
pip install -e . --no-deps          # flexllmgen 패키지 (의존성은 위에서 고정)
```

OPT-6.7B 가중치는 첫 실행 때 FlexGen이 `~/opt_weights/opt-6.7b-np`로 자동
변환·저장합니다 (HF에서 다운로드, ~13GB).

## 빠른 시작

```bash
# 단위 테스트 (GPU 불필요, 논문 Figure 예제 검증)
python -m flexllmgen.impress.test_radix_tree
python -m flexllmgen.impress.test_reordering
python -m flexllmgen.impress.test_token_cache

# 통합 서빙 3-mode 비교 (IMPRESS / FullLoad / ReComp, GPU 필요)
python -m flexllmgen.impress.verify_impress_e2e

# 논문 §6.1 조건 재현 (prefix pool 57–64GB 생성, 디스크 공간 필요)
python -m flexllmgen.impress.verify_paper_condition --dataset copa
python -m flexllmgen.impress.verify_paper_condition --dataset rte

# 하이퍼파라미터 sweep (α / probe head 수 / chunk size)
python -m flexllmgen.impress.sweep_impress --num-samples 20
```

전체 실험 명령·옵션 치트시트: [IMPRESS_experiments.md](IMPRESS_experiments.md)

## 문서

- [IMPRESS_flexgen_survey.md](IMPRESS_flexgen_survey.md) — FlexGen 코드 조사
  (attention 경로, KV 레이아웃, offloading 구조)와 구현 지점 설계
- [IMPRESS_experiments.md](IMPRESS_experiments.md) — 실험 실행 가이드
  (명령어, 캐시 기반 빠른 재실행, GPU/CPU 메모리 설정)
- [IMPRESS_paper_condition.md](IMPRESS_paper_condition.md) — §6.1 조건 재현
  설계·결과·논문과의 이탈(deviation) 목록
- [analysis/fig7b/README.md](analysis/fig7b/README.md) — Figure 7(b)/8 재현
  및 query-조건부 유사도 분석

## 알려진 한계

- OPT-6.7B 단일 모델 (13B/30B 미지원), 생성 길이 1 토큰(분류 태스크 기준)
- 논문의 "10분 주기" KV reordering은 요청 횟수 기반 동기 실행으로 대체
- 디스크 read는 page-cache warm 상태 측정 — 절대값보다 상대 비교가 유효
- 4.8–5.7k prefix 연장 방식은 논문 미명세 → wikitext 문서 컨텍스트로 추정 구현

## License

Apache-2.0 (FlexGen 원 라이선스를 따름). FlexGen은
[FMInference/FlexGen](https://github.com/FMInference/FlexGen)의 저작물입니다.
