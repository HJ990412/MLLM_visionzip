# Figure 7(b) 재현 — 직접 실행 가이드

head 간 important-token index set의 Jaccard 유사도 heatmap (OPT-6.7B).
importance = attention weight column 합(H2O, §4.1), 모든 head 실계산.

## 기본 사용법

```bash
PY=/home/dblab/anaconda3/envs/hj/bin/python
cd /home/dblab/hj/analysis/fig7b

# 비율 50%, 논문 스타일 스케일(0.8~1.0)
$PY fig7b_jaccard.py --ratios 0.5

# 비율 90%, 스케일 0~1, 32개 레이어 전부
$PY fig7b_jaccard.py --ratios 0.9 --vmin 0 --vmax 1 --grid-all
```

- **첫 실행만 GPU 사용(~2분)**: importance score를 뜨고
  `output/scores_21shot_seed0.pt`에 자동 캐시.
- **이후 실행은 ~8초, GPU 불필요**: 같은 prefix(=같은 --num-shots/--seed)면
  캐시를 읽어 비율/스케일/그림만 다시 계산. GPU 서버가 바쁠 때도 돌아감.

## 주요 옵션

| 옵션 | 기본 | 설명 |
|---|---|---|
| `--ratios` | `0.5,0.1,0.4` | 쉼표 목록. **첫 값이 그림에 사용**되고, 표에는 전부 나옴. 예: `--ratios 0.7` / `--ratios 0.9,0.5,0.25` |
| `--vmin/--vmax` | 0.8 / 1.0 | 컬러바 범위. 논문 스타일=0.8~1.0, 전체 보기=0~1 |
| `--grid-all` | off | 32개 레이어 전부를 4×8 그리드로 렌더 |
| `--num-shots` | 21 | 입력 prefix의 shot 수 (RTE train, lm-eval 템플릿) |
| `--seed` | 0 | shot 샘플링 시드. **바꾸면 새 prefix → 첫 실행은 GPU 필요** |
| `--scores-cache` | auto | `off`로 캐시 비활성, 경로 지정도 가능 |

## 출력물 (`output/`)

- `fig7b_L16_r<ratio>.png` — middle layer 단독 heatmap (논문 형태)
- `fig7b_layers_1_16_32_r<ratio>.png` — L1/L16/L32 3장 비교
- `fig7b_all_layers_r<ratio>.png` — `--grid-all` 시 32개 전부
- `avg_jaccard_all_layers.csv` — 레이어×비율별 평균 (매 실행 갱신)
- 터미널 표: 레이어별 평균 + **무작위 기대치 E(J)=r/(2−r)** 병기

## 해석할 때 볼 것

1. 평균값은 **E(J) 대비 초과분**으로 읽을 것 — 비율을 올리면 값 자체는
   자명하게 1에 수렴함 (r=90%: E(J)=0.818 / r=50%: 0.333 / r=25%: 0.143).
2. 대각선=1.0은 스크립트가 assert로 자동 확인.
3. 지금까지의 측정: L16@50%=0.451, L1@90%=0.944 (논문 Fig 7(b)의 ">0.95"는
   미명세 설정으로 재현 불가; 다만 논문 Fig 8의 수치 체계와는 우리 측정이
   방향 일치 — 상세는 대화 기록/`diag_length.py` 참조).

## 참고: 길이 민감도 진단

```bash
$PY diag_length.py   # 32~1642 tokens에서 L16/L1 평균 변화 (GPU 필요, ~3분)
```
