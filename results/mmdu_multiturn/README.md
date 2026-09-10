# MMDU multi-turn status

현재 MMDU 결과는 **correctness gate와 workload/capacity feasibility**까지만
포함한다. 기존 single-image SSD prefix store를 독립 이미지별로 만든 뒤 tensor
concat하는 방식은 사용하지 않았고, MMDU의 FullLoad/Static+Diverse
TTFT·SSD·quality 수치는 생성하지 않았다.

## Correctness gate

- 고정 subset: `mmdu:70`, `mmdu:35`, 각 첫 3 turns
- token sequence, visual span, image order, cache length: 6/6 exact
- greedy first token, generated token IDs, response: 6/6 exact
- 사전 허용치: max absolute logit error <= 0.125, mean absolute <= 0.01
- 관측 최댓값: max 0.15625, mean 0.021940438
- `append_only_dynamic_cache_gate_passed=false`
- `static_diverse_mmdu_gate_passed=false`

상세 결과는 `correctness_progressive_seed1234_v2/`에 있다.

## Full workload feasibility

- 110 dialogs, 1,645 turns
- 421 canonical image records, 419 unique resolved paths
- active images/turn: mean 3.249848, max 20
- 실제 cache geometry: 1,176 visual tokens/image, 524,288 B/token,
  616,562,688 B/image
- 최대 dialog visual working set: 12.331254 GB
- per-ID 전체 cache: 259.572892 GB; path-deduplicated: 258.339766 GB
- `prompt + 16 <= 4096`: 340/1,645 turns; 모든 turn이 가능한 dialog 1/110

1,645개 context 길이는 fixed-336 tokenizer projection이다. 실제 LLaVA-NeXT
processor sequence와의 exact 교차검증 범위는 correctness gate의 6 turns다.
상세 표와 그래프는 `feasibility_seed1234/`에 있다.

## Reproduce

```bash
conda activate mllm_ft
python scripts/17_validate_mmdu_cache.py
CUDA_VISIBLE_DEVICES='' python scripts/19_analyze_mmdu_feasibility.py
```

MMDU SSD system run을 열려면 먼저 contextual multi-image span, absolute
position/cache-position, interleaved text dependency를 보존하는 store abstraction을
구현하고 같은 numeric gate를 통과해야 한다.
