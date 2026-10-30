# 원본과 선택 집합 분포

| 항목 | 공식 validation 전체 | 선택 25k |
|---|---:|---:|
| 이미지 | 40,504 | 25,000 |
| 질문 | 214,354 | 75,000 |
| JPG bytes | 6,636,659,503 | 4,092,807,587 |
| width min | 120.000 | 120.000 |
| width mean | 576.541 | 576.438 |
| width p50 | 640.000 | 640.000 |
| width p95 | 640.000 | 640.000 |
| width max | 640.000 | 640.000 |
| height min | 111.000 | 111.000 |
| height mean | 485.070 | 485.147 |
| height p50 | 480.000 | 480.000 |
| height p95 | 640.000 | 640.000 |
| height max | 640.000 | 640.000 |
| pixels min | 14400.000 | 14400.000 |
| pixels mean | 275856.343 | 275894.443 |
| pixels p50 | 273920.000 | 273920.000 |
| pixels p95 | 357760.000 | 359040.000 |
| pixels max | 409600.000 | 409600.000 |
| answer type: number | 28,134 (13.125%) | 9,704 (12.939%) |
| answer type: other | 105,679 (49.301%) | 37,453 (49.937%) |
| answer type: yes/no | 80,541 (37.574%) | 27,843 (37.124%) |
| question_characters min | 10.000 | 10.000 |
| question_characters mean | 30.832 | 30.947 |
| question_characters p50 | 29.000 | 29.000 |
| question_characters p95 | 49.000 | 50.000 |
| question_characters max | 100.000 | 100.000 |
| question_whitespace_words min | 2.000 | 2.000 |
| question_whitespace_words mean | 6.185 | 6.207 |
| question_whitespace_words p50 | 6.000 | 6.000 |
| question_whitespace_words p95 | 10.000 | 10.000 |
| question_whitespace_words max | 23.000 | 22.000 |

원본 이미지당 질문은 3–132개, 평균 5.292개, p50=4, p95=13이다. 선택 workload는 이미지당 정확히 3문항이다. 동일 문장이 다른 image/question ID에 나타나도 제거하지 않았다.

집계 원본 SHA256: `866fa37c4190b008d455bb24fbdfb66190b12ea35cb0aa02c7af417958de8b1f`.
원본·선택 question type 전체 histogram, shape, 해상도 histogram과 overlap ID 목록은 해당 dataset_stats.json에 보존되어 있다.
