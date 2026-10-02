## 측정된 store bytes에 근거한 용량 추정

종료 시점 여유 175.146 GiB에서 30 GiB reserve를 남긴 가정상 KV 전용 여유는 145.146 GiB다. 기존 raw·검증 payload·다른 파일은 그대로 두며 추가 쓰기가 없다는 가정이다. 이미지별 실제 allocated bytes를 사용했다.

| Method | 398-image stores GiB | Mean-size capacity estimate (images) | Fits KV-only headroom |
|---|---:|---:|---|
| llava / fullload | 468.781 | 123 | False |
| llava / mpic32_ssd | 440.280 | 131 | False |
| llava / sparsevlm_ssd_kv25_allhead | 468.781 | 123 | False |
| llava / ours_prefixstore_kv25 | 114.789 | 503 | True |

FullLoad/AllHead 행은 같은 canonical bundle을 각 단독 배포의 footprint로 표시한 것이며 실제 물리 총량에 두 번 더하지 않는다. AllHead 미사용 probe sidecar도 포함한다. Mean-size capacity는 이미지 크기 분포가 같다는 환산이며, 실제 workload 전체의 sum과 fit 여부를 별도 표시했다. 현재 실험은 한 이미지씩 생성·감사·정리했으므로 이것은 실제 동시 적재·eviction·hit-rate 실험 결과가 아니다.
