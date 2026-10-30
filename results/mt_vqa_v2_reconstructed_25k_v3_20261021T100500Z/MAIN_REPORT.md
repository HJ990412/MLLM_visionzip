# MT-VQA-v2-reconstructed-25k 본실험 상태

| Model | GPU gate | Smoke | 기존250 pilot | 새40 pilot | Main | Main 채택/목표 |
|---|---|---|---|---|---|---:|
| llava | PASS | COMPLETE | COMPLETE | COMPLETE | COMPLETE | 375,000/375,000 |
| qwen | PASS | COMPLETE | COMPLETE | COMPLETE | COMPLETE | 225,000/225,000 |
| glm | PASS | COMPLETE | COMPLETE | COMPLETE | COMPLETE | 225,000/225,000 |

목표는 825,000 final requests다. Smoke/pilot/진단/실패 요청은 main 분모에 포함하지 않는다. 실행 중/미실행 모델의 VQA score·paired 차이·속도/IO 감소·setup 포함 session은 **NOT_MEASURED**다.

[llava pilot250 독립 분석](llava/pilot250/REPORT.md)

[llava pilot40 독립 분석](llava/pilot40/REPORT.md)

[llava main 독립 분석](llava/main/REPORT.md)

[qwen pilot250 독립 분석](qwen/pilot250/REPORT.md)

[qwen pilot40 독립 분석](qwen/pilot40/REPORT.md)

[qwen main 독립 분석](qwen/main/REPORT.md)

[glm pilot250 독립 분석](glm/pilot250/REPORT.md)

[glm pilot40 독립 분석](glm/pilot40/REPORT.md)

[glm main 독립 분석](glm/main/REPORT.md)

Primary는 공식 VQA score와 followup TTFT, 첫 persistence/activation을 포함한3-turn session이다. AllHead의 공유 build 귀속0을 독립 배포 무료로 해석하지 않는다. 모델간 효과를 합치지 않는다.
