# MT-VQA-v2-reconstructed-25k 본실험 상태

| Model | GPU gate | Smoke | 기존250 pilot | 새40 pilot | Main | Main 채택/목표 |
|---|---|---|---|---|---|---:|
| llava | RUNNING | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | 0/375,000 |
| qwen | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | 0/225,000 |
| glm | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | 0/225,000 |

목표는 825,000 final requests다. Smoke/pilot/진단/실패 요청은 main 분모에 포함하지 않는다. 실행 중/미실행 모델의 VQA score·paired 차이·속도/IO 감소·setup 포함 session은 **NOT_MEASURED**다.

Primary는 공식 VQA score와 followup TTFT, 첫 persistence/activation을 포함한3-turn session이다. AllHead의 공유 build 귀속0을 독립 배포 무료로 해석하지 않는다. 모델간 효과를 합치지 않는다.
