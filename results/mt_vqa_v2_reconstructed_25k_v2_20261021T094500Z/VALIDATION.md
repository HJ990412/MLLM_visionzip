# 검증 상태

Dataset/official scorer/관련 CPU 검증은 PASS다. GPU gate는 각 모델의 새 receipt로만 판단한다.

| Model | GPU gate | Smoke | 기존250 pilot | 새40 pilot | Main | Main 채택/목표 |
|---|---|---|---|---|---|---:|
| llava | RUNNING | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | 0/375,000 |
| qwen | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | 0/225,000 |
| glm | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | 0/225,000 |


Qwen/GLM의 legacy FP32 FAIL은 기존 임계값 그대로 별도 보존하며 native same-shape/semantic gate와 구분한다. 기존 MT-GQA PASS를 새 GPU PASS로 복사하지 않는다.
