# LLaVA 실제 출력과 공식 scorer 입력

검증 데이터에서 image ID 순 첫 두 이미지의 ReComp 3턴을 그대로 표시했다. 성능 추정용 표본이 아니다.

| Question ID | Native raw | Scorer answer | 공식 정규화 후 | VQA score % |
|---|---|---|---|---:|
| 118638002 | `Batting</s>` | `Batting` | `batting` | 60.0 |
| 118638000 | `Baseball</s>` | `Baseball` | `Baseball` | 0.0 |
| 118638001 | `Baseball field</s>` | `Baseball field` | `baseball field` | 100.0 |
| 164001 | `White</s>` | `White` | `white` | 100.0 |
| 164002 | `Yes</s>` | `Yes` | `yes` | 100.0 |
| 164003 | `Pink</s>` | `Pink` | `pink` | 100.0 |

EOS control은 scorer 입력에서 제외하지만 단어를 잘라내거나 대소문자를 사전 변경하지 않는다. 생성 history에도 해당 방법의 실제 native answer를 쓴다.

고정 공식 upstream revision은 GT 10개가 모두 같을 때 추가 정규화를 건너뛴다. 따라서 이 분기의 대소문자 차이는 점수에 영향을 준다. 이는 본 작업에서 만든 정규화 정책이 아니다. 공식 reference를 수정하지 않고 해당 결과를 그대로 primary로 보존했다. Native raw의 EOS 때문에 raw EM이 달라지는 현상과도 구분해야 한다.

원본 GT, token IDs, commit full hash는 llava_answer_examples.json에 있다. 공식 source는 mmimpress/mtvqa25k/vendor에 원본·Python3 diff와 함께 고정되어 있다.
