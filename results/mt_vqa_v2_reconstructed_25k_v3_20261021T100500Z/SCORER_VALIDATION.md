# 공식 VQA scorer 검증

Reference revision `a013f0043c1e2cdc995922dfe257f7149aa9af06`, 원본 SHA256
`f08edfcad5be0112500993e245c706b6cb928eadebe203f89f838e5e0d04bec8`. Python3 변경은 print문 2개뿐이다.
원본 blob SHA1, port SHA256, diff는 mmimpress/mtvqa25k/vendor에 있다.

합성 fixture18개와 기존 raw3000개 문항별 공식 reference/호환 점수 비교 PASS, 불일치0.
0/1/2/3/4/10 일치수는0/.3/.6/.9/1/1. 숫자·관사·구두점·소수·축약·공백·다중단어,
빈 답변과 첫 단어만 고르는 잘못된 처리도 검사했다. 기존 결과를 덮어쓰지 않았다.
이 3000개 재채점은 새 방법의 250-image GPU 파일럿이 아니다.

고정 공식 revision은 정답10개가 공백 처리 후 모두 같으면 추가 정규화를 생략한다.
예를 들어 unanimous 'cat'에 'Cat'은0점이다. 이 동작을 수정하거나 normalized EM으로
바꾸지 않았다. Primary는 이 고정 reference의 VQA score (%)다.

신규 GPU raw에서는 생성-only IDs를 native tokenizer로 독립 decode한 전체 raw와
scorer answer를 함께 보존한다. GLM은 v6의 native 답변 box span과 literal history body를
구분한다. 실제 새 출력 검증의 완료 여부는 각 모델 gate/pilot receipt를 따른다.
공식 evaluator: https://github.com/GT-Vision-Lab/VQA/tree/a013f0043c1e2cdc995922dfe257f7149aa9af06
