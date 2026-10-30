# MT-VQA-v2-reconstructed-25k 데이터 검증

공식 VQAv2 validation 질문/annotation 214,354쌍과 COCO val2014 JPG를 사용했다.
이미지 파일 40,504개 전부 디코드 가능하며 eligible 40,504개,
제외 사유 집계 {}다. 선택 이미지 **25,000**, 대화 **25,000**,
질문 **75,000**개다. 선택 이미지 전부를 독립 재디코드하고 공식 정답·ID·순서를 검증했다.

CPython random.Random(1234)/MT19937, 숫자 ID canonical 정렬에서 sample(25000),
같은 RNG로 이미지 순서대로 정렬된 question_id에서 sample(3)을 사용했다. 첫 질문을
제외하지 않았다. 원본 독립 질문을 이미지별로 묶은 workload이며 저자 subset/순서 재현이 아니다.
사람 정답 10개와 answer_id/confidence, answer_type/question_type, 원본 annotation을 보존했다.

기존 250-image 개발 데이터와 이미지 166개·질문
199개 중복이다. 새 holdout이라고 주장하지 않는다.
추가 개발 이미지의 exact 파일/RGB 비교는 최초 run 결과의 development_overlap.json에 있다.
ConvBench source와도 선택 이미지 87개에 해당하는 파일이 겹친다. 이 비교는 리사이즈된
이미지의 지각적 유사성이나 로컬 밖 데이터까지 배제하지 않는다.

선택 JPG 실측 합계: 4,092,807,587 bytes.
전체/선택 answer type, 해상도, 이미지 모양, 질문 수/길이 histogram은 dataset_stats.json에 있다.
모델별 예상 N 분포와 native helper source hash는 run/geometry_*.json에 있다.
LLaVA N960–2880, Qwen N256–529, GLM N16–529. 실제 GPU N과 예산은 별도 감사한다.

Ordered workload SHA256: `eb3c5e812fd6937cc0741cac6dd85d6839a7d00288dffbfa7ea8294272269cc8`.
Dataset config SHA256: `1b99478b0c0f253810453f169b91f9843973c05dd815b3cc46aeabcc2a07b933`.
원본 archive URL·파일명·바이트·실제 SHA256·버전/HTTP 헤더와 CRC 기록은 source/ 및
source_manifest.json, 이미지별 full hash는 source_image_manifest.json에 있다.
공식 원본: https://visualqa.org/download.html
