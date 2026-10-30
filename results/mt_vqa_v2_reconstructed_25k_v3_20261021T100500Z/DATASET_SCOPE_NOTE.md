# Phase별 데이터 식별

현재 raw의 `dataset` 필드는 파일럿에도 실험 계열명 `MT-VQA-v2-reconstructed-25k`를 기록한다. **pilot250의 실제 데이터는 기존 250-image PNG와 원래 질문·턴 순서다.** 이 일반 태그만으로 membership을 판단하면 안 된다.

실제 membership은 각 raw의 `phase`, `manifest_sha256`과 해당 immutable manifest의 `dataset` 및 ID·정답으로 독립 감사한다. `dataset_scope_registry.json`에 phase별 경로·full hash·정확한 데이터 이름을 연결했다. 기존250 파일럿, 새40 파일럿, smoke와 main을 합산하지 않는다. 원본 raw는 수정하지 않았다.
