# 감사 결과 위치

이번 GPU 후속 감사의 전체 판정·논문–코드 대응표·한계·수정 제안은 [REPORT.md](REPORT.md)에 통합했다. [REPRODUCE.md](REPRODUCE.md)에 환경과 검증 명령이 있다.

현재 새 다중 이미지 경로는 **MPIC-32 (adapted, global k / per-image k)**로 사용할 수 있다. 실제 pretrained GPU와 MMDU 소규모 생성은 검증됐으며, 공식 MPIC 구현·exact multi-image k 규칙·GPU hit-transfer 병렬화·원 논문 judge/성능 재현을 인증하지 않는다. 기존 single-image main의 별도 계약은 이전 감사에 남아 있다.
