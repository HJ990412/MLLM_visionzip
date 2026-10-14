# GLM-4.6V-Flash 추가를 위한 기존 실험 독립 감사

이 문서는 기존 artifact를 읽기 전용으로 감사한 결과다. GLM 실행의 GPU gate,
pilot 또는 main 성공을 대신하지 않는다. 프로젝트 실제 경로는
`/home/dblab/hj/mllm_v2`다. 2026-10-07에 감사했다.

## 기존 완료 실행과 실패 실행 구분

`results/mt_gqa_B_llava5_qwen3_20261001T093921Z/REPORT.md`의 LLaVA는
60,915/60,915 요청 완료이고, 그 실행의 Qwen은 G7에서 BLOCKED이며 main은
0/36,549다. 이후 별도의
`results/qwen_mt_gqa_prefixstore3_20261002T074856Z/REPORT.md`에서 Qwen
3-arm main은 36,549/36,549로 완료됐다. 두 상태를 대체하거나 합치지 않는다.
이번 감사는 두 모델을 실행하지 않았다.

최신 Qwen G7 review는 legacy FP32 dense/compact FAIL을 보존한다.
same-shape native BF16 SSD/memory KV bits, position, mask, logits, 생성 ID를
직접 비교하고, 연산 **전에** FP64로 바꾼 all-layer 독립 attention이
atol=rtol=1e-12를 만족하는지 별도 검사한다. QK, softmax, V aggregation의
shape별 rounding을 분리했다. 이 Qwen 판정은 GLM gate의 PASS 근거가 아니다.
GLM은 자기 모델의 native 연산과 고정 입력으로 같은 종류의 검증을 새로 수행해야 한다.

## 실제 workload와 고정 pilot membership

아래 hash는 실제 파일 전체 bytes에서 독립 계산했다. ordered hash는
각 원래 dialogue 순서와 turn 순서에 대해
`dialog_id + "\t" + turn_id + "\t" + question_id + "\n"`을 이어 붙인
UTF-8 bytes의 SHA256이다.

| 용도 | 실제 파일 | SHA256 | membership |
|---|---|---|---|
| Main index | `data/mt_gqa/dialogues.json` | `2c47cfad2a7ccbb673042b400304d7f3ca03d6fbe59d04fa83db50708c924224` | 398 images, 4,061 dialogues, 3 turns |
| 최신 Qwen manifest | `runs/qwen_mt_gqa_prefixstore3_20261002T074856Z/manifest.json` | `64d71acb240977907ed443cd59b0955be13c7168599942e63c13e7e14fc51b72` | index dialogues와 exact equality |
| 기존 GQA pilot | `runs/prefix25_persistence_20261001T050113Z/gqa_manifest.json` | `40e23fa7aeb22495717fc7eede1227de3a6b4811fde022d9cda71ef73c5e3895` | 40 images × 6 questions |
| 기존 MT pilot | `runs/prefix25_persistence_20261001T050113Z/mt_manifest.json` | `d935dd7f65adee6c05a7c9ce5dbe38f22dbdc9fda9ac989703892817e8118e46` | 40 dialogues × 3 turns, 40 images |

Main ordered hash는 계약의
`0287e0c57813800c781633b969c5cff336b3a3c1a1bdcdbb56d63f6ddab0ca62`와
일치한다. 방법당 12,183 요청, 8,122 T2/T3 followups, 세 방법 총 36,549
final 요청이다. 첫 dialogue는 `mtgqa_000001 / n229548`, 마지막은
`mtgqa_004061 / n90294`다. 이미지별 대화 수는 1–30이다.

두 pilot의 `results/prefix25_persistence_20261001T050113Z/` 사본은
각 `runs/` 파일과 bytes/hash가 같다. 기존 manifest가 있으므로 새 표본을 뽑을
필요가 없다. GQA는 방법당 240, 총 720 요청이며 `history=none`이다.
질문 순서 1–6은 여섯 개의 독립 질문을 나타내며 generated six-turn dialogue가
아니다. MT는 방법당 120, 총 360 요청이며 방법 자체의 생성 history를 사용한다.

GQA 첫/마지막 이미지는 `n355567 / n494918`, MT 첫/마지막 dialogue는
`mtgqa_002749 / mtgqa_003603`다. 이 hash로 고정된 파일에는 모든 question ID,
질문 원문, gold, image ID/path/hash와 순서가 포함된다. MT 40 dialogues의 모든
질문과 gold[0]는 main index의 같은 dialogue와 exact 일치했다.
Main 이미지 398개와 GQA/MT pilot 이미지 각각 40개를 실제 파일로 열어 전체
SHA256을 계산했고 기존 image hash와 모두 일치했다.

Pilot 원본 source manifest도 실제 파일이 있고 기록된 hash와 일치했다:

| 파일 | SHA256 |
|---|---|
| `runs/qwen25_port_20260928T054537Z/gqa_manifest/manifest.json` | `e29d593ef83b106f2088ceceb607a9fb22dff90a542775cf19f68735739c52f2` |
| `runs/qwen25_port_20260928T054537Z/mt_manifest/manifest.json` | `b76425302ad7de6000b9ac3079341120b367c8c5e70a1478469383709a9252b6` |

## 재사용 가능한 lifecycle과 모델별 교체 지점

`110_eval_qwen_mt_gqa_B.py`에서 유지할 계약은 원래 global dialogue ordinal의
3-way rotation을 세 turn 내내 고정하고 image-major 실행하는 것이다. 각 이미지의
실행 대상 중 첫 source dialogue T1에만 store capture/build를 수행하고 나머지
동일 이미지 dialogue T1도 모든 방법에서 full pixels를 실행한다. 각 dialogue에
별도 방법별 history를 만들고 source logical/physical request를 기록한다.

Store writer-return은 request 바깥에서 별도로 측정한다. source T1의 score 및
clone tail은 source request E2E에 한 번만 들어간다. activation은 별도 setup
event다. 실제 pread 반환 bytes/calls/ranges를 raw에 남기고 매 followup에 새로
읽는다. source capture를 release하여 resident full KV가 SSD를 대체하지 않게 한다.

이미지 attempt별로 raw/events를 flush+fsync하고 full payload hash, 독립 audit,
원자적 COMMITTED receipt를 남긴 다음 새 run-owned 정확 allowlist만 정리한다.
부분 이미지 attempt의 raw는 final 분모에 넣지 않고 전체 이미지를 새 attempt로
다시 실행한다. resume는 source/config/manifest hash와 채택 evidence를 재검사한다.
이 흐름은 GLM용 새 runner/auditor에서 재사용할 수 있다.

Qwen의 기존 110/111을 GLM에 직접 호출하면 안 된다. 다음이 모델에 고정돼 있다:
28 decoder layers, 4 native KV heads, S=21 구조 행, Qwen prompt/tokenization,
Qwen MRoPE reconstruction, 57 payload preads, 특정 store meta와 module 경로.
`116_qwen_prefixstore_semantic.py`도 observer가 28 layers에서 멈추고 GQA
참조의 native heads=4가 고정돼 있다. GLM layer/head/position/mapping은 실제
checkpoint/config/native source로 다시 검증해야 한다.

## 독립 분석과 persistence endpoint 주의

`118_report_qwen_prefixstore3.py`는 `113_report_mt_gqa_B.py`의 통계 기능을
사용한다. strict normalized EM은 소문자화, punctuation 제거, 공백 정리,
영어 관사 a/an/the 제거 뒤 raw gold[0]와 exact 비교한다. cluster bootstrap은
seed1234, 이미지 단위 10,000회이며 이미지의 모든 방법/대화/turn을 같이
resample한 pooled sum/count로 request 가중치를 유지한다.

기존 113/118은 persistence의 inner `service_persistence_ms`를 사용하는
부분이 있다. 최신 Qwen root REPORT의 authoritative session은
`reporting_details/`의 **outer writer return** amendment다. 따라서 GLM
analyzer는 `outer_writer_wall_ms`를 우선 사용하고 inner 값을 조용히 대체하지
않는다. source score/clone tail을 setup에 다시 더하지 않는다.

최신 Qwen `reporting_details/session_accounting_audit.json`은 attribution 합과
실제 setup 합 모두 51,552.15482553467 ms, build/activation 각각 796개를
확인했다. 이는 과거 Qwen 수치이며 GLM 측정이 아니다. standalone-equivalent는
이미지별 setup을 매 dialogue에 재부과한 DERIVED 값이고, 비-source T1 capture
증분은 NOT_MEASURED다.

## CPU에서 수행한 일과 GPU admission 구분

이번 독립 감사에서는 파일/순서/hash/count, 실제 image identity, pilot/main
membership, 기준 report/contract/source와 reporting endpoint를 확인했다.
GLM용 manifest, store 형식/negative fixture, 독립 고정 attention 참조, raw
history/I/O audit와 bootstrap 구현은 GPU 없이 개발·검증할 수 있다.
실제 T1 capture 보존, native logits/ID, GPU memory 적합성, non-thinking 생성,
pilot/main latency/정확도는 GLM GPU 실행의 별도 gate와 receipt가 필요하다.
Sandbox 내부 GPU 접근 결과만으로 물리 GPU가 없다고 단정하지 않는다.
