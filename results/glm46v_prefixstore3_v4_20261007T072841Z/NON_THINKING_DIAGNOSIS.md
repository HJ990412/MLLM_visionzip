# v4 main non-thinking 실패의 native 대조 진단

대상은 `main:mtgqa_000040:recompute:T2`, 이미지 `n204894`이다. V4 main은 이 요청의 생성 형식 이상을 감지하여 중단했고 실패 기록을 보존했다. 이 진단은 pilot/main final 요청에서 제외한다. 동결 source, checkpoint, NF4/BF16/SDPA, native 해상도, seed1234, greedy, cap32, system/question instruction 및 token filter를 바꾸지 않았다.

`main/images/n204894/attempt_001/raw.jsonl`에 있는 해당 방법의 실제 T1 질문·생성 ID·prediction을 사용했다. T1 prediction은 `Crown`, native 생성은 `<|begin_of_box|>Crown<|end_of_box|><|user|>`이다. T2 질문 `Who is the crown worn on?`은 동결 main manifest에서 읽었다. 두 inference 경로와 answer/history 구성에 gold를 전달하지 않았다.

변경하지 않은 v4 runner의 수동 decode와, 같은 로드 모델의 **공식 `model.generate(do_sample=False,max_new_tokens=32,output_logits=True,return_dict_in_generate=True)`**를 대조했다. Native processor/chat template의 `enable_thinking=False`와 같은 pixel preparation을 사용했다. Official generate의 최초 4축 position tensor에는 physical text 축이 추가되므로 나머지 native 3축 RoPE 위치를 수동 경로의 위치와 비교했다.

| Actual T1에서 파생한 history | 두 경로의 prompt/입력 IDs/3축 위치 | 첫 logits bits | 전체 generated IDs | Native 출력 |
|---|---|---|---|---|
| V4 parsed final answer `Crown` | EXACT | EXACT, max abs 0 | EXACT | `The kid (or child) is wearing the crown.</think>The kid (or child) is wearing the crown.<\|end_of_box\|><\|user\|>` |
| Actual native decoded T1 body `<\|begin_of_box\|>Crown<\|end_of_box\|>` | EXACT | EXACT, max abs 0 | EXACT | `<\|begin_of_box\|>The kid<\|end_of_box\|><\|user\|>` |

첫 번째 history의 공식 생성에서도 원래 실패 문자열이 그대로 재현됐다. 생성된 begin-box는 0개, end-box 1개, `<think>` 0개, `</think>` 1개이며 답변 문장이 반복된다. **이 fixture의 수동 decode 위치 포팅 오류는 대조 결과로 배제된다.** Native non-thinking 입력에서 발생한 malformed 출력이며, closing marker만으로 모델의 내부 reasoning 상태를 단정하지 않는다.

두 번째 history는 T1의 **실제 generated IDs**를 native tokenizer의 `skip_special_tokens=True`로 decode했다. Registered EOS/user boundary는 제거되지만, 일반 added token인 native answer-box 경계는 유지된다. 새 답변·gold·임의 prefix를 넣지 않았다. 두 경로 모두 begin/end box 한 쌍과 `The kid`를 생성했고 think marker는 없다. 이 단일 controlled case는 parsed answer만 history에 넣는 v4 경로가 native assistant grammar를 보존하지 못했으며, 실제 native generated body를 보존하면 해당 형식 이상이 사라진다는 근거다. 전체 workload의 모든 native 형식 이상이 해결됐다는 증명은 아니다.

V4의 scorer용 `prediction`과 다음 native assistant history를 같은 문자열로 취급한 점이 protocol 수정 대상이다. 새 history 규칙을 적용하면 prompt IDs·positions가 바뀌므로 v4의 기존 요청과 섞어서 같은 frozen main으로 채택하면 안 된다. 수정은 별도 contract/source freeze/run 및 correctness/pilot 재검증이 필요하다. V4의 실패, partial attempt와 이전 adopted 요청은 그대로 보존한다. 이 진단 자체가 main guard를 완화하거나 token을 제거하지 않는다.

실행한 derived script: [diagnose_non_thinking.py](diagnose_non_thinking.py).

상세 기록: [receipt.json](non_thinking_diagnostics/20261007T075327702092Z/receipt.json), [v4 history 대조](non_thinking_diagnostics/20261007T075327702092Z/v4_final_answer_history.json), [native body history 대조](non_thinking_diagnostics/20261007T075327702092Z/native_generated_t1_body_history.json). 모든 생성 ID·literal text·prompt·실제 history·forward positions·첫 logits hashes를 포함한다. GPU 진단 4요청이 정상 종료했고 source 파일을 수정하지 않았다.
