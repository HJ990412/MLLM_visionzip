# Qwen KV25 pilot timing field clarification

The frozen pilot timing addendum's phrase that `image_file_decode_ms` appears in `result` and `timing_ms` for every raw request applies to normal-pixel requests only. For all four arms at T1 and ReComp at later turns, image file decode is measured per request and the duration appears at the raw top level, in `result.image_file_decode_ms`, and in `result.timing_ms.image_file_decode`; it is included in both TTFT and request E2E.

SSD cache-hit rows do not decode the image file. Their raw top-level `image_file_decode_ms` is `null`, and their `result` and `result.timing_ms` omit that key. The report audit checks this distinction. This clarification preserves the already hash-bound pilot source and timing addendum.
