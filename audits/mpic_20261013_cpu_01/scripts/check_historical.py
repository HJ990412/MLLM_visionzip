"""Independently check one recorded dialogue; never execute historical runners."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
DEFAULT = Path(__file__).resolve().parents[1] / "evidence"
SNAPSHOT = Path("/home/dblab/.cache/huggingface/hub/models--llava-hf--llava-v1.6-vicuna-7b-hf/snapshots/c916e6cdcd760b4cecd1dd4907f84ac649f93b23")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--evidence", type=Path, default=DEFAULT)
    a = p.parse_args()
    rows = [json.loads(x) for x in (a.evidence / "historical_raw_9_requests.jsonl").read_text().splitlines()]
    by = {(r["method_id"], r["turn_id"]): r for r in rows}
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    checked = []
    for r in rows:
        method, turn, s, g = r["method_id"], r["turn_id"], r["result"], r["geometry"]
        expected_history = []
        for t in range(1, turn):
            prior = by[method, t]
            expected_history += [f"Q{t}: {prior['question'].strip()}", f"A{t}: {prior['prediction']}"]
            e = r["history_entries"][t - 1]
            assert e["source_method_id"] == method and e["source_dialog_id"] == r["dialog_id"]
            assert e["answer"] == prior["prediction"]
            assert e["source_physical_execution_id"] == prior["physical_execution_id"]
        history = "\n".join(expected_history)
        assert r["history"] == history
        body = ([history, ""] if history else []) + [f"Current question Q{turn}: {r['question'].strip()}",
            "Answer the current question with a single word or short phrase. ASSISTANT:"]
        expected_prompt = "USER: <image>\n" + "\n".join(body)
        assert r["prompt"] == expected_prompt
        ids = tokenizer(expected_prompt).input_ids
        image_at = ids.index(32000)
        suffix = ids[image_at + 1:]
        assert r["suffix_input_ids"] == suffix
        assert r["prompt_sha256"] == hashlib.sha256(expected_prompt.encode()).hexdigest()
        recomputed = (r["first_token_at_s"] - r["request_started_at_s"]) * 1000
        assert abs(recomputed - r["ttft_ms"]) < 1e-5
        expected_bytes = 0
        if turn > 1 and method != "recompute":
            kv = g["num_layers"] * 2 * g["v_token_num"] * g["num_heads"] * g["head_dim"] * 2
            expected = Counter()
            for li in range(g["num_layers"]):
                for name in ("k.bin", "v.bin"):
                    length = g["v_token_num"] * g["num_heads"] * g["head_dim"] * 2
                    expected[(f"layer_{li:02d}", name, 0, length, length)] += 1
            embedding = 32 * g["num_heads"] * g["head_dim"] * 2 if method == "mpic32_ssd" else 0
            if embedding:
                expected[("visual", "visual_input.bin", 0, embedding, embedding)] += 1
            actual = Counter()
            for e in s["os_pread_trace"]:
                path = Path(e["path"])
                parent = "visual" if path.name == "visual_input.bin" else path.parent.name
                actual[(parent, path.name, e["offset"], e["requested"], e["returned"])] += 1
            assert actual == expected
            expected_bytes = kv + embedding
            assert r["ssd_read_bytes"] == expected_bytes and r["ssd_preads"] == sum(expected.values())
            assert all(e["status"] == "SUCCESS" for e in s["conditioning_events"])
            assert s["page_cache_conditioning_excluded_from_ttft"]
            if method == "mpic32_ssd":
                assert s["selected_image_local_rows"] == list(range(32))
                assert s["same_source_target_context"] and s["same_source_target_positions"]
                assert s["attention_key_length_per_layer"] == [image_at + g["v_token_num"] + len(suffix)] * g["num_layers"]
                assert s["active_rows_per_layer"] == [image_at + 32 + len(suffix)] * g["num_layers"]
                assert s["recomputed_text_rows_per_layer"] == [image_at + len(suffix)] * g["num_layers"]
                assert s["recomputed_image_rows_per_layer"] == [32] * g["num_layers"]
                assert s["decode_cache_append_exact"] and s["decoder_prefill_pass_count"] == 1
        assert r["ssd_read_bytes"] == expected_bytes
        checked.append({"method": method, "turn": turn, "prompt_history_lineage": "PASS",
            "ttft_timestamp_identity": "PASS", "recorded_pread_ranges": "PASS",
            "suffix_tokens": len(suffix), "ttft_ms_historical": r["ttft_ms"], "ssd_bytes": r["ssd_read_bytes"],
            "preads": r["ssd_preads"], "image_tokens_including_structure": g["v_token_num"],
            "prediction": r["prediction"],
            "observed_QKV_shapes": "NOT VERIFIED: historical logs contain counters, not captured tensors"})
    with (a.evidence / "historical_sample_checks.json").open("x") as f:
        json.dump({"status": "PASS", "scope": "recorded arithmetic, provenance, inputs, byte ranges only",
                   "fresh_GPU_run": False, "requests": checked}, f, indent=2)
        f.write("\n")
    original = ROOT / json.loads((a.evidence / "historical_provenance.json").read_text())["raw_path"]
    for name in ("mpic_meta.json", "mpic_persistence.json", "mpic_hashes.json", "mpic32_ssd_activation.json",
                 "raster_meta.json", "fullload_activation.json", "store_integrity_after.json"):
        source = original.parent / name
        if source.is_file():
            dest = a.evidence / "historical_store_records" / name
            dest.parent.mkdir(exist_ok=True)
            with dest.open("xb") as f:
                f.write(source.read_bytes())
    print(json.dumps({"recorded_requests_checked": len(checked), "fresh_GPU_run": False,
                      "minimum_suffix_length_in_sample": min(x["suffix_tokens"] for x in checked)}))


if __name__ == "__main__":
    main()
