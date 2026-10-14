"""Verify audit evidence and that protected source/selected prior raw stayed unchanged."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
AUDIT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main():
    e = AUDIT / "evidence"
    before = json.loads((e / "source_sha256_before.json").read_text())
    after = {name: sha(ROOT / name) for name in before}
    assert before == after
    prior = json.loads((e / "historical_provenance.json").read_text())
    assert sha(ROOT / prior["raw_path"]) == prior["raw_sha256"]
    write(e / "source_sha256_after.json", after)
    write(e / "source_preservation.json", {
        "status": "PASS", "scope": "249 source/doc/test/reference files and selected historical raw",
        "files_checked": len(before), "changes": [], "historical_raw_unchanged": True,
        "production_source_edited": False, "existing_experiments_rerun": False})
    probe = e / "cpu_probe_v4"
    s = json.loads((probe / "summary.json").read_text())
    assert len(s["mechanism_cases"]) == 15 and s["generated_history_requests"] == 9
    assert s["legacy_fullload_single_suffix"] == "FAIL"
    rows = [json.loads(x) for x in (probe / "layer_traces.jsonl").read_text().splitlines()]
    assert len(rows) == 15 and sum(len(r["layers"]) for r in rows) == 45
    assert all(all(d["original_position_causal_mask_correct"] and d["future_probability_max"] == 0
                   for d in r["layers"]) for r in rows)
    core_errors = max(r["checks"]["logits_vs_independent_selective"]["max_abs"] for r in rows)
    report = (AUDIT / "AUDIT_REPORT.md").read_text()
    reproduce = (AUDIT / "REPRODUCE.md").read_text()
    import re
    for doc in (report, reproduce):
        for target in re.findall(r"\]\(([^)]+)\)", doc):
            if not target.startswith(("https://", "http://")):
                assert (AUDIT / target).is_file(), target
    write(e / "audit_verification.json", {
        "source_preservation": "PASS", "mechanism_cases": 15, "observed_layers": 45,
        "maximum_selective_oracle_logit_error": core_errors,
        "existing_unit_tests": {"MPIC": 17, "main_history_safety": 8},
        "historical_requests_independently_checked": 9,
        "overall_baseline": "MPIC-32 (adapted), restricted single-image contract",
        "overall_paper_system_fidelity": "FAIL (multi-image and mixed hit/miss unsupported)",
        "current_pretrained_GPU": "NOT VERIFIED", "report_links": "PASS"})
    manifest = {str(p.relative_to(AUDIT)): sha(p) for p in sorted(AUDIT.rglob("*"))
                if p.is_file() and p.name != "artifact_sha256.json"}
    write(AUDIT / "artifact_sha256.json", manifest)
    print(json.dumps({"source_files_preserved": len(before), "cpu_cases": 15, "layer_traces": 45,
                      "selective_oracle_max_logit_error": core_errors, "report_links": "PASS"}))


if __name__ == "__main__":
    main()
