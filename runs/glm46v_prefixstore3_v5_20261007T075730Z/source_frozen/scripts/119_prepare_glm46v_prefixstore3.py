#!/usr/bin/env python3
"""Freeze CPU processor geometry, validation membership, sources and resources."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mmimpress.glm46v.runner import MODEL_ID, REVISION, IMAGE_SIZE, METHODS, MAX_NEW_TOKENS


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"immutable freeze exists: {path}")
    with path.open("x") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def cmd(args):
    p = subprocess.run(args, text=True, capture_output=True)
    return dict(command=args, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)


def prepare(run):
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor, AutoConfig
    checkpoint = run / "checkpoint"
    model_config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True,
                                            size=IMAGE_SIZE, use_fast=True)
    identity = json.loads((run / "manifests/main_image_identity.json").read_text())
    main = json.loads((run / "manifests/main_manifest.json").read_text())
    dialogues = main["dialogues"]
    from mmimpress.mt_gqa import workload_sha256
    observed = workload_sha256(dialogues)
    assert observed == "0287e0c57813800c781633b969c5cff336b3a3c1a1bdcdbb56d63f6ddab0ca62"
    geometry = {}
    for image_id, item in identity.items():
        image_path = (ROOT / item["image_path"]).resolve()
        assert sha(image_path) == item["sha256"]
        with Image.open(image_path) as image:
            original_hw = [image.height, image.width]
            inputs = processor.image_processor(images=image.convert("RGB"), return_tensors="pt")
        grid = inputs["image_grid_thw"].tolist()
        n = int(inputs["image_grid_thw"].prod()) // model_config.vision_config.spatial_merge_size**2
        assert 0 < n <= 1024
        geometry[image_id] = dict(image_path=str(image_path), image_sha256=item["sha256"],
                                 original_hw=original_hw, image_grid_thw=grid, N_content=n,
                                 k=(n+3)//4, k_mod_64=((n+3)//4)%64,
                                 native_pixels_tensor_bytes=inputs["pixel_values"].numel()*inputs["pixel_values"].element_size())
    dump(run / "geometry_manifest.json", geometry)
    first = {}
    for ordinal, dialogue in enumerate(dialogues):
        first.setdefault(dialogue["image_id"], (ordinal, dialogue))
    sample_ids = []
    criteria = {}
    def choose(label, key):
        chosen = min(geometry, key=lambda iid: (key(iid), iid))
        criteria[label] = chosen
        if chosen not in sample_ids:
            sample_ids.append(chosen)
    choose("minimum_N", lambda iid: geometry[iid]["N_content"])
    choose("maximum_N", lambda iid: -geometry[iid]["N_content"])
    choose("k_mod64_lower", lambda iid: geometry[iid]["k_mod_64"])
    choose("k_mod64_upper", lambda iid: 64-geometry[iid]["k_mod_64"])
    longest = max(dialogues, key=lambda d: max(len(processor.tokenizer.encode(t["question"])) for t in d["turns"]))
    criteria["longest_question_sequence"] = longest["image_id"]
    if longest["image_id"] not in sample_ids:
        sample_ids.append(longest["image_id"])
    samples = []
    for iid in sample_ids:
        ordinal, dialogue = first[iid]
        if iid == longest["image_id"]:
            dialogue = longest
        t1, t2 = dialogue["turns"][:2]
        samples.append(dict(**geometry[iid], image_id=iid, dialog_id=dialogue["dialog_id"],
                            question=t1["question"], other_question=t2["question"],
                            long_question=max(dialogue["turns"],key=lambda t:len(processor.tokenizer.encode(t["question"])))['question'],
                            history=[], canonical_ordinal=ordinal))
    counts = {}
    for d in dialogues:
        counts[d["image_id"]] = counts.get(d["image_id"], 0) + 1
    shared = min(iid for iid in counts if counts[iid] > 1)
    shared_dialogues = [d for d in dialogues if d["image_id"] == shared][:2]
    history_fixture = next(d for d in dialogues if d['dialog_id']=='mtgqa_000040')
    fixture_geometry = geometry[history_fixture['image_id']]
    dump(run / "validation_manifest.json", dict(selected_before_model_outputs=True, seed=1234,
            selection_rule="min/max processor N, each side k%64, longest single native question; ID tie break",
            criteria=criteria, samples=samples, same_image_dialogues=shared_dialogues,
            same_image_identity=geometry[shared], source_workload_sha256=observed,
            synthetic_gpu_stress=dict(seed=1234,height=896,width=896,expected_N=1024,
               question=max(longest['turns'],key=lambda t:len(processor.tokenizer.encode(t['question'])))['question']),
            native_history_regression_fixture=dict(image_id=history_fixture['image_id'],
               image_path=fixture_geometry['image_path'],image_sha256=fixture_geometry['image_sha256'],
               dialog_id=history_fixture['dialog_id'],t1_question=history_fixture['turns'][0]['question'],
               followup_question=history_fixture['turns'][1]['question'],
               selection_reason='Known v4 native assistant-body/history protocol failure; no gold input; excluded from quality denominator'),
            synthetic_chunk_boundaries_N=[1,4,5,249,252,253,256,257,508,509,512,513],
            tolerance=dict(native_bits="exact", same_shape_logits="exact", generated_ids="exact",
                           attention_fp64_atol=1e-12,attention_fp64_rtol=1e-12,
                           fp32_diagnostic_atol=1e-5,fp32_diagnostic_rtol=1e-5)))
    size = shutil.disk_usage(run)
    resource = dict(disk_total_bytes=size.total, disk_free_bytes=size.free,
            minimum_reserve_bytes=30<<30, maximum_new_checkpoint_bytes=22<<30,
            maximum_main_store_bytes=40<<30, maximum_raw_bytes=10<<30,
            maximum_validation_bytes=24<<30, maximum_failed_attempt_bytes=10<<30,
            largest_content_store_bytes=1024*40960,
            mount=cmd(["findmnt", "-J", "-T", str(run)]),
            nvidia=cmd(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,utilization.gpu", "--format=csv"]),
            other_gpu=cmd(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv"]),
            note="sandbox masks CUDA; actual GPU execution requires authorized unsandboxed command")
    dump(run / "resource_preflight.json", resource)
    if size.free < (30+40+10+24+10)<<30:
        raise RuntimeError("insufficient reserve for conservative checkpoint-excluded working set")
    sources = list((ROOT / "mmimpress/glm46v").glob("*.py"))
    sources += list((ROOT / "scripts").glob("*glm46v*.py"))
    sources += list((ROOT / "tests").glob("test_glm46v*.py"))
    sources += list((ROOT / "docs").glob("glm46v*.md"))
    source_hashes = {}
    for path in sorted(sources):
        rel = str(path.relative_to(ROOT))
        source_hashes[rel] = sha(path)
        dest = run / "source_frozen" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    contract = ROOT / "docs/glm46v_prefixstore3_contract.md"
    shutil.copy2(contract, run / "CONTRACT.md")
    metadata_files = {p.name: dict(sha256=sha(p), bytes=p.stat().st_size)
                      for p in checkpoint.iterdir() if p.is_file() and p.suffix != ".safetensors"}
    weight_files = {}
    for p in sorted(checkpoint.glob('*.safetensors')):
        digest = sha(p)
        download_meta = checkpoint/'.cache/huggingface/download'/(p.name+'.metadata')
        origin = download_meta.read_text().splitlines() if download_meta.exists() else []
        if len(origin)>1 and len(origin[1])==64 and origin[1]!=digest:
            raise ValueError('official weight download SHA mismatch:'+p.name)
        weight_files[p.name]=dict(sha256=digest,bytes=p.stat().st_size,download_receipt=origin)
    if len(weight_files)!=4:raise ValueError('full pinned checkpoint shards are required')
    tf_source = Path(transformers.__file__).parent
    tf_files = [*tf_source.glob("models/glm4v/*.py"), *tf_source.glob("models/glm46v/*.py"),
                tf_source / "cache_utils.py", tf_source / "masking_utils.py",
                tf_source / "integrations/sdpa_attention.py"]
    environment = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
            packages={p:importlib.metadata.version(p) for p in
                      ["torch","torchvision","transformers","bitsandbytes","accelerate","huggingface_hub","tokenizers","Pillow","numpy","pytest"]},
            cuda_build=torch.version.cuda, cuda_available=torch.cuda.is_available(),
            transformers_sources={str(p.relative_to(tf_source)):sha(p) for p in tf_files},
            transformers_origin="PyPI wheel transformers==5.13.1; exact installed source hashes",
            inherited_environment="read-only mllm_ft Torch and dependencies; GLM venv overrides Transformers only",
            pip_freeze=cmd([sys.executable,"-m","pip","freeze"]))
    dump(run / "environment.json", environment)
    dump(run / "config.json", dict(model_id=MODEL_ID,checkpoint_revision=REVISION,processor_revision=REVISION,
            processor_settings=dict(size=IMAGE_SIZE,use_fast=True,enable_thinking=False),
            checkpoint_metadata=metadata_files,checkpoint_weights=weight_files,
            methods=METHODS,seed=1234,greedy=True,max_new_tokens=MAX_NEW_TOKENS,
            answer_extraction='native-final-answer-box-v1; raw generated IDs/text preserved; strict scorer unchanged',
            history_policy='actual-generated-body/terminal-EOS-only-v1; native chat template serializes prior content',
            quantization="decoder-only NF4 double quantization/BF16 compute; BF16 unquantized vision/merger/lm_head",
            backend="sdpa",native_KV=dict(layers=40,heads=2,head_dim=128,dtype="BF16",bytes_per_token=40960),
            source_sha256=source_hashes,contract_sha256=sha(contract),
            workload_sha256=observed,validation_manifest_sha256=sha(run/"validation_manifest.json"),
            created_at=datetime.now(timezone.utc).isoformat(),gpu_validation="REQUIRED_BEFORE_PILOT"))
    dump(run / "freeze.json",dict(source_sha256=source_hashes,config_sha256=sha(run/"config.json"),
             environment_sha256=sha(run/"environment.json"),contract_sha256=sha(run/"CONTRACT.md"),
             validation_manifest_sha256=sha(run/"validation_manifest.json"),
             manifests={str(p.relative_to(run)):sha(p) for p in (run/"manifests").glob("*.json")}))
    print(json.dumps(dict(status="CPU_PREFLIGHT_PASS",images=len(geometry),
         N_min=min(x['N_content'] for x in geometry.values()),N_max=max(x['N_content'] for x in geometry.values()),
         validation_samples=len(samples),run=str(run))))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-dir",required=True,type=Path)
    args=parser.parse_args()
    prepare(args.run_dir.resolve())
