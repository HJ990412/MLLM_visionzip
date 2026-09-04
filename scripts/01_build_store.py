"""Build the disk-resident prefix-KV store: one image == one prefix.

For every image in data/index.json this runs a single prefill over
[system tokens | expanded image block] -- deliberately WITHOUT any question, so
the cached KV is a prefix of every request about that image -- and writes it
head-major (see mmimpress/store.py).  The pre-layer-0 hidden states of the
visual block are saved too: they are question-independent, and having them on
disk lets the serving path pick SparseVLM raters without ever running the
vision tower.

  python scripts/01_build_store.py [--limit N]
"""
import argparse
import gc
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mmimpress.config import CHUNK_SIZE, PROJECT_ROOT, STORE_DIR
from mmimpress.dataset import load_index
from mmimpress.model import LlavaRunner, cache_layers
from mmimpress.serve import suffix_ids_for
from mmimpress.store import write_image_store


@torch.no_grad()
def build_one(runner, entry, out_root: Path):
    img = Image.open(PROJECT_ROOT / entry["image_path"]).convert("RGB")
    q0 = entry["questions"][0]["question"]
    enc = runner.encode(img, q0)
    ids = enc["input_ids"][0]
    v0, vn = runner.visual_span(ids)
    prefix_len = v0 + vn
    base, hi_h, hi_w, nl = runner.anyres_layout(
        enc["image_sizes"][0].tolist(), vn)

    # the question-side ids the server will reconstruct must line up with what
    # the processor produced here, or the stored prefix is not a real prefix
    suf = suffix_ids_for(runner, q0)
    assert torch.equal(ids[prefix_len:], suf.to(ids.device)), \
        "suffix reconstruction differs from the processor output"

    box = {}

    def grab(module, args, kwargs):
        if "h" not in box:
            box["h"] = (args[0] if args else kwargs["hidden_states"]).detach()

    hook = runner.layers[0].register_forward_pre_hook(grab, with_kwargs=True)
    try:
        out = runner.prefix_forward(enc, prefix_len)
    finally:
        hook.remove()

    layers = cache_layers(out.past_key_values)
    assert layers[0][0].shape[2] == prefix_len, layers[0][0].shape

    out_dir = out_root / str(entry["image_id"])
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(box["h"][0, v0:prefix_len].to(torch.float16).cpu(),
               out_dir / "v_hidden.pt")
    meta = write_image_store(
        out_dir, layers, v0, vn, ids[:prefix_len].tolist(), nl,
        chunk_size=CHUNK_SIZE,
        extra={"image_id": entry["image_id"], "model": runner.model_id,
               "base_grid": base, "hires_grid": [hi_h, hi_w],
               "n_questions": len(entry["questions"])})
    del out, layers, box
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=str(STORE_DIR))
    ap.add_argument("--index", default=None)
    ap.add_argument("--only", default=None,
                    help="comma-separated image_ids to build (shard)")
    args = ap.parse_args()

    index = load_index(args.index)
    if args.only:
        keep = set(args.only.split(","))
        index = [e for e in index if str(e["image_id"]) in keep]
    if args.limit:
        index = index[:args.limit]
    out_root = Path(args.out)
    runner = LlavaRunner().load()

    total = 0
    for i, e in enumerate(index):
        d = out_root / str(e["image_id"])
        if (d / "meta.json").exists():
            print(f"[{i+1}/{len(index)}] {e['image_id']}: exists, skip")
            continue
        t0 = time.time()
        m = build_one(runner, e, out_root)
        total += m["bytes_visual_kv"]
        print(f"[{i+1}/{len(index)}] {e['image_id']}: v_num={m['v_token_num']} "
              f"chunks={m['n_chunks_per_layer']} "
              f"{m['bytes_visual_kv']/1e9:.2f}GB+{m['bytes_probe_sidecar']/1e9:.2f} ({time.time()-t0:.1f}s)")
        if (i + 1) % 4 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    print(f"store: {total/1e9:.1f} GB visual KV in {out_root}")


if __name__ == "__main__":
    main()
