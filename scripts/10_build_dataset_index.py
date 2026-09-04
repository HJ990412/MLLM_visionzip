"""Evaluation indices for the large-scale / cross-dataset validation.

One entry per IMAGE, with all of that image's questions attached, because the
whole point of the store is that an image's visual KV is built once and reused
by every question about it.  Sampling is at the image level for the same
reason: sampling questions would scatter them across images and destroy the
reuse the system is built to exploit.

  gqa_large  GQA testdev_balanced -- every image, a fixed number of questions
             each, so image diversity is maximal rather than a few images
             carrying most of the questions
  vqav2      VQAv2 validation (streamed; the split is far too large to
             materialise) -- general VQA
  textvqa    TextVQA validation -- small text in the scene, where dropping the
             wrong patch should hurt most

Config (seed, counts, per-image question statistics) is written next to the
index so the subset is reproducible.

  python scripts/10_build_dataset_index.py --dataset gqa_large
"""
import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mmimpress.config import DATA_DIR, PROJECT_ROOT


def _save(img, path):
    if not path.exists():
        img.convert("RGB").save(path)


def stats(index):
    n = [len(e["questions"]) for e in index]
    return {"n_images": len(index), "n_questions": sum(n),
            "q_per_image_mean": sum(n) / max(len(n), 1),
            "q_per_image_min": min(n) if n else 0,
            "q_per_image_max": max(n) if n else 0}


def build_gqa(out_dir, seed, max_images, q_per_image):
    from datasets import load_dataset
    imgs = load_dataset("lmms-lab/GQA", "testdev_balanced_images",
                        split="testdev")
    qs = load_dataset("lmms-lab/GQA", "testdev_balanced_instructions",
                      split="testdev")
    by_img = collections.defaultdict(list)
    for row, iid in enumerate(qs["imageId"]):
        by_img[iid].append(row)
    # only images that can supply the full question quota, so every image
    # contributes the same number of questions and none is over-represented
    picked = sorted(i for i in by_img if len(by_img[i]) >= q_per_image)
    rng = random.Random(seed)
    rng.shuffle(picked)
    picked = picked[:max_images]
    want = set(picked)
    id_to_row = {i: r for r, i in enumerate(imgs["id"]) if i in want}
    qid, qtext, qans = qs["id"], qs["question"], qs["answer"]
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for iid in picked:
        rows = by_img[iid]
        rng.shuffle(rows)
        rows = rows[:q_per_image]
        p = img_dir / f"{iid}.png"
        _save(imgs[id_to_row[iid]]["image"], p)
        index.append({"image_id": iid,
                      "image_path": str(p.relative_to(PROJECT_ROOT)),
                      "questions": [{"question_id": str(qid[r]),
                                     "question": qtext[r],
                                     "answers": [str(qans[r])]} for r in rows]})
    return index


def build_textvqa(out_dir, seed, max_images, q_per_image):
    from datasets import load_dataset
    d = load_dataset("lmms-lab/textvqa", split="validation")
    by_img = collections.defaultdict(list)
    for row, iid in enumerate(d["image_id"]):
        by_img[iid].append(row)
    # prefer images that carry more than one question so the KV really is reused
    order = sorted(by_img, key=lambda i: (-len(by_img[i]), i))
    rng = random.Random(seed)
    top = [i for i in order if len(by_img[i]) >= q_per_image]
    rng.shuffle(top)
    picked = top[:max_images]
    qid, qtext, qans = d["question_id"], d["question"], d["answers"]
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for iid in picked:
        rows = by_img[iid][:q_per_image]
        p = img_dir / f"{iid}.png"
        _save(d[rows[0]]["image"], p)
        index.append({"image_id": iid,
                      "image_path": str(p.relative_to(PROJECT_ROOT)),
                      "questions": [{"question_id": str(qid[r]),
                                     "question": qtext[r],
                                     "answers": [str(a) for a in qans[r]]}
                                    for r in rows]})
    return index


def build_vqav2(out_dir, seed, max_images, q_per_image, scan_rows=60000):
    """VQAv2 validation, streamed.

    The split is ~214k questions; streaming a prefix and grouping it by image
    gives enough multi-question images without materialising the whole thing.
    """
    from datasets import load_dataset
    it = load_dataset("lmms-lab/VQAv2", split="validation", streaming=True)
    by_img = collections.defaultdict(list)
    keep = {}
    for n, r in enumerate(it):
        iid = str(r["image_id"])
        by_img[iid].append({"question_id": str(r["question_id"]),
                            "question": r["question"],
                            "answers": [str(a["answer"]) if isinstance(a, dict)
                                        else str(a) for a in r["answers"]]})
        if iid not in keep:
            keep[iid] = r["image"]
        if n + 1 >= scan_rows:
            break
    order = [i for i in by_img if len(by_img[i]) >= q_per_image]
    rng = random.Random(seed)
    rng.shuffle(order)
    picked = order[:max_images]
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for iid in picked:
        p = img_dir / f"{iid}.png"
        _save(keep[iid], p)
        index.append({"image_id": iid,
                      "image_path": str(p.relative_to(PROJECT_ROOT)),
                      "questions": by_img[iid][:q_per_image]})
    return index


BUILDERS = {"gqa_large": build_gqa, "textvqa": build_textvqa,
            "vqav2": build_vqav2}
DEFAULTS = {"gqa_large": (398, 5), "textvqa": (500, 2), "vqav2": (250, 5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(BUILDERS), required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--q-per-image", type=int, default=None)
    ap.add_argument("--scan-rows", type=int, default=6000,
                    help="VQAv2 only: streamed rows to scan before sampling")
    args = ap.parse_args()

    mi, qpi = DEFAULTS[args.dataset]
    mi = args.max_images or mi
    qpi = args.q_per_image or qpi
    out_dir = DATA_DIR / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    kw = {"scan_rows": args.scan_rows} if args.dataset == "vqav2" else {}
    index = BUILDERS[args.dataset](out_dir, args.seed, mi, qpi, **kw)
    with open(out_dir / "index.json", "w") as f:
        json.dump(index, f)
    cfg = {"dataset": args.dataset, "seed": args.seed, "max_images": mi,
           "q_per_image": qpi, **stats(index)}
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=1)
    print(json.dumps(cfg, indent=1))


if __name__ == "__main__":
    main()
