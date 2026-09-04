"""GQA workload: images are the reusable prefixes, questions are the queries.

GQA testdev_balanced is a natural fit for a prefix-KV study and needs none of
IMPRESS's synthetic reuse model: it has ~400 distinct images with many
questions each, so prefix reuse frequency is whatever the benchmark says it is.
lmms-lab ships images and instructions as separate configs, so the (small)
image table is materialised once and the instruction rows are joined against it.

Index format (data/index.json):
  [{image_id, image_path, questions: [{question_id, question, answer}]}]
"""
from __future__ import annotations

import collections
import json
import random
from pathlib import Path

from mmimpress.config import DATA_DIR

HF_REPO = "lmms-lab/GQA"


def build_index(num_images: int = 40, min_questions: int = 2, seed: int = 0,
                split: str = "testdev_balanced", out_dir: Path = None):
    from datasets import load_dataset
    out_dir = Path(out_dir or DATA_DIR)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    imgs = load_dataset(HF_REPO, f"{split}_images", split="testdev")
    qs = load_dataset(HF_REPO, f"{split}_instructions", split="testdev")

    by_image = collections.defaultdict(list)
    for row, iid in enumerate(qs["imageId"]):
        by_image[iid].append(row)

    eligible = sorted(i for i, r in by_image.items()
                      if len(r) >= min_questions)
    picked = random.Random(seed).sample(eligible,
                                        min(num_images, len(eligible)))
    want = set(picked)
    id_to_row = {iid: r for r, iid in enumerate(imgs["id"]) if iid in want}

    qid = qs["id"]
    qtext = qs["question"]
    qans = qs["answer"]
    index = []
    for iid in picked:
        p = img_dir / f"{iid}.png"
        if not p.exists():
            imgs[id_to_row[iid]]["image"].convert("RGB").save(p)
        index.append({
            "image_id": iid,
            "image_path": str(p.relative_to(out_dir.parent)),
            "questions": [{"question_id": str(qid[r]),
                           "question": qtext[r],
                           "answer": str(qans[r])}
                          for r in by_image[iid]],
        })
    out = out_dir / "index.json"
    with open(out, "w") as f:
        json.dump(index, f, indent=1)
    return index, out


def load_index(path: Path = None):
    with open(Path(path or DATA_DIR / "index.json")) as f:
        return json.load(f)


# ------------------------------------------------------------- accuracy
def gqa_accuracy(pred: str, answer: str) -> float:
    """GQA is single-word/short-phrase with one gold answer: exact match after
    lowercase + punctuation/article stripping."""
    import re
    art = {"a", "an", "the"}

    def norm(s):
        s = re.sub(r"[^\w\s]", " ", str(s).lower())
        return " ".join(w for w in s.split() if w not in art)

    p, g = norm(pred), norm(answer)
    return float(p == g or (len(g) > 0 and p.split()[:len(g.split())] == g.split()))


# ------------------------------------------------- multi-dataset scoring
_ART = {"a", "an", "the"}


def _norm(s):
    import re
    s = re.sub(r"[^\w\s]", " ", str(s).lower())
    return " ".join(w for w in s.split() if w not in _ART)


def vqa_score(pred, answers):
    """Official VQA accuracy: min(#annotators agreeing / 3, 1).

    VQAv2 and TextVQA both ship ten human answers per question and are scored
    this way; using exact match against one of them would understate every
    method equally but would not be the published metric.
    """
    p = _norm(pred)
    return min(sum(_norm(a) == p for a in answers) / 3.0, 1.0)


def exact_score(pred, answers):
    """GQA: one gold answer, exact match after the same normalisation."""
    g = _norm(answers[0] if isinstance(answers, (list, tuple)) else answers)
    p = _norm(pred)
    return float(p == g or (g and p.split()[:len(g.split())] == g.split()))


METRICS = {"gqa": exact_score, "vqa": vqa_score}


def question_answers(q):
    """Index entries carry `answers` (new) or `answer` (the 40/240 index)."""
    if "answers" in q:
        return q["answers"]
    return [q["answer"]]
