"""Canonical datasets and prompt policy for multi-turn MLLM evaluation.

This module deliberately contains no model- or SSD-serving implementation.
VisDial reuses the existing single-image store and :mod:`mmimpress.serve`;
MMDU first goes through a correctness gate because independently computed
image-prefix KV tensors are not composable in an interleaved causal sequence.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from mmimpress.config import PROJECT_ROOT

SCHEMA_VERSION = "multiturn-index-v1"
IMAGE_MARKER = "<ImageHere>"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _stable_key(seed: int, value) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _relative(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def resolve_image_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_canonical(path: Path):
    with open(path) as f:
        payload = json.load(f)
    if isinstance(payload, list):
        dialogs = payload
    else:
        assert payload.get("schema_version") == SCHEMA_VERSION, payload.keys()
        dialogs = payload["dialogs"]
    validate_canonical(dialogs)
    return dialogs


def _visdial_selection(dialogs, dense_by_image, limit, seed):
    """Deterministic selection, balanced over the one dense round per image."""
    limit = min(limit if limit is not None else len(dialogs), len(dialogs))
    by_round = defaultdict(list)
    for d in dialogs:
        ann = dense_by_image.get(str(d["image_id"]))
        by_round[int(ann["round_id"]) if ann else 0].append(d)

    selected = []
    # The main 100-dialog workload gets ten dense examples for every turn,
    # making the optional NDCG slice turn-balanced rather than accidental.
    if limit >= 10 and all(by_round.get(r) for r in range(1, 11)):
        quota = limit // 10
        for r in range(1, 11):
            group = sorted(by_round[r], key=lambda d: _stable_key(seed, d["image_id"]))
            selected.extend(group[:quota])
    chosen = {str(d["image_id"]) for d in selected}
    remaining = sorted((d for d in dialogs if str(d["image_id"]) not in chosen),
                       key=lambda d: _stable_key(seed, d["image_id"]))
    selected.extend(remaining[:limit - len(selected)])
    return sorted(selected, key=lambda d: _stable_key(seed, d["image_id"]))


def build_visdial_index(annotation_path: Path, image_dir: Path,
                        dense_path: Path | None = None, max_dialogs: int = 100,
                        seed: int = 1234):
    annotation_path, image_dir = Path(annotation_path), Path(image_dir)
    if not annotation_path.exists():
        raise FileNotFoundError(annotation_path)
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)
    with open(annotation_path) as f:
        src = json.load(f)
    assert src.get("version") == "1.0", src.get("version")
    assert str(src.get("split", "")).startswith("val"), src.get("split")
    data = src["data"]
    questions, answers = data["questions"], data["answers"]

    dense = []
    if dense_path is not None:
        dense_path = Path(dense_path)
        if not dense_path.exists():
            raise FileNotFoundError(dense_path)
        with open(dense_path) as f:
            dense = json.load(f)
    dense_by_image = {str(x["image_id"]): x for x in dense}
    picked = _visdial_selection(data["dialogs"], dense_by_image,
                                max_dialogs, seed)

    dialogs = []
    for src_d in picked:
        iid = str(src_d["image_id"])
        image_id = f"visdial:{iid}"
        image_path = image_dir / f"VisualDialog_val2018_{int(iid):012d}.jpg"
        assert image_path.exists(), image_path
        dense_ann = dense_by_image.get(iid)
        turns = []
        for ti, row in enumerate(src_d["dialog"], 1):
            opts = [str(answers[int(i)]) for i in row["answer_options"]]
            gt_index = int(row["gt_index"])
            assert 0 <= gt_index < len(opts) == 100
            gold = str(answers[int(row["answer"])])
            assert opts[gt_index] == gold
            t = {
                "turn_id": ti,
                "source_question_index": int(row["question"]),
                "source_answer_index": int(row["answer"]),
                "question": str(questions[int(row["question"])]),
                "gold_answer": gold,
                "new_image_ids": [image_id] if ti == 1 else [],
                "active_image_ids": [image_id],
                "candidate_answer_indices": [int(i) for i in row["answer_options"]],
                "candidate_answers": opts,
                "gt_index": gt_index,
            }
            if dense_ann and int(dense_ann["round_id"]) == ti:
                rel = [float(x) for x in dense_ann["gt_relevance"]]
                assert len(rel) == len(opts)
                t["dense_relevance"] = rel
            turns.append(t)
        dialogs.append({
            "dataset": "visdial_v1.0_val",
            "dialog_id": f"visdial:{iid}",
            "image_ids": [image_id],
            "images": [{"image_id": image_id, "image_path": _relative(image_path),
                        "source_index": 1}],
            "caption": str(src_d["caption"]),
            "turns": turns,
            # This is the only text allowed to reach importance reordering.
            # Evaluation turns are intentionally absent from store_index.json.
            "calibration_contexts": [{"kind": "caption", "text": str(src_d["caption"])}],
        })

    validate_canonical(dialogs)
    config = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "visdial_v1.0_val",
        "seed": seed,
        "selection": "sha256(seed:image_id), stratified by dense round when limit>=10",
        "max_dialogs": max_dialogs,
        "n_dialogs": len(dialogs),
        "n_turns": sum(len(d["turns"]) for d in dialogs),
        "n_images": len(dialogs),
        "history_policy": "gold_teacher_forced",
        "calibration_policy": "caption_only_pre_dialog",
        "future_turn_calibration_count": 0,
        "dense_round_counts": dict(sorted(Counter(
            t["turn_id"] for d in dialogs for t in d["turns"]
            if "dense_relevance" in t).items())),
        "source_annotation": str(annotation_path.resolve()),
        "source_annotation_sha256": sha256_file(annotation_path),
        "source_dense": str(dense_path.resolve()) if dense_path else None,
        "source_dense_sha256": sha256_file(dense_path) if dense_path else None,
    }
    return dialogs, config


def build_visdial_store_index(dialogs):
    """Legacy image-index adapter containing caption calibration only.

    Keeping ``turns`` out of this file makes future-turn leakage impossible
    even if the reorder script is accidentally given a calibration count > 1.
    """
    out = []
    for d in dialogs:
        cal = d["calibration_contexts"]
        assert len(cal) == 1 and cal[0]["kind"] == "caption"
        out.append({
            "image_id": d["image_ids"][0],
            "image_path": d["images"][0]["image_path"],
            "questions": [{
                "question_id": f"{d['dialog_id']}:caption-calibration",
                "question": f"Image caption: {cal[0]['text']}",
                "answers": [cal[0]["text"]],
                "calibration_only": True,
            }],
        })
    return out


def _mmdu_selection(items, limit, seed):
    ordered = sorted(items, key=lambda d: _stable_key(seed, d["id"]))
    return ordered[:min(limit if limit is not None else len(ordered), len(ordered))]


def build_mmdu_index(benchmark_path: Path, image_root: Path,
                     max_dialogs: int | None = None, seed: int = 1234,
                     dialog_ids=None):
    benchmark_path, image_root = Path(benchmark_path), Path(image_root)
    if not benchmark_path.exists():
        raise FileNotFoundError(benchmark_path)
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)
    by_basename = defaultdict(list)
    for candidate in image_root.rglob("*"):
        if candidate.is_file():
            by_basename[candidate.name].append(candidate)
    with open(benchmark_path) as f:
        source = json.load(f)
    items = list(source.values()) if isinstance(source, dict) else list(source)
    if dialog_ids is not None:
        wanted = [str(x) for x in dialog_ids]
        by_id = {str(x["id"]): x for x in items}
        missing = [x for x in wanted if x not in by_id]
        assert not missing, f"unknown MMDU dialog ids: {missing}"
        picked = [by_id[x] for x in wanted]
    else:
        picked = _mmdu_selection(items, max_dialogs, seed)
    dialogs = []
    for item in picked:
        did = str(item["id"])
        images = []
        for ii, raw in enumerate(item["image"], 1):
            raw_path = Path(str(raw))
            candidates = []
            if raw_path.is_absolute():
                candidates.append(raw_path)
            candidates.append(image_root / str(raw).lstrip("/"))
            direct = next((p for p in candidates if p.exists()), None)
            if direct is None:
                matches = by_basename.get(raw_path.name, [])
                if len(matches) != 1:
                    raise FileNotFoundError(
                        f"MMDU image {raw!r}: direct path absent and basename "
                        f"has {len(matches)} matches under {image_root}")
                p = matches[0]
            else:
                p = direct
            images.append({"image_id": f"mmdu:{did}:image{ii}",
                           "image_path": _relative(p), "source_index": ii,
                           "source_path": str(raw)})
        image_ids = [x["image_id"] for x in images]
        conv = item["conversations"]
        assert len(conv) % 2 == 0
        turns, cursor = [], 0
        for ci in range(0, len(conv), 2):
            user, assistant = conv[ci], conv[ci + 1]
            assert user["from"] == "user" and assistant["from"] == "assistant"
            q = str(user["value"])
            n_new = q.count(IMAGE_MARKER)
            assert cursor + n_new <= len(images), (did, ci, cursor, n_new)
            new_ids = image_ids[cursor:cursor + n_new]
            cursor += n_new
            turns.append({
                "turn_id": len(turns) + 1,
                "question": q,
                "gold_answer": str(assistant["value"]),
                "new_image_ids": new_ids,
                "active_image_ids": image_ids[:cursor],
                "candidate_answers": [],
                "source_user_index": ci,
                "image_marker_count": n_new,
            })
        assert cursor == len(images), \
            f"dialog {did}: introduced {cursor}/{len(images)} images"
        dialogs.append({
            "dataset": "mmdu",
            "dialog_id": f"mmdu:{did}",
            "image_ids": image_ids,
            "images": images,
            "caption": "",
            "turns": turns,
            "calibration_contexts": [],
            "source_set": item.get("set"),
        })
    validate_canonical(dialogs)
    config = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "mmdu",
        "seed": seed,
        "selection": ("explicit ordered dialog_ids" if dialog_ids is not None
                      else "sha256(seed:dialog_id)"),
        "explicit_dialog_ids": ([str(x) for x in dialog_ids]
                                if dialog_ids is not None else None),
        "max_dialogs": max_dialogs,
        "n_dialogs": len(dialogs),
        "n_turns": sum(len(d["turns"]) for d in dialogs),
        "n_images": sum(len(d["images"]) for d in dialogs),
        "images_per_dialog_min": min(len(d["images"]) for d in dialogs),
        "images_per_dialog_max": max(len(d["images"]) for d in dialogs),
        "history_policy": "gold_teacher_forced",
        "calibration_policy": "none_correctness_gate_only",
        "source_benchmark": str(benchmark_path.resolve()),
        "source_benchmark_sha256": sha256_file(benchmark_path),
    }
    return dialogs, config


def validate_canonical(dialogs):
    seen_dialogs = set()
    for d in dialogs:
        assert d["dialog_id"] not in seen_dialogs
        seen_dialogs.add(d["dialog_id"])
        image_ids = list(d["image_ids"])
        assert image_ids == [x["image_id"] for x in d["images"]]
        introduced = []
        for expected, t in enumerate(d["turns"], 1):
            assert t["turn_id"] == expected
            for iid in t["new_image_ids"]:
                assert iid in image_ids and iid not in introduced
                introduced.append(iid)
            assert t["active_image_ids"] == introduced
            assert len(t["candidate_answers"]) in (0, 100)
        assert introduced == image_ids
    return True


def visdial_history_text(dialog, turn_id: int, generated=None):
    """Caption + teacher-forced prior Q/A + the current question."""
    assert 1 <= turn_id <= len(dialog["turns"])
    lines = [f"Image caption: {dialog['caption']}", ""]
    for t in dialog["turns"][:turn_id - 1]:
        answer = (generated or {}).get(t["turn_id"], t["gold_answer"])
        lines.extend([f"Q{t['turn_id']}: {t['question']}",
                      f"A{t['turn_id']}: {answer}"])
    cur = dialog["turns"][turn_id - 1]
    lines.extend(["", f"Current question Q{turn_id}: {cur['question']}"])
    return "\n".join(lines)


def visdial_prior_history_text(dialog, turn_id: int, generated=None):
    """Caption and completed turns only (excludes the current question)."""
    assert 1 <= turn_id <= len(dialog["turns"])
    lines = [f"Image caption: {dialog['caption']}"]
    for t in dialog["turns"][:turn_id - 1]:
        answer = (generated or {}).get(t["turn_id"], t["gold_answer"])
        lines.extend([f"Q{t['turn_id']}: {t['question']}",
                      f"A{t['turn_id']}: {answer}"])
    return "\n".join(lines)


def visdial_prompt(dialog, turn_id: int, generated=None):
    return ("USER: <image>\n" + visdial_history_text(dialog, turn_id, generated)
            + "\nAnswer the current question concisely. ASSISTANT:")


def mmdu_question_text(turn):
    return turn["question"].replace(IMAGE_MARKER, "<image>\n").lstrip()


def mmdu_prompt(dialog, turn_id: int, generated=None):
    """Append-only Vicuna-style prompt with gold history by default."""
    assert 1 <= turn_id <= len(dialog["turns"])
    parts = []
    for t in dialog["turns"][:turn_id]:
        parts.append(f"USER: {mmdu_question_text(t)} ASSISTANT:")
        if t["turn_id"] < turn_id:
            ans = (generated or {}).get(t["turn_id"], t["gold_answer"])
            parts[-1] += f" {ans}</s>"
    return " ".join(parts)


_ARTICLES = {"a", "an", "the"}


def normalize_answer(text: str) -> str:
    text = re.sub(r"[^\w\s]", " ", str(text).lower())
    return " ".join(x for x in text.split() if x not in _ARTICLES)


def generative_match(prediction: str, gold: str) -> float:
    """Auxiliary normalized exact/prefix match; never an official score."""
    p, g = normalize_answer(prediction), normalize_answer(gold)
    return float(p == g or (g and p.split()[:len(g.split())] == g.split()))


def retrieval_metrics(ranks):
    """VisDial sparse retrieval metrics from 1-based ground-truth ranks."""
    ranks = [float(r) for r in ranks]
    if not ranks:
        return {k: None for k in ("mrr", "r@1", "r@5", "r@10", "mean_rank")}
    n = len(ranks)
    return {"mrr": sum(1.0 / r for r in ranks) / n,
            "r@1": sum(r <= 1 for r in ranks) / n,
            "r@5": sum(r <= 5 for r in ranks) / n,
            "r@10": sum(r <= 10 for r in ranks) / n,
            "mean_rank": sum(ranks) / n}


def ndcg(scores, relevance):
    """NDCG for one densely annotated VisDial round."""
    assert len(scores) == len(relevance)
    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
    ideal = sorted(range(len(relevance)), key=lambda i: (-float(relevance[i]), i))

    k = sum(float(x) > 0 for x in relevance)

    def dcg(idx):
        import math
        return sum(float(relevance[i]) / math.log2(rank + 2)
                   for rank, i in enumerate(idx[:k]))

    denom = dcg(ideal)
    return dcg(order) / denom if denom else 0.0
