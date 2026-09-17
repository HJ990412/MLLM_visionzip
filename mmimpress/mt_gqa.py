"""Deterministic, auditable reconstruction of the three-turn MT-GQA workload.

The source paper reports 4,061 three-turn dialogues derived from GQA
``testdev-balanced``, but the official MetaCompress repository did not publish
the dialogue artifact at the provenance revision recorded below.  This module
therefore builds a deliberately named ``MT-GQA-reconstructed`` workload.  It
contains data preparation and validation only; importing it never loads a
model or touches a GPU.

The reconstruction is intentionally fail-closed for the canonical workload:

* the exact local GQA source JSON is identified by SHA256;
* every image contributes ``floor(original_question_count / 3)`` dialogues;
* image-local source order and contiguous triple membership are preserved;
* exact normalized repeats are permitted across different dialogues, but are
  never permitted inside one dialogue; and
* every ordering decision uses a domain-separated SHA256 rank rather than
  Python's process-randomized ``hash`` or implementation-specific RNG state.
"""
from __future__ import annotations

import hashlib
import json
import os
import statistics
import unicodedata
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from mmimpress.config import PROJECT_ROOT


SCHEMA_VERSION = "mt-gqa-reconstructed-v1"
BENCHMARK_TYPE = "MT-GQA-reconstructed"
DEFAULT_SEED = 1234
RANKING_DOMAIN = "mt-gqa-reconstructed-sha256-rank-v1"
QUESTION_NORMALIZATION = "Unicode NFKC + whitespace collapse + casefold"
IMAGE_MARKER = "<image>"
SHORT_ANSWER_INSTRUCTION = (
    "Answer the current question with a single word or short phrase."
)

EXPECTED_SOURCE_SHA256 = (
    "14039069c0b3c797c7aa9bcd5f4c2aa4b5976e02c0b6773e1a584d942a03a318"
)
EXPECTED_SOURCE_QUESTIONS = 12_578
EXPECTED_IMAGES = 398
EXPECTED_DIALOGUES = 4_061
EXPECTED_SELECTED_QUESTIONS = EXPECTED_DIALOGUES * 3
EXPECTED_SOURCE_DUPLICATE_EXTRAS = 8
EXPECTED_SELECTED_DUPLICATE_EXTRAS = 8
EXPECTED_DUPLICATE_IMAGES = (
    "n145498", "n283587", "n356822", "n379991",
    "n382416", "n460385", "n501609", "n52544",
)

METACOMPRESS_REPOSITORY = "https://github.com/MArSha1147/MetaCompress"
METACOMPRESS_COMMIT = "70a620eaec3d03368bfff9b85c2e278871c595ea"
METACOMPRESS_README_SHA256 = (
    "36bdc94be163aadea0f489e55495de29e8f2c4b6f54285d87dfa2412f096f1a1"
)
DISCLAIMER = (
    "The source paper specifies GQA testdev-balanced, 4,061 three-turn "
    "dialogues, but the exact dialogue artifact was unavailable at evaluation "
    "time. We therefore use a deterministic reconstruction and do not claim "
    "exact benchmark identity."
)

ARTIFACT_FILENAMES = (
    "dialogues.json",
    "config.json",
    "dataset_stats.json",
    "dataset_provenance.json",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    """Return the byte representation used for every emitted JSON artifact."""
    text = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    )
    return (text + "\n").encode("utf-8")


def workload_sha256(dialogs: Sequence[Mapping[str, Any]]) -> str:
    """Hash the canonical dialogue/turn/question request sequence."""
    payload = "".join(
        f"{dialog['dialog_id']}\t{turn['turn_id']}\t{turn['question_id']}\n"
        for dialog in dialogs
        for turn in dialog["turns"]
    ).encode("utf-8")
    return sha256_bytes(payload)


def normalize_question(text: str) -> str:
    """Normalize only for exact-question duplicate detection.

    Punctuation is deliberately preserved: this is not an answer scorer and
    must not merge semantically different GQA questions merely because their
    punctuation differs.
    """
    return " ".join(unicodedata.normalize("NFKC", str(text)).split()).casefold()


def _turn_for_id(
    dialog: Mapping[str, Any], turn_id: int,
) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    """Return only the causally available prefix and requested turn."""
    turns = dialog.get("turns")
    if not isinstance(turns, list) or len(turns) != 3:
        raise ValueError("MT-GQA dialogue must contain exactly three turns")
    if not isinstance(turn_id, int) or isinstance(turn_id, bool):
        raise ValueError("turn_id must be an integer in [1, 3]")
    if turn_id < 1 or turn_id > len(turns):
        raise ValueError(f"turn_id out of range: {turn_id}")
    for expected, turn in enumerate(turns, 1):
        if not isinstance(turn, Mapping) or turn.get("turn_id") != expected:
            raise ValueError("MT-GQA turns must be ordered and 1-indexed")
    return turns[:turn_id - 1], turns[turn_id - 1]


def mt_gqa_prior_history_text(
    dialog: Mapping[str, Any], turn_id: int,
) -> str:
    """Render teacher-forced history strictly before ``turn_id``.

    This helper deliberately slices before rendering, making future questions
    and answers structurally inaccessible to the prompt construction path.
    """
    prior, _ = _turn_for_id(dialog, turn_id)
    lines: list[str] = []
    for turn in prior:
        question = str(turn.get("question", "")).strip()
        answers = turn.get("answers")
        if not question or not isinstance(answers, list) or len(answers) != 1:
            raise ValueError("history turn requires one question and one answer")
        answer = str(answers[0]).strip()
        if not answer:
            raise ValueError("history answer must be nonempty")
        tid = int(turn["turn_id"])
        lines.extend((f"Q{tid}: {question}", f"A{tid}: {answer}"))
    return "\n".join(lines)


def mt_gqa_prompt(dialog: Mapping[str, Any], turn_id: int) -> str:
    """Render the canonical causal MT-GQA prompt for one evaluated turn."""
    prior, current = _turn_for_id(dialog, turn_id)
    question = str(current.get("question", "")).strip()
    if not question:
        raise ValueError("current question must be nonempty")
    body: list[str] = []
    if prior:
        # Reuse the public renderer so the runner and tests share one history
        # policy rather than independently formatting gold context.
        body.extend((mt_gqa_prior_history_text(dialog, turn_id), ""))
    body.append(f"Current question Q{turn_id}: {question}")
    body.append(f"{SHORT_ANSWER_INSTRUCTION} ASSISTANT:")
    # Match the established LLaVA/Vicuna raw-prompt convention.  The image
    # placeholder is the first multimodal marker and appears in the opening
    # USER message; the generation cue is always the terminal token span.
    return f"USER: {IMAGE_MARKER}\n" + "\n".join(body)


def stable_rank(seed: int, namespace: str, *parts: Any) -> str:
    """Portable, unambiguous, domain-separated SHA256 ranking key."""
    fields = (RANKING_DOMAIN, str(int(seed)), str(namespace),
              *(str(part) for part in parts))
    payload = bytearray()
    for field in fields:
        encoded = field.encode("utf-8")
        payload.extend(len(encoded).to_bytes(8, byteorder="big", signed=False))
        payload.extend(encoded)
    return sha256_bytes(bytes(payload))


def _rank_tuple(seed: int, namespace: str, *parts: Any) -> tuple[Any, ...]:
    # The original fields are a deterministic tie breaker even under a
    # theoretical SHA256 collision.
    rendered = tuple(str(part) for part in parts)
    return (stable_rank(seed, namespace, *rendered), *rendered)


def _portable_image_path(path: Path) -> str:
    """Represent image paths relative to the project when possible."""
    resolved = path.resolve()
    try:
        return os.path.relpath(resolved, PROJECT_ROOT.resolve())
    except ValueError:
        # Different Windows drives are irrelevant on the experiment host, but
        # retaining a correct absolute path makes this helper portable.
        return str(resolved)


def resolve_image_path(path: str | Path | Mapping[str, Any]) -> Path:
    """Resolve either an emitted path string or a canonical dialogue."""
    if isinstance(path, Mapping):
        raw_path = path.get("image_path")
        if raw_path is None:
            images = path.get("images")
            if (isinstance(images, list) and len(images) == 1
                    and isinstance(images[0], Mapping)):
                raw_path = images[0].get("image_path")
        if raw_path is None:
            raise ValueError("dialogue has no image_path")
        value = Path(str(raw_path))
    else:
        value = Path(path)
    return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def load_gqa_questions(path: Path | str) -> dict[str, dict[str, Any]]:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"source questions must be a regular file: {source}")
    with source.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not payload:
        raise ValueError("GQA source must be a nonempty question-id mapping")

    rows: dict[str, dict[str, Any]] = {}
    for source_position, (raw_qid, raw_row) in enumerate(payload.items()):
        qid = str(raw_qid)
        if not qid or qid in rows:
            raise ValueError(f"invalid or duplicate question ID: {qid!r}")
        if not isinstance(raw_row, dict):
            raise ValueError(f"question {qid} is not an object")
        missing = {"imageId", "question", "answer"} - set(raw_row)
        if missing:
            raise ValueError(f"question {qid} is missing fields: {sorted(missing)}")
        image_id = str(raw_row["imageId"])
        question = str(raw_row["question"])
        answer = str(raw_row["answer"])
        if (not image_id or Path(image_id).name != image_id
                or "/" in image_id or "\\" in image_id):
            raise ValueError(f"unsafe image ID for question {qid}: {image_id!r}")
        if not normalize_question(question):
            raise ValueError(f"question {qid} normalizes to an empty string")
        if not answer.strip():
            raise ValueError(f"question {qid} has an empty answer")
        rows[qid] = {
            "question_id": qid,
            "image_id": image_id,
            "question": question,
            "answer": answer,
            "source_position": source_position,
        }
    return rows


def _select_image_questions(
    image_id: str,
    questions: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Take source-order contiguous triples up to the exact floor quota."""
    original_count = len(questions)
    dialogue_quota = original_count // 3
    selected_count = dialogue_quota * 3
    if dialogue_quota < 1:
        raise ValueError(f"image {image_id} has fewer than three questions")

    by_normalized: dict[str, list[dict[str, str]]] = defaultdict(list)
    for raw in questions:
        row = dict(raw)
        row["normalized_question"] = normalize_question(row["question"])
        by_normalized[row["normalized_question"]].append(row)

    unique_count = len(by_normalized)
    selected = [dict(row) for row in questions[:selected_count]]
    for row in selected:
        row["normalized_question"] = normalize_question(row["question"])
    selected_unique = len({row["normalized_question"] for row in selected})
    selected_duplicate_extras = len(selected) - selected_unique

    selection_stats = {
        "original_questions": original_count,
        "dialogue_quota": dialogue_quota,
        "selected_questions": selected_count,
        "discarded_questions": original_count - selected_count,
        "unique_normalized_questions": unique_count,
        "source_duplicate_extras": original_count - unique_count,
        "selected_duplicate_extras": selected_duplicate_extras,
        "duplicate_extras_discarded_as_remainder": (
            original_count - unique_count - selected_duplicate_extras),
    }
    return selected, selection_stats


def _pack_dialogues(
    image_id: str,
    selected: Sequence[Mapping[str, Any]],
    dialogue_quota: int,
) -> list[list[dict[str, Any]]]:
    """Preserve source-order contiguous triples and reject ambiguous triples."""
    if len(selected) != dialogue_quota * 3:
        raise ValueError(f"image {image_id}: selected-question quota mismatch")
    bins = [
        [dict(row) for row in selected[offset:offset + 3]]
        for offset in range(0, len(selected), 3)
    ]
    for bucket in bins:
        if len({row["normalized_question"] for row in bucket}) != 3:
            raise ValueError(
                f"image {image_id}: source-order triple contains an exact "
                "normalized question duplicate")
    return bins


def reconstruct_dialogues(
    questions: Mapping[str, Mapping[str, Any]],
    image_dir: Path | str,
    *,
    seed: int = DEFAULT_SEED,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reconstruct deterministic same-image triples from validated GQA rows."""
    image_root = Path(image_dir).resolve()
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)

    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_qids: set[str] = set()
    image_source_positions = Counter()
    for source_position, (raw_qid, raw) in enumerate(questions.items()):
        qid = str(raw.get("question_id", raw_qid))
        if qid in seen_qids:
            raise ValueError(f"duplicate question ID: {qid}")
        seen_qids.add(qid)
        image_id = str(raw.get("image_id", raw.get("imageId", "")))
        question = str(raw.get("question", ""))
        if (not image_id or not normalize_question(question)
                or "answer" not in raw or not str(raw["answer"]).strip()):
            raise ValueError(f"invalid normalized source row: {qid}")
        image_source_position = image_source_positions[image_id]
        image_source_positions[image_id] += 1
        by_image[image_id].append({
            "question_id": qid,
            "image_id": image_id,
            "question": question,
            "answer": str(raw["answer"]),
            "source_position": int(raw.get("source_position", source_position)),
            "image_source_position": image_source_position,
        })
    if not by_image:
        raise ValueError("no images in source questions")

    image_paths: dict[str, Path] = {}
    missing_images = []
    for image_id in by_image:
        path = image_root / f"{image_id}.jpg"
        if not path.is_file():
            missing_images.append(str(path))
        else:
            image_paths[image_id] = path
    if missing_images:
        preview = ", ".join(missing_images[:5])
        raise FileNotFoundError(
            f"missing {len(missing_images)} GQA image(s): {preview}")

    image_order = sorted(
        by_image,
        key=lambda image_id: _rank_tuple(seed, "image-order", image_id),
    )
    dialogs: list[dict[str, Any]] = []
    per_image_stats: dict[str, dict[str, Any]] = {}
    for image_id in image_order:
        selected, selection_stats = _select_image_questions(
            image_id, by_image[image_id])
        buckets = _pack_dialogues(
            image_id, selected, selection_stats["dialogue_quota"])
        per_image_stats[image_id] = selection_stats
        image_path = _portable_image_path(image_paths[image_id])
        for image_dialogue_index, bucket in enumerate(buckets, 1):
            turns = [
                {
                    "turn_id": turn_id,
                    "question_id": row["question_id"],
                    "question": row["question"],
                    "answers": [row["answer"]],
                    "source_position": row["source_position"],
                    "image_source_position": row["image_source_position"],
                }
                for turn_id, row in enumerate(bucket, 1)
            ]
            dialogs.append({
                # Assigned below after the globally deterministic image order
                # and the deterministic per-image bucket order are complete.
                "dialog_id": "",
                "image_id": image_id,
                "image_path": image_path,
                "image_dialogue_index": image_dialogue_index,
                "turns": turns,
            })

    for index, dialog in enumerate(dialogs, 1):
        dialog["dialog_id"] = f"mtgqa_{index:06d}"

    build_stats = _selection_statistics(per_image_stats)
    build_stats["per_image"] = per_image_stats
    return dialogs, build_stats


def _numeric_summary(values: Sequence[int]) -> dict[str, float | int]:
    ordered = sorted(values)
    p95_index = max(0, (95 * len(ordered) + 99) // 100 - 1)
    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "median": statistics.median(values),
        "p95_nearest_rank": ordered[p95_index],
    }


def _selection_statistics(
    per_image: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    original = [int(row["original_questions"]) for row in per_image.values()]
    quotas = [int(row["dialogue_quota"]) for row in per_image.values()]
    duplicate_images = sorted(
        image_id for image_id, row in per_image.items()
        if int(row["source_duplicate_extras"]) > 0)
    selected_duplicate_images = sorted(
        image_id for image_id, row in per_image.items()
        if int(row["selected_duplicate_extras"]) > 0)
    source_extras = sum(int(row["source_duplicate_extras"])
                        for row in per_image.values())
    selected_extras = sum(int(row["selected_duplicate_extras"])
                          for row in per_image.values())
    return {
        "source_questions": sum(original),
        "unique_images": len(per_image),
        "dialogues": sum(quotas),
        "selected_questions": sum(quotas) * 3,
        "discarded_questions": sum(original) - sum(quotas) * 3,
        "questions_per_image": _numeric_summary(original),
        "dialogues_per_image": {
            **_numeric_summary(quotas),
            "histogram": {
                str(key): value
                for key, value in sorted(Counter(quotas).items())
            },
        },
        "duplicates": {
            "normalization": QUESTION_NORMALIZATION,
            "images_with_source_duplicates": duplicate_images,
            "source_duplicate_extras": source_extras,
            "selected_duplicate_extras": selected_extras,
            "images_with_selected_duplicates": selected_duplicate_images,
            "duplicate_extras_discarded_as_remainder": (
                source_extras - selected_extras),
            "within_dialogue_normalized_question_duplicates": 0,
        },
    }


def validate_dialogues(
    dialogs: Sequence[Mapping[str, Any]],
    *,
    expected_quotas: Mapping[str, int] | None = None,
    check_images: bool = True,
) -> dict[str, Any]:
    """Validate the complete reconstructed-dialogue contract."""
    if not isinstance(dialogs, Sequence) or isinstance(dialogs, (str, bytes)):
        raise ValueError("dialogues must be a sequence")
    if not dialogs:
        raise ValueError("dialogues must be nonempty")
    seen_dialog_ids: set[str] = set()
    seen_question_ids: set[str] = set()
    seen_triplets: set[tuple[str, tuple[str, ...]]] = set()
    by_image = Counter()
    normalized_by_image: dict[str, Counter[str]] = defaultdict(Counter)

    for ordinal, dialog in enumerate(dialogs, 1):
        dialog_id = str(dialog.get("dialog_id", ""))
        if dialog_id != f"mtgqa_{ordinal:06d}":
            raise ValueError(f"noncanonical dialogue ID/order: {dialog_id}")
        if dialog_id in seen_dialog_ids:
            raise ValueError(f"duplicate dialogue ID: {dialog_id}")
        seen_dialog_ids.add(dialog_id)

        image_id = str(dialog.get("image_id", ""))
        if not image_id:
            raise ValueError(f"{dialog_id}: missing image ID")
        image_path = resolve_image_path(str(dialog.get("image_path", "")))
        if check_images and not image_path.is_file():
            raise FileNotFoundError(f"{dialog_id}: {image_path}")
        if image_path.name != f"{image_id}.jpg":
            raise ValueError(f"{dialog_id}: image path/ID mismatch")
        expected_image_dialogue_index = by_image[image_id] + 1
        if dialog.get("image_dialogue_index") != expected_image_dialogue_index:
            raise ValueError(f"{dialog_id}: noncontiguous per-image dialogue index")

        turns = dialog.get("turns")
        if not isinstance(turns, list) or len(turns) != 3:
            raise ValueError(f"{dialog_id}: expected exactly three turns")
        local_qids: list[str] = []
        local_texts: list[str] = []
        for turn_id, turn in enumerate(turns, 1):
            if turn.get("turn_id") != turn_id:
                raise ValueError(f"{dialog_id}: invalid turn order")
            qid = str(turn.get("question_id", ""))
            if not qid or qid in local_qids:
                raise ValueError(f"{dialog_id}: duplicate/empty question ID")
            if qid in seen_question_ids:
                raise ValueError(f"question reused across dialogues: {qid}")
            seen_question_ids.add(qid)
            local_qids.append(qid)
            normalized = normalize_question(str(turn.get("question", "")))
            if not normalized or normalized in local_texts:
                raise ValueError(
                    f"{dialog_id}: exact normalized question duplicate")
            local_texts.append(normalized)
            answers = turn.get("answers")
            if (not isinstance(answers, list) or len(answers) != 1
                    or not str(answers[0]).strip()):
                raise ValueError(f"{dialog_id}: expected one GQA gold answer")
            normalized_by_image[image_id][normalized] += 1
            expected_image_source_position = (
                (expected_image_dialogue_index - 1) * 3 + turn_id - 1)
            if turn.get("image_source_position") != expected_image_source_position:
                raise ValueError(
                    f"{dialog_id}: source-order contiguous membership violated")

        signature = (image_id, tuple(sorted(local_qids)))
        if signature in seen_triplets:
            raise ValueError(f"duplicate question triplet: {dialog_id}")
        seen_triplets.add(signature)
        by_image[image_id] += 1

    if expected_quotas is not None:
        expected = Counter({str(key): int(value)
                            for key, value in expected_quotas.items()})
        if by_image != expected:
            raise ValueError("per-image floor quota mismatch")
    retained_duplicates = sum(
        sum(count - 1 for count in counts.values() if count > 1)
        for counts in normalized_by_image.values()
    )
    retention_images = sorted(
        image_id for image_id, counts in normalized_by_image.items()
        if any(count > 1 for count in counts.values()))
    return {
        "passed": True,
        "dialogues": len(dialogs),
        "turns": len(dialogs) * 3,
        "unique_images": len(by_image),
        "unique_question_ids": len(seen_question_ids),
        "duplicate_triplets": 0,
        "within_dialogue_duplicate_question_ids": 0,
        "within_dialogue_normalized_question_duplicates": 0,
        "retained_normalized_duplicate_extras": retained_duplicates,
        "images_with_retained_normalized_duplicates": retention_images,
        "all_images_exist": bool(check_images),
    }


def _strict_canonical_checks(
    source_sha256: str,
    seed: int,
    build_stats: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> None:
    checks = {
        "source SHA256": (source_sha256, EXPECTED_SOURCE_SHA256),
        "seed": (seed, DEFAULT_SEED),
        "source questions": (
            build_stats["source_questions"], EXPECTED_SOURCE_QUESTIONS),
        "images": (build_stats["unique_images"], EXPECTED_IMAGES),
        "dialogues": (build_stats["dialogues"], EXPECTED_DIALOGUES),
        "selected questions": (
            build_stats["selected_questions"], EXPECTED_SELECTED_QUESTIONS),
        "source duplicate extras": (
            build_stats["duplicates"]["source_duplicate_extras"],
            EXPECTED_SOURCE_DUPLICATE_EXTRAS),
        "selected duplicate extras": (
            build_stats["duplicates"]["selected_duplicate_extras"],
            EXPECTED_SELECTED_DUPLICATE_EXTRAS),
        "validated selected duplicate extras": (
            validation["retained_normalized_duplicate_extras"],
            EXPECTED_SELECTED_DUPLICATE_EXTRAS),
    }
    for label, (observed, expected) in checks.items():
        if observed != expected:
            raise ValueError(
                f"canonical {label} mismatch: expected {expected}, got {observed}")
    observed_images = tuple(
        build_stats["duplicates"]["images_with_selected_duplicates"])
    if observed_images != EXPECTED_DUPLICATE_IMAGES:
        raise ValueError(
            "canonical duplicate-bearing images mismatch: "
            f"expected {EXPECTED_DUPLICATE_IMAGES}, "
            f"got {observed_images}")


def build_artifacts(
    source_questions: Path | str,
    image_dir: Path | str,
    *,
    seed: int = DEFAULT_SEED,
    strict_canonical: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build all four canonical JSON values without writing them."""
    source_path = Path(source_questions).resolve()
    image_root = Path(image_dir).resolve()
    source_sha = sha256_file(source_path)
    if strict_canonical and source_sha != EXPECTED_SOURCE_SHA256:
        raise ValueError(
            "canonical source SHA256 mismatch: expected "
            f"{EXPECTED_SOURCE_SHA256}, got {source_sha}")
    if strict_canonical and int(seed) != DEFAULT_SEED:
        raise ValueError(
            f"canonical seed mismatch: expected {DEFAULT_SEED}, got {seed}")
    questions = load_gqa_questions(source_path)
    dialogs, build_stats = reconstruct_dialogues(
        questions, image_root, seed=seed)
    quotas = {
        image_id: int(row["dialogue_quota"])
        for image_id, row in build_stats["per_image"].items()
    }
    validation = validate_dialogues(
        dialogs, expected_quotas=quotas, check_images=True)
    if strict_canonical:
        _strict_canonical_checks(source_sha, seed, build_stats, validation)

    dialogues_value = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_type": BENCHMARK_TYPE,
        "seed": int(seed),
        "dialogues": dialogs,
    }
    dialogues_sha = sha256_bytes(canonical_json_bytes(dialogues_value))
    workload_sha = workload_sha256(dialogs)

    dataset_stats = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_type": BENCHMARK_TYPE,
        "seed": int(seed),
        "quota_rule": "floor(original_questions_per_image / 3)",
        "turns_per_dialogue": 3,
        "source_questions": build_stats["source_questions"],
        "unique_images": build_stats["unique_images"],
        "dialogues": build_stats["dialogues"],
        "selected_questions": build_stats["selected_questions"],
        "discarded_questions": build_stats["discarded_questions"],
        "questions_per_image": build_stats["questions_per_image"],
        "dialogues_per_image": build_stats["dialogues_per_image"],
        "dialogue_quota_by_image": quotas,
        "duplicates": build_stats["duplicates"],
        "validation": validation,
        "dialogues_sha256": dialogues_sha,
        "workload_sha256": workload_sha,
    }

    provenance = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_type": BENCHMARK_TYPE,
        "official_benchmark_identity_claimed": False,
        "disclaimer": DISCLAIMER,
        "paper": {
            "title": "Rethinking Token Reduction for Large Vision-Language Models",
            "arxiv": "2603.21701",
            "reported_source": "GQA testdev-balanced",
            "reported_dialogues": EXPECTED_DIALOGUES,
            "reported_turns_per_dialogue": 3,
        },
        "source": {
            "dataset": "GQA",
            "split": "testdev-balanced",
            "questions_path": str(source_path),
            "questions_sha256": source_sha,
            "questions": build_stats["source_questions"],
            "image_dir": str(image_root),
            "images_required": build_stats["unique_images"],
            "images_verified": validation["unique_images"],
        },
        "metacompress_repository_evidence": {
            "repository": METACOMPRESS_REPOSITORY,
            "commit": METACOMPRESS_COMMIT,
            "branch": "main",
            "tags_observed": [],
            "tree_files": ["README.md"],
            "readme_sha256": METACOMPRESS_README_SHA256,
            "evidence_scope": "README-only",
            "readme_status": (
                "Code implementation is being organized and will be released "
                "as soon as possible."
            ),
            "official_dialogue_artifact_observed": False,
        },
        "reconstruction": {
            "seed": int(seed),
            "ranking": RANKING_DOMAIN,
            "seed_role": "SHA256 image ordering only",
            "membership": (
                "For each image, preserve source JSON occurrence order and "
                "take consecutive non-overlapping triples up to "
                "floor(original_questions / 3)."
            ),
            "question_normalization": QUESTION_NORMALIZATION,
            "quota_rule": "floor(original_questions_per_image / 3)",
            "same_image_per_dialogue": True,
            "turns_per_dialogue": 3,
            "duplicate_policy": (
                "Preserve source-order membership. Exact normalized repeats "
                "across different dialogues are retained and disclosed; any "
                "repeat inside one source-order triple fails validation."
            ),
            "future_question_or_answer_used_for_layout": False,
        },
    }

    stats_sha = sha256_bytes(canonical_json_bytes(dataset_stats))
    provenance_sha = sha256_bytes(canonical_json_bytes(provenance))
    config = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_type": BENCHMARK_TYPE,
        "seed": int(seed),
        "strict_canonical": bool(strict_canonical),
        "history_policy": "gold_teacher_forced",
        "future_question_access_count_for_layout": 0,
        "future_answer_access_count_for_layout": 0,
        "calibration_count": 0,
        "n_dialogues": len(dialogs),
        "n_turns": len(dialogs) * 3,
        "n_unique_images": validation["unique_images"],
        "turns_per_dialogue": 3,
        "source_questions_path": str(source_path),
        "source_questions_sha256": source_sha,
        "image_dir": str(image_root),
        "ranking_algorithm": RANKING_DOMAIN,
        "seed_role": "SHA256 image ordering only",
        "membership_rule": "source-order contiguous triples within each image",
        "question_normalization": QUESTION_NORMALIZATION,
        "official_benchmark_identity_claimed": False,
        "disclaimer": DISCLAIMER,
        "artifact_sha256": {
            "dialogues.json": dialogues_sha,
            "dataset_stats.json": stats_sha,
            "dataset_provenance.json": provenance_sha,
        },
        "dialogues_sha256": dialogues_sha,
        "workload_sha256": workload_sha,
    }
    artifacts = {
        "dialogues.json": dialogues_value,
        "config.json": config,
        "dataset_stats.json": dataset_stats,
        "dataset_provenance.json": provenance,
    }
    summary = {
        "benchmark_type": BENCHMARK_TYPE,
        "seed": int(seed),
        "source_questions_sha256": source_sha,
        "dialogues_sha256": dialogues_sha,
        "workload_sha256": workload_sha,
        "n_dialogues": len(dialogs),
        "n_turns": len(dialogs) * 3,
        "n_unique_images": validation["unique_images"],
        "selected_duplicate_extras": validation[
            "retained_normalized_duplicate_extras"],
        "artifact_sha256": {
            name: sha256_bytes(canonical_json_bytes(value))
            for name, value in artifacts.items()
        },
    }
    return artifacts, summary


def write_artifacts_no_clobber(
    output_dir: Path | str,
    artifacts: Mapping[str, Any],
) -> dict[str, str]:
    """Durably publish the artifact set with atomic no-clobber file links.

    Each filename becomes visible atomically.  A crash may leave a partial set,
    which every CLI entry point detects and refuses rather than completing or
    overwriting implicitly.
    """
    if set(artifacts) != set(ARTIFACT_FILENAMES):
        raise ValueError(
            f"artifact set must be exactly {sorted(ARTIFACT_FILENAMES)}")
    output = Path(output_dir)
    if output.is_symlink():
        raise ValueError(f"output directory must not be a symlink: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if not output.is_dir():
        raise ValueError(f"output path is not a directory: {output}")
    targets = {name: output / name for name in ARTIFACT_FILENAMES}
    existing = [str(path) for path in targets.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing MT-GQA artifact(s): "
            + ", ".join(existing))

    staged: dict[str, Path] = {}
    published: list[tuple[Path, int, int]] = []
    try:
        for name in ARTIFACT_FILENAMES:
            temporary = output / (
                f".{name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
            with temporary.open("xb") as handle:
                handle.write(canonical_json_bytes(artifacts[name]))
                handle.flush()
                os.fsync(handle.fileno())
            staged[name] = temporary

        # A hard link is an atomic, no-replace publication primitive on the
        # same filesystem.  Unlike os.replace, it cannot clobber a target that
        # appears after the initial existence check.
        for name in ARTIFACT_FILENAMES:
            target = targets[name]
            os.link(staged[name], target)
            stat = target.stat()
            published.append((target, stat.st_dev, stat.st_ino))
            staged[name].unlink()

        directory_fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        # Roll back only links whose inode is provably the one published by
        # this invocation; never remove an independently replaced target.
        for target, device, inode in reversed(published):
            try:
                stat = target.stat()
                if stat.st_dev == device and stat.st_ino == inode:
                    target.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        for temporary in staged.values():
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    return {name: sha256_file(path) for name, path in targets.items()}


def validate_artifact_directory(
    output_dir: Path | str,
    *,
    source_questions: Path | str | None = None,
    image_dir: Path | str | None = None,
    strict_canonical: bool = True,
) -> dict[str, Any]:
    """Validate emitted files, declared hashes, source, images and rebuild."""
    output = Path(output_dir)
    values: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    for name in ARTIFACT_FILENAMES:
        path = output / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing regular artifact: {path}")
        raw_bytes = path.read_bytes()
        try:
            values[name] = json.loads(raw_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid UTF-8 JSON artifact: {path}") from exc
        if raw_bytes != canonical_json_bytes(values[name]):
            raise ValueError(f"noncanonical JSON encoding: {name}")
        hashes[name] = sha256_file(path)

    config = values["config.json"]
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("config schema version mismatch")
    if config.get("benchmark_type") != BENCHMARK_TYPE:
        raise ValueError("benchmark type mismatch")
    declared = config.get("artifact_sha256", {})
    for name in ("dialogues.json", "dataset_stats.json",
                 "dataset_provenance.json"):
        if declared.get(name) != hashes[name]:
            raise ValueError(f"artifact hash mismatch: {name}")

    payload = values["dialogues.json"]
    if (payload.get("schema_version") != SCHEMA_VERSION
            or payload.get("benchmark_type") != BENCHMARK_TYPE):
        raise ValueError("dialogues metadata mismatch")
    quotas = values["dataset_stats.json"].get("dialogue_quota_by_image")
    validation = validate_dialogues(
        payload.get("dialogues", []), expected_quotas=quotas,
        check_images=True)
    observed_workload_sha = workload_sha256(payload["dialogues"])
    if config.get("dialogues_sha256") != hashes["dialogues.json"]:
        raise ValueError("config dialogues_sha256 mismatch")
    if config.get("workload_sha256") != observed_workload_sha:
        raise ValueError("config workload_sha256 mismatch")
    if config.get("n_dialogues") != validation["dialogues"]:
        raise ValueError("config dialogue count mismatch")
    if config.get("n_turns") != validation["turns"]:
        raise ValueError("config turn count mismatch")
    if config.get("n_unique_images") != validation["unique_images"]:
        raise ValueError("config image count mismatch")
    stats = values["dataset_stats.json"]
    if stats.get("dialogues_sha256") != hashes["dialogues.json"]:
        raise ValueError("dataset_stats dialogues_sha256 mismatch")
    if stats.get("workload_sha256") != observed_workload_sha:
        raise ValueError("dataset_stats workload_sha256 mismatch")
    provenance = values["dataset_provenance.json"]
    if (provenance.get("benchmark_type") != BENCHMARK_TYPE
            or provenance.get("official_benchmark_identity_claimed") is not False
            or provenance.get("disclaimer") != DISCLAIMER):
        raise ValueError("dataset provenance/disclaimer mismatch")

    source_path = Path(source_questions or config["source_questions_path"])
    image_root = Path(image_dir or config["image_dir"])
    if sha256_file(source_path) != config.get("source_questions_sha256"):
        raise ValueError("source questions SHA256 mismatch")

    expected, _ = build_artifacts(
        source_path,
        image_root,
        seed=int(config["seed"]),
        strict_canonical=strict_canonical,
    )
    for name in ARTIFACT_FILENAMES:
        if values[name] != expected[name]:
            raise ValueError(f"artifact differs from deterministic rebuild: {name}")
    return {
        **validation,
        "benchmark_type": BENCHMARK_TYPE,
        "source_questions_sha256": config["source_questions_sha256"],
        "dialogues_sha256": hashes["dialogues.json"],
        "workload_sha256": observed_workload_sha,
        "artifact_sha256": hashes,
        "deterministic_rebuild_match": True,
    }
