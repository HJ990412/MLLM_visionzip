"""Deterministic local reconstruction of a three-turn MT-VQA-v2 workload.

The project does not contain an official/released MT-VQA-v2 dialogue file.
It does contain a frozen VQAv2 validation slice (250 images, five questions
per image) used by the previous image-only generalization experiment.  This
module turns that exact local slice into one non-overlapping three-turn
dialogue per image and deliberately calls it ``MT-VQA-v2-reconstructed``.

The prior VQAv2 experiment evaluated ``questions[1:5]``.  We preserve that
established slice, take its first source-order triple (``questions[1:4]``),
and discard the final remainder question.  Question membership never depends
on model outputs or experimental results.
"""
from __future__ import annotations

import hashlib
import json
import os
import unicodedata
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from mmimpress.config import PROJECT_ROOT


SCHEMA_VERSION = "mt-vqa-v2-reconstructed-v1"
BENCHMARK_TYPE = "MT-VQA-v2-reconstructed"
DEFAULT_SEED = 1234
SOURCE_DATASET = "VQAv2"
SOURCE_SPLIT = "validation"
SOURCE_INDEX_SHA256 = (
    "b83d5fa288fcb722ca073e261d3fec9086629ed0db2e568a2d5d6a24ef1589d7"
)
SOURCE_CONFIG_SHA256 = (
    "3bb4db281be6de46d42353c8becf3d5f32ea8643035104ef08161d76b2323467"
)
SOURCE_IMAGE_MANIFEST_SHA256 = (
    "f1724a035d0a52a5a100c84091d6ac63fe5a7f04b94acf2359f47332963f13e2"
)
SOURCE_IMAGE_TOTAL_BYTES = 118_362_026
EXPECTED_IMAGES = 250
EXPECTED_SOURCE_QUESTIONS_PER_IMAGE = 5
EXPECTED_DIALOGUES = 250
EXPECTED_TURNS = 750
EVALUATION_SLICE_START = 1
EVALUATION_SLICE_STOP = 5
TURNS_PER_DIALOGUE = 3
QUESTION_NORMALIZATION = "Unicode NFKC + whitespace collapse + casefold"
DISCLAIMER = (
    "No official/released MT-VQA-v2 dialogue artifact was present locally. "
    "This evaluation uses a deterministic reconstruction from the project's "
    "frozen VQAv2 validation slice and does not claim official benchmark "
    "identity or exact MetaCompress protocol identity."
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
    return (json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2,
    ) + "\n").encode("utf-8")


def workload_sha256(dialogues: Sequence[Mapping[str, Any]]) -> str:
    payload = "".join(
        f"{dialog['dialog_id']}\t{turn['turn_id']}\t{turn['question_id']}\n"
        for dialog in dialogues for turn in dialog["turns"]
    ).encode("utf-8")
    return sha256_bytes(payload)


def normalize_question(value: Any) -> str:
    return " ".join(
        unicodedata.normalize("NFKC", str(value)).split()).casefold()


def resolve_image_path(value: str | Path | Mapping[str, Any]) -> Path:
    if isinstance(value, Mapping):
        raw = value.get("image_path")
        if raw is None:
            raise ValueError("dialogue has no image_path")
        path = Path(str(raw))
    else:
        path = Path(value)
    return path.resolve() if path.is_absolute() else (
        PROJECT_ROOT / path).resolve()


def _validate_source_index(
    source_index: Path | str,
    source_config: Path | str,
    *,
    strict_canonical: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    index_path = Path(source_index).resolve()
    config_path = Path(source_config).resolve()
    for path, label in ((index_path, "source index"),
                        (config_path, "source config")):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{label} must be a regular file: {path}")
    index_sha = sha256_file(index_path)
    config_sha = sha256_file(config_path)
    if strict_canonical and index_sha != SOURCE_INDEX_SHA256:
        raise ValueError(
            f"source index SHA256 mismatch: {index_sha}")
    if strict_canonical and config_sha != SOURCE_CONFIG_SHA256:
        raise ValueError(
            f"source config SHA256 mismatch: {config_sha}")
    rows = json.loads(index_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("source VQAv2 index must be a nonempty list")
    if not isinstance(config, Mapping):
        raise ValueError("source VQAv2 config must be an object")
    if strict_canonical and (
        len(rows) != EXPECTED_IMAGES
        or config.get("dataset") != "vqav2"
        or int(config.get("seed", -1)) != DEFAULT_SEED
        or int(config.get("max_images", -1)) != EXPECTED_IMAGES
        or int(config.get("q_per_image", -1))
        != EXPECTED_SOURCE_QUESTIONS_PER_IMAGE
    ):
        raise ValueError("source VQAv2 slice metadata is not canonical")
    return [dict(row) for row in rows], {
        "index_path": str(index_path),
        "index_sha256": index_sha,
        "config_path": str(config_path),
        "config_sha256": config_sha,
        "config": dict(config),
    }


def reconstruct_dialogues(
    source_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dialogues: list[dict[str, Any]] = []
    seen_images: set[str] = set()
    seen_source_qids: set[str] = set()
    selected_qids: set[str] = set()
    held_out_qids: list[str] = []
    remainder_qids: list[str] = []
    for ordinal, raw in enumerate(source_rows):
        image_id = str(raw.get("image_id", ""))
        image_path = str(raw.get("image_path", ""))
        questions = raw.get("questions")
        if (not image_id or Path(image_id).name != image_id
                or image_id in seen_images):
            raise ValueError(f"invalid or duplicate image ID: {image_id!r}")
        if not isinstance(questions, list) or len(questions) != 5:
            raise ValueError(f"{image_id}: expected exactly five source questions")
        path = resolve_image_path(image_path)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        if path.stem != image_id:
            raise ValueError(f"{image_id}: image path/ID mismatch")
        normalized: set[str] = set()
        parsed: list[dict[str, Any]] = []
        for source_position, question in enumerate(questions):
            if not isinstance(question, Mapping):
                raise ValueError(f"{image_id}: question is not an object")
            qid = str(question.get("question_id", ""))
            text = str(question.get("question", "")).strip()
            answers = question.get("answers")
            normalized_text = normalize_question(text)
            if (not qid or qid in seen_source_qids or not normalized_text
                    or normalized_text in normalized):
                raise ValueError(f"{image_id}: invalid/duplicate question {qid!r}")
            if (not isinstance(answers, list) or len(answers) != 10
                    or any(not str(answer).strip() for answer in answers)):
                raise ValueError(f"{image_id}/{qid}: expected ten nonempty answers")
            seen_source_qids.add(qid)
            normalized.add(normalized_text)
            parsed.append({
                "question_id": qid,
                "question": text,
                "answers": [str(answer) for answer in answers],
                "source_position": source_position,
            })
        local_slice = parsed[EVALUATION_SLICE_START:EVALUATION_SLICE_STOP]
        selected = local_slice[:TURNS_PER_DIALOGUE]
        if len(selected) != 3 or len(local_slice) != 4:
            raise AssertionError("canonical evaluation slice is malformed")
        held_out_qids.append(parsed[0]["question_id"])
        remainder_qids.append(local_slice[3]["question_id"])
        turns = []
        for turn_id, question in enumerate(selected, 1):
            qid = question["question_id"]
            if qid in selected_qids:
                raise ValueError(f"question reused across dialogues: {qid}")
            selected_qids.add(qid)
            turns.append({
                "turn_id": turn_id,
                "question_id": qid,
                "question": question["question"],
                "answers": question["answers"],
                "source_position": question["source_position"],
            })
        dialogues.append({
            "dialog_id": f"mtvqav2_{ordinal + 1:06d}",
            "global_dialog_ordinal": ordinal,
            "image_id": image_id,
            "image_path": image_path,
            "image_dialogue_index": 1,
            "turns": turns,
        })
        seen_images.add(image_id)
    return dialogues, {
        "source_images": len(source_rows),
        "source_questions": len(seen_source_qids),
        "selected_questions": len(selected_qids),
        "dialogues": len(dialogues),
        "held_out_questions": len(held_out_qids),
        "discarded_slice_remainder_questions": len(remainder_qids),
        "question_overlap_between_dialogues": 0,
        "dialogue_overlap": False,
        "held_out_question_ids_sha256": sha256_bytes(
            "\n".join(held_out_qids).encode("utf-8")),
        "remainder_question_ids_sha256": sha256_bytes(
            "\n".join(remainder_qids).encode("utf-8")),
    }


def validate_dialogues(
    dialogues: Sequence[Mapping[str, Any]], *, check_images: bool = True,
) -> dict[str, Any]:
    if not dialogues:
        raise ValueError("dialogue workload is empty")
    seen_dialogs: set[str] = set()
    seen_images: set[str] = set()
    seen_questions: set[str] = set()
    for ordinal, dialog in enumerate(dialogues):
        did = str(dialog.get("dialog_id", ""))
        image_id = str(dialog.get("image_id", ""))
        expected_did = f"mtvqav2_{ordinal + 1:06d}"
        if (did != expected_did or did in seen_dialogs
                or int(dialog.get("global_dialog_ordinal", -1)) != ordinal):
            raise ValueError(f"noncanonical dialogue identity/order: {did!r}")
        if not image_id or image_id in seen_images:
            raise ValueError(f"each image must occur in exactly one dialogue: {image_id!r}")
        image_path = resolve_image_path(dialog)
        if check_images and (image_path.is_symlink() or not image_path.is_file()):
            raise FileNotFoundError(image_path)
        if image_path.stem != image_id:
            raise ValueError(f"{did}: image path/ID mismatch")
        turns = dialog.get("turns")
        if not isinstance(turns, list) or len(turns) != 3:
            raise ValueError(f"{did}: expected exactly three turns")
        local_questions: set[str] = set()
        for turn_id, turn in enumerate(turns, 1):
            if int(turn.get("turn_id", -1)) != turn_id:
                raise ValueError(f"{did}: invalid turn ordering")
            qid = str(turn.get("question_id", ""))
            question = normalize_question(turn.get("question", ""))
            answers = turn.get("answers")
            expected_source_position = turn_id
            if (not qid or qid in seen_questions or not question
                    or question in local_questions):
                raise ValueError(f"{did}: invalid/reused question {qid!r}")
            if (not isinstance(answers, list) or len(answers) != 10
                    or any(not str(answer).strip() for answer in answers)):
                raise ValueError(f"{did}/{qid}: invalid answer annotations")
            if int(turn.get("source_position", -1)) != expected_source_position:
                raise ValueError(f"{did}: source-order membership changed")
            seen_questions.add(qid)
            local_questions.add(question)
        seen_dialogs.add(did)
        seen_images.add(image_id)
    return {
        "passed": True,
        "dialogues": len(dialogues),
        "turns": len(dialogues) * 3,
        "unique_images": len(seen_images),
        "unique_question_ids": len(seen_questions),
        "all_dialogues_exactly_three_turns": True,
        "all_turns_share_dialogue_image": True,
        "question_overlap_between_dialogues": 0,
        "all_images_exist": bool(check_images),
    }


def build_artifacts(
    source_index: Path | str,
    source_config: Path | str,
    *,
    seed: int = DEFAULT_SEED,
    strict_canonical: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if strict_canonical and int(seed) != DEFAULT_SEED:
        raise ValueError(f"canonical seed must be {DEFAULT_SEED}")
    source, source_meta = _validate_source_index(
        source_index, source_config, strict_canonical=strict_canonical)
    image_manifest_lines: list[str] = []
    image_total_bytes = 0
    for row in source:
        image_id = str(row["image_id"])
        image_path_text = str(row["image_path"])
        image_path = resolve_image_path(image_path_text)
        image_manifest_lines.append(
            f"{image_id}\t{image_path_text}\t{sha256_file(image_path)}\n")
        image_total_bytes += image_path.stat().st_size
    image_manifest_sha = sha256_bytes(
        "".join(image_manifest_lines).encode("utf-8"))
    if strict_canonical and (
        image_manifest_sha != SOURCE_IMAGE_MANIFEST_SHA256
        or image_total_bytes != SOURCE_IMAGE_TOTAL_BYTES
    ):
        raise ValueError("referenced VQAv2 image manifest changed")
    dialogues, construction = reconstruct_dialogues(source)
    validation = validate_dialogues(dialogues, check_images=True)
    if strict_canonical and (
        validation["dialogues"] != EXPECTED_DIALOGUES
        or validation["turns"] != EXPECTED_TURNS
        or validation["unique_images"] != EXPECTED_IMAGES
    ):
        raise ValueError("reconstructed canonical counts changed")
    dialogue_value = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_type": BENCHMARK_TYPE,
        "seed": int(seed),
        "dialogues": dialogues,
    }
    dialogues_sha = sha256_bytes(canonical_json_bytes(dialogue_value))
    workload_sha = workload_sha256(dialogues)
    dataset_stats = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_type": BENCHMARK_TYPE,
        "seed": int(seed),
        "source_images": len(source),
        "source_questions": sum(len(row["questions"]) for row in source),
        "evaluation_slice": {
            "start": EVALUATION_SLICE_START,
            "stop": EVALUATION_SLICE_STOP,
            "rule": "questions[1:5] (established local VQAv2 evaluation slice)",
        },
        "dialogue_rule": (
            "one source-order non-overlapping triple questions[1:4] per image"
        ),
        "remainder_policy": "discard questions[4] after the local slice",
        "dialogue_overlap": False,
        "question_reuse_across_dialogues": False,
        **construction,
        "validation": validation,
        "dialogues_sha256": dialogues_sha,
        "workload_sha256": workload_sha,
    }
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_type": BENCHMARK_TYPE,
        "official_benchmark_identity_claimed": False,
        "exact_metacompress_reproduction_claimed": False,
        "disclaimer": DISCLAIMER,
        "source": {
            "dataset": SOURCE_DATASET,
            "split": SOURCE_SPLIT,
            "local_index": source_meta["index_path"],
            "local_index_sha256": source_meta["index_sha256"],
            "local_config": source_meta["config_path"],
            "local_config_sha256": source_meta["config_sha256"],
            "source_slice_seed": int(source_meta["config"]["seed"]),
            "source_images": len(source),
            "source_questions_per_image": 5,
            "human_answers_per_question": 10,
            "referenced_image_manifest_framing": (
                "index-order image_id\\tindex_image_path\\tfile_sha256\\n"
            ),
            "referenced_image_manifest_sha256": image_manifest_sha,
            "referenced_image_total_bytes": image_total_bytes,
        },
        "availability_audit": {
            "official_released_mt_vqa_v2_dialogue_artifact_found_locally": False,
            "construction_required": True,
        },
        "reconstruction": {
            "seed": int(seed),
            "seed_role": (
                "inherited frozen source-slice identity; no new random sampling"
            ),
            "same_image_per_dialogue": True,
            "turns_per_dialogue": 3,
            "source_order_preserved": True,
            "membership": "questions[1:4] for every source image",
            "overlap_allowed": False,
            "questions_used_more_than_once": 0,
            "results_used_to_choose_membership": False,
        },
    }
    stats_sha = sha256_bytes(canonical_json_bytes(dataset_stats))
    provenance_sha = sha256_bytes(canonical_json_bytes(provenance))
    config = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_type": BENCHMARK_TYPE,
        "seed": int(seed),
        "strict_canonical": bool(strict_canonical),
        "history_protocol": "generated_history_only",
        "n_dialogues": len(dialogues),
        "n_turns": len(dialogues) * 3,
        "n_unique_images": validation["unique_images"],
        "turns_per_dialogue": 3,
        "source_index_sha256": source_meta["index_sha256"],
        "source_config_sha256": source_meta["config_sha256"],
        "evaluation_slice": "questions[1:5]",
        "dialogue_membership": "questions[1:4]",
        "dialogue_overlap": False,
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
        "dialogues.json": dialogue_value,
        "config.json": config,
        "dataset_stats.json": dataset_stats,
        "dataset_provenance.json": provenance,
    }
    summary = {
        "benchmark_type": BENCHMARK_TYPE,
        "seed": int(seed),
        "source_index_sha256": source_meta["index_sha256"],
        "dialogues_sha256": dialogues_sha,
        "workload_sha256": workload_sha,
        "n_dialogues": len(dialogues),
        "n_turns": len(dialogues) * 3,
        "n_unique_images": validation["unique_images"],
        "artifact_sha256": {
            name: sha256_bytes(canonical_json_bytes(value))
            for name, value in artifacts.items()
        },
    }
    return artifacts, summary


def write_artifacts_no_clobber(
    output_dir: Path | str, artifacts: Mapping[str, Any],
) -> dict[str, str]:
    if set(artifacts) != set(ARTIFACT_FILENAMES):
        raise ValueError("incomplete MT-VQA-v2 artifact set")
    output = Path(output_dir)
    if output.is_symlink():
        raise ValueError("output directory may not be a symlink")
    output.mkdir(parents=True, exist_ok=True)
    targets = {name: output / name for name in ARTIFACT_FILENAMES}
    existing = [str(path) for path in targets.values() if os.path.lexists(path)]
    if existing:
        raise FileExistsError("refusing to overwrite: " + ", ".join(existing))
    staged: dict[str, Path] = {}
    published: list[tuple[Path, int, int]] = []
    try:
        for name in ARTIFACT_FILENAMES:
            temporary = output / f".{name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
            with temporary.open("xb") as handle:
                handle.write(canonical_json_bytes(artifacts[name]))
                handle.flush()
                os.fsync(handle.fileno())
            staged[name] = temporary
        for name in ARTIFACT_FILENAMES:
            target = targets[name]
            os.link(staged[name], target)
            stat = target.stat()
            published.append((target, stat.st_dev, stat.st_ino))
            staged[name].unlink()
        descriptor = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
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
    source_index: Path | str,
    source_config: Path | str,
    strict_canonical: bool = True,
) -> dict[str, Any]:
    output = Path(output_dir)
    observed: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    for name in ARTIFACT_FILENAMES:
        path = output / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing regular artifact: {path}")
        raw = path.read_bytes()
        value = json.loads(raw)
        if raw != canonical_json_bytes(value):
            raise ValueError(f"noncanonical artifact encoding: {name}")
        observed[name] = value
        hashes[name] = sha256_file(path)
    rebuilt, summary = build_artifacts(
        source_index, source_config, seed=DEFAULT_SEED,
        strict_canonical=strict_canonical)
    rebuilt_bytes = {name: canonical_json_bytes(value)
                     for name, value in rebuilt.items()}
    for name in ARTIFACT_FILENAMES:
        if (output / name).read_bytes() != rebuilt_bytes[name]:
            raise ValueError(f"artifact differs from deterministic rebuild: {name}")
    validation = validate_dialogues(
        observed["dialogues.json"]["dialogues"], check_images=True)
    return {
        "passed": True,
        **summary,
        "validation": validation,
        "observed_artifact_sha256": hashes,
    }
