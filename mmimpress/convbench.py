"""Official ConvBench XLSX adapter with source-row-preserving validation.

The source workbook has 578 populated data rows. At the pinned source commit,
one row (ID 131) references an image absent from the official image directory.
The resulting 577-image-available population is explicit rather than silently
dropping a row; ``source_row_index`` always addresses the original XLSX row and
the corresponding index of ``ConvBenchEval/pairwise.npy``.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from PIL import Image

from mmimpress.config import PROJECT_ROOT


SCHEMA_VERSION = "convbench-official-image-available-v1"
EXPECTED_SOURCE_ROWS = 578
EXPECTED_CONVERSATIONS = 577
EXPECTED_MISSING_SOURCE_ID = "131"
HEADERS = (
    "ID", "instruction_category", "image_id",
    "instruction-conditioned-caption", "The_first_turn_instruction",
    "First_turn_instruction_category", "first_turn_answer",
    "The_second_turn_instruction", "Second_turn_instruction_category",
    "second_turn_answer", "The_third_turn_instruction",
    "Third_turn_instruction_category", "third_turn_answer",
    "third_turn_demands",
)
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def _column_number(reference: str) -> int:
    letters = re.match(r"^[A-Z]+", reference)
    if letters is None:
        raise ValueError(f"invalid XLSX cell reference: {reference}")
    number = 0
    for char in letters.group():
        number = number * 26 + ord(char) - ord("A") + 1
    return number - 1


def read_workbook_rows(path: Path) -> list[dict[str, str]]:
    """Read the sole official worksheet using stdlib; preserve XLSX row order."""
    with zipfile.ZipFile(path) as archive:
        strings_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        strings = ["".join(node.text or "" for node in item.findall(".//m:t", NS))
                   for item in strings_root.findall("m:si", NS)]
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    rows: list[list[str]] = []
    for row in sheet.findall(".//m:sheetData/m:row", NS):
        cells = [""] * len(HEADERS)
        for cell in row.findall("m:c", NS):
            column = _column_number(cell.attrib["r"])
            if column >= len(HEADERS):
                continue
            value = cell.find("m:v", NS)
            if value is None:
                inline = cell.find("m:is", NS)
                if inline is None:
                    continue
                rendered = "".join(node.text or "" for node in inline.findall(".//m:t", NS))
            elif cell.get("t") == "s":
                rendered = strings[int(value.text or "0")]
            else:
                rendered = value.text or ""
            cells[column] = rendered
        if any(value != "" for value in cells):
            rows.append(cells)
    if not rows or tuple(rows[0]) != HEADERS:
        raise ValueError("official ConvBench XLSX column header mismatch")
    return [dict(zip(HEADERS, values, strict=True)) for values in rows[1:]]


def _source_revision(source_dir: Path) -> str:
    revision = subprocess.check_output(
        ["git", "-C", str(source_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("official ConvBench commit is unavailable")
    return revision


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def build_index(source_dir: Path, *, verify_images: bool = True) -> tuple[dict, dict, dict]:
    source_dir = source_dir.resolve()
    workbook = source_dir / "ConvBench.xlsx"
    image_dir = source_dir / "visit_bench_images"
    pairwise = source_dir / "ConvBenchEval/pairwise.npy"
    for path in (workbook, pairwise):
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
    if not image_dir.is_dir() or image_dir.is_symlink():
        raise FileNotFoundError(image_dir)
    rows = read_workbook_rows(workbook)
    if len(rows) != EXPECTED_SOURCE_ROWS:
        raise ValueError(f"expected {EXPECTED_SOURCE_ROWS} populated source rows, got {len(rows)}")
    seen_ids: set[str] = set()
    conversations: list[dict] = []
    excluded: list[dict] = []
    for source_row_index, row in enumerate(rows):
        source_id = row["ID"].strip()
        image_id = row["image_id"].strip()
        if not source_id or source_id in seen_ids:
            raise ValueError(f"duplicate/empty source ID: {source_id!r}")
        seen_ids.add(source_id)
        if not image_id or Path(image_id).name != image_id or "\\" in image_id:
            raise ValueError(f"unsafe image ID: {image_id!r}")
        for field in HEADERS[3:]:
            if not row[field].strip():
                raise ValueError(f"source ID {source_id}: empty {field}")
        image_path = image_dir / image_id
        if not image_path.is_file():
            excluded.append({"source_row_index": source_row_index,
                             "source_id": source_id, "image_id": image_id,
                             "reason": "image_missing_from_official_repository"})
            continue
        if image_path.is_symlink():
            raise ValueError(f"symlink image refused: {image_path}")
        if verify_images:
            with Image.open(image_path) as image:
                image.verify()
        questions = [row["The_first_turn_instruction"],
                     row["The_second_turn_instruction"],
                     row["The_third_turn_instruction"]]
        answers = [row["first_turn_answer"], row["second_turn_answer"],
                   row["third_turn_answer"]]
        categories = [row["First_turn_instruction_category"],
                      row["Second_turn_instruction_category"],
                      row["Third_turn_instruction_category"]]
        turns = [
            {"turn_id": i + 1, "question": questions[i],
             "reference_answer": answers[i], "category": categories[i]}
            for i in range(3)
        ]
        conversation = {
            "conversation_id": f"convbench:{source_id}",
            "source_id": source_id,
            "source_row_index": source_row_index,
            "image_id": image_id,
            "image_path": _relative(image_path),
            "instruction_category": row["instruction_category"],
            "instruction_conditioned_caption": row["instruction-conditioned-caption"],
            "third_turn_demands": row["third_turn_demands"],
            "turn1_category": categories[0],
            "turn2_category": categories[1],
            "turn3_category": categories[2],
            "Q1": questions[0], "Q2": questions[1], "Q3": questions[2],
            "reference_A1": answers[0], "reference_A2": answers[1],
            "reference_A3": answers[2],
            "turns": turns,
        }
        conversations.append(conversation)
    if len(conversations) != EXPECTED_CONVERSATIONS:
        raise ValueError(f"expected {EXPECTED_CONVERSATIONS} image-available conversations, got {len(conversations)}")
    if len(excluded) != 1 or excluded[0]["source_id"] != EXPECTED_MISSING_SOURCE_ID:
        raise ValueError(f"unexpected official missing-image set: {excluded}")
    image_counts = Counter(row["image_id"] for row in conversations)
    duplicate_groups = [
        {"image_id": image_id,
         "conversation_ids": [row["conversation_id"] for row in conversations
                              if row["image_id"] == image_id], "count": count}
        for image_id, count in sorted(image_counts.items()) if count > 1
    ]
    revision = _source_revision(source_dir)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "official_repository": "https://github.com/shirlyliu64/ConvBench",
        "official_commit": revision,
        "source_workbook": _relative(workbook),
        "source_workbook_sha256": sha256_file(workbook),
        "official_pairwise": _relative(pairwise),
        "official_pairwise_sha256": sha256_file(pairwise),
        "source_populated_rows": len(rows),
        "excluded_rows": excluded,
        "selection_policy": "all official XLSX rows with an existing official image; preserve source_row_index for pairwise.npy",
    }
    config = {
        "schema_version": SCHEMA_VERSION,
        "seed": 1234,
        "source_rows": len(rows),
        "conversations": len(conversations),
        "turns_per_conversation": 3,
        "logical_method_turn_requests": len(conversations) * 3 * 4,
        "cache_hit_requests_per_method": len(conversations) * 2,
        "unique_images": len(image_counts),
        "duplicate_image_groups": duplicate_groups,
        "duplicate_image_excess_conversations": sum(c - 1 for c in image_counts.values()),
        "missing_image_source_ids": [item["source_id"] for item in excluded],
    }
    index = {"schema_version": SCHEMA_VERSION,
             "official_commit": revision,
             "conversations": conversations}
    return index, config, provenance


def validate_index(index: dict, config: dict, provenance: dict,
                   source_dir: Path) -> dict:
    expected = build_index(source_dir)
    for name, actual, wanted in zip(("index", "config", "provenance"),
                                    (index, config, provenance), expected,
                                    strict=True):
        if canonical_json(actual) != canonical_json(wanted):
            raise ValueError(f"{name} does not match pinned official source")
    return {
        "passed": True,
        "conversations": config["conversations"],
        "source_rows": config["source_rows"],
        "unique_images": config["unique_images"],
        "duplicate_image_groups": len(config["duplicate_image_groups"]),
        "excluded_rows": provenance["excluded_rows"],
        "official_commit": provenance["official_commit"],
    }


def write_artifacts(output_dir: Path, artifacts: dict[str, dict]) -> dict[str, str]:
    """Publish a new all-or-nothing dataset directory without replacing files."""
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite dataset artifacts: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".convbench-index-", dir=output_dir.parent) as tmp:
        staging = Path(tmp)
        hashes = {}
        for name, payload in artifacts.items():
            body = canonical_json(payload)
            path = staging / name
            with path.open("xb") as handle:
                handle.write(body)
                handle.flush()
                os.fsync(handle.fileno())
            hashes[name] = hashlib.sha256(body).hexdigest()
        staging.rename(output_dir)
    return hashes
