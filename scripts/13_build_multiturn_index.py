"""Build deterministic canonical indices for VisDial v1.0 or MMDU.

Examples:
  python scripts/13_build_multiturn_index.py --dataset visdial --profile smoke
  python scripts/13_build_multiturn_index.py --dataset visdial --profile main
  python scripts/13_build_multiturn_index.py --dataset mmdu --profile full
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mmimpress.config import PROJECT_ROOT
from mmimpress.multiturn import (SCHEMA_VERSION, build_mmdu_index,
                                 build_visdial_index,
                                 build_visdial_store_index, sha256_file)


DEFAULT_LIMIT = {
    ("visdial", "smoke"): 2,
    ("visdial", "main"): 100,
    ("visdial", "full"): 2064,
    ("mmdu", "smoke"): 2,
    ("mmdu", "main"): 20,
    ("mmdu", "full"): 110,
}


def write_json_atomic(path, value, force=False):
    """Create deterministically; never silently replace a different index."""
    path = Path(path)
    # Normalize integer dict keys and tuples exactly as JSON will represent
    # them before comparing an existing artifact.
    value = json.loads(json.dumps(value))
    if path.exists():
        with open(path) as f:
            old = json.load(f)
        if old == value:
            return False
        if not force:
            raise FileExistsError(
                f"{path} already exists with different content; use --force "
                "only for an intentional canonical-index revision")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(value, f, indent=1)
    tmp.replace(path)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("visdial", "mmdu"), required=True)
    ap.add_argument("--profile", choices=("smoke", "main", "full"),
                    default="main")
    ap.add_argument("--max-dialogs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--annotation", default=None)
    ap.add_argument("--dense-annotation", default=None)
    ap.add_argument("--image-dir", default=None)
    ap.add_argument("--dialog-ids", default=None,
                    help="MMDU only: explicit ordered ids, e.g. 84,69")
    ap.add_argument("--selection-note", default=None,
                    help="provenance note for an explicit feasibility subset")
    ap.add_argument("--force", action="store_true",
                    help="replace a different canonical index intentionally")
    args = ap.parse_args()

    limit = (args.max_dialogs if args.max_dialogs is not None else
             DEFAULT_LIMIT[(args.dataset, args.profile)])
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        base = (PROJECT_ROOT / "data" /
                ("visdial_v1.0" if args.dataset == "visdial" else "mmdu"))
        out_dir = base / "subsets" / f"{args.profile}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "visdial":
        base = PROJECT_ROOT / "data" / "visdial_v1.0"
        annotation = Path(args.annotation or base / "visdial_1.0_val.json")
        dense = Path(args.dense_annotation or
                     base / "raw" / "visdial_1.0_val_dense_annotations.json")
        image_dir = Path(args.image_dir or base / "VisualDialog_val2018")
        dialogs, config = build_visdial_index(
            annotation, image_dir, dense, limit, args.seed)
    else:
        base = PROJECT_ROOT / "data" / "mmdu"
        annotation = Path(args.annotation or base / "raw" / "benchmark.json")
        image_dir = Path(args.image_dir or base / "mmdu_pics")
        explicit = ([x.strip() for x in args.dialog_ids.split(",") if x.strip()]
                    if args.dialog_ids else None)
        dialogs, config = build_mmdu_index(
            annotation, image_dir, limit, args.seed, dialog_ids=explicit)

    payload = {"schema_version": SCHEMA_VERSION, "dialogs": dialogs}
    index_path = out_dir / "index.json"
    write_json_atomic(index_path, payload, args.force)
    config.update({
        "profile": args.profile,
        "requested_limit": limit,
        "selected_dialog_ids": [d["dialog_id"] for d in dialogs],
        "index_path": str(index_path.resolve()),
        "index_sha256": sha256_file(index_path),
        "selection_note": args.selection_note,
    })
    if args.dataset == "visdial":
        store_index = build_visdial_store_index(dialogs)
        store_path = out_dir / "store_index.json"
        write_json_atomic(store_path, store_index, args.force)
        config["store_index_path"] = str(store_path.resolve())
        config["store_index_sha256"] = sha256_file(store_path)
    write_json_atomic(out_dir / "config.json", config, args.force)

    print(json.dumps({k: config[k] for k in (
        "dataset", "profile", "seed", "n_dialogs", "n_turns", "n_images",
        "index_sha256")}, indent=1))


if __name__ == "__main__":
    main()
