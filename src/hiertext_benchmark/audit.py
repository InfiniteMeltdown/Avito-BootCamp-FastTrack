from __future__ import annotations

import collections
import json
from pathlib import Path

from PIL import Image

from common import read_csv, read_json, sha256, write_json


def validate_manifest(rows):
    ids, groups, pairs = set(), {}, collections.defaultdict(list)
    for row in rows:
        if row["example_id"] in ids:
            raise ValueError("Duplicate example ID")
        ids.add(row["example_id"])
        group = row["source_group"]
        assignment = row["split"], str(row["fold"])
        if group in groups and groups[group] != assignment:
            raise ValueError("Source group crosses split/fold boundaries")
        groups[group] = assignment
        if int(row["rotation"]) not in (0, 180) or int(row["label"]) != int(int(row["rotation"]) == 180):
            raise ValueError("Incorrect rotation label")
        pairs[row["pair_id"]].append(row)
    for pair in pairs.values():
        if len(pair) != 2 or {int(row["rotation"]) for row in pair} != {0, 180}:
            raise ValueError("Incomplete or repeated rotation pair")
        for key in ["image_path", "source_id", "source_group", "split", "fold", "width", "height", "cohort", "crop_type", "file_sha256"]:
            if pair[0][key] != pair[1][key]:
                raise ValueError(f"Rotation pair metadata differs: {key}")
    return pairs


def audit(benchmark: Path):
    rows = read_csv(benchmark / "manifest.csv")
    pairs = validate_manifest(rows)
    summary = read_json(benchmark / "summary.json")
    if sha256(benchmark / "manifest.csv") != summary["manifest_sha256"]:
        raise ValueError("Frozen manifest was modified")
    root = benchmark.resolve()
    for pair in pairs.values():
        row = pair[0]
        path = (benchmark / row["image_path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Crop path escapes benchmark directory")
        if sha256(path) != row["file_sha256"]:
            raise ValueError(f"Crop changed: {path}")
        with Image.open(path) as im:
            if im.size != (int(row["width"]), int(row["height"])):
                raise ValueError(f"Crop size mismatch: {path}")
            im.verify()
    crop_annotations = read_json(benchmark / "crop_annotations.json")
    hashes = [row["pixel_sha256"] for row in crop_annotations]
    if len(hashes) != len(set(hashes)):
        raise ValueError("Duplicate crop pixels in benchmark")
    result = {"status": "passed", "examples": len(rows), "pairs": len(pairs),
              "source_groups": len({row['source_group'] for row in rows}),
              "split_examples": dict(collections.Counter(row["split"] for row in rows)),
              "checks": ["manifest hash", "crop checksums and sizes", "unique IDs/crop pixels",
                         "rotation pair labels", "source group separation", "contained crop paths"]}
    write_json(benchmark / "audit.json", result)
    print(json.dumps(result, indent=2))
