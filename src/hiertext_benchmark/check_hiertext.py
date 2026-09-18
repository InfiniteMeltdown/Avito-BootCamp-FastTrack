"""Check the saved dataset against HierText annotations and original image pixels."""

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image

from hiertext import ANNOTATION_SHA256, eligible, line_quad, rectify, size_bin
from audit import audit, validate_manifest
from common import read_csv, read_json, sha256, write_json


def check(root, benchmark):
    # File integrity, paired labels, unique IDs, dimensions and split leakage.
    audit(benchmark)
    pairs = validate_manifest(read_csv(benchmark / "manifest.csv"))
    crops = read_json(benchmark / "crop_annotations.json")
    summary = read_json(benchmark / "summary.json")
    sources = {s["source_id"]: s for s in read_csv(benchmark / "sources.csv")}
    annotation_path = root / "raw/validation.jsonl.gz"
    if sha256(annotation_path) != ANNOTATION_SHA256:
        raise ValueError("Original annotations differ from the pinned HierText version")
    with gzip.open(annotation_path, "rt", encoding="utf-8") as handle:
        annotations = {a["image_id"]: a for a in json.load(handle)["annotations"]}
    if len(crops) != len(pairs) or {c["crop_id"] for c in crops} != set(pairs):
        raise ValueError("Crop annotations do not match the manifest")

    by_source = defaultdict(list)
    for crop in crops:
        by_source[crop["source_id"]].append(crop)
    for index, (source_id, source_crops) in enumerate(by_source.items(), 1):
        source = sources[source_id]
        path = root / "images" / f"{source_id}.jpg"
        if sha256(path) != source["file_sha256"]:
            raise ValueError(f"Source image changed: {path}")
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"))
        for crop in source_crops:
            crop_id = crop["crop_id"]
            match = re.fullmatch(r"([0-9a-f]+)_p(\d+)_l(\d+)_(word|line)(-?\d+)", crop_id)
            if not match or match[1] != source_id:
                raise ValueError(f"Invalid crop ID: {crop_id}")
            line = annotations[source_id]["paragraphs"][int(match[2])]["lines"][int(match[3])]
            element = line if match[4] == "line" else line["words"][int(match[5])]
            if not eligible(element):
                raise ValueError(f"Ineligible crop: {crop_id}")
            vertices = line_quad(line) if match[4] == "line" else np.float32(element["vertices"])
            if not np.allclose(vertices, crop["vertices"], rtol=0, atol=1e-4):
                raise ValueError(f"Reading-order geometry changed: {crop_id}")
            expected = rectify(pixels, vertices)
            with Image.open(benchmark / crop["image_path"]) as image:
                actual = np.asarray(image.convert("RGB"))
            if not np.array_equal(expected, actual):
                raise ValueError(f"Crop does not reproduce original annotation: {crop_id}")
            digest = min(hashlib.sha256(view.tobytes()).hexdigest()
                         for view in [actual, np.rot90(actual, 2)])
            if digest != crop["pixel_sha256"]:
                raise ValueError(f"Crop pixel hash differs: {crop_id}")
            row = pairs[crop_id][0]
            for key in ["source_id", "source_group", "split", "fold", "image_path",
                        "file_sha256", "width", "height", "cohort", "crop_type", "size_bin"]:
                if str(crop[key]) != str(row[key]):
                    raise ValueError(f"Crop/manifest mismatch ({key}): {crop_id}")
            for key in ["source_group", "split", "fold"]:
                if str(crop[key]) != source[key]:
                    raise ValueError(f"Source assignment mismatch ({key}): {crop_id}")
            if (crop["split"], int(crop["fold"])) not in {
                ("development", 0), ("development", 1), ("development", 2), ("validation", -1)
            }:
                raise ValueError(f"Invalid split/fold: {crop_id}")
            if crop["size_bin"] != size_bin(crop["width"], crop["height"]):
                raise ValueError(f"Incorrect size bin: {crop_id}")
        if index % 200 == 0:
            print(f"Reproduced crops from {index}/{len(by_source)} sources", flush=True)

    if max(Counter(c["source_group"] for c in crops).values()) > summary["source_cap"]:
        raise ValueError("Source crop cap exceeded")
    counts = Counter(c["cohort"] for c in crops)
    if (counts["matched"], counts["natural"], 2 * len(crops)) != (
        summary["matched_crops"], summary["natural_crops"], summary["labeled_examples"]
    ):
        raise ValueError("Summary counts do not match the dataset")
    result = {"status": "passed", "reproduced_crops": len(crops),
              "labeled_examples": 2 * len(crops), "source_images": len(by_source)}
    write_json(benchmark / "check.json", result)
    print(f"PASS: all {len(crops):,} crops reproduce their source annotations; "
          f"{2 * len(crops):,} labels and grouped splits verified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/hiertext"))
    parser.add_argument("--benchmark", type=Path, default=Path("artifacts/hiertext"))
    args = parser.parse_args()
    check(args.root, args.benchmark)
