from __future__ import annotations

import collections
import gzip
import hashlib
import json
import math
import random
import shutil
import tarfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from common import read_json, sha256, write_csv, write_json

ARCHIVE_URL = "https://open-images-dataset.s3.amazonaws.com/ocr/validation.tgz"
ANNOTATION_REVISION = "70b6620b2b112597d8219e11eee9773a1403827c"
ANNOTATION_SHA256 = "ce7085d8bf24c2f13df852715c34494e6748cafcabb4bfc7a6e2bb320f58d23a"
ARCHIVE_SHA256 = "1903446ab63a9be482b7f279cc54198904987efe2b30d3004225d48d1b84c4bf"
HEIGHT_EDGES = [16, 24, 40, 64, 128]
RATIO_EDGES = [2, 4, 8, 16]


def download(root: Path) -> None:
    """Download only the validation split; interrupted downloads never look complete."""
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    provenance_path = raw / "provenance.json"
    if provenance_path.exists():
        provenance = read_json(provenance_path)
    else:
        provenance = {"revision": ANNOTATION_REVISION, "archive_url": ARCHIVE_URL,
                      "annotation_url": f"https://raw.githubusercontent.com/google-research-datasets/hiertext/{ANNOTATION_REVISION}/gt/validation.jsonl.gz",
                      "annotation_sha256": ANNOTATION_SHA256, "archive_sha256": ARCHIVE_SHA256}
    for filename, key in [("validation.jsonl.gz", "annotation"), ("validation.tgz", "archive")]:
        path = raw / filename
        if not path.exists():
            print(f"Downloading {filename}", flush=True)
            temporary = path.with_suffix(path.suffix + ".part")
            with urllib.request.urlopen(provenance[key + "_url"], timeout=120) as response, temporary.open("wb") as out:
                shutil.copyfileobj(response, out, length=1024 * 1024)
            temporary.replace(path)
        digest = sha256(path)
        expected = provenance.get(key + "_sha256")
        if expected and expected != digest:
            raise ValueError(f"Checksum mismatch: {path}")
        provenance[key + "_sha256"] = digest
        write_json(provenance_path, provenance)
    print("Downloads verified.", flush=True)


def quad_size(vertices) -> tuple[int, int]:
    v = np.asarray(vertices, dtype=np.float32)
    if v.shape != (4, 2) or not np.isfinite(v).all():
        raise ValueError("Expected four finite vertices")
    if not cv2.isContourConvex(v) or abs(cv2.contourArea(v)) < 4:
        raise ValueError("Degenerate/non-convex quadrilateral")
    width = max(np.linalg.norm(v[1] - v[0]), np.linalg.norm(v[2] - v[3]))
    height = max(np.linalg.norm(v[3] - v[0]), np.linalg.norm(v[2] - v[1]))
    if min(width, height) < 2 or max(width, height) > 10000:
        raise ValueError("Invalid crop dimensions")
    return max(2, round(float(width))), max(2, round(float(height)))


def rectify(image: np.ndarray, vertices) -> np.ndarray:
    """Source vertices are TEXT-relative TL, TR, BR, BL. Never image-sort them."""
    width, height = quad_size(vertices)
    destination = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    matrix = cv2.getPerspectiveTransform(np.float32(vertices), destination)
    return cv2.warpPerspective(image, matrix, (width, height),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def eligible(word) -> bool:
    return bool(word.get("legible") and not word.get("vertical")
                and word.get("text", "").strip() and len(word["vertices"]) == 4)


def line_quad(line) -> np.ndarray:
    """Recover line direction from ordered WORD vertices, not line corner order."""
    if not eligible(line) or not line["words"] or not all(eligible(w) for w in line["words"]):
        raise ValueError("Line contains unsupported word geometry/layout")
    directions = []
    for word in line["words"]:
        quad_size(word["vertices"])
        v = np.asarray(word["vertices"], dtype=float)
        # Both edges run left-to-right in the word's reading coordinate frame.
        direction = (v[1] - v[0]) + (v[2] - v[3])
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            raise ValueError("Undefined reading direction")
        directions.append(direction / norm)
    u = np.mean(directions, axis=0)
    if np.linalg.norm(u) < 1e-6:
        raise ValueError("Mixed reading directions")
    u /= np.linalg.norm(u)
    if np.min(np.asarray(directions) @ u) < math.cos(math.radians(15)):
        raise ValueError("Inconsistent word directions")
    down = np.array([-u[1], u[0]])
    points = np.asarray(line["vertices"], dtype=float)
    x, y = points @ u, points @ down
    corners = np.array([u * a + down * b for a, b in
                        [(x.min(), y.min()), (x.max(), y.min()),
                         (x.max(), y.max()), (x.min(), y.max())]], dtype=np.float32)
    quad_size(corners)
    return corners


def size_bin(width, height) -> str:
    h = int(np.searchsorted(HEIGHT_EDGES, height, side="right"))
    r = int(np.searchsorted(RATIO_EDGES, width / height, side="right"))
    return f"h{h}_r{r}"


def _extract_images(root, annotations):
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    wanted = {item["image_id"] for item in annotations}
    missing = {key for key in wanted if not (images / f"{key}.jpg").exists()}
    if not missing:
        return images
    print(f"Extracting {len(missing)} source images", flush=True)
    seen = set()
    with tarfile.open(root / "raw" / "validation.tgz", "r|gz") as archive:
        for member in archive:
            # Extract only expected regular JPEGs to constructed destinations.
            name = Path(member.name).name
            key = Path(name).stem
            if not member.isfile() or key not in missing or Path(name).suffix.lower() != ".jpg":
                continue
            if key in seen:
                raise ValueError(f"Duplicate archive entry: {key}")
            seen.add(key)
            target = images / f"{key}.jpg"
            temporary = target.with_suffix(".jpg.part")
            with archive.extractfile(member) as incoming, temporary.open("wb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)
            temporary.replace(target)
    if missing - seen:
        raise ValueError(f"Missing source images: {sorted(missing - seen)[:5]}")
    return images


def _source_index(images, annotations, seed):
    """Group exact and conservative near duplicates before assigning splits."""
    parent = list(range(len(annotations)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    records, thumbnails, signatures = [], [], []
    exact = {}
    for i, ann in enumerate(annotations):
        path = images / f"{ann['image_id']}.jpg"
        with Image.open(path) as im:
            im = im.convert("RGB")
            if im.size != (ann["image_width"], ann["image_height"]):
                raise ValueError(f"Annotation/image size mismatch: {path}")
            pixel_hash = hashlib.sha256(im.tobytes()).hexdigest()
            thumb = np.asarray(im.resize((32, 32))).astype(np.int16)
            grey = np.asarray(im.convert("L").resize((9, 8)))
            signature = int.from_bytes(np.packbits(grey[:, 1:] > grey[:, :-1]).tobytes(), "big")
        if pixel_hash in exact:
            parent[find(i)] = find(exact[pixel_hash])
        else:
            for j, old_signature in enumerate(signatures):
                ratio = ann["image_width"] / ann["image_height"]
                old = records[j]
                if abs(math.log(ratio / (old["width"] / old["height"]))) > .02:
                    continue
                if (signature ^ old_signature).bit_count() <= 4 and np.mean(np.abs(thumb - thumbnails[j])) <= 4:
                    parent[find(i)] = find(j)
                    break
        exact[pixel_hash] = i
        signatures.append(signature)
        thumbnails.append(thumb)
        records.append({"source_id": ann["image_id"], "width": ann["image_width"],
                        "height": ann["image_height"], "pixel_sha256": pixel_hash,
                        "file_sha256": sha256(path)})
    groups = sorted({find(i) for i in range(len(records))})
    random.Random(seed).shuffle(groups)
    development = set(groups[:len(groups) // 2])
    folds = {group: i % 3 for i, group in enumerate(groups[:len(groups) // 2])}
    for i, record in enumerate(records):
        group = find(i)
        record.update(source_group=records[group]["source_id"],
                      split="development" if group in development else "validation",
                      fold=folds.get(group, -1))
    return records


def _target_sizes(folder, output):
    paths = sorted(folder.glob("*.png"))
    if not paths:
        raise ValueError(f"No PNGs in target dimension reference {folder}")
    cache = output / "target_sizes.json"
    # Dimensions only; never inspect or assign competition labels.
    inventory_hash = hashlib.sha256("\n".join(f"{p.name}:{p.stat().st_size}:{p.stat().st_mtime_ns}" for p in paths).encode()).hexdigest()
    if cache.exists():
        saved = read_json(cache)
        if saved["inventory_sha256"] == inventory_hash:
            return saved["sizes"], inventory_hash
    sizes = []
    for path in paths:
        with Image.open(path) as im:
            sizes.append(list(im.size))
    write_json(cache, {"inventory_sha256": inventory_hash, "sizes": sizes})
    return sizes, inventory_hash


def build(root: Path, output: Path, target_images: Path, count=5000, natural_count=1000, seed=42, source_cap=12):
    if (output / "manifest.csv").exists():
        raise ValueError("Benchmark already frozen. Use another --output directory to rebuild.")
    output.mkdir(parents=True, exist_ok=True)
    provenance = read_json(root / "raw" / "provenance.json")
    for filename, key in [("validation.jsonl.gz", "annotation"), ("validation.tgz", "archive")]:
        if sha256(root / "raw" / filename) != provenance[key + "_sha256"]:
            raise ValueError(f"Checksum mismatch: {filename}")
    with gzip.open(root / "raw" / "validation.jsonl.gz", "rt", encoding="utf-8") as handle:
        annotations = json.load(handle)["annotations"]
    if len({a["image_id"] for a in annotations}) != len(annotations):
        raise ValueError("Duplicate annotation source IDs")
    # Only hexadecimal source IDs are admitted as filesystem names.
    if any(not a["image_id"] or any(c not in "0123456789abcdef" for c in a["image_id"]) for a in annotations):
        raise ValueError("Unexpected source ID format")
    images = _extract_images(root, annotations)
    print("Indexing and grouping source images", flush=True)
    sources = _source_index(images, annotations, seed)
    source_map = {row["source_id"]: row for row in sources}
    write_csv(output / "sources.csv", sources)
    candidates, rejected = [], collections.Counter()
    for ann in annotations:
        source_id = ann["image_id"]
        for pi, paragraph in enumerate(ann["paragraphs"]):
            for li, line in enumerate(paragraph["lines"]):
                elements = [("word", wi, word) for wi, word in enumerate(line["words"])]
                elements.append(("line", -1, line))
                for kind, wi, element in elements:
                    try:
                        if not eligible(element):
                            raise ValueError("Illegible/vertical/non-quad/empty")
                        vertices = line_quad(element) if kind == "line" else np.float32(element["vertices"])
                        width, height = quad_size(vertices)
                    except ValueError as error:
                        rejected[f"{kind}: {error}"] += 1
                        continue
                    crop_id = f"{source_id}_p{pi}_l{li}_{kind}{wi}"
                    candidates.append({"crop_id": crop_id, "source_id": source_id, "crop_type": kind,
                                       "width": width, "height": height, "size_bin": size_bin(width, height),
                                       "vertices": vertices.tolist(), "text": element["text"]})
    print(f"Eligible crops: {len(candidates)}. Reading competition dimensions only.", flush=True)
    target, target_hash = _target_sizes(target_images, output)
    bins = collections.Counter(size_bin(w, h) for w, h in target)
    quotas = {b: int(count * n / len(target)) for b, n in bins.items()}
    remainder = count - sum(quotas.values())
    for b in sorted(bins, key=lambda b: (-(count * bins[b] / len(target) - quotas[b]), b))[:remainder]:
        quotas[b] += 1
    rng = random.Random(seed)
    rng.shuffle(candidates)
    source_counts, matched_counts = collections.Counter(), collections.Counter()
    selected, selected_ids, pixel_hashes = [], set(), set()
    crops_dir = output / "crops"
    crops_dir.mkdir(exist_ok=True)
    image_cache = collections.OrderedDict()

    def accept(candidate, cohort):
        source = source_map[candidate["source_id"]]
        if source_counts[source["source_group"]] >= source_cap:
            return False
        if candidate["crop_id"] in selected_ids:
            return False
        key = candidate["source_id"]
        if key not in image_cache:
            with Image.open(images / f"{key}.jpg") as im:
                image_cache[key] = np.asarray(im.convert("RGB"))
            if len(image_cache) > 16:
                image_cache.popitem(last=False)
        crop = rectify(image_cache[key], candidate["vertices"])
        # Rotation-canonical hashes keep identical crops out of opposing splits.
        hashes = [hashlib.sha256(crop.tobytes()).hexdigest(),
                  hashlib.sha256(np.rot90(crop, 2).tobytes()).hexdigest()]
        pixel_hash = min(hashes)
        if pixel_hash in pixel_hashes:
            rejected["duplicate crop pixels"] += 1
            return False
        pixel_hashes.add(pixel_hash)
        path = crops_dir / f"{candidate['crop_id']}.png"
        Image.fromarray(crop).save(path)
        selected.append({**candidate, "cohort": cohort, "source_group": source["source_group"],
                         "split": source["split"], "fold": source["fold"],
                         "image_path": path.relative_to(output).as_posix(), "pixel_sha256": pixel_hash,
                         "file_sha256": sha256(path)})
        selected_ids.add(candidate["crop_id"])
        source_counts[source["source_group"]] += 1
        return True

    for candidate in candidates:
        b = candidate["size_bin"]
        if matched_counts[b] < quotas.get(b, 0) and accept(candidate, "matched"):
            matched_counts[b] += 1
    natural_selected = 0
    # A disjoint randomly sampled reference cohort, with the same source cap.
    rng.shuffle(candidates)
    for candidate in candidates:
        if natural_selected >= natural_count:
            break
        if accept(candidate, "natural"):
            natural_selected += 1
    if not selected:
        raise ValueError("No valid crops were selected")
    manifest = []
    for crop in selected:
        for rotation in [0, 180]:
            manifest.append({"example_id": f"{crop['crop_id']}_r{rotation}",
                             "pair_id": crop["crop_id"], "source_id": crop["source_id"],
                             "source_group": crop["source_group"], "crop_type": crop["crop_type"],
                             "cohort": crop["cohort"], "split": crop["split"], "fold": crop["fold"],
                             "width": crop["width"], "height": crop["height"], "size_bin": crop["size_bin"],
                             "rotation": rotation, "label": int(rotation == 180),
                             "image_path": crop["image_path"], "file_sha256": crop["file_sha256"]})
    write_json(output / "crop_annotations.json", selected)
    write_csv(output / "manifest.csv", manifest)
    a = np.array([[c["width"], c["height"]] for c in selected if c["cohort"] == "matched"])
    summary = {"dataset": "HierText official validation split", "provenance": provenance,
               "seed": seed, "requested_matched_crops": count, "matched_crops": sum(matched_counts.values()),
               "natural_crops": natural_selected, "labeled_examples": len(manifest),
               "source_images": len(sources), "source_groups": len({s['source_group'] for s in sources}),
               "selected_source_groups": len(source_counts), "source_cap": source_cap,
               "target_inventory_sha256": target_hash, "height_edges": HEIGHT_EDGES, "aspect_ratio_edges": RATIO_EDGES,
               "target_bin_counts": dict(bins), "requested_bin_quotas": quotas,
               "selected_bin_counts": dict(matched_counts),
               "unfilled_quotas": {b: n - matched_counts[b] for b, n in quotas.items() if n > matched_counts[b]},
               "matched_median_width_height": np.median(a, axis=0).tolist(),
               "crop_counts": dict(collections.Counter(f"{c['cohort']}/{c['split']}/{c['crop_type']}" for c in selected)),
               "rejections": dict(rejected), "manifest_sha256": sha256(output / "manifest.csv"),
               "limitations": ["English-dominant dataset; no claim of Cyrillic coverage.",
                               "Ground-truth rectified crops differ from detector crops.",
                               "Near-duplicate detection is conservative, not exhaustive.",
                               "Natural cohort is source-capped and disjoint from matched cohort."]}
    write_json(output / "summary.json", summary)
    # Representative extraction QA only; no human labels are added or changed.
    chosen = random.Random(seed).sample(selected, min(48, len(selected)))
    sheet = Image.new("RGB", (1200, 120 * math.ceil(len(chosen) / 4)), "#dce0e5")
    draw = ImageDraw.Draw(sheet)
    for i, crop in enumerate(chosen):
        x, y = i % 4 * 300, i // 4 * 120
        draw.text((x + 5, y + 4), f"{crop['crop_type']} {crop['width']}x{crop['height']} {crop['split']}", fill="black")
        with Image.open(output / crop["image_path"]) as im:
            preview = ImageOps.contain(im, (290, 92))
            sheet.paste(preview, (x + 5, y + 24))
    sheet.save(output / "contact_sheet.jpg")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download HierText or build upright text crops and rotation-pair labels.")
    parser.add_argument("command", choices=["download", "build"])
    parser.add_argument("--root", type=Path, default=Path("data/hiertext"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/hiertext"))
    parser.add_argument("--target-images", type=Path, default=Path("test/images"))
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--natural-count", type=int, default=1000)
    parser.add_argument("--source-cap", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.command == "download":
        download(args.root)
    else:
        if min(args.count, args.source_cap) < 1 or args.natural_count < 0:
            parser.error("Crop count/source cap must be positive; natural count must be nonnegative")
        build(args.root, args.output, args.target_images, args.count,
              args.natural_count, args.seed, args.source_cap)
