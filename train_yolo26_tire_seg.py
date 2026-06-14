"""
train_yolo26_tire_seg.py

Standalone training script for the tire-tread workflow discussed in chat.

Purpose:
    1. Download a Roboflow object-detection dataset.
    2. Inspect the exported COCO annotation JSONs for hidden polygon/mask annotations.
    3. Convert any real polygon/RLE annotations into Ultralytics YOLO segmentation format.
    4. Optionally filter classes to NORMAL_Tyres and BALD_Tyres.
    5. Optionally fall back to rectangular box polygons if no true masks are exposed.
    6. Train a YOLO26 segmentation model.
    7. Validate the best checkpoint.
    8. Save best.pt, last.pt, data.yaml, metrics CSV and conversion summary.

Why this exists:
    The Roboflow tire tread dataset can visually appear to show polygon/mask-like overlays
    in the website, while the project itself is classified as object detection. In that case,
    normal YOLO exports may contain only box labels:
        class_id x_center y_center width height

    YOLO segmentation training expects polygon labels:
        class_id x1 y1 x2 y2 x3 y3 ...

    This script downloads the allowed COCO object-detection export and checks whether the
    raw JSON contains any segmentation/polygon fields that can be converted.

Install:
    pip install -r requirements_train.txt

Minimum packages:
    pip install ultralytics roboflow opencv-python numpy pandas pyyaml pycocotools

Example:
    export ROBOFLOW_API_KEY="your_key"

    python train_yolo26_tire_seg.py \
        --workspace mark-aft7n \
        --project tire-tread \
        --version 1 \
        --model yolo26n-seg.pt \
        --epochs 80 \
        --imgsz 640 \
        --batch 8 \
        --filter-normal-bald

If the script says no real polygon objects were found, either:
    1. Train YOLO detection instead,
    2. Use YOLO boxes + SAM/SAM2 pseudo-masks,
    3. Or rerun with --allow-bbox-rectangle-fallback to train rectangle masks
       knowing that this is not true tread segmentation.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import yaml
from pycocotools import mask as mask_utils
from roboflow import Roboflow
from ultralytics import YOLO


# Common image extensions used when copying converted datasets.
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------
# Basic file and JSON helpers
# ---------------------------------------------------------------------

def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file as a Python dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    """Save a Python object as pretty JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def find_jsons(root: Path) -> List[Path]:
    """Find all JSON files below a directory."""
    return sorted(root.rglob("*.json"))


def split_name_from_json_path(json_path: Path) -> str:
    """
    Infer YOLO split name from a COCO JSON path.

    Roboflow commonly exports:
        train/_annotations.coco.json
        valid/_annotations.coco.json
        test/_annotations.coco.json
    """
    parent = json_path.parent.name.lower()

    if parent in {"valid", "val", "validation"}:
        return "val"
    if parent == "test":
        return "test"
    return "train"


def find_image_file(split_dir: Path, coco_root: Path, file_name: str) -> Optional[Path]:
    """
    Find the image file referenced by a COCO image record.

    First check the JSON's split directory, then search the full dataset root.
    """
    direct = split_dir / file_name
    if direct.exists():
        return direct

    basename = Path(file_name).name

    split_matches = list(split_dir.rglob(basename))
    if split_matches:
        return split_matches[0]

    root_matches = list(coco_root.rglob(basename))
    if root_matches:
        return root_matches[0]

    return None


# ---------------------------------------------------------------------
# Roboflow download
# ---------------------------------------------------------------------

def download_roboflow_coco(
    api_key: str,
    workspace: str,
    project: str,
    version: int,
    export_format: str = "coco",
) -> Path:
    """
    Download a Roboflow dataset.

    For object-detection projects, "coco" is allowed while "coco-segmentation"
    usually is not. We intentionally use "coco" and inspect whether hidden
    segmentation fields exist.
    """
    rf = Roboflow(api_key=api_key)
    project_obj = rf.workspace(workspace).project(project)
    version_obj = project_obj.version(version)

    dataset = version_obj.download(export_format)
    return Path(dataset.location)


# ---------------------------------------------------------------------
# Annotation schema inspection
# ---------------------------------------------------------------------

def scan_annotation_schema(json_paths: Iterable[Path]) -> Dict[str, Any]:
    """
    Count which annotation keys are present and whether segmentation-like fields exist.

    Useful for diagnosing whether the Roboflow export contains real polygons/masks.
    """
    key_counts = Counter()
    segmentation_count = 0
    nonempty_segmentation_count = 0
    rle_count = 0
    polygon_list_count = 0
    bbox_count = 0
    candidate_polygon_keys = Counter()

    possible_keys = [
        "segmentation",
        "polygon",
        "polygons",
        "points",
        "vertices",
        "mask",
        "masks",
        "path",
        "paths",
        "contour",
        "contours",
    ]

    for jp in json_paths:
        data = load_json(jp)
        for ann in data.get("annotations", []):
            key_counts.update(ann.keys())

            if ann.get("bbox") is not None:
                bbox_count += 1

            seg = ann.get("segmentation", None)
            if seg is not None:
                segmentation_count += 1
                if seg not in ([], {}, "", None):
                    nonempty_segmentation_count += 1

                    if isinstance(seg, dict):
                        rle_count += 1
                    elif isinstance(seg, list):
                        # Standard COCO polygons are usually list-of-flat-lists.
                        if any(isinstance(x, list) and len(x) >= 6 for x in seg):
                            polygon_list_count += 1
                        # Some exporters may store one flat polygon directly.
                        elif len(seg) >= 6 and all(isinstance(x, (int, float)) for x in seg):
                            polygon_list_count += 1

            for key in possible_keys:
                if key in ann and ann.get(key) not in (None, [], {}, ""):
                    candidate_polygon_keys[key] += 1

    return {
        "annotation_key_counts": dict(key_counts),
        "bbox_count": bbox_count,
        "segmentation_count": segmentation_count,
        "nonempty_segmentation_count": nonempty_segmentation_count,
        "rle_count": rle_count,
        "polygon_list_count": polygon_list_count,
        "candidate_polygon_keys": dict(candidate_polygon_keys),
    }


def print_dataset_overview(json_paths: List[Path]) -> None:
    """Print split/category/annotation overview for debugging."""
    print("\nFound COCO JSON files:")
    for p in json_paths:
        print(f"  - {p}")

    for jp in json_paths:
        data = load_json(jp)
        images = data.get("images", [])
        annotations = data.get("annotations", [])
        categories = data.get("categories", [])

        print(f"\n== {jp} ==")
        print(f"images: {len(images)} | annotations: {len(annotations)} | categories: {len(categories)}")
        print("categories:")
        for cat in categories:
            print(f"  {cat}")

        if annotations:
            print("first annotation keys:", list(annotations[0].keys()))
            print("first annotation sample:")
            print(json.dumps(annotations[0], indent=2)[:2000])


# ---------------------------------------------------------------------
# Polygon and RLE conversion helpers
# ---------------------------------------------------------------------

def polygon_area(poly: List[float]) -> float:
    """
    Compute polygon area in pixel units from flat [x1,y1,x2,y2,...] coordinates.
    """
    if len(poly) < 6:
        return 0.0

    pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
    x = pts[:, 0]
    y = pts[:, 1]

    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def bbox_to_rectangle_polygon(bbox: List[float]) -> List[float]:
    """
    Convert a COCO bbox [x, y, width, height] into a rectangle polygon.

    This is only a fallback and is not true segmentation.
    """
    x, y, w, h = map(float, bbox)
    return [x, y, x + w, y, x + w, y + h, x, y + h]


def points_to_flat_polygon(obj: Any) -> List[float]:
    """
    Convert a few common point/polygon representations into a flat polygon.

    Supported examples:
        [x1, y1, x2, y2, ...]
        [[x1, y1], [x2, y2], ...]
        [{"x": x1, "y": y1}, {"x": x2, "y": y2}, ...]
    """
    if obj is None:
        return []

    if isinstance(obj, list):
        # Already flat.
        if len(obj) >= 6 and all(isinstance(v, (int, float)) for v in obj):
            return [float(v) for v in obj]

        # List of points.
        flat: List[float] = []
        ok = True

        for item in obj:
            if isinstance(item, dict) and "x" in item and "y" in item:
                flat.extend([float(item["x"]), float(item["y"])])
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                flat.extend([float(item[0]), float(item[1])])
            else:
                ok = False
                break

        if ok and len(flat) >= 6:
            return flat

    return []


def rle_to_polygons(segmentation: Dict[str, Any], height: int, width: int) -> List[List[float]]:
    """
    Convert a COCO RLE mask into external contour polygons.
    """
    polygons: List[List[float]] = []

    try:
        rle = segmentation

        # Uncompressed COCO RLE may need conversion.
        if isinstance(rle.get("counts"), list):
            rle = mask_utils.frPyObjects(rle, height, width)

        decoded = mask_utils.decode(rle)

        # Some RLE decodes to HxWxN.
        if decoded.ndim == 3:
            decoded = np.any(decoded, axis=2).astype(np.uint8)
        else:
            decoded = decoded.astype(np.uint8)

        contours, _ = cv2.findContours(decoded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cnt.shape[0] < 3:
                continue

            # Simplify contour slightly to avoid extremely long YOLO label rows.
            epsilon = 0.002 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            if approx.shape[0] >= 3:
                poly = approx.reshape(-1, 2).astype(float).flatten().tolist()
                polygons.append(poly)

    except Exception:
        # If RLE decode fails, just return no polygons.
        pass

    return polygons


def annotation_to_polygons(
    ann: Dict[str, Any],
    height: int,
    width: int,
    allow_bbox_fallback: bool = False,
) -> Tuple[List[List[float]], bool]:
    """
    Extract polygon(s) from a COCO annotation.

    Returns:
        polygons, used_bbox_fallback

    The function checks:
        - standard COCO segmentation lists
        - COCO RLE segmentation dicts
        - common non-standard polygon-like keys
        - optional bbox rectangle fallback
    """
    polygons: List[List[float]] = []
    used_bbox_fallback = False

    # Standard COCO segmentation.
    seg = ann.get("segmentation", None)

    if isinstance(seg, list) and seg:
        # Sometimes stored as one flat polygon directly.
        if all(isinstance(x, (int, float)) for x in seg) and len(seg) >= 6:
            polygons.append([float(x) for x in seg])
        else:
            # Standard case: list of polygons.
            for part in seg:
                flat = points_to_flat_polygon(part)
                if len(flat) >= 6:
                    polygons.append(flat)

    elif isinstance(seg, dict) and seg:
        polygons.extend(rle_to_polygons(seg, height, width))

    # Try possible non-standard polygon keys.
    for key in ["polygon", "polygons", "points", "vertices", "contour", "contours", "path", "paths"]:
        if key not in ann or ann.get(key) in (None, [], {}, ""):
            continue

        val = ann.get(key)

        if isinstance(val, list):
            if val and all(isinstance(x, (int, float)) for x in val):
                flat = points_to_flat_polygon(val)
                if len(flat) >= 6:
                    polygons.append(flat)
            else:
                for part in val:
                    flat = points_to_flat_polygon(part)
                    if len(flat) >= 6:
                        polygons.append(flat)
        else:
            flat = points_to_flat_polygon(val)
            if len(flat) >= 6:
                polygons.append(flat)

    # Optional fallback: convert bbox into a rectangular polygon.
    if not polygons and allow_bbox_fallback and ann.get("bbox") is not None:
        polygons.append(bbox_to_rectangle_polygon(ann["bbox"]))
        used_bbox_fallback = True

    return polygons, used_bbox_fallback


# ---------------------------------------------------------------------
# COCO -> YOLO segmentation conversion
# ---------------------------------------------------------------------

def collect_category_names(json_paths: Iterable[Path]) -> Dict[int, str]:
    """Collect category id -> category name across all COCO JSONs."""
    category_name_by_id: Dict[int, str] = {}

    for jp in json_paths:
        data = load_json(jp)
        for cat in data.get("categories", []):
            category_name_by_id[int(cat["id"])] = cat["name"]

    return category_name_by_id


def convert_coco_to_yolo_seg(
    coco_root: Path,
    out_root: Path,
    keep_classes: Optional[List[str]] = None,
    allow_bbox_fallback: bool = False,
    min_polygon_points: int = 3,
    min_area_px: float = 10.0,
) -> Tuple[Path, Dict[str, Any], Dict[str, Any]]:
    """
    Convert COCO object-detection export into YOLO segmentation format when possible.

    If keep_classes is provided, only those classes are kept and remapped to 0..N-1.

    Output dataset layout:
        out_root/
            data.yaml
            train/images/*.jpg
            train/labels/*.txt
            val/images/*.jpg
            val/labels/*.txt
            test/images/*.jpg
            test/labels/*.txt
    """
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    json_paths = find_jsons(coco_root)
    if not json_paths:
        raise FileNotFoundError(f"No JSON files found under {coco_root}")

    category_name_by_id = collect_category_names(json_paths)
    available_names = sorted(set(category_name_by_id.values()))

    print("\nAvailable classes:")
    for name in available_names:
        print(f"  - {name}")

    if keep_classes:
        missing = [c for c in keep_classes if c not in available_names]
        if missing:
            raise ValueError(f"Requested keep classes not found: {missing}\nAvailable: {available_names}")
        final_names = list(keep_classes)
    else:
        final_names = available_names

    name_to_new_id = {name: idx for idx, name in enumerate(final_names)}

    print("\nFinal YOLO classes:")
    for name, idx in name_to_new_id.items():
        print(f"  {idx}: {name}")

    split_counts: Dict[str, Any] = {}
    total_real_polygon_objects = 0
    total_bbox_fallback_objects = 0

    for json_path in json_paths:
        split = split_name_from_json_path(json_path)
        split_dir = json_path.parent
        data = load_json(json_path)

        images = {int(img["id"]): img for img in data.get("images", [])}

        anns_by_image: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for ann in data.get("annotations", []):
            anns_by_image[int(ann["image_id"])].append(ann)

        out_img_dir = out_root / split / "images"
        out_lbl_dir = out_root / split / "labels"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        kept_images = 0
        kept_objects = 0
        real_polygon_objects = 0
        bbox_fallback_objects = 0
        skipped_no_class = 0
        skipped_no_polygon = 0
        missing_images = 0

        for image_id, img in images.items():
            file_name = img["file_name"]
            width = int(img.get("width", 0))
            height = int(img.get("height", 0))

            if width <= 0 or height <= 0:
                continue

            img_path = find_image_file(split_dir, coco_root, file_name)
            if img_path is None:
                missing_images += 1
                continue

            yolo_lines: List[str] = []

            for ann in anns_by_image.get(image_id, []):
                cat_id = int(ann["category_id"])
                cat_name = category_name_by_id.get(cat_id)

                if cat_name not in name_to_new_id:
                    skipped_no_class += 1
                    continue

                polygons, used_bbox_fallback = annotation_to_polygons(
                    ann,
                    height=height,
                    width=width,
                    allow_bbox_fallback=allow_bbox_fallback,
                )

                if not polygons:
                    skipped_no_polygon += 1
                    continue

                for poly in polygons:
                    if len(poly) < min_polygon_points * 2:
                        continue

                    if polygon_area(poly) < min_area_px:
                        continue

                    coords = np.array(poly, dtype=np.float32).reshape(-1, 2)

                    # Normalize from pixel coordinates to 0..1 YOLO coordinates.
                    coords[:, 0] = np.clip(coords[:, 0] / width, 0.0, 1.0)
                    coords[:, 1] = np.clip(coords[:, 1] / height, 0.0, 1.0)

                    if coords.shape[0] < min_polygon_points:
                        continue

                    cls_id = name_to_new_id[cat_name]
                    flat = coords.flatten().tolist()

                    line = str(cls_id) + " " + " ".join(f"{v:.6f}" for v in flat)
                    yolo_lines.append(line)

                    if used_bbox_fallback:
                        bbox_fallback_objects += 1
                    else:
                        real_polygon_objects += 1

            if not yolo_lines:
                continue

            # Copy image and write its YOLO segmentation label file.
            out_img_path = out_img_dir / Path(file_name).name
            shutil.copy2(img_path, out_img_path)

            label_path = out_lbl_dir / (Path(file_name).stem + ".txt")
            with open(label_path, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines) + "\n")

            kept_images += 1
            kept_objects += len(yolo_lines)

        split_counts[split] = {
            "kept_images": kept_images,
            "kept_objects": kept_objects,
            "real_polygon_objects": real_polygon_objects,
            "bbox_fallback_objects": bbox_fallback_objects,
            "skipped_no_class": skipped_no_class,
            "skipped_no_polygon": skipped_no_polygon,
            "missing_images": missing_images,
        }

        total_real_polygon_objects += real_polygon_objects
        total_bbox_fallback_objects += bbox_fallback_objects

    # Build Ultralytics data.yaml.
    data_yaml: Dict[str, Any] = {
        "path": str(out_root),
        "train": "train/images",
        "val": "val/images" if (out_root / "val" / "images").exists() else "train/images",
        "names": {idx: name for idx, name in enumerate(final_names)},
    }

    if (out_root / "test" / "images").exists():
        data_yaml["test"] = "test/images"

    data_yaml_path = out_root / "data.yaml"
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    conversion_meta = {
        "total_real_polygon_objects": total_real_polygon_objects,
        "total_bbox_fallback_objects": total_bbox_fallback_objects,
        "allow_bbox_fallback": allow_bbox_fallback,
        "final_names": final_names,
    }

    return data_yaml_path, split_counts, conversion_meta


def validate_yolo_seg_labels(data_yaml_path: Path) -> Dict[str, int]:
    """
    Basic check that label rows look like YOLO segmentation rows.

    Detection row:
        class + 4 coords = 5 values

    Segmentation row:
        class + polygon coords = >5 values
    """
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg.get("path", data_yaml_path.parent))

    counts = {
        "files_checked": 0,
        "rows": 0,
        "seg_rows": 0,
        "box_like_rows": 0,
    }

    for split in ["train", "val", "test"]:
        if split not in cfg:
            continue

        img_dir = root / cfg[split]
        label_dir = Path(str(img_dir).replace("/images", "/labels"))

        if not label_dir.exists():
            continue

        for lbl in label_dir.rglob("*.txt"):
            counts["files_checked"] += 1

            with open(lbl, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue

                    counts["rows"] += 1

                    if len(parts) > 5:
                        counts["seg_rows"] += 1
                    else:
                        counts["box_like_rows"] += 1

    return counts


# ---------------------------------------------------------------------
# Training and saving
# ---------------------------------------------------------------------

def train_yolo_seg(
    data_yaml_path: Path,
    model_weights: str,
    epochs: int,
    imgsz: int,
    batch: int,
    patience: int,
    project_dir: Path,
    run_name: str,
    workers: int = 2,
) -> Tuple[YOLO, Path]:
    """
    Train a YOLO segmentation model with Ultralytics.

    Returns:
        trained model object and run directory.
    """
    model = YOLO(model_weights)

    model.train(
        data=str(data_yaml_path),
        task="segment",
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        cache=False,
        workers=workers,
        plots=True,
    )

    run_dir = project_dir / run_name
    return model, run_dir


def validate_best_checkpoint(best_pt: Path, data_yaml_path: Path, imgsz: int, batch: int) -> Dict[str, Any]:
    """
    Validate best.pt and return metric dictionary when available.
    """
    model = YOLO(str(best_pt))
    metrics = model.val(
        data=str(data_yaml_path),
        task="segment",
        imgsz=imgsz,
        batch=batch,
        plots=True,
    )

    try:
        return dict(metrics.results_dict)
    except Exception:
        return {"metrics_repr": repr(metrics)}


def save_training_artifacts(
    run_dir: Path,
    data_yaml_path: Path,
    output_dir: Path,
    conversion_summary: Dict[str, Any],
    validation_metrics: Dict[str, Any],
) -> Path:
    """
    Save important training artifacts into a compact output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_dir = run_dir / "weights"
    best_pt = weights_dir / "best.pt"
    last_pt = weights_dir / "last.pt"
    results_csv = run_dir / "results.csv"

    if best_pt.exists():
        shutil.copy2(best_pt, output_dir / "best.pt")
    else:
        print(f"WARNING: best.pt not found at {best_pt}")

    if last_pt.exists():
        shutil.copy2(last_pt, output_dir / "last.pt")
    else:
        print(f"WARNING: last.pt not found at {last_pt}")

    if data_yaml_path.exists():
        shutil.copy2(data_yaml_path, output_dir / "data.yaml")

    if results_csv.exists():
        shutil.copy2(results_csv, output_dir / "results.csv")

    save_json(conversion_summary, output_dir / "conversion_summary.json")
    save_json(validation_metrics, output_dir / "validation_metrics.json")

    zip_path = output_dir.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()

    shutil.make_archive(str(zip_path).replace(".zip", ""), "zip", output_dir)

    return zip_path


def print_results_tail(run_dir: Path, n: int = 10) -> None:
    """Print last N rows of Ultralytics results.csv, if it exists."""
    results_csv = run_dir / "results.csv"

    if not results_csv.exists():
        print("No results.csv found.")
        return

    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]

    print("\nLast training metrics:")
    print(df.tail(n).to_string(index=False))

    print("\nAvailable metric columns:")
    for col in df.columns:
        print(f"  - {col}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO26-seg on Roboflow tire tread data.")

    # Roboflow arguments.
    parser.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY", ""), help="Roboflow API key. Defaults to ROBOFLOW_API_KEY env var.")
    parser.add_argument("--workspace", default="mark-aft7n", help="Roboflow workspace id.")
    parser.add_argument("--project", default="tire-tread", help="Roboflow project id.")
    parser.add_argument("--version", type=int, default=1, help="Roboflow dataset version number.")
    parser.add_argument("--format", default="coco", help="Roboflow download format. For this workflow use 'coco'.")

    # Class filtering.
    parser.add_argument("--filter-normal-bald", action="store_true", help="Keep only NORMAL_Tyres and BALD_Tyres.")
    parser.add_argument("--keep-classes", nargs="*", default=["NORMAL_Tyres", "BALD_Tyres"], help="Classes to keep when filtering.")

    # Conversion behavior.
    parser.add_argument("--allow-bbox-rectangle-fallback", action="store_true", help="If no polygons are found, convert bboxes into rectangle polygons. Not true segmentation.")
    parser.add_argument("--min-area-px", type=float, default=10.0, help="Minimum polygon area in pixels.")
    parser.add_argument("--converted-dir", default="/content/tire_tread_yolo_seg_converted", help="Where to write converted YOLO-seg dataset.")

    # Training configuration.
    parser.add_argument("--model", default="yolo26n-seg.pt", help="YOLO segmentation weights, e.g. yolo26n-seg.pt.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--project-dir", default="/content/yolo26_tire_runs")
    parser.add_argument("--run-name", default="tire_tread_yolo26_seg")

    # Output.
    parser.add_argument("--save-dir", default="/content/trained_tire_tread_yolo26_seg", help="Where to save best.pt, last.pt, metrics and summaries.")
    parser.add_argument("--skip-train", action="store_true", help="Only download/convert/validate labels; do not train.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.api_key:
        raise ValueError("Roboflow API key missing. Pass --api-key or set ROBOFLOW_API_KEY.")

    # -----------------------------------------------------------------
    # 1. Download Roboflow dataset in allowed COCO object-detection format.
    # -----------------------------------------------------------------
    print("\nDownloading Roboflow dataset...")
    coco_root = download_roboflow_coco(
        api_key=args.api_key,
        workspace=args.workspace,
        project=args.project,
        version=args.version,
        export_format=args.format,
    )
    print(f"Downloaded dataset to: {coco_root}")

    # -----------------------------------------------------------------
    # 2. Inspect annotations for hidden polygons/masks.
    # -----------------------------------------------------------------
    json_paths = find_jsons(coco_root)
    if not json_paths:
        raise FileNotFoundError(f"No JSON annotation files found under {coco_root}")

    print_dataset_overview(json_paths)

    scan = scan_annotation_schema(json_paths)
    print("\nAnnotation schema scan:")
    print(json.dumps(scan, indent=2)[:5000])

    # -----------------------------------------------------------------
    # 3. Convert COCO JSON to YOLO segmentation labels.
    # -----------------------------------------------------------------
    keep_classes = args.keep_classes if args.filter_normal_bald else None

    print("\nConverting annotations to YOLO segmentation format...")
    data_yaml_path, split_counts, conversion_meta = convert_coco_to_yolo_seg(
        coco_root=coco_root,
        out_root=Path(args.converted_dir),
        keep_classes=keep_classes,
        allow_bbox_fallback=args.allow_bbox_rectangle_fallback,
        min_area_px=args.min_area_px,
    )

    print("\nConversion split counts:")
    print(json.dumps(split_counts, indent=2))

    print("\nConversion meta:")
    print(json.dumps(conversion_meta, indent=2))

    # If no real polygons were found and fallback is disabled, stop early with a clear message.
    if conversion_meta["total_real_polygon_objects"] == 0:
        if conversion_meta["total_bbox_fallback_objects"] > 0:
            print("\nWARNING: No real polygons found. Labels were created from bounding boxes only.")
            print("This is rectangle-mask training, not true segmentation.")
        else:
            raise ValueError(
                "No real polygon/mask annotations found in the COCO export. "
                "Roboflow may show masks visually in the UI but not expose them through the object-detection export. "
                "Rerun with --allow-bbox-rectangle-fallback if you intentionally want rectangle masks, "
                "or use YOLO detection + SAM/SAM2 pseudo-masks."
            )

    # -----------------------------------------------------------------
    # 4. Validate converted YOLO segmentation label structure.
    # -----------------------------------------------------------------
    label_counts = validate_yolo_seg_labels(data_yaml_path)
    print("\nConverted label validation:")
    print(json.dumps(label_counts, indent=2))

    if label_counts["rows"] == 0:
        raise ValueError("No YOLO label rows found after conversion.")

    if label_counts["seg_rows"] == 0:
        raise ValueError("No YOLO segmentation rows found after conversion.")

    conversion_summary = {
        "coco_root": str(coco_root),
        "data_yaml_path": str(data_yaml_path),
        "scan": scan,
        "split_counts": split_counts,
        "conversion_meta": conversion_meta,
        "label_counts": label_counts,
        "args": vars(args),
    }

    # Save conversion summary even if skipping train.
    converted_summary_path = Path(args.converted_dir) / "conversion_summary.json"
    save_json(conversion_summary, converted_summary_path)
    print(f"\nSaved conversion summary to: {converted_summary_path}")

    if args.skip_train:
        print("\n--skip-train was set. Stopping after conversion.")
        return

    # -----------------------------------------------------------------
    # 5. Train YOLO segmentation.
    # -----------------------------------------------------------------
    print("\nStarting YOLO segmentation training...")
    _, run_dir = train_yolo_seg(
        data_yaml_path=data_yaml_path,
        model_weights=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        project_dir=Path(args.project_dir),
        run_name=args.run_name,
        workers=args.workers,
    )

    print(f"\nTraining run directory: {run_dir}")
    print_results_tail(run_dir)

    # -----------------------------------------------------------------
    # 6. Validate best checkpoint.
    # -----------------------------------------------------------------
    best_pt = run_dir / "weights" / "best.pt"
    if best_pt.exists():
        print("\nValidating best checkpoint...")
        validation_metrics = validate_best_checkpoint(
            best_pt=best_pt,
            data_yaml_path=data_yaml_path,
            imgsz=args.imgsz,
            batch=args.batch,
        )
        print("\nValidation metrics:")
        print(json.dumps(validation_metrics, indent=2))
    else:
        validation_metrics = {"error": f"best.pt not found at {best_pt}"}
        print(f"\nWARNING: {validation_metrics['error']}")

    # -----------------------------------------------------------------
    # 7. Save artifacts.
    # -----------------------------------------------------------------
    print("\nSaving training artifacts...")
    zip_path = save_training_artifacts(
        run_dir=run_dir,
        data_yaml_path=data_yaml_path,
        output_dir=Path(args.save_dir),
        conversion_summary=conversion_summary,
        validation_metrics=validation_metrics,
    )

    print(f"\nSaved artifacts to: {args.save_dir}")
    print(f"Saved zip to: {zip_path}")


if __name__ == "__main__":
    main()
