"""
tire_patch_estimator.py

Quick standalone script for:

1) Loading a trained YOLO segmentation model.
2) Taking one image and running parent tire/tread segmentation.
3) Splitting each detected parent mask into a near-square grid, 4 patches by default.
4) Running independent YOLO inference on each patch crop.
5) Returning a structured result:
   - overall parent prediction
   - per-patch predictions and rough tread-depth proxy values
6) Rendering the image with the parent mask and patch boxes drawn on top.

Install:
    pip install ultralytics opencv-python numpy

Example:
    python tire_patch_estimator.py \
        --model trained_tire_tread_yolo26_seg/best.pt \
        --image tire.jpg \
        --out tire_patches_rendered.jpg
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


# -----------------------------
# Default rough depth priors.
# These are NOT physical gauge measurements.
# They are visual proxy values mapped from model classes.
# -----------------------------
NORMAL_DEPTH_MM = 5.5
BALD_DEPTH_MM = 1.2
BAD_DEPTH_MM = 3.0
BASE_UNCERTAINTY_MM = 1.2


def get_class_name(names: Any, class_id: int) -> str:
    """Safely resolve a YOLO class id into a class name."""
    if isinstance(names, dict):
        return str(names.get(int(class_id), class_id))
    if isinstance(names, list):
        return str(names[int(class_id)])
    return str(class_id)


def class_to_depth_prior(class_name: str) -> Tuple[float, float, str]:
    """
    Map a model class name to:
      - rough depth prior in mm
      - rough P(bald)
      - human-readable status

    This assumes classes like:
      NORMAL_Tyres
      BALD_Tyres
      BAD_Tyres

    Edit this function if your class names differ.
    """
    name = class_name.lower()

    if "normal" in name or "good" in name:
        return NORMAL_DEPTH_MM, 0.05, "normal-looking"

    if "bald" in name:
        return BALD_DEPTH_MM, 0.95, "bald-looking"

    if "bad" in name or "damage" in name or "defect" in name:
        return BAD_DEPTH_MM, 0.50, "damage/uncertain"

    return BAD_DEPTH_MM, 0.50, "unknown-class"


def depth_from_p_bald(p_bald: float) -> float:
    """
    Convert P(bald) into a rough depth proxy by interpolating between:
      normal depth prior and bald depth prior.
    """
    p = float(np.clip(p_bald, 0.0, 1.0))
    return (1.0 - p) * NORMAL_DEPTH_MM + p * BALD_DEPTH_MM


def uncertainty_from_patch(
    p_bald: float,
    mask_fraction: float,
    inference_confidence: float,
    used_parent_fallback: bool,
) -> float:
    """
    Simple uncertainty estimate.

    Increases when:
      - patch prediction is ambiguous, near P(bald)=0.5
      - patch contains little parent-mask area
      - patch inference confidence is low
      - patch had to fall back to parent prediction
    """
    p = float(np.clip(p_bald, 0.0, 1.0))

    # Highest ambiguity near 0.5, lowest near 0 or 1.
    probability_ambiguity = 0.6 * (1.0 - abs(p - 0.5) * 2.0)

    # Penalize patches that barely overlap the parent mask.
    coverage_penalty = 0.4 * max(0.0, 0.35 - float(mask_fraction)) / 0.35

    # Penalize low-confidence patch inference.
    conf_penalty = 0.5 * max(0.0, 0.50 - float(inference_confidence)) / 0.50

    # Fallback means this patch was not independently detected.
    fallback_penalty = 0.5 if used_parent_fallback else 0.0

    return float(BASE_UNCERTAINTY_MM + probability_ambiguity + coverage_penalty + conf_penalty + fallback_penalty)


def ensure_binary_mask(mask: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    """Convert YOLO mask tensor/array into a binary uint8 mask matching the image size."""
    h, w = shape_hw

    mask = np.asarray(mask)
    mask = (mask > 0.5).astype(np.uint8)

    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    return mask


def mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Return bounding box around a binary mask as x1, y1, x2, y2."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def choose_near_square_grid(n_patches: int, bbox_w: int, bbox_h: int) -> Tuple[int, int]:
    """
    Pick rows/cols for a grid that is as close to square as possible.

    For example:
      n=4  -> 2x2
      n=6  -> usually 2x3 or 3x2 depending on bbox aspect
      n=9  -> 3x3
      n=12 -> 3x4 or 4x3 depending on bbox aspect
    """
    n = max(1, int(n_patches))
    best = None

    for rows in range(1, n + 1):
        cols = math.ceil(n / rows)
        cells = rows * cols

        # Aspect ratio of a single patch cell.
        patch_aspect = (bbox_w / cols) / max(1e-6, (bbox_h / rows))

        # Prefer square patch cells and avoid too many extra cells.
        square_penalty = abs(math.log(max(patch_aspect, 1e-6)))
        extra_penalty = 0.25 * (cells - n)
        balance_penalty = 0.05 * abs(rows - cols)

        score = square_penalty + extra_penalty + balance_penalty

        if best is None or score < best[0]:
            best = (score, rows, cols)

    return int(best[1]), int(best[2])


def split_mask_into_grid_patches(
    mask: np.ndarray,
    n_patches: int = 4,
    min_mask_fraction: float = 0.05,
    min_patch_width: int = 32,
    min_patch_height: int = 32,
) -> Tuple[List[Dict[str, Any]], Tuple[int, int]]:
    """
    Split the parent mask's bounding box into a near-square grid.

    Patches are rectangles, but each patch stores how much of that rectangle
    is actually inside the parent mask.
    """
    bbox = mask_bbox(mask)
    if bbox is None:
        return [], (0, 0)

    x1, y1, x2, y2 = bbox
    bbox_w = x2 - x1
    bbox_h = y2 - y1

    rows, cols = choose_near_square_grid(n_patches, bbox_w, bbox_h)

    x_edges = np.linspace(x1, x2, cols + 1).astype(int)
    y_edges = np.linspace(y1, y2, rows + 1).astype(int)

    patches: List[Dict[str, Any]] = []
    patch_index = 1

    for r in range(rows):
        for c in range(cols):
            px1, px2 = int(x_edges[c]), int(x_edges[c + 1])
            py1, py2 = int(y_edges[r]), int(y_edges[r + 1])

            patch_w = px2 - px1
            patch_h = py2 - py1

            patch_mask = mask[py1:py2, px1:px2]
            rect_area = max(1, patch_w * patch_h)
            mask_area = int(patch_mask.sum())
            mask_fraction = mask_area / rect_area

            valid = (
                patch_index <= n_patches
                and patch_w >= min_patch_width
                and patch_h >= min_patch_height
                and mask_fraction >= min_mask_fraction
            )

            patches.append({
                "patch_index": patch_index,
                "grid_row": r,
                "grid_col": c,
                "grid_rows": rows,
                "grid_cols": cols,
                "x1": px1,
                "y1": py1,
                "x2": px2,
                "y2": py2,
                "width": patch_w,
                "height": patch_h,
                "mask_area": mask_area,
                "rect_area": rect_area,
                "mask_fraction": float(mask_fraction),
                "valid": bool(valid),
            })

            patch_index += 1

    return patches, (rows, cols)


def crop_patch_with_parent_mask(
    image_bgr: np.ndarray,
    parent_mask: np.ndarray,
    patch: Dict[str, Any],
) -> np.ndarray:
    """
    Crop a patch rectangle from the original image and black out pixels
    outside the parent mask.
    """
    x1, y1, x2, y2 = patch["x1"], patch["y1"], patch["x2"], patch["y2"]

    crop = image_bgr[y1:y2, x1:x2].copy()
    patch_mask = parent_mask[y1:y2, x1:x2].astype(np.uint8)

    masked_crop = crop.copy()
    masked_crop[patch_mask == 0] = 0

    return masked_crop


def choose_parent_predictions(
    result: Any,
    selection: str = "largest_mask",
    process_all_parent_masks: bool = False,
) -> List[Dict[str, Any]]:
    """
    Choose which full-image parent masks to process.

    If process_all_parent_masks=False:
      - largest_mask: process the largest detected mask
      - highest_conf: process the highest-confidence detection
    """
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return []

    masks = result.masks.data.detach().cpu().numpy()
    boxes = result.boxes

    items = []
    for i in range(len(boxes)):
        conf = float(boxes.conf[i].detach().cpu().item())
        cls_id = int(boxes.cls[i].detach().cpu().item())
        cls_name = get_class_name(result.names, cls_id)
        raw_mask = masks[i]
        area = float((raw_mask > 0.5).sum())

        items.append({
            "parent_detection_index": i,
            "parent_class_id": cls_id,
            "parent_class_name": cls_name,
            "parent_confidence": conf,
            "raw_mask": raw_mask,
            "raw_mask_area": area,
        })

    if process_all_parent_masks:
        return items

    if selection == "largest_mask":
        return [max(items, key=lambda x: x["raw_mask_area"])]

    if selection == "highest_conf":
        return [max(items, key=lambda x: x["parent_confidence"])]

    raise ValueError(f"Unknown parent selection mode: {selection}")


def infer_single_patch(
    model: YOLO,
    patch_bgr: np.ndarray,
    parent_class_name: str,
    parent_confidence: float,
    imgsz: int = 640,
    patch_conf_threshold: float = 0.05,
    prediction_mode: str = "weighted_probs",
    use_parent_fallback: bool = True,
) -> Dict[str, Any]:
    """
    Run independent YOLO inference on one patch crop.

    Returns patch-level:
      - predicted class
      - confidence
      - P(bald)
      - rough depth estimate

    If no patch detection is found, optionally falls back to parent prediction.
    """
    if patch_bgr is None or patch_bgr.size == 0:
        return {
            "patch_inference_status": "empty_patch",
            "patch_prediction_class": None,
            "patch_prediction_confidence": 0.0,
            "p_bald": None,
            "estimated_depth_mm": None,
            "used_parent_fallback": False,
        }

    # Ultralytics can accept numpy images. Convert BGR -> RGB for consistency.
    patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)

    results = model.predict(
        source=patch_rgb,
        imgsz=imgsz,
        conf=patch_conf_threshold,
        task="segment",
        verbose=False,
    )

    res = results[0]

    if res.boxes is None or len(res.boxes) == 0:
        if not use_parent_fallback:
            return {
                "patch_inference_status": "no_patch_detection",
                "patch_prediction_class": None,
                "patch_prediction_confidence": 0.0,
                "p_bald": None,
                "estimated_depth_mm": None,
                "used_parent_fallback": False,
            }

        # Use parent prediction, but mark it as fallback and push confidence through prior.
        _, parent_p_bald_prior, _ = class_to_depth_prior(parent_class_name)
        p_bald = parent_p_bald_prior * parent_confidence + 0.5 * (1.0 - parent_confidence)

        return {
            "patch_inference_status": "no_patch_detection_parent_fallback",
            "patch_prediction_class": parent_class_name,
            "patch_prediction_confidence": float(parent_confidence),
            "p_bald": float(np.clip(p_bald, 0.0, 1.0)),
            "estimated_depth_mm": depth_from_p_bald(p_bald),
            "used_parent_fallback": True,
        }

    detections = []
    for i in range(len(res.boxes)):
        conf = float(res.boxes.conf[i].detach().cpu().item())
        cls_id = int(res.boxes.cls[i].detach().cpu().item())
        cls_name = get_class_name(res.names, cls_id)
        _, p_bald_prior, status = class_to_depth_prior(cls_name)

        detections.append({
            "class_name": cls_name,
            "confidence": conf,
            "p_bald_prior": p_bald_prior,
            "status": status,
        })

    if prediction_mode == "highest_conf":
        best = max(detections, key=lambda d: d["confidence"])

        # Confidence-weighted prior. Low confidence moves toward neutral 0.5.
        p_bald = best["p_bald_prior"] * best["confidence"] + 0.5 * (1.0 - best["confidence"])

        return {
            "patch_inference_status": "patch_detected_highest_conf",
            "patch_prediction_class": best["class_name"],
            "patch_prediction_confidence": float(best["confidence"]),
            "p_bald": float(np.clip(p_bald, 0.0, 1.0)),
            "estimated_depth_mm": depth_from_p_bald(p_bald),
            "used_parent_fallback": False,
            "patch_detections": detections,
        }

    if prediction_mode == "weighted_probs":
        confs = np.array([d["confidence"] for d in detections], dtype=np.float32)
        priors = np.array([d["p_bald_prior"] for d in detections], dtype=np.float32)

        if confs.sum() <= 1e-8:
            p_bald = 0.5
            agg_conf = 0.0
        else:
            p_bald = float((confs * priors).sum() / confs.sum())
            agg_conf = float(np.clip(confs.max(), 0.0, 1.0))

        top = max(detections, key=lambda d: d["confidence"])

        return {
            "patch_inference_status": "patch_detected_weighted_probs",
            "patch_prediction_class": top["class_name"],
            "patch_prediction_confidence": agg_conf,
            "p_bald": float(np.clip(p_bald, 0.0, 1.0)),
            "estimated_depth_mm": depth_from_p_bald(p_bald),
            "used_parent_fallback": False,
            "patch_detections": detections,
        }

    raise ValueError(f"Unknown patch prediction mode: {prediction_mode}")


def draw_result(
    image_bgr: np.ndarray,
    parent_results: List[Dict[str, Any]],
    alpha: float = 0.28,
) -> np.ndarray:
    """
    Render parent mask overlay and patch rectangles/labels onto the image.
    """
    out = image_bgr.copy()

    for parent in parent_results:
        parent_mask = parent.get("parent_mask")
        patches = parent.get("patches", [])

        if parent_mask is not None:
            overlay = out.copy()
            overlay[parent_mask > 0] = (180, 0, 255)  # pink overlay
            out = cv2.addWeighted(out, 1.0 - alpha, overlay, alpha, 0)

        for p in patches:
            x1, y1, x2, y2 = p["x1"], p["y1"], p["x2"], p["y2"]

            if not p.get("valid", False):
                color = (128, 128, 128)
            else:
                depth = p.get("estimated_depth_mm")
                if depth is None:
                    color = (128, 128, 128)
                elif depth <= 2.0:
                    color = (0, 0, 255)       # red
                elif depth <= 3.5:
                    color = (0, 165, 255)     # orange
                else:
                    color = (0, 200, 0)       # green

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            label = f"#{p['patch_index']}"
            if p.get("estimated_depth_mm") is not None:
                label += f" {p['estimated_depth_mm']:.1f}mm"
            if p.get("uncertainty_mm") is not None:
                label += f"±{p['uncertainty_mm']:.1f}"

            cv2.putText(
                out,
                label,
                (x1 + 5, max(y1 + 22, 22)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

    return out


class TirePatchEstimator:
    """
    Small wrapper class:
      - loads YOLO model once
      - exposes estimate_image(...)
      - optionally renders output
    """

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.model = YOLO(str(self.model_path))

    def estimate_image(
        self,
        image_path: str | Path,
        n_patches: int = 4,
        full_conf_threshold: float = 0.25,
        patch_conf_threshold: float = 0.05,
        imgsz: int = 640,
        parent_selection: str = "largest_mask",
        process_all_parent_masks: bool = False,
        patch_prediction_mode: str = "weighted_probs",
        use_parent_fallback: bool = True,
        min_mask_fraction: float = 0.05,
        min_patch_width: int = 32,
        min_patch_height: int = 32,
        render_output_path: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        """
        Run the full parent-mask + independent patch-inference workflow on one image.

        Returns a structured dict:
          {
            "image": "...",
            "parents": [
              {
                "overall": {...},
                "patches": [...]
              }
            ],
            "rendered_image_path": "..."
          }
        """
        image_path = Path(image_path)
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise ValueError(f"Could not read image: {image_path}")

        h, w = image_bgr.shape[:2]

        # Full-image segmentation/detection.
        results = self.model.predict(
            source=str(image_path),
            imgsz=imgsz,
            conf=full_conf_threshold,
            task="segment",
            verbose=False,
        )

        result = results[0]
        parent_items = choose_parent_predictions(
            result,
            selection=parent_selection,
            process_all_parent_masks=process_all_parent_masks,
        )

        output: Dict[str, Any] = {
            "image": str(image_path),
            "image_width": w,
            "image_height": h,
            "model_path": str(self.model_path),
            "num_parent_masks": len(parent_items),
            "parents": [],
            "rendered_image_path": None,
        }

        parent_results_for_render = []

        for parent in parent_items:
            parent_mask = ensure_binary_mask(parent["raw_mask"], (h, w))

            patches, grid_shape = split_mask_into_grid_patches(
                parent_mask,
                n_patches=n_patches,
                min_mask_fraction=min_mask_fraction,
                min_patch_width=min_patch_width,
                min_patch_height=min_patch_height,
            )

            grid_rows, grid_cols = grid_shape

            patch_results = []
            valid_depths = []
            valid_uncertainties = []
            valid_p_bald = []

            for patch in patches:
                patch_record = dict(patch)

                if not patch["valid"]:
                    patch_record.update({
                        "patch_inference_status": "invalid_patch",
                        "patch_prediction_class": None,
                        "patch_prediction_confidence": None,
                        "p_bald": None,
                        "estimated_depth_mm": None,
                        "uncertainty_mm": None,
                        "used_parent_fallback": False,
                    })
                    patch_results.append(patch_record)
                    continue

                patch_crop = crop_patch_with_parent_mask(image_bgr, parent_mask, patch)

                patch_pred = infer_single_patch(
                    model=self.model,
                    patch_bgr=patch_crop,
                    parent_class_name=parent["parent_class_name"],
                    parent_confidence=parent["parent_confidence"],
                    imgsz=imgsz,
                    patch_conf_threshold=patch_conf_threshold,
                    prediction_mode=patch_prediction_mode,
                    use_parent_fallback=use_parent_fallback,
                )

                patch_record.update(patch_pred)

                if patch_record["estimated_depth_mm"] is not None:
                    uncertainty = uncertainty_from_patch(
                        p_bald=patch_record["p_bald"],
                        mask_fraction=patch_record["mask_fraction"],
                        inference_confidence=patch_record["patch_prediction_confidence"] or 0.0,
                        used_parent_fallback=bool(patch_record["used_parent_fallback"]),
                    )
                    patch_record["uncertainty_mm"] = uncertainty

                    valid_depths.append(float(patch_record["estimated_depth_mm"]))
                    valid_uncertainties.append(float(uncertainty))
                    valid_p_bald.append(float(patch_record["p_bald"]))

                patch_results.append(patch_record)

            if valid_depths:
                overall_depth = float(np.median(valid_depths))
                overall_uncertainty = float(np.mean(valid_uncertainties))
                overall_min_depth = float(np.min(valid_depths))
                overall_mean_p_bald = float(np.mean(valid_p_bald))
            else:
                overall_depth = None
                overall_uncertainty = None
                overall_min_depth = None
                overall_mean_p_bald = None

            parent_output = {
                "overall": {
                    "parent_detection_index": parent["parent_detection_index"],
                    "parent_class_name": parent["parent_class_name"],
                    "parent_confidence": parent["parent_confidence"],
                    "grid_rows": grid_rows,
                    "grid_cols": grid_cols,
                    "requested_patches": n_patches,
                    "valid_patch_count": len(valid_depths),
                    "median_depth_mm": overall_depth,
                    "mean_uncertainty_mm": overall_uncertainty,
                    "min_depth_mm": overall_min_depth,
                    "mean_p_bald": overall_mean_p_bald,
                },
                "patches": patch_results,
            }

            output["parents"].append(parent_output)

            # Keep render-only fields separate from JSON-friendly output.
            parent_results_for_render.append({
                "parent_mask": parent_mask,
                "patches": patch_results,
            })

        if render_output_path is not None:
            rendered = draw_result(image_bgr, parent_results_for_render)
            render_output_path = Path(render_output_path)
            render_output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(render_output_path), rendered)
            output["rendered_image_path"] = str(render_output_path)

        return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained YOLO segmentation checkpoint, e.g. best.pt")
    parser.add_argument("--image", required=True, help="Path to input tire image")
    parser.add_argument("--out", default="tire_patches_rendered.jpg", help="Path to save rendered image")
    parser.add_argument("--json", default="tire_patch_estimates.json", help="Path to save structured JSON")
    parser.add_argument("--patches", type=int, default=4, help="Number of grid patches, default 4")
    parser.add_argument("--process-all", action="store_true", help="Process all parent masks instead of largest only")
    args = parser.parse_args()

    estimator = TirePatchEstimator(args.model)

    result = estimator.estimate_image(
        image_path=args.image,
        n_patches=args.patches,
        process_all_parent_masks=args.process_all,
        render_output_path=args.out,
    )

    json_path = Path(args.json)
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"\nRendered image saved to: {args.out}")
    print(f"JSON saved to: {args.json}")


if __name__ == "__main__":
    main()
