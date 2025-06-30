#!/usr/bin/env python3
"""
progressive_yolo_sam_tool.py

A command-line tool that uses YOLO and Segment Anything to detect and segment
Xylem, Vascular bundles, and Total root in root cross-section images, then
computes metrics like area and maximum diameter for each class and per-instance
details for xylem. Now using full-precision to avoid dtype mismatches.
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image, ImageDraw

# ───────────────────────── CONFIG ─────────────────────────
ALPHA        = 0.65     # mask fill opacity
OUTLINE_W    = 2        # mask outline width (px)
MIN_ROOT_PX  = 2000     # min pixels for a valid root mask
POS_PTS      = 36       # number of positive prompt points
NEG_EDGE     = 20       # negative prompt offset

YOLO_WEIGHTS = "weight/YOLO/best.pt"
SAM_TYPE     = "vit_l"
SAM_CKPT     = "weight/SAM/sam_vit_l_0b3195.pth"

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# Pre-compute dilation kernel
_DILATE_KERN = cv2.getStructuringElement(
    cv2.MORPH_RECT, (OUTLINE_W, OUTLINE_W)
)

# Model placeholders (initialized in main)
yolo_model: YOLO = None  # type: ignore
sam_predictor: SamPredictor = None  # type: ignore


def class_color_cycle(cls: str, k: int) -> tuple[int,int,int]:
    """Color for each mask instance: random for Xylem, fixed cycling for others."""
    if cls == "Xylem":
        return tuple(random.randint(0, 255) for _ in range(3))
    base = {
        "Vascular bundle": np.array([0, 255,   0]),
        "Total root":       np.array([0,   0, 255]),
    }[cls]
    factor = 0.85 if k % 2 == 0 else 1.15
    return tuple(int(v) for v in np.clip(base * factor, 0, 255))


def blend_mask(img: np.ndarray, mask: np.ndarray, rgb, alpha=ALPHA) -> np.ndarray:
    """Blend a boolean mask into an RGB image with an outline."""
    out = img.astype(np.float32)
    for c in range(3):
        out[..., c][mask] = (1 - alpha) * out[..., c][mask] + alpha * rgb[c]
    edge = cv2.dilate(mask.astype(np.uint8), _DILATE_KERN, 1).astype(bool) ^ mask
    out[edge] = rgb
    return out.astype(np.uint8)


def draw_boxes(img: np.ndarray, boxes, labels) -> np.ndarray:
    """Draw bounding boxes and labels onto an RGB image."""
    pil = Image.fromarray(img)
    if len(boxes) > 0:
        dr = ImageDraw.Draw(pil)
        for bx, lb in zip(boxes, labels):
            x1, y1, x2, y2 = map(int, bx)
            dr.rectangle([x1, y1, x2, y2], outline=(0, 0, 0), width=2)
            dr.text((x1, y1), str(lb), fill=(255, 255, 0))
    return np.array(pil)


def prompt_points(box, H, W, n_pos=POS_PTS, neg_edge=NEG_EDGE):
    """Generate SAM point prompts inside and just outside a box."""
    x1, y1, x2, y2 = map(int, box)
    xs = np.linspace(x1 + 5, x2 - 5, int(np.sqrt(n_pos)))
    ys = np.linspace(y1 + 5, y2 - 5, int(np.sqrt(n_pos)))
    pos = np.array([(x, y) for y in ys for x in xs])
    neg = np.array([
        (max(0, x1 - neg_edge), y1),
        (min(W - 1, x2 + neg_edge), y1),
        (x1, max(0, y1 - neg_edge)),
        (x1, min(H - 1, y2 + neg_edge))
    ])
    coords = np.vstack([pos, neg])
    labels = np.array([1] * len(pos) + [0] * len(neg))
    return coords, labels


def _choose_mask(box_np, pts=None, pt_labels=None):
    """
    Given box (and optional point prompts), return the highest-scoring SAM mask.
    Assumes sam_predictor.set_image(...) was already called with the correct image.
    """
    masks, scores, _ = sam_predictor.predict(
        box=box_np, point_coords=pts, point_labels=pt_labels, multimask_output=True
    )
    scores = scores.squeeze()
    return masks[scores.argmax()]


def robust_root_mask(img_rgb, masked_rgb, box, H, W):
    """
    Fallback strategy for Total root: box-only, then with prompts.
    Switches between masked_rgb and original img_rgb as needed.
    """
    box_np = np.array(box, float)

    # 1) box-only on masked image
    sam_predictor.set_image(masked_rgb)
    m1 = _choose_mask(box_np)
    if m1.sum() >= MIN_ROOT_PX:
        return m1

    # 2) box + points on masked image
    pts, lbls = prompt_points(box, H, W)
    sam_predictor.set_image(masked_rgb)
    m2 = _choose_mask(box_np, pts, lbls)
    if m2.sum() >= MIN_ROOT_PX:
        return m2

    # 3) box + points on original image
    sam_predictor.set_image(img_rgb)
    return _choose_mask(box_np, pts, lbls)


def refine_masks(x_masks, vb_masks, root_masks):
    """Subtract overlaps: remove xylem from VB, and both from root."""
    x_comb = np.any(x_masks, axis=0) if x_masks else np.zeros_like(root_masks[0], bool)
    vb_ref = [m & ~x_comb for m in vb_masks]
    vb_comb = np.any(vb_ref, axis=0) if vb_ref else np.zeros_like(root_masks[0], bool)
    root_ref = [m & ~(x_comb | vb_comb) for m in root_masks]
    return x_masks, vb_ref, root_ref


def compute_props(masks):
    """
    For each boolean mask, compute area and approximate max diameter via
    the minimum enclosing circle. Returns list of dicts:
      {'id': idx, 'area': px_count, 'diameter': px_diameter}.
    """
    props = []
    for idx, m in enumerate(masks):
        cnts, _ = cv2.findContours(
            m.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        if not cnts:
            props.append({'id': idx, 'area': 0.0, 'diameter': 0.0})
            continue

        cnt = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        (_, _), radius = cv2.minEnclosingCircle(cnt)
        diameter = 2.0 * radius

        props.append({
            'id': idx,
            'area': float(area),
            'diameter': float(diameter)
        })
    return props


def progressive_yolo_sam(img_path, out_dir="results", save_images=False):
    """
    Runs YOLO detection + SAM segmentation on img_path, refines masks,
    computes metrics, and—optionally—saves overlays, compressed masks,
    and a JSON summary into out_dir.
    Returns a dict of overall metrics.
    """
    p = Path(img_path)
    bgr = cv2.imread(str(p))
    if bgr is None:
        raise FileNotFoundError(f"Image not found: {p}")
    H, W = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    with torch.no_grad():
        # 1) YOLO detection
        det = yolo_model.predict(bgr, verbose=False)[0]
        boxes = det.boxes.xyxy.cpu().numpy()
        labels = [yolo_model.model.names[int(i)] for i in det.boxes.cls.cpu().numpy()]

        # 2) Group boxes by class
        cls2bx = {"Xylem": [], "Vascular bundle": [], "Total root": []}
        for bx, lb in zip(boxes, labels):
            if lb in cls2bx:
                cls2bx[lb].append(bx)

        # 3) Segment Xylem
        sam_predictor.set_image(rgb)
        x_masks = [
            _choose_mask(np.array(bx, float))
            for bx in cls2bx["Xylem"]
        ]
        no_x = rgb.copy()
        for m in x_masks:
            no_x[m] = 0

        # 4) Segment VBs
        sam_predictor.set_image(no_x)
        vb_masks = [
            _choose_mask(np.array(bx, float))
            for bx in cls2bx["Vascular bundle"]
        ]
        no_vb = no_x.copy()
        for m in vb_masks:
            no_vb[m] = 0

        # 5) Segment Roots with fallback
        root_masks = [
            robust_root_mask(rgb, no_vb, bx, H, W)
            for bx in cls2bx["Total root"]
        ]

    # 6) Refine overlap between classes
    x_masks, vb_masks, root_masks = refine_masks(x_masks, vb_masks, root_masks)

    # 7) Compute properties
    x_props    = compute_props(x_masks)
    vb_props   = compute_props(vb_masks)
    root_props = compute_props(root_masks)
    metrics = {
        'xylem_count':       len(x_props),
        'vb_total_area':     sum(p['area'] for p in vb_props),
        'vb_max_diameter':   max((p['diameter'] for p in vb_props), default=0.0),
        'root_total_area':   sum(p['area'] for p in root_props),
        'root_max_diameter': max((p['diameter'] for p in root_props), default=0.0),
        'xylem_details':     x_props
    }

    # 8) Save outputs if requested
    if save_images:
        out_root = Path(out_dir)
        (out_root / "overlays").mkdir(parents=True, exist_ok=True)
        (out_root / "masks").mkdir(parents=True, exist_ok=True)

        # overlay visualization
        overlay = rgb.copy()
        colour_meta = []
        for cls, masks in [("Xylem", x_masks),
                           ("Vascular bundle", vb_masks),
                           ("Total root", root_masks)]:
            for k, m in enumerate(masks):
                col = class_color_cycle(cls, k)
                overlay = blend_mask(overlay, m, col)
                colour_meta.append({'class': cls, 'inst': k, 'rgb': col})

        overlay = draw_boxes(overlay, boxes, labels)
        Image.fromarray(overlay).save(out_root / "overlays" / f"{p.stem}_overlay.png")

        # masks archive
        np.savez_compressed(
            out_root / "masks" / f"{p.stem}_masks.npz",
            xylem=np.stack(x_masks) if x_masks else np.empty((0, H, W), bool),
            vascular_bundle=np.stack(vb_masks) if vb_masks else np.empty((0, H, W), bool),
            total_root=np.stack(root_masks) if root_masks else np.empty((0, H, W), bool),
        )

        # JSON summary
        summary = {
            'file':    p.name,
            'n_xylem': len(x_masks),
            'n_vb':    len(vb_masks),
            'n_root':  len(root_masks),
            'metrics': metrics,
            'colours': colour_meta
        }
        with open(out_root / "masks" / f"{p.stem}.json", "w") as f:
            json.dump(summary, f, indent=2)

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Efficient YOLO+SAM root analyzer: detects Xylem, VBs, and root area."
    )
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--output-dir", default="results2", help="Directory to save outputs")
    parser.add_argument(
        "--save-images", action="store_true", default='overlays',
        help="Enable saving overlays, masks, and JSON summary"
    )
    args = parser.parse_args()

    global yolo_model, sam_predictor
    # Load YOLO in full precision to avoid conv/BN dtype mismatch
    yolo_model = YOLO(YOLO_WEIGHTS).to(DEVICE).eval()

    # Load SAM in full precision as well
    sam = sam_model_registry[SAM_TYPE](checkpoint=SAM_CKPT).to(DEVICE).eval()
    sam_predictor = SamPredictor(sam)

    print("✅ Models ready on", DEVICE)
    metrics = progressive_yolo_sam(
        args.image, args.output_dir, args.save_images
    )
    print("Result metrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
