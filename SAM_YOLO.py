#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
from analysis.configs import *
from analysis.utils import *

yolo_model: YOLO = None
sam_predictor: SamPredictor = None


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


def progressive_yolo_sam(img_path, out_dir="temp_results"):
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


yolo_model = YOLO(YOLO_WEIGHTS).to(DEVICE).eval()

sam = sam_model_registry[SAM_TYPE](checkpoint=SAM_CKPT).to(DEVICE).eval()
sam_predictor = SamPredictor(sam)

print("✅ Models ready on", DEVICE)
metrics = progressive_yolo_sam(
    "/home/mzr0167/Desktop/For_tools/Hasib1st order/images/1101-1d_png.rf.43496866626e1347cb3837d38656a3d9.jpg"
)
print("Result metrics:")
print(json.dumps(metrics, indent=2))
