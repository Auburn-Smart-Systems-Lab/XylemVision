#!/usr/bin/env python3
import numpy as np
import torch
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
from analysis.configs import *
from analysis.utils import *

yolo_model: YOLO = None
sam_predictor: SamPredictor = None

yolo_model = YOLO(YOLO_WEIGHTS).to(DEVICE).eval()

sam = sam_model_registry[SAM_TYPE](checkpoint=SAM_CKPT).to(DEVICE).eval()
sam_predictor = SamPredictor(sam)

print("✅ Models ready on", DEVICE)


def _choose_mask(box_np, pts=None, pt_labels=None):
    masks, scores, _ = sam_predictor.predict(
        box=box_np, point_coords=pts, point_labels=pt_labels, multimask_output=True
    )
    scores = scores.squeeze()
    return masks[scores.argmax()]


def robust_root_mask(img_rgb, masked_rgb, box, H, W):
    box_np = np.array(box, float)

    sam_predictor.set_image(masked_rgb)
    m1 = _choose_mask(box_np)
    if m1.sum() >= MIN_ROOT_PX:
        return m1

    pts, lbls = prompt_points(box, H, W)
    sam_predictor.set_image(masked_rgb)
    m2 = _choose_mask(box_np, pts, lbls)
    if m2.sum() >= MIN_ROOT_PX:
        return m2

    sam_predictor.set_image(img_rgb)
    return _choose_mask(box_np, pts, lbls)


def detect_objects(bgr):
    with torch.no_grad():
        det = yolo_model.predict(bgr, verbose=False)[0]
    boxes = det.boxes.xyxy.cpu().numpy()
    labels = [yolo_model.model.names[int(i)] for i in det.boxes.cls.cpu().numpy()]
    return boxes, labels

def apply_masking_sequence(rgb, cls2bx):
    sam_predictor.set_image(rgb)
    x_masks = [_choose_mask(np.array(bx, float)) for bx in cls2bx["Xylem"]]

    no_x = rgb.copy()
    for m in x_masks:
        no_x[m] = 0

    sam_predictor.set_image(no_x)
    vb_masks = [_choose_mask(np.array(bx, float)) for bx in cls2bx["Vascular bundle"]]

    no_vb = no_x.copy()
    for m in vb_masks:
        no_vb[m] = 0

    root_masks = [
        robust_root_mask(rgb, no_vb, bx, *rgb.shape[:2])
        for bx in cls2bx["Total root"]
    ]

    return x_masks, vb_masks, root_masks


def calculate_metrics(x_masks, vb_masks, root_masks):
    x_props = compute_props(x_masks)
    vb_props = compute_props(vb_masks)
    root_props = compute_props(root_masks)

    return {
        'xylem_count':       len(x_props),
        'vb_total_area':     sum(p['area'] for p in vb_props),
        'vb_max_diameter':   max((p['diameter'] for p in vb_props), default=0.0),
        'root_total_area':   sum(p['area'] for p in root_props),
        'root_max_diameter': max((p['diameter'] for p in root_props), default=0.0),
        'xylem_details':     x_props
    }

def progressive_yolo_sam(pil_image):
    rgb = np.array(pil_image)
    bgr = rgb[..., ::-1]

    boxes, labels = detect_objects(bgr)
    cls2bx = group_boxes_by_class(boxes, labels)

    x_masks, vb_masks, root_masks = apply_masking_sequence(rgb, cls2bx)
    x_masks, vb_masks, root_masks = refine_masks(x_masks, vb_masks, root_masks)

    metrics = calculate_metrics(x_masks, vb_masks, root_masks)
    colour_meta = []

    masks_dict = {
        'Xylem': x_masks,
        'Vascular bundle': vb_masks,
        'Total root': root_masks,
        'meta': {'name': 'in_memory_image'}
    }

    overlay = rgb.copy()
    for cls in ["Xylem", "Vascular bundle", "Total root"]:
        masks = masks_dict.get(cls, [])
        for k, m in enumerate(masks):
            col = class_color_cycle(cls, k)
            overlay = blend_mask(overlay, m, col)
            colour_meta.append({'class': cls, 'inst': k, 'rgb': col})

    overlay = draw_boxes(overlay, boxes, labels)
    overlay_pil = Image.fromarray(overlay)

    return {
        'file': 'in_memory_image',
        'n_xylem': len(x_masks),
        'n_vb': len(vb_masks),
        'n_root': len(root_masks),
        'metrics': metrics,
        'colours': colour_meta
    }, pil_image, overlay_pil
