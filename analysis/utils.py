from .configs import _DILATE_KERN
from pathlib import Path
from .configs import *
from PIL import Image, ImageDraw
import numpy as np
import random
import cv2

def load_image(img_path):
    p = Path(img_path)
    bgr = cv2.imread(str(p))
    if bgr is None:
        raise FileNotFoundError(f"Image not found: {p}")
    return p, bgr, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def group_boxes_by_class(boxes, labels):
    cls2bx = {"Xylem": [], "Vascular bundle": [], "Total root": []}
    for box, label in zip(boxes, labels):
        if label in cls2bx:
            cls2bx[label].append(box)
    return cls2bx

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

def blend_mask(img: np.ndarray, mask: np.ndarray, rgb, alpha=ALPHA) -> np.ndarray:
    """Blend a boolean mask into an RGB image with an outline."""
    out = img.astype(np.float32)
    for c in range(3):
        out[..., c][mask] = (1 - alpha) * out[..., c][mask] + alpha * rgb[c]
    edge = cv2.dilate(mask.astype(np.uint8), _DILATE_KERN, 1).astype(bool) ^ mask
    out[edge] = rgb
    return out.astype(np.uint8)

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