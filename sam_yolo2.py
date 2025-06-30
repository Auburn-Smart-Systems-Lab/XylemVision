from pathlib import Path
import cv2, json, numpy as np
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
import torch
import random
import pandas as pd

# ─── CONFIGURATION ───
YOLO_WEIGHTS = "SSL/15k_self/weights/best.pt"
SAM_TYPE = "vit_l"
SAM_CKPT = "sam_weigths/sam_vit_l_0b3195.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize models
yolo = YOLO(YOLO_WEIGHTS).to(DEVICE)
sam_model = sam_model_registry[SAM_TYPE](checkpoint=SAM_CKPT).to(DEVICE)
sam_predict = SamPredictor(sam_model)

# Helper functions

def random_color():
    return tuple(random.randint(0, 255) for _ in range(3))

def blend_mask(img, mask, rgb, alpha=0.65):
    img = img.copy().astype(np.float32)
    img[mask] = (1-alpha)*img[mask] + alpha*np.array(rgb)
    return img.astype(np.uint8)

def sam_best_mask(pred, img_rgb, box_np):
    pred.set_image(img_rgb)
    masks, scores, _ = pred.predict(box=box_np, multimask_output=True)
    return masks[np.argmax(scores)]

def get_metrics(mask):
    area = mask.sum()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    diameter = max(cv2.minEnclosingCircle(cnt)[1]*2 for cnt in contours) if contours else 0
    return area, diameter

# Main function with refined mask logic
def process_image(img_path, out_dir="results"):
    p = Path(img_path)
    img_bgr = cv2.imread(str(p))
    if img_bgr is None: raise FileNotFoundError(p)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]

    detections = yolo.predict(img_bgr, verbose=False)[0]
    boxes = detections.boxes.xyxy.cpu().numpy()
    labels = [yolo.model.names[int(i)] for i in detections.boxes.cls.cpu().numpy()]

    # Prepare mask holders
    masks_dict = {"Xylem": [], "Vascular bundle": [], "Total root": []}

    for bx, lb in zip(boxes, labels):
        if lb in masks_dict:
            mask = sam_best_mask(sam_predict, img_rgb, bx)
            masks_dict[lb].append(mask)

    overlay = img_rgb.copy()

    # Combine masks
    xylem_combined = np.any(masks_dict["Xylem"], axis=0) if masks_dict["Xylem"] else np.zeros((H,W), bool)
    vb_combined = np.any(masks_dict["Vascular bundle"], axis=0) if masks_dict["Vascular bundle"] else np.zeros((H,W), bool)
    total_root_combined = np.any(masks_dict["Total root"], axis=0) if masks_dict["Total root"] else np.zeros((H,W), bool)

    # Refine masks
    vb_refined = vb_combined & ~xylem_combined
    total_root_refined = total_root_combined & ~(xylem_combined | vb_refined)

    # Excel data
    summary_data, xylem_data = [], []

    # Xylem metrics
    for idx, m in enumerate(masks_dict["Xylem"], 1):
        col = random_color()
        overlay = blend_mask(overlay, m, col)
        area, dia = get_metrics(m)
        xylem_data.append([p.name, idx, area, dia])

    # Vascular bundle metrics
    vb_area, vb_dia = get_metrics(vb_refined)
    overlay = blend_mask(overlay, vb_refined, random_color())

    # Total root metrics
    root_area, root_dia = get_metrics(total_root_refined)
    overlay = blend_mask(overlay, total_root_refined, random_color())

    summary_data.append([p.name, len(masks_dict["Xylem"]), vb_area, vb_dia, root_area, root_dia])

    # Save overlay image
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(out_root / f"{p.stem}_overlay.png")

    # Save Excel
    summary_df = pd.DataFrame(summary_data, columns=["Image", "Xylem Count", "Vascular Bundle Area", "Vascular Bundle Diameter", "Total Root Area", "Total Root Diameter"])
    xylem_df = pd.DataFrame(xylem_data, columns=["Image", "Xylem ID", "Xylem Area", "Xylem Diameter"])

    summary_df.to_excel(out_root / f"{p.stem}_summary.xlsx", index=False)
    xylem_df.to_excel(out_root / f"{p.stem}_xylem_details.xlsx", index=False)

    print(f"✓ Saved overlay: {p.stem}_overlay.png")
    print(f"✓ Saved summary Excel: {p.stem}_summary.xlsx")
    print(f"✓ Saved Xylem details Excel: {p.stem}_xylem_details.xlsx")


# Example Usage
if __name__ == "__main__":
    process_image(r"Data\Hasib1st order\images\1202-1b_png.rf.1142046558841f55e5968c06bb41bf44.jpg")