import cv2
import torch


ALPHA        = 0.65
OUTLINE_W    = 2
MIN_ROOT_PX  = 2000
POS_PTS      = 36
NEG_EDGE     = 20

YOLO_WEIGHTS = "weight/YOLO/best.pt"
SAM_TYPE     = "vit_l"
SAM_CKPT     = "weight/SAM/sam_vit_l_0b3195.pth"

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

_DILATE_KERN = cv2.getStructuringElement(
    cv2.MORPH_RECT, (OUTLINE_W, OUTLINE_W)
)