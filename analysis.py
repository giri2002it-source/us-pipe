import os
import torch

# ------------------ FORCE SAFE RENDER SETTINGS ------------------
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"
torch.set_num_threads(1)

import cv2
import numpy as np
from torchvision.ops import nms
from ultralytics import YOLO
import pymupdf as fitz
from PIL import Image


# ------------------ LOAD MODEL ONCE ------------------
_MODEL = None

def get_model(model_path):
    global _MODEL
    if _MODEL is None:
        print("🔄 Loading YOLO model once...")
        _MODEL = YOLO(model_path)
        _MODEL.to("cpu")
        print("✅ YOLO model loaded on CPU")
    return _MODEL


# ------------------ COLOR PER CLASS ------------------
def get_color_for_class(cls_id):
    COLORS = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]
    return COLORS[cls_id % len(COLORS)]


# ------------------ DETECTION ------------------
def detect_symbols_in_image(
    model,
    img_np,
    window_size=640,
    stride=640,          # ⬅️ reduced overlap
    conf_threshold=0.3,
    iou_threshold=0.45
):
    height, width = img_np.shape[:2]
    all_dets = []

    with torch.no_grad():   # ⬅️ critical
        for y in range(0, height, stride):
            for x in range(0, width, stride):
                window = img_np[y:y+window_size, x:x+window_size]

                if window.shape[0] < window_size or window.shape[1] < window_size:
                    padded = np.zeros((window_size, window_size, 3), dtype=np.uint8)
                    padded[:window.shape[0], :window.shape[1]] = window
                    window = padded

                window_rgb = cv2.cvtColor(window, cv2.COLOR_BGR2RGB)
                window_pil = Image.fromarray(window_rgb)

                results = model.predict(
                    window_pil,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    verbose=False
                )

                for r in results:
                    for b in r.boxes:
                        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                        all_dets.append([
                            x + x1,
                            y + y1,
                            x + x2,
                            y + y2,
                            float(b.conf[0]),
                            int(b.cls[0])
                        ])

    if not all_dets:
        return {}, []

    boxes = torch.tensor([d[:4] for d in all_dets])
    scores = torch.tensor([d[4] for d in all_dets])
    keep = nms(boxes, scores, iou_threshold)

    kept_dets = [all_dets[i] for i in keep]

    counts = {}
    for d in kept_dets:
        name = model.names.get(d[5], f"class_{d[5]}")
        counts[name] = counts.get(name, 0) + 1

    return counts, kept_dets


# ------------------ MAIN PROCESS ------------------
def process_pdf_for_symbols(
    pdf_path,
    model_path,
    output_dir="output",
    dpi=150
):
    model = get_model(model_path)
    doc = fitz.open(pdf_path)
    os.makedirs(output_dir, exist_ok=True)

    total_counts = {}
    all_output_images = []

    for page_idx, page in enumerate(doc):
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)

        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        page_counts, page_boxes = detect_symbols_in_image(model, img_np)

        for k, v in page_counts.items():
            total_counts[k] = total_counts.get(k, 0) + v

        del pix   # ⬅️ memory safety

    doc.close()
    return total_counts, all_output_images, os.path.basename(model_path), model.names
