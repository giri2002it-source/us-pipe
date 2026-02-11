import os

# ------------------ FORCE SAFE RENDER SETTINGS ------------------
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

import cv2
import numpy as np
import torch
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
        print("✅ YOLO model loaded")
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
    stride=512,
    conf_threshold=0.3,
    iou_threshold=0.45
):
    height, width = img_np.shape[:2]
    all_dets = []

    for y in range(0, height, stride):
        for x in range(0, width, stride):
            y_end = min(y + window_size, height)
            x_end = min(x + window_size, width)
            window = img_np[y:y_end, x:x_end]

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
                    xyxy = b.xyxy[0].cpu().numpy()
                    conf = float(b.conf[0].cpu())
                    cls_id = int(b.cls[0].cpu())

                    all_dets.append([
                        x + xyxy[0],
                        y + xyxy[1],
                        x + xyxy[2],
                        y + xyxy[3],
                        conf,
                        cls_id
                    ])

    if not all_dets:
        return {}, []

    boxes = torch.tensor([d[:4] for d in all_dets])
    scores = torch.tensor([d[4] for d in all_dets])
    keep = nms(boxes, scores, iou_threshold)

    kept_dets = [all_dets[i] for i in keep]

    counts = {}
    for d in kept_dets:
        cls_id = int(d[5])
        name = model.names.get(cls_id, f"class_{cls_id}")
        counts[name] = counts.get(name, 0) + 1

    return counts, kept_dets


# ------------------ SAVE PER CLASS IMAGE ------------------
def save_per_class_visualizations(img_np, page_boxes, class_names, output_dir, page_num):
    page_dir = os.path.join(output_dir, f"page_{page_num}")
    os.makedirs(page_dir, exist_ok=True)

    saved_images = {}
    boxes_by_class = {}

    for box in page_boxes:
        cls_id = int(box[5])
        boxes_by_class.setdefault(cls_id, []).append(box)

    for cls_id, boxes in boxes_by_class.items():
        cls_name = class_names.get(cls_id, f"class_{cls_id}")
        vis_img = img_np.copy()
        color = get_color_for_class(cls_id)

        for x1, y1, x2, y2, conf, _ in boxes:
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)

        out_path = os.path.join(page_dir, f"{cls_name}.png")
        cv2.imwrite(out_path, vis_img)
        saved_images.setdefault(cls_name, []).append(out_path)

    return saved_images


# ------------------ MAIN PROCESS ------------------
def process_pdf_for_symbols(
    pdf_path,
    model_path,
    output_dir="output",
    dpi=150   # ⬅️ lower DPI = safer memory
):
    model = get_model(model_path)
    doc = fitz.open(pdf_path)
    os.makedirs(output_dir, exist_ok=True)

    total_counts = {}
    all_output_images = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)

        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        page_counts, page_boxes = detect_symbols_in_image(model, img_np)

        for k, v in page_counts.items():
            total_counts[k] = total_counts.get(k, 0) + v

        if page_boxes:
            images = save_per_class_visualizations(
                img_np,
                page_boxes,
                model.names,
                output_dir,
                page_idx + 1
            )
            for img_list in images.values():
                all_output_images.extend(img_list)

    doc.close()
    return total_counts, all_output_images, os.path.basename(model_path), model.names