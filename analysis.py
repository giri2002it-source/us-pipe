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
        (255, 0, 0),     # Red
        (0, 255, 0),     # Green
        (0, 0, 255),     # Blue
        (255, 255, 0),   # Yellow
        (255, 0, 255),   # Magenta
        (0, 255, 255),   # Cyan
    ]
    return COLORS[cls_id % len(COLORS)]


# ------------------ DETECTION ------------------
def detect_symbols_in_image(
    model,
    img_np,
    window_size=640,
    stride=640,
    conf_threshold=0.3,
    iou_threshold=0.45
):
    height, width = img_np.shape[:2]
    all_dets = []

    with torch.no_grad():
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
        return {}, [], img_np

    # -------- NMS --------
    boxes = torch.tensor([d[:4] for d in all_dets])
    scores = torch.tensor([d[4] for d in all_dets])
    keep = nms(boxes, scores, iou_threshold)

    kept_dets = [all_dets[i] for i in keep]

    # -------- COUNT PER CLASS --------
    counts = {}
    for d in kept_dets:
        name = model.names.get(d[5], f"class_{d[5]}")
        counts[name] = counts.get(name, 0) + 1

    # -------- DRAW BOXES (MULTI COLOR) --------
    annotated = img_np.copy()
    for x1, y1, x2, y2, conf, cls_id in kept_dets:
        color = get_color_for_class(cls_id)
        label = model.names.get(cls_id, str(cls_id))

        cv2.rectangle(
            annotated,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color,
            2
        )
        cv2.putText(
            annotated,
            label,
            (int(x1), int(y1) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA
        )

    return counts, kept_dets, annotated


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
    output_images = []

    for page_idx, page in enumerate(doc):
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)

        img_np = np.frombuffer(
            pix.samples, dtype=np.uint8
        ).reshape(pix.h, pix.w, pix.n)

        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        page_counts, _, annotated_img = detect_symbols_in_image(model, img_np)

        for k, v in page_counts.items():
            total_counts[k] = total_counts.get(k, 0) + v

        # Save annotated image
        out_path = os.path.join(output_dir, f"page_{page_idx+1}.jpg")
        cv2.imwrite(out_path, annotated_img)
        output_images.append(out_path)

        del pix  # memory safety

    doc.close()

    return {
        "counts": total_counts,
        "images": output_images,
        "model": os.path.basename(model_path),
        "classes": model.names
    }
