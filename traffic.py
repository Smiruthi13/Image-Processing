import cv2
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ---------- LOAD IMAGE ----------
image = cv2.imread("traffic.jpeg")
if image is None:
    print("Error: Image not found!")
    exit()
original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ---------- (zebra / grayscale part: keep your code as is) ----------
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
threshold = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 31, -10)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
processed = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
zebra_result = image.copy()
for c in contours:
    if cv2.contourArea(c) > 300:
        x, y, w, h = cv2.boundingRect(c)
        if w > 25 and h > 8:
            cv2.rectangle(zebra_result, (x, y), (x + w, y + h), (0, 255, 255), 2)

# ---------- IMPROVED YOLO DETECTION ----------
model = YOLO("yolo11x.pt")          # biggest model = best accuracy (use yolo11m.pt if slow)
device = 0 if torch.cuda.is_available() else "cpu"

CLASSES = {0: "Person", 2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
CONF = 0.20                         # lower so partially hidden people are kept

def run_yolo(img, imgsz, offset=(0, 0)):
    res = model(img, conf=CONF, iou=0.6, imgsz=imgsz, max_det=1000,
                classes=list(CLASSES.keys()), augment=False,
                device=device, verbose=False)[0]
    out = []
    ox, oy = offset
    for b in res.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        out.append([x1 + ox, y1 + oy, x2 + ox, y2 + oy,
                    float(b.conf[0]), int(b.cls[0])])
    return out

def tiled_detect(img, tile=640, overlap=0.25):
    """Slice the image into overlapping tiles so small/far people become bigger."""
    H, W = img.shape[:2]
    step = int(tile * (1 - overlap))
    dets = []
    for y in range(0, max(H - tile, 0) + step, step):
        for x in range(0, max(W - tile, 0) + step, step):
            x2, y2 = min(x + tile, W), min(y + tile, H)
            x1, y1 = max(x2 - tile, 0), max(y2 - tile, 0)
            crop = img[y1:y2, x1:x2]
            dets += run_yolo(crop, imgsz=tile, offset=(x1, y1))
    return dets

# Pass 1: whole image at high resolution, Pass 2: tiles
detections = run_yolo(image, imgsz=1280)
detections += tiled_detect(image, tile=640, overlap=0.25)

# Merge duplicates from all passes (class-wise NMS)
if detections:
    d = torch.tensor(detections)
    keep = torchvision.ops.batched_nms(d[:, :4], d[:, 4], d[:, 5].long(), iou_threshold=0.5)
    detections = d[keep].tolist()

object_result = image.copy()
detected_people = detected_cars = 0

for x1, y1, x2, y2, conf, cls in detections:
    cls = int(cls)
    label = CLASSES[cls]
    if cls == 0:
        detected_people += 1
        color = (0, 255, 0)
    elif cls == 2:
        detected_cars += 1
        color = (255, 128, 0)
    else:
        color = (0, 165, 255)

    cv2.rectangle(object_result, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    cv2.putText(object_result, f"{label} {conf:.2f}", (int(x1), max(int(y1) - 6, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# ---------- DISPLAY ----------
plt.figure(figsize=(15, 10))
for i, (img, title, cmap) in enumerate([
    (original, "Original Image", None),
    (gray, "Grayscale Image", "gray"),
    (processed, "Thresholded Image", "gray"),
    (cv2.cvtColor(zebra_result, cv2.COLOR_BGR2RGB), "Potential Road Markings", None),
    (cv2.cvtColor(object_result, cv2.COLOR_BGR2RGB), "YOLO Detection (Tiled)", None),
], 1):
    plt.subplot(2, 3, i)
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis("off")
plt.tight_layout()
plt.show()

print("TRAFFIC ANALYSIS RESULTS")
print("------------------------")
print("People detected:", detected_people)
print("Cars detected:", detected_cars)

cv2.imwrite("grayscale.jpg", gray)
cv2.imwrite("zebra_candidates.jpg", zebra_result)
cv2.imwrite("object_detection.jpg", object_result)
print("Results saved successfully!")