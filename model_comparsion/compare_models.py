
import cv2
import torch
import time
import numpy as np
import matplotlib.pyplot as plt

from ultralytics import YOLO
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights
)
from torchvision.transforms.functional import to_tensor
from PIL import Image


# ----------------------------------
# 1. SETTINGS
# ----------------------------------

IMAGE_PATH = "traffic.jpeg"

DEVICE = "cpu"

CONFIDENCE_THRESHOLD = 0.35

REPETITIONS = 10

# COCO classes we want to detect
TRAFFIC_CLASSES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck"
}


# ----------------------------------
# 2. LOAD IMAGE
# ----------------------------------

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        "Could not find traffic.jpg. Check the image path."
    )

original = image.copy()

rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print("Image loaded successfully:", image.shape)


# ----------------------------------
# 3. LOAD YOLO11n
# ----------------------------------

print("\nLoading YOLO11n...")

yolo_model = YOLO("yolo11n.pt")

print("YOLO11n loaded!")


# ----------------------------------
# 4. LOAD FASTER R-CNN
# ----------------------------------

print("\nLoading Faster R-CNN...")

weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT

faster_model = fasterrcnn_resnet50_fpn(
    weights=weights
)

faster_model.eval()
faster_model.to(DEVICE)

# Get Faster R-CNN's COCO class names
faster_classes = weights.meta["categories"]

print("Faster R-CNN loaded!")


# ----------------------------------
# 5. YOLO DETECTION FUNCTION
# ----------------------------------

def run_yolo():

    results = yolo_model.predict(
        source=rgb_image,
        conf=CONFIDENCE_THRESHOLD,
        imgsz=800,
        device=DEVICE,
        verbose=False
    )

    detections = []

    for box in results[0].boxes:

        class_id = int(box.cls[0])

        class_name = yolo_model.names[class_id]

        confidence = float(box.conf[0])

        if class_name not in TRAFFIC_CLASSES:
            continue

        x1, y1, x2, y2 = map(
            int, box.xyxy[0].tolist()
        )

        detections.append({
            "class": class_name,
            "confidence": confidence,
            "box": (x1, y1, x2, y2)
        })

    return detections


# ----------------------------------
# 6. FASTER R-CNN DETECTION FUNCTION
# ----------------------------------

# Convert the same image into a PyTorch tensor
image_tensor = to_tensor(
    Image.fromarray(rgb_image)
).to(DEVICE)


def run_faster_rcnn():

    with torch.inference_mode():

        predictions = faster_model(
            [image_tensor]
        )

    detections = []

    boxes = predictions[0]["boxes"]
    labels = predictions[0]["labels"]
    scores = predictions[0]["scores"]

    for box, label, score in zip(
        boxes, labels, scores
    ):

        class_id = int(label.item())

        class_name = faster_classes[class_id]

        confidence = float(score.item())

        if confidence < CONFIDENCE_THRESHOLD:
            continue

        if class_name not in TRAFFIC_CLASSES:
            continue

        x1, y1, x2, y2 = map(
            int, box.tolist()
        )

        detections.append({
            "class": class_name,
            "confidence": confidence,
            "box": (x1, y1, x2, y2)
        })

    return detections


# ----------------------------------
# 7. MEASURE INFERENCE TIME
# ----------------------------------

def benchmark(model_function, model_name):

    print(f"\nBenchmarking {model_name}...")

    # Warm-up: exclude initial model setup
    for _ in range(2):
        model_function()

    times = []

    for _ in range(REPETITIONS):

        start = time.perf_counter()

        detections = model_function()

        end = time.perf_counter()

        elapsed = (end - start) * 1000

        times.append(elapsed)

    average_time = np.mean(times)

    fps = 1000 / average_time

    print(f"{model_name} average time: "
          f"{average_time:.2f} ms")

    print(f"{model_name} estimated FPS: "
          f"{fps:.2f}")

    return detections, average_time, fps


# ----------------------------------
# 8. RUN BOTH MODELS
# ----------------------------------

yolo_detections, yolo_time, yolo_fps = benchmark(
    run_yolo,
    "YOLO11n"
)

faster_detections, faster_time, faster_fps = benchmark(
    run_faster_rcnn,
    "Faster R-CNN"
)


# ----------------------------------
# 9. DRAW DETECTION BOXES
# ----------------------------------

def draw_detections(image, detections, model_name):

    output = image.copy()

    for detection in detections:

        x1, y1, x2, y2 = detection["box"]

        class_name = detection["class"]

        confidence = detection["confidence"]

        label = f"{class_name}: {confidence:.2f}"

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            output,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )

    return cv2.cvtColor(
        output,
        cv2.COLOR_BGR2RGB
    )


yolo_output = draw_detections(
    original,
    yolo_detections,
    "YOLO11n"
)

faster_output = draw_detections(
    original,
    faster_detections,
    "Faster R-CNN"
)


# ----------------------------------
# 10. DISPLAY COMPARISON
# ----------------------------------

plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)

plt.imshow(yolo_output)

plt.title(
    f"YOLO11n\n"
    f"Detections: {len(yolo_detections)} | "
    f"Time: {yolo_time:.2f} ms"
)

plt.axis("off")


plt.subplot(1, 2, 2)

plt.imshow(faster_output)

plt.title(
    f"Faster R-CNN\n"
    f"Detections: {len(faster_detections)} | "
    f"Time: {faster_time:.2f} ms"
)

plt.axis("off")

plt.tight_layout()

plt.savefig("model_comparison.png")

plt.show()


# ----------------------------------
# 11. FINAL COMPARISON
# ----------------------------------

print("\n========== FINAL COMPARISON ==========")

print("YOLO11n")
print("Detections:", len(yolo_detections))
print(f"Average inference: {yolo_time:.2f} ms")
print(f"Estimated FPS: {yolo_fps:.2f}")

print("\nFaster R-CNN")
print("Detections:", len(faster_detections))
print(f"Average inference: {faster_time:.2f} ms")
print(f"Estimated FPS: {faster_fps:.2f}")

print("\n======================================")

if yolo_time < faster_time:
    print("YOLO11n was faster in this experiment.")
else:
    print("Faster R-CNN was faster in this experiment.")

print("\nComparison image saved as model_comparison.png")
