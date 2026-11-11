
import os
import json
import cv2
import torch
import numpy as np

from collections import defaultdict
from ultralytics import YOLO
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights
)

# ---------------- CONFIGURATION ----------------

IMAGE_DIR = "dataset/val2017"
ANNOTATION_FILE = "dataset/annotations/instances_val2017.json"

NUM_IMAGES = 10
CONFIDENCE_THRESHOLD = 0.35
IOU_THRESHOLD = 0.50

DEVICE = "cpu"

TRAFFIC_CLASSES = {
    "person", "bicycle", "car",
    "motorcycle", "bus", "truck"
}

# ---------------- LOAD DATASET ----------------

with open(ANNOTATION_FILE, "r") as file:
    coco_data = json.load(file)

images = coco_data["images"]
annotations = coco_data["annotations"]
categories = coco_data["categories"]

category_name_to_id = {
    category["name"]: category["id"]
    for category in categories
}

selected_categories = {
    category_name_to_id[name]: name
    for name in TRAFFIC_CLASSES
}

# Group ground-truth annotations by image
ground_truth = defaultdict(list)

for ann in annotations:

    category_id = ann["category_id"]

    if category_id not in selected_categories:
        continue

    if ann.get("iscrowd", 0) == 1:
        continue

    x, y, w, h = ann["bbox"]

    # Convert COCO xywh to xyxy
    box = [x, y, x + w, y + h]

    ground_truth[ann["image_id"]].append(
        (category_id, box)
    )

available_images = [
    img for img in images
    if os.path.exists(
        os.path.join(IMAGE_DIR, img["file_name"])
    )
]

# Use the same 10 images for both models
test_images = available_images[:NUM_IMAGES]

print("Images selected for evaluation:", len(test_images))

# ---------------- IOU FUNCTION ----------------

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union == 0:
        return 0

    return intersection / union


# ---------------- METRIC CALCULATION ----------------

def calculate_metrics(predictions, actual_objects):

    TP = 0
    FP = 0
    FN = 0

    for category_id in selected_categories:

        gt_boxes = [
            box for cat, box in actual_objects
            if cat == category_id
        ]

        pred_boxes = [
            (score, box)
            for cat, score, box in predictions
            if cat == category_id
        ]

        # Match high-confidence predictions first
        pred_boxes.sort(key=lambda item: item[0], reverse=True)

        matched_gt = set()

        for score, pred_box in pred_boxes:

            best_iou = 0
            best_gt_index = -1

            for index, gt_box in enumerate(gt_boxes):

                if index in matched_gt:
                    continue

                iou = calculate_iou(pred_box, gt_box)

                if iou > best_iou:
                    best_iou = iou
                    best_gt_index = index

            if best_iou >= IOU_THRESHOLD:
                TP += 1
                matched_gt.add(best_gt_index)
            else:
                FP += 1

        FN += len(gt_boxes) - len(matched_gt)

    precision = TP / (TP + FP) if TP + FP else 0
    recall = TP / (TP + FN) if TP + FN else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    return TP, FP, FN, precision, recall, f1


# ---------------- LOAD YOLO ----------------

print("\nLoading YOLO11n...")

yolo_model = YOLO("yolo11n.pt")

yolo_name_to_id = {
    name.lower(): category_id
    for category_id, name in selected_categories.items()
}

# ---------------- LOAD FASTER R-CNN ----------------

print("Loading Faster R-CNN...")

weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT

faster_model = fasterrcnn_resnet50_fpn(
    weights=weights
)

faster_model.to(DEVICE)
faster_model.eval()

torchvision_categories = weights.meta["categories"]

torchvision_label_to_id = {
    name.lower(): category_id
    for category_id, name in selected_categories.items()
}

# ---------------- EVALUATE BOTH MODELS ----------------

all_predictions = {
    "YOLO11n": [],
    "Faster R-CNN": []
}

all_actual_objects = []

for i, image_info in enumerate(test_images):

    image_id = image_info["id"]

    image_path = os.path.join(
        IMAGE_DIR, image_info["file_name"]
    )

    print(f"\nProcessing image {i + 1}/{len(test_images)}")

    # Ground truth for this image
    actual_objects = ground_truth[image_id]

    all_actual_objects.extend(actual_objects)

    # ---------- YOLO PREDICTION ----------

    yolo_results = yolo_model.predict(
        image_path,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False,
        device=DEVICE
    )

    yolo_predictions = []

    for result in yolo_results:

        for box in result.boxes:

            class_index = int(box.cls[0].item())
            class_name = yolo_model.names[class_index].lower()

            if class_name not in yolo_name_to_id:
                continue

            category_id = yolo_name_to_id[class_name]

            confidence = float(box.conf[0].item())
            coordinates = box.xyxy[0].cpu().tolist()

            yolo_predictions.append(
                (category_id, confidence, coordinates)
            )

    all_predictions["YOLO11n"].append(
        (yolo_predictions, actual_objects)
    )

    # ---------- FASTER R-CNN PREDICTION ----------

    image = cv2.imread(image_path)

    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_tensor = (
        torch.from_numpy(image_rgb)
        .permute(2, 0, 1)
        .float() / 255.0
    )

    with torch.inference_mode():

        output = faster_model(
            [image_tensor.to(DEVICE)]
        )[0]

    faster_predictions = []

    for box, label, score in zip(
        output["boxes"],
        output["labels"],
        output["scores"]
    ):

        confidence = float(score.item())

        if confidence < CONFIDENCE_THRESHOLD:
            continue

        label_index = int(label.item())

        class_name = torchvision_categories[label_index].lower()

        if class_name not in torchvision_label_to_id:
            continue

        category_id = torchvision_label_to_id[class_name]

        coordinates = box.cpu().tolist()

        faster_predictions.append(
            (category_id, confidence, coordinates)
        )

    all_predictions["Faster R-CNN"].append(
        (faster_predictions, actual_objects)
    )


# ---------------- FINAL RESULTS ----------------

print("\n========== MODEL EVALUATION RESULTS ==========")

for model_name, image_results in all_predictions.items():

    total_TP = 0
    total_FP = 0
    total_FN = 0

    for predictions, actual_objects in image_results:

        TP, FP, FN, _, _, _ = calculate_metrics(
            predictions, actual_objects
        )

        total_TP += TP
        total_FP += FP
        total_FN += FN

    precision = (
        total_TP / (total_TP + total_FP)
        if total_TP + total_FP else 0
    )

    recall = (
        total_TP / (total_TP + total_FN)
        if total_TP + total_FN else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    print(f"\n{model_name}")
    print("-----------------------------")
    print("True Positives:", total_TP)
    print("False Positives:", total_FP)
    print("False Negatives:", total_FN)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

print("\n==============================================")
