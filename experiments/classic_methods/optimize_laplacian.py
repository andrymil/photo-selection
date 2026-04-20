import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from ultralytics import YOLO
from utils.seed import set_seed
from utils.data import prepare_dataset
from utils.path import calculate_path
from utils.vision import get_laplacian_score, get_largest_bbox

BASE_DIR = "datasets"
RESULTS_PATH = calculate_path(__file__)

set_seed(42)


paths, labels, _ = prepare_dataset(BASE_DIR)

_, val_paths, _, val_labels = train_test_split(
    paths, labels, test_size=0.2, random_state=42, stratify=labels
)

print("Loading YOLOv8...")
yolo = YOLO("models/pretrained/yolov8n.pt")


raw_global_scores = []
raw_roi_scores = []

print(f"Calculating variance for {len(val_paths)} images...")
for path in tqdm(val_paths):
    img_cv2 = cv2.imread(path)

    if img_cv2 is None:
        continue

    raw_global_scores.append(get_laplacian_score(img_cv2))

    yolo_results = yolo(img_cv2, verbose=False)[0]
    bbox = get_largest_bbox(yolo_results, img_cv2.shape)

    if bbox:
        x1, y1, x2, y2 = bbox
        crop_cv2 = img_cv2[y1:y2, x1:x2]
    else:
        crop_cv2 = img_cv2

    raw_roi_scores.append(get_laplacian_score(crop_cv2))


thresholds = np.arange(10, 1500, 10)

global_f1s, roi_f1s = [], []

print("Searching for optimal threshold...")
for t in thresholds:
    pred_global = [1 if score < t else 0 for score in raw_global_scores]
    pred_roi = [1 if score < t else 0 for score in raw_roi_scores]

    global_f1s.append(f1_score(val_labels, pred_global, zero_division=0))
    roi_f1s.append(f1_score(val_labels, pred_roi, zero_division=0))

best_global_idx = np.argmax(global_f1s)
best_roi_idx = np.argmax(roi_f1s)

best_global_th = thresholds[best_global_idx]
best_roi_th = thresholds[best_roi_idx]

print("RESULTS")
print("=" * 50)
print(
    f"Best threshold for global Laplacian: {best_global_th} (F1-Score: {global_f1s[best_global_idx]:.4f})"
)
print(
    f"Best threshold for ROI Laplacian:      {best_roi_th} (F1-Score: {roi_f1s[best_roi_idx]:.4f})"
)

plt.figure(figsize=(10, 6))
plt.plot(thresholds, global_f1s, label="Global Laplacian", color="red", alpha=0.7)
plt.plot(thresholds, roi_f1s, label="ROI Laplacian (YOLO)", color="blue", linewidth=2)

plt.axvline(x=best_global_th, color="red", linestyle="--", alpha=0.5)
plt.axvline(x=best_roi_th, color="blue", linestyle="--", alpha=0.5)
plt.scatter(
    [best_global_th], [global_f1s[best_global_idx]], color="red", s=100, zorder=5
)
plt.scatter([best_roi_th], [roi_f1s[best_roi_idx]], color="blue", s=100, zorder=5)

plt.title("Laplacian variance threshold optimization", fontsize=14)
plt.xlabel("Threshold", fontsize=12)
plt.ylabel("F1-Score", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(RESULTS_PATH / "laplacian_threshold_optimization.png")
plt.show()
