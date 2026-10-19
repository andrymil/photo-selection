import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import f1_score
from ultralytics import YOLO

from common.seed import set_seed
from common.paths import YOLO_WEIGHTS, ensure_dir
from common.vision import get_laplacian_score, get_largest_bbox
from sharpness.config import SEED, CLASSIC_RESULTS_DIR
from sharpness.data import load_val_set


def main():
    set_seed(SEED)

    results_path = ensure_dir(CLASSIC_RESULTS_DIR)

    val_paths, val_labels = load_val_set()

    print("Loading YOLOv8...")
    yolo = YOLO(str(YOLO_WEIGHTS))

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
    plt.plot(
        thresholds, roi_f1s, label="ROI Laplacian (YOLO)", color="blue", linewidth=2
    )

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
    plt.savefig(results_path / "laplacian_threshold_optimization.png")
    plt.show()


if __name__ == "__main__":
    main()
