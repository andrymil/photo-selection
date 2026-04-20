import cv2
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from utils.data import prepare_dataset
from utils.vision import get_largest_bbox

BASE_DIR = "datasets"
TARGET_DIR = "datasets_roi"

print("Loading YOLOv8...")
yolo = YOLO("models/pretrained/yolov8n.pt")

paths, _, _ = prepare_dataset(BASE_DIR)


for img_path in tqdm(paths):
    target_path = img_path.replace(BASE_DIR, TARGET_DIR)

    Path(target_path).parent.mkdir(parents=True, exist_ok=True)

    img_cv2 = cv2.imread(img_path)

    if img_cv2 is None:
        continue

    yolo_results = yolo(img_cv2, verbose=False)[0]
    bbox = get_largest_bbox(yolo_results, img_cv2.shape)

    if bbox:
        x1, y1, x2, y2 = bbox
        crop_cv2 = img_cv2[y1:y2, x1:x2]
        cv2.imwrite(target_path, crop_cv2)
    else:
        cv2.imwrite(target_path, img_cv2)

print(f"ROI dataset saved to: {TARGET_DIR}/")
