import cv2
import shutil
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

from common.paths import ROOT_DIR, DATASETS_DIR, YOLO_WEIGHTS
from common.vision import get_smart_bbox
from sharpness.data import prepare_dataset

TARGET_DIR = ROOT_DIR / "datasets_roi"


def main():
    if TARGET_DIR.exists():
        print(f"Cleaning old directory {TARGET_DIR}...")
        shutil.rmtree(TARGET_DIR)

    print("Loading YOLOv8...")
    yolo = YOLO(str(YOLO_WEIGHTS))

    paths, _, _ = prepare_dataset()

    for img_path in tqdm(paths):
        target_path = TARGET_DIR / Path(img_path).relative_to(DATASETS_DIR)

        target_path.parent.mkdir(parents=True, exist_ok=True)

        img_cv2 = cv2.imread(img_path)

        if img_cv2 is None:
            continue

        yolo_results = yolo(img_cv2, verbose=False)[0]
        bbox = get_smart_bbox(yolo_results, img_cv2.shape)

        if bbox:
            x1, y1, x2, y2 = bbox
            crop_cv2 = img_cv2[y1:y2, x1:x2]
            cv2.imwrite(str(target_path), crop_cv2)
        else:
            cv2.imwrite(str(target_path), img_cv2)

    print(f"ROI dataset saved to: {TARGET_DIR}/")


if __name__ == "__main__":
    main()
