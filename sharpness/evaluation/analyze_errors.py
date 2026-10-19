import torch
import torch.nn.functional as F
import os
import shutil
from tqdm import tqdm
from PIL import Image

from common.seed import set_seed
from sharpness.config import SEED, IMAGE_SIZE, BEST_CHECKPOINT, ERROR_ANALYSIS_DIR
from sharpness.data import load_val_set
from sharpness.model import get_efficientnet, get_val_transforms

ERROR_DIR = ERROR_ANALYSIS_DIR


def analyze_errors():
    set_seed(SEED)
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    val_paths, val_labels = load_val_set()

    weights_path = BEST_CHECKPOINT

    if not weights_path.exists():
        print(f"Error: Model weights not found in {weights_path}")
        return

    print(f"Loading model weights from: {weights_path}")
    model = get_efficientnet(DEVICE, weights_path=str(weights_path), eval_mode=True)
    transform = get_val_transforms(img_size=IMAGE_SIZE)

    if os.path.exists(ERROR_DIR):
        shutil.rmtree(ERROR_DIR)

    fp_dir = os.path.join(ERROR_DIR, "False_Positives")
    fn_dir = os.path.join(ERROR_DIR, "False_Negatives")

    os.makedirs(fp_dir, exist_ok=True)
    os.makedirs(fn_dir, exist_ok=True)

    fp_count = 0
    fn_count = 0

    print("\nStarting error analysis on validation dataset...")

    with torch.no_grad():
        for img_path, true_label in tqdm(
            zip(val_paths, val_labels), total=len(val_paths)
        ):
            try:
                image_pil = Image.open(img_path).convert("RGB")
            except Exception as error:
                print(f"Couldn't open image {img_path}: {error}")
                continue

            img_t = transform(image_pil).unsqueeze(0).to(DEVICE)

            outputs = model(img_t)
            probabilities = F.softmax(outputs, dim=1)

            confidence, predicted = torch.max(probabilities, 1)
            predicted_label = predicted.item()
            conf_percent = confidence.item() * 100

            if predicted_label != true_label:
                original_filename = os.path.basename(img_path)
                new_filename = f"conf_{int(conf_percent):02d}_{original_filename}"

                if true_label == 0 and predicted_label == 1:
                    target_path = os.path.join(fp_dir, new_filename)
                    shutil.copy2(img_path, target_path)
                    fp_count += 1

                elif true_label == 1 and predicted_label == 0:
                    target_path = os.path.join(fn_dir, new_filename)
                    shutil.copy2(img_path, target_path)
                    fn_count += 1

    print("\nAnalysis finished!")
    print("-" * 30)
    print(f"Saved False Positives (GT: Sharp, Pred: Blurred): {fp_count}")
    print(f"Saved False Negatives (GT: Blurred, Pred: Sharp): {fn_count}")
    print(f"Saved images in: {ERROR_DIR}/")


if __name__ == "__main__":
    analyze_errors()
