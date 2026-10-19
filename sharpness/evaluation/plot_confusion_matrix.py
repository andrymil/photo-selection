import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from PIL import Image
from tqdm import tqdm

from common.seed import set_seed
from common.paths import ensure_dir
from sharpness.config import SEED, IMAGE_SIZE, BEST_CHECKPOINT, EVALUATION_RESULTS_DIR
from sharpness.data import load_val_set
from sharpness.model import get_efficientnet, get_val_transforms, predict_ai

EXPERIMENT = BEST_CHECKPOINT.stem


def main():
    set_seed(SEED)
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    val_paths, val_labels = load_val_set()

    print("Loading model...")

    weights_path = BEST_CHECKPOINT

    model = get_efficientnet(DEVICE, weights_path=str(weights_path))
    transform = get_val_transforms(img_size=IMAGE_SIZE)

    y_pred = []
    print("Generating predictions for the confusion matrix...")
    for path in tqdm(val_paths):
        img_pil = Image.open(path).convert("RGB")
        pred = predict_ai(model, img_pil, transform, DEVICE)
        y_pred.append(pred)

    cm = confusion_matrix(val_labels, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        annot_kws={"size": 16, "weight": "bold"},
        xticklabels=["Sharp (0)", "Blurred (1)"],
        yticklabels=["Sharp (0)", "Blurred (1)"],
    )

    plt.title("Confusion Matrix", fontsize=15, pad=15)
    plt.xlabel("Model prediction", fontsize=13, fontweight="bold")
    plt.ylabel("Real class", fontsize=13, fontweight="bold")
    plt.tight_layout()

    results_path = ensure_dir(EVALUATION_RESULTS_DIR)
    path = results_path / f"{EXPERIMENT}_CM.png"
    plt.savefig(path, dpi=300)
    print(f"Saved confusion matrix to: {path}")
    plt.show()


if __name__ == "__main__":
    main()
