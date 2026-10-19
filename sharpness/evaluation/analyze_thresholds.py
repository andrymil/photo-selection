import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score
from PIL import Image
from tqdm import tqdm

from common.seed import set_seed
from common.paths import ensure_dir
from sharpness.config import SEED, IMAGE_SIZE, BEST_CHECKPOINT, EVALUATION_RESULTS_DIR
from sharpness.data import load_val_set
from sharpness.model import get_efficientnet, get_val_transforms

EXPERIMENT = BEST_CHECKPOINT.stem


def main():
    set_seed(SEED)
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    val_paths, val_labels = load_val_set()

    weights_path = BEST_CHECKPOINT

    print(f"Loading model from: {weights_path}")
    model = get_efficientnet(DEVICE, weights_path=str(weights_path))
    transform = get_val_transforms(img_size=IMAGE_SIZE)

    blur_probs = []
    print("Evaluating validation set to get probabilities...")

    with torch.no_grad():
        for path in tqdm(val_paths):
            img_pil = Image.open(path).convert("RGB")
            img_t = transform(img_pil).unsqueeze(0).to(DEVICE)

            outputs = model(img_t)
            probs = F.softmax(outputs, dim=1)

            blur_probs.append(probs[0, 1].item())

    thresholds = np.linspace(0.0, 1.0, 100)

    precisions = []
    recalls = []
    f1_scores = []

    for t in thresholds:
        y_pred = [1 if p >= t else 0 for p in blur_probs]

        precisions.append(precision_score(val_labels, y_pred, zero_division=0))
        recalls.append(recall_score(val_labels, y_pred, zero_division=0))
        f1_scores.append(f1_score(val_labels, y_pred, zero_division=0))

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 10))

    plt.plot(
        thresholds,
        precisions,
        label="Precision",
        color="green",
        linewidth=2.5,
    )
    plt.plot(
        thresholds,
        recalls,
        label="Recall",
        color="red",
        linewidth=2.5,
    )
    plt.plot(
        thresholds,
        f1_scores,
        label="F1-Score",
        color="blue",
        linestyle="--",
        linewidth=2,
    )

    plt.axvline(x=0.5, color="gray", linestyle=":", label="Default Threshold (0.5)")

    plt.title(
        "Metrics across different Decision Thresholds", fontsize=16, fontweight="bold"
    )
    plt.xlabel("Decision Threshold", fontsize=13)
    plt.ylabel("Score", fontsize=13)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.8, 1.05)
    plt.xticks(np.arange(0, 1.05, 0.1))
    plt.legend(loc="lower center", fontsize=11)

    plt.tight_layout()

    results_path = ensure_dir(EVALUATION_RESULTS_DIR)
    plot_path = results_path / f"{EXPERIMENT}_threshold_analysis.png"
    plt.savefig(plot_path, dpi=300)
    print(f"\nPlot saved successfully to: {plot_path}")
    plt.show()


if __name__ == "__main__":
    main()
