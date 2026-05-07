import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from utils.seed import set_seed
from utils.data import prepare_dataset
from utils.path import calculate_path
from utils.model import get_efficientnet, get_val_transforms

IMAGE_SIZE = 384
EXPERIMENT = "EfficientNetV2S_384px_batch16_with_EBB_BEST"
BASE_DIR = "datasets"


def main():
    set_seed(42)
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    paths, labels, _ = prepare_dataset(BASE_DIR)
    _, val_paths, _, val_labels = train_test_split(
        paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    checkpoints_path = Path("models/checkpoints")
    weights_path = checkpoints_path / f"{EXPERIMENT}.pth"

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

    results_path = calculate_path(__file__)
    plot_path = results_path / f"{EXPERIMENT}_threshold_analysis.png"
    plt.savefig(plot_path, dpi=300)
    print(f"\nPlot saved successfully to: {plot_path}")
    plt.show()


if __name__ == "__main__":
    main()
