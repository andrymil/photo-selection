import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from utils.seed import set_seed
from utils.data import prepare_dataset
from utils.path import calculate_path
from utils.model import get_resnet18, get_standard_transforms, predict_ai

EXPERIMENT_GLOBAL = "ResNet18_224px_batch32"
BASE_DIR = "datasets"


def main():
    set_seed(42)
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    paths, labels, _ = prepare_dataset(BASE_DIR)
    _, val_paths, _, val_labels = train_test_split(
        paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print("Loading Global AI...")
    model = get_resnet18(
        DEVICE, weights_path=f"models/checkpoints/{EXPERIMENT_GLOBAL}.pth"
    )
    transform = get_standard_transforms()

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

    plt.title("Confusion Matrix - Global AI (ResNet18)", fontsize=15, pad=15)
    plt.xlabel("Model prediction", fontsize=13, fontweight="bold")
    plt.ylabel("Real class", fontsize=13, fontweight="bold")
    plt.tight_layout()

    results_path = calculate_path(__file__)
    path = results_path / "confusion_matrix_global_ai.png"
    plt.savefig(path, dpi=300)
    print(f"Saved confusion matrix to: {path}")
    plt.show()


if __name__ == "__main__":
    main()
