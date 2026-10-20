import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from torch.utils.data import DataLoader

from common.paths import RESULTS_DIR, ensure_dir
from composition.config import (
    BATCH_SIZE,
    NUM_WORKERS,
    EXPERIMENT_NAME,
    BEST_CHECKPOINT,
    IMAGES_DIR,
    CADB_DIR,
)
from composition.data import CompositionDataset, load_scores, load_splits
from composition.metrics import compute_metrics
from composition.model import get_composition_model, get_eval_transforms, predict_scores

TEST_RESULTS_DIR = RESULTS_DIR / "composition" / "test"
SCENES_PATH = CADB_DIR / "scene_categories.json"
GOOD_COMPOSITION_THRESHOLD = 2.6
GALLERY_SIZE = 6

# Zhang et al., "Image Composition Assessment with Saliency-augmented
# Multi-pattern Pooling", BMVC 2021, Table 2 (CADB test set).
SAMP_NET_RESULTS = {"srcc": 0.6564, "plcc": 0.6709, "mse": 0.3867}


def predict_test_set(model, device):
    scores, splits = load_scores(), load_splits()
    dataset = CompositionDataset(splits["test"], scores, get_eval_transforms())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    predictions, targets = predict_scores(model, loader, device)

    return pd.DataFrame(
        {"image": splits["test"], "human_score": targets, "model_score": predictions}
    )


def compute_accuracy(predictions, targets):
    predicted_good = predictions > GOOD_COMPOSITION_THRESHOLD
    actually_good = targets > GOOD_COMPOSITION_THRESHOLD

    return float((predicted_good == actually_good).mean())


def compute_test_metrics(results):
    predictions = results["model_score"].to_numpy()
    targets = results["human_score"].to_numpy()

    our_results = compute_metrics(predictions, targets)
    our_results["accuracy"] = compute_accuracy(predictions, targets)

    return pd.DataFrame(
        [
            {"method": "EfficientNetV2-S (ours)", **our_results},
            {"method": "SAMP-Net (Zhang et al., 2021)", **SAMP_NET_RESULTS},
        ]
    )


def compute_scene_metrics(results):
    with open(SCENES_PATH) as file:
        scenes = json.load(file)

    results = results.assign(scene=results["image"].map(scenes))
    rows = []

    for scene, group in results.groupby("scene"):
        metrics = compute_metrics(
            group["model_score"].to_numpy(), group["human_score"].to_numpy()
        )
        rows.append(
            {
                "scene": scene,
                "images": len(group),
                "human_mean": group["human_score"].mean(),
                "model_mean": group["model_score"].mean(),
                **metrics,
            }
        )

    return pd.DataFrame(rows).sort_values("images", ascending=False)


def plot_scatter(results, metrics, path):
    ours = metrics.iloc[0]

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 8))
    plt.scatter(results["human_score"], results["model_score"], alpha=0.4, s=20)
    plt.plot([1, 5], [1, 5], "k--", label="Idealna zgodność")

    plt.title(
        f"Zbiór testowy CADB: SRCC = {ours['srcc']:.3f}, PLCC = {ours['plcc']:.3f}",
        fontweight="bold",
    )
    plt.xlabel("Ocena ludzi (średnia z 5 ocen)")
    plt.ylabel("Ocena modelu")
    plt.xlim(1, 5)
    plt.ylim(1, 5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_gallery(results, path):
    ranked = results.sort_values("model_score", ascending=False)
    rows = {
        "Najlepsza kompozycja według modelu": ranked.head(GALLERY_SIZE),
        "Najgorsza kompozycja według modelu": ranked.tail(GALLERY_SIZE),
    }

    fig = plt.figure(figsize=(3 * GALLERY_SIZE, 8), layout="constrained")

    for subfig, (title, images) in zip(fig.subfigures(2, 1), rows.items()):
        subfig.suptitle(title, fontweight="bold", fontsize=14)

        for ax, image in zip(subfig.subplots(1, GALLERY_SIZE), images.itertuples()):
            ax.imshow(Image.open(IMAGES_DIR / image.image).convert("RGB"))
            ax.set_xlabel(
                f"model: {image.model_score:.2f} | ludzie: {image.human_score:.2f}"
            )
            ax.set_xticks([])
            ax.set_yticks([])

    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_results(results, metrics, scene_metrics):
    results_dir = ensure_dir(TEST_RESULTS_DIR)

    results.to_csv(results_dir / f"{EXPERIMENT_NAME}_predictions.csv", index=False)
    metrics.to_csv(results_dir / f"{EXPERIMENT_NAME}_metrics.csv", index=False)
    scene_metrics.to_csv(results_dir / f"{EXPERIMENT_NAME}_scenes.csv", index=False)
    plot_scatter(results, metrics, results_dir / f"{EXPERIMENT_NAME}_scatter.png")
    plot_gallery(results, results_dir / f"{EXPERIMENT_NAME}_gallery.png")

    print(f"Results saved to {results_dir}")


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading model from: {BEST_CHECKPOINT}")

    model = get_composition_model(device, weights_path=BEST_CHECKPOINT)
    results = predict_test_set(model, device)
    metrics = compute_test_metrics(results)
    scene_metrics = compute_scene_metrics(results)

    print(metrics.round(4).to_string(index=False))
    print(scene_metrics.round(3).to_string(index=False))
    save_results(results, metrics, scene_metrics)


if __name__ == "__main__":
    main()
