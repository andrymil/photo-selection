import pandas as pd
import torch
from torch.utils.data import DataLoader

from common.paths import ensure_dir
from composition.config import (
    BATCH_SIZE,
    NUM_WORKERS,
    EXPERIMENT_NAME,
    BEST_CHECKPOINT,
    TEST_RESULTS_DIR,
    GALLERY_SIZE,
    SAMP_NET_RESULTS,
)
from composition.data import CompositionDataset, load_scores, load_splits
from composition.metrics import compute_metrics
from composition.model import get_composition_model, get_eval_transforms, predict_scores
from composition.plots import plot_gallery, plot_scatter

GOOD_COMPOSITION_THRESHOLD = 2.6


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


def find_best_and_worst(results):
    ranked = results.sort_values("model_score", ascending=False)

    return {
        "Najlepsza kompozycja według modelu": ranked.head(GALLERY_SIZE),
        "Najgorsza kompozycja według modelu": ranked.tail(GALLERY_SIZE),
    }


def save_results(results, metrics):
    results_dir = ensure_dir(TEST_RESULTS_DIR)

    results.to_csv(results_dir / f"{EXPERIMENT_NAME}_predictions.csv", index=False)
    metrics.to_csv(results_dir / f"{EXPERIMENT_NAME}_metrics.csv", index=False)
    plot_scatter(results, metrics, results_dir / f"{EXPERIMENT_NAME}_scatter.png")
    plot_gallery(
        find_best_and_worst(results), results_dir / f"{EXPERIMENT_NAME}_gallery.png"
    )

    print(f"Results saved to {results_dir}")


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading model from: {BEST_CHECKPOINT}")

    model = get_composition_model(device, weights_path=BEST_CHECKPOINT)
    results = predict_test_set(model, device)
    metrics = compute_test_metrics(results)

    print(metrics.round(4).to_string(index=False))
    save_results(results, metrics)


if __name__ == "__main__":
    main()
