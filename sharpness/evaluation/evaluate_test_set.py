import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from common.seed import set_seed
from common.paths import DATASETS_DIR, ensure_dir
from common.vision import get_difference_hash, get_laplacian_score
from sharpness.config import (
    SEED,
    IMAGE_SIZE,
    BEST_CHECKPOINT,
    TEST_RESULTS_DIR,
)
from sharpness.data import (
    prepare_dataset,
    split_dataset,
    load_certh_natural_blur,
    load_ebb_validation,
)
from sharpness.model import get_efficientnet, get_val_transforms

EXPERIMENT = BEST_CHECKPOINT.stem
MODEL_THRESHOLDS = [0.5, 0.75]
LAPLACIAN_THRESHOLDS = np.arange(10, 1500, 10)
DUPLICATE_MAX_DISTANCE = 2

VAL = "Validation split"
NATURAL_BLUR = "CERTH NaturalBlur"
NATURAL_BLUR_CLEAN = "CERTH NaturalBlur (no duplicates)"
EBB_ORIGINAL = "EBB validation original"
EBB_BOKEH = "EBB validation bokeh"


def find_training_duplicates(test_paths, train_paths):
    train_hashes = np.stack(
        [get_difference_hash(p) for p in tqdm(train_paths, desc="Hashing train")]
    )

    is_duplicate = []
    for path in tqdm(test_paths, desc="Hashing test"):
        distances = (train_hashes != get_difference_hash(path)).sum(axis=1)
        is_duplicate.append(distances.min() <= DUPLICATE_MAX_DISTANCE)

    return is_duplicate


def score_images(paths, model, transform, device, desc):
    blur_probs, laplacian_scores = [], []

    with torch.no_grad():
        for path in tqdm(paths, desc=desc):
            img_pil = Image.open(path).convert("RGB")
            img_t = transform(img_pil).unsqueeze(0).to(device)

            outputs = model(img_t)
            blur_probs.append(F.softmax(outputs, dim=1)[0, 1].item())

            laplacian_scores.append(get_laplacian_score(cv2.imread(path)))

    return blur_probs, laplacian_scores


def tune_laplacian_threshold(df):
    f1_scores = [
        f1_score(df["label"], (df["laplacian"] < t).astype(int), zero_division=0)
        for t in LAPLACIAN_THRESHOLDS
    ]

    return LAPLACIAN_THRESHOLDS[int(np.argmax(f1_scores))]


def get_predictions(df, laplacian_threshold):
    predictions = {
        ("EfficientNetV2-S", t): (df["blur_prob"] >= t).astype(int)
        for t in MODEL_THRESHOLDS
    }
    predictions[("Laplacian variance", laplacian_threshold)] = (
        df["laplacian"] < laplacian_threshold
    ).astype(int)

    return predictions


def classification_metrics(df, dataset, laplacian_threshold):
    auc = {
        "EfficientNetV2-S": roc_auc_score(df["label"], df["blur_prob"]),
        "Laplacian variance": roc_auc_score(df["label"], -df["laplacian"]),
    }

    rows = []
    for (method, threshold), y_pred in get_predictions(df, laplacian_threshold).items():
        rows.append(
            {
                "dataset": dataset,
                "images": len(df),
                "method": method,
                "threshold": threshold,
                "accuracy": accuracy_score(df["label"], y_pred),
                "precision": precision_score(df["label"], y_pred, zero_division=0),
                "recall": recall_score(df["label"], y_pred, zero_division=0),
                "f1": f1_score(df["label"], y_pred, zero_division=0),
                "auc": auc[method],
            }
        )

    return rows


def false_rejection_rates(df, dataset, laplacian_threshold):
    return [
        {
            "dataset": dataset,
            "images": len(df),
            "method": method,
            "threshold": threshold,
            "false_rejection_rate": y_pred.mean(),
        }
        for (method, threshold), y_pred in get_predictions(
            df, laplacian_threshold
        ).items()
    ]


def plot_confusion_matrix(df, title, path):
    y_pred = (df["blur_prob"] >= 0.5).astype(int)
    cm = confusion_matrix(df["label"], y_pred)

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

    plt.title(title, fontsize=15, pad=15)
    plt.xlabel("Model prediction", fontsize=13, fontweight="bold")
    plt.ylabel("Real class", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_roc_curves(datasets, path):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 8))

    for name, df in datasets.items():
        for method, scores, linestyle in [
            ("EfficientNetV2-S", df["blur_prob"], "-"),
            ("Laplacian variance", -df["laplacian"], "--"),
        ]:
            fpr, tpr, _ = roc_curve(df["label"], scores)
            auc = roc_auc_score(df["label"], scores)
            plt.plot(
                fpr,
                tpr,
                linestyle=linestyle,
                linewidth=2,
                label=f"{method} – {name} (AUC = {auc:.3f})",
            )

    plt.plot([0, 1], [0, 1], color="gray", linestyle=":")
    plt.title("ROC curves", fontsize=16, fontweight="bold")
    plt.xlabel("False positive rate (sharp photos rejected)", fontsize=13)
    plt.ylabel("True positive rate (blurred photos rejected)", fontsize=13)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_thresholds(df, title, path):
    thresholds = np.linspace(0.0, 1.0, 100)

    precisions, recalls, f1_scores = [], [], []
    for t in thresholds:
        y_pred = (df["blur_prob"] >= t).astype(int)
        precisions.append(precision_score(df["label"], y_pred, zero_division=0))
        recalls.append(recall_score(df["label"], y_pred, zero_division=0))
        f1_scores.append(f1_score(df["label"], y_pred, zero_division=0))

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 10))

    plt.plot(thresholds, precisions, label="Precision", color="green", linewidth=2.5)
    plt.plot(thresholds, recalls, label="Recall", color="red", linewidth=2.5)
    plt.plot(
        thresholds,
        f1_scores,
        label="F1-Score",
        color="blue",
        linestyle="--",
        linewidth=2,
    )
    plt.axvline(x=0.5, color="gray", linestyle=":", label="Default Threshold (0.5)")

    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel("Decision Threshold", fontsize=13)
    plt.ylabel("Score", fontsize=13)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.5, 1.05)
    plt.xticks(np.arange(0, 1.05, 0.1))
    plt.legend(loc="lower center", fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def main():
    set_seed(SEED)
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    results_path = ensure_dir(TEST_RESULTS_DIR)

    paths, labels, _ = prepare_dataset()
    _, val_paths, _, val_labels = split_dataset(paths, labels)
    nb_paths, nb_labels = load_certh_natural_blur()
    ebb_original_paths, ebb_bokeh_paths = load_ebb_validation()

    nb_duplicates = find_training_duplicates(nb_paths, paths)
    print(f"{NATURAL_BLUR}: {sum(nb_duplicates)} images duplicated in training data")

    print(f"Loading model from: {BEST_CHECKPOINT}")
    model = get_efficientnet(DEVICE, weights_path=str(BEST_CHECKPOINT))
    transform = get_val_transforms(img_size=IMAGE_SIZE)

    sources = {
        VAL: (val_paths, val_labels, None),
        NATURAL_BLUR: (nb_paths, nb_labels, nb_duplicates),
        EBB_ORIGINAL: (ebb_original_paths, [0] * len(ebb_original_paths), None),
        EBB_BOKEH: (ebb_bokeh_paths, [0] * len(ebb_bokeh_paths), None),
    }

    frames = []
    for name, (source_paths, source_labels, duplicates) in sources.items():
        blur_probs, laplacian_scores = score_images(
            source_paths, model, transform, DEVICE, desc=name
        )
        frames.append(
            pd.DataFrame(
                {
                    "dataset": name,
                    "path": [os.path.relpath(p, DATASETS_DIR) for p in source_paths],
                    "label": source_labels,
                    "duplicate_in_training": duplicates,
                    "blur_prob": blur_probs,
                    "laplacian": laplacian_scores,
                }
            )
        )

    scores = pd.concat(frames, ignore_index=True)
    scores.to_csv(results_path / f"{EXPERIMENT}_scores.csv", index=False)

    val_df = scores[scores["dataset"] == VAL]
    nb_df = scores[scores["dataset"] == NATURAL_BLUR]
    nb_clean_df = nb_df[~nb_df["duplicate_in_training"].astype(bool)]

    laplacian_threshold = tune_laplacian_threshold(val_df)
    print(f"Laplacian threshold tuned on validation split: {laplacian_threshold}")

    metrics = pd.DataFrame(
        classification_metrics(val_df, VAL, laplacian_threshold)
        + classification_metrics(nb_df, NATURAL_BLUR, laplacian_threshold)
        + classification_metrics(nb_clean_df, NATURAL_BLUR_CLEAN, laplacian_threshold)
    )
    metrics.to_csv(results_path / f"{EXPERIMENT}_metrics.csv", index=False)

    ebb_original_df = scores[scores["dataset"] == EBB_ORIGINAL]
    ebb_bokeh_df = scores[scores["dataset"] == EBB_BOKEH]

    rejections = pd.DataFrame(
        false_rejection_rates(ebb_original_df, EBB_ORIGINAL, laplacian_threshold)
        + false_rejection_rates(ebb_bokeh_df, EBB_BOKEH, laplacian_threshold)
    )
    rejections.to_csv(
        results_path / f"{EXPERIMENT}_ebb_false_rejections.csv", index=False
    )

    plot_confusion_matrix(
        nb_clean_df,
        f"Confusion Matrix – {NATURAL_BLUR_CLEAN}",
        results_path / f"{EXPERIMENT}_CERTH_NaturalBlur_CM.png",
    )
    plot_roc_curves(
        {"validation": val_df, "NaturalBlur": nb_clean_df},
        results_path / f"{EXPERIMENT}_ROC.png",
    )
    plot_thresholds(
        nb_clean_df,
        f"Metrics across Decision Thresholds – {NATURAL_BLUR_CLEAN}",
        results_path / f"{EXPERIMENT}_CERTH_NaturalBlur_threshold_analysis.png",
    )

    pd.set_option("display.width", 200)
    print("\nRESULTS")
    print("=" * 50)
    print(metrics.round(4).to_string(index=False))
    print()
    print(rejections.round(4).to_string(index=False))
    print(f"\nSaved results to: {results_path}")


if __name__ == "__main__":
    main()
