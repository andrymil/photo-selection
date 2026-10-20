import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common.paths import ensure_dir
from composition.config import (
    SEED,
    EXPERIMENT_NAME,
    SCORES_PATH,
    SCENES_PATH,
    TEST_RESULTS_DIR,
    GALLERY_SIZE,
    SAMP_NET_RESULTS,
)
from composition.metrics import compute_metrics
from composition.plots import plot_agreement, plot_gallery, plot_scene_srcc

PREDICTIONS_PATH = TEST_RESULTS_DIR / f"{EXPERIMENT_NAME}_predictions.csv"
BOOTSTRAP_SAMPLES = 2000

SCENE_NAMES = {
    "human": "ludzie",
    "static": "martwa natura",
    "animal": "zwierzęta",
    "landscape": "krajobraz",
    "plant": "rośliny",
    "architecture": "architektura",
    "cityscape": "miasto",
    "other": "inne",
    "night": "noc",
    "indoor": "wnętrza",
}


def load_json(path):
    with open(path) as file:
        return json.load(file)


def compute_srcc(results):
    return compute_metrics(
        results["model_score"].to_numpy(), results["human_score"].to_numpy()
    )["srcc"]


def compute_scene_metrics(results):
    scenes = load_json(SCENES_PATH)
    results = results.assign(scene=results["image"].map(scenes))
    rows = []

    for scene, group in results.groupby("scene"):
        metrics = compute_metrics(
            group["model_score"].to_numpy(), group["human_score"].to_numpy()
        )
        rows.append(
            {
                "scene": scene,
                "scene_pl": SCENE_NAMES[scene],
                "images": len(group),
                "human_mean": group["human_score"].mean(),
                "model_mean": group["model_score"].mean(),
                **metrics,
            }
        )

    return pd.DataFrame(rows).sort_values("images", ascending=False)


def bootstrap_srcc(results):
    rng = np.random.default_rng(SEED)
    samples = []

    for _ in range(BOOTSTRAP_SAMPLES):
        indices = rng.integers(0, len(results), len(results))
        samples.append(compute_srcc(results.iloc[indices]))

    return np.percentile(samples, [2.5, 97.5])


def load_ratings(image_names):
    annotations = load_json(SCORES_PATH)
    return np.array([annotations[name]["scores"] for name in image_names])


def mean_of_others(ratings, rater):
    return np.delete(ratings, rater, axis=1).mean(axis=1)


def compute_rater_agreement(ratings):
    return [
        float(spearmanr(ratings[:, rater], mean_of_others(ratings, rater)).statistic)
        for rater in range(ratings.shape[1])
    ]


def compute_model_agreement(results, ratings):
    model_scores = results["model_score"].to_numpy()
    agreement = [
        spearmanr(model_scores, mean_of_others(ratings, rater)).statistic
        for rater in range(ratings.shape[1])
    ]

    return float(np.mean(agreement))


def compute_agreement(results):
    ratings = load_ratings(results["image"])

    return {
        "samp_net": SAMP_NET_RESULTS["srcc"],
        "ours": compute_srcc(results),
        "ours_ci": bootstrap_srcc(results),
        "raters": compute_rater_agreement(ratings),
        "ours_vs_raters": compute_model_agreement(results, ratings),
    }


def agreement_table(agreement):
    low, high = agreement["ours_ci"]
    rows = [
        {"name": "SAMP-Net (Zhang et al., 2021)", "srcc": agreement["samp_net"]},
        {"name": "Ours", "srcc": agreement["ours"], "ci_low": low, "ci_high": high},
        {
            "name": "Ours vs mean of 4 raters (average)",
            "srcc": agreement["ours_vs_raters"],
        },
    ]
    rows += [
        {"name": f"Rater {number} vs mean of others", "srcc": srcc}
        for number, srcc in enumerate(agreement["raters"], start=1)
    ]

    return pd.DataFrame(rows)


def find_largest_errors(results):
    errors = results.assign(error=results["model_score"] - results["human_score"])
    ranked = errors.sort_values("error")

    return {
        "Model ocenia za wysoko": ranked.tail(GALLERY_SIZE).iloc[::-1],
        "Model ocenia za nisko": ranked.head(GALLERY_SIZE),
    }


def main():
    results = pd.read_csv(PREDICTIONS_PATH)
    results_dir = ensure_dir(TEST_RESULTS_DIR)

    scene_metrics = compute_scene_metrics(results)
    agreement = compute_agreement(results)

    scene_metrics.to_csv(results_dir / f"{EXPERIMENT_NAME}_scenes.csv", index=False)
    agreement_table(agreement).to_csv(
        results_dir / f"{EXPERIMENT_NAME}_agreement.csv", index=False
    )

    plot_scene_srcc(
        scene_metrics,
        agreement["ours"],
        results_dir / f"{EXPERIMENT_NAME}_scenes.png",
    )
    plot_agreement(agreement, results_dir / f"{EXPERIMENT_NAME}_agreement.png")
    plot_gallery(
        find_largest_errors(results), results_dir / f"{EXPERIMENT_NAME}_errors.png"
    )

    print(scene_metrics.round(3).to_string(index=False))
    print(agreement_table(agreement).round(3).to_string(index=False))
    print(f"Results saved to {results_dir}")


if __name__ == "__main__":
    main()
