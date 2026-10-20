import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from composition.config import IMAGES_DIR


def plot_training_history(history, path):
    sns.set_theme(style="whitegrid")
    fig, (ax_loss, ax_corr) = plt.subplots(1, 2, figsize=(14, 5))

    ax_loss.plot(history["epoch"], history["train_mse"], marker="o", label="Trening")
    ax_loss.plot(history["epoch"], history["val_mse"], marker="o", label="Walidacja")
    ax_loss.set_title("Błąd średniokwadratowy (MSE)", fontweight="bold")
    ax_loss.set_xlabel("Epoka")
    ax_loss.set_ylabel("MSE")
    ax_loss.legend()

    ax_corr.plot(history["epoch"], history["val_srcc"], marker="o", label="SRCC")
    ax_corr.plot(history["epoch"], history["val_plcc"], marker="s", label="PLCC")
    ax_corr.set_title("Zgodność z ocenami ludzi (walidacja)", fontweight="bold")
    ax_corr.set_xlabel("Epoka")
    ax_corr.set_ylabel("Korelacja")
    ax_corr.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


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


def plot_gallery(rows, path):
    columns = max(len(images) for images in rows.values())
    fig = plt.figure(figsize=(3 * columns, 4 * len(rows)), layout="constrained")

    for subfig, (title, images) in zip(fig.subfigures(len(rows), 1), rows.items()):
        subfig.suptitle(title, fontweight="bold", fontsize=14)

        for ax, image in zip(subfig.subplots(1, columns), images.itertuples()):
            ax.imshow(Image.open(IMAGES_DIR / image.image).convert("RGB"))
            ax.set_xlabel(
                f"model: {image.model_score:.2f} | ludzie: {image.human_score:.2f}"
            )
            ax.set_xticks([])
            ax.set_yticks([])

    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_scene_srcc(scene_metrics, overall_srcc, path):
    ordered = scene_metrics.sort_values("srcc")
    labels = [
        f"{name} ({count})"
        for name, count in zip(ordered["scene_pl"], ordered["images"])
    ]

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(9, 6))
    plt.barh(labels, ordered["srcc"], color="#4C72B0")
    plt.axvline(
        overall_srcc,
        color="black",
        linestyle="--",
        label=f"Cały zbiór testowy ({overall_srcc:.3f})",
    )

    plt.title("Zgodność z ocenami ludzi według kategorii sceny", fontweight="bold")
    plt.xlabel("SRCC")
    plt.ylabel("Kategoria sceny (liczba zdjęć)")
    plt.xlim(0, 1)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


OUR_MODEL = "Nasz model"
SAMP_NET = "SAMP-Net"
AGREEMENT_COLORS = {OUR_MODEL: "#4C72B0", SAMP_NET: "#999999"}
RATER_COLOR = "#55A868"


def plot_agreement(agreement, path):
    judges = {
        f"Oceniający {number}": srcc
        for number, srcc in enumerate(agreement["raters"], start=1)
    }
    judges[OUR_MODEL] = agreement["ours_vs_raters"]
    judges[SAMP_NET] = agreement["samp_net"]
    ranking = sorted(judges.items(), key=lambda judge: judge[1])

    names = [name for name, _ in ranking]
    values = [value for _, value in ranking]
    colors = [AGREEMENT_COLORS.get(name, RATER_COLOR) for name in names]

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(9, 5))
    plt.barh(names, values, color=colors)

    for y, value in enumerate(values):
        plt.text(value + 0.01, y, f"{value:.3f}", va="center")

    plt.title("Zgodność z oceną ludzi (zbiór testowy CADB)", fontweight="bold")
    plt.xlabel("Zgodność kolejności zdjęć ze średnią oceną pozostałych 4 osób (SRCC)")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
