import matplotlib.pyplot as plt
import seaborn as sns


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
