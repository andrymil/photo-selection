import matplotlib.pyplot as plt
import seaborn as sns


def plot_training_history(df, save_path):
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training Metrics History", fontsize=18, fontweight="bold", y=1.02)

    def draw_subplot(ax, metric_key, title, ylabel):
        sns.lineplot(
            data=df,
            x="epoch",
            y=metric_key,
            hue="phase",
            marker="o",
            linewidth=2,
            ax=ax,
        )
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xticks(df["epoch"].unique())
        ax.legend(title="Phase", fontsize=11)

    draw_subplot(axes[0, 0], "loss", "Model Loss (Cross Entropy)", "Loss")

    draw_subplot(axes[0, 1], "f1", "F1-Score", "Score")

    draw_subplot(axes[1, 0], "accuracy", "Accuracy", "Score")

    ax_pr = axes[1, 1]

    df_pr = df.melt(
        id_vars=["epoch", "phase"],
        value_vars=["recall", "precision"],
        var_name="metric",
        value_name="score",
    )

    df_pr["metric"] = df_pr["metric"].str.capitalize()

    sns.lineplot(
        data=df_pr,
        x="epoch",
        y="score",
        hue="phase",
        style="metric",
        markers=["o", "s"],
        dashes=True,
        linewidth=2,
        ax=ax_pr,
    )

    ax_pr.set_title("Precision vs Recall", fontsize=14, fontweight="bold")
    ax_pr.set_xlabel("Epoch", fontsize=12)
    ax_pr.set_ylabel("Score", fontsize=12)
    ax_pr.set_xticks(df["epoch"].unique())
    ax_pr.legend(title="Phase & Metric", fontsize=10, loc="lower right")

    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved successfully to {save_path}")
