import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.paths import ensure_dir
from common.seed import set_seed
from composition.config import (
    SEED,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_WORKERS,
    EXPERIMENT_NAME,
    BEST_CHECKPOINT,
    TRAINING_RESULTS_DIR,
)
from composition.data import CompositionDataset, load_scores, load_splits
from composition.metrics import compute_metrics
from composition.model import (
    get_composition_model,
    get_eval_transforms,
    get_train_transforms,
    predict_scores,
)
from composition.plots import plot_training_history


def create_loaders(scores, splits):
    train_dataset = CompositionDataset(splits["train"], scores, get_train_transforms())
    val_dataset = CompositionDataset(splits["val"], scores, get_eval_transforms())

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for images, targets in tqdm(loader, desc="Training"):
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        loss = criterion(model(images).squeeze(1), targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def train(model, train_loader, val_loader, device):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    ensure_dir(BEST_CHECKPOINT.parent)
    history = []
    best_srcc = -1.0

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        train_mse = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = compute_metrics(*predict_scores(model, val_loader, device))
        scheduler.step(val_metrics["srcc"])

        print(
            f"Train MSE: {train_mse:.4f} | Val MSE: {val_metrics['mse']:.4f} | "
            f"Val SRCC: {val_metrics['srcc']:.4f} | Val PLCC: {val_metrics['plcc']:.4f}"
        )
        history.append(
            {
                "epoch": epoch,
                "train_mse": train_mse,
                **{f"val_{name}": value for name, value in val_metrics.items()},
            }
        )

        if val_metrics["srcc"] > best_srcc:
            best_srcc = val_metrics["srcc"]
            torch.save(model.state_dict(), BEST_CHECKPOINT)
            print("New best SRCC! Model saved.")

    print(f"\nBest val SRCC: {best_srcc:.4f}")

    return pd.DataFrame(history)


def save_history(history):
    results_dir = ensure_dir(TRAINING_RESULTS_DIR)

    history.to_csv(results_dir / f"{EXPERIMENT_NAME}.csv", index=False)
    plot_training_history(history, results_dir / f"{EXPERIMENT_NAME}_plot.png")
    print(f"Results saved to {results_dir}")


def main():
    set_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    scores, splits = load_scores(), load_splits()
    train_loader, val_loader = create_loaders(scores, splits)

    train_mean = float(np.mean([scores[name] for name in splits["train"]]))
    model = get_composition_model(device, initial_score=train_mean)

    history = train(model, train_loader, val_loader, device)
    save_history(history)


if __name__ == "__main__":
    main()
