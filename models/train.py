import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from utils.seed import set_seed
from utils.data import prepare_dataset
from utils.path import calculate_path
from utils.model import get_efficientnet, get_train_transforms, get_val_transforms
from utils.plot import plot_training_history

IMAGE_SIZE = 384
BATCH_SIZE = 16
ACCUMULATION_STEPS = 1
VIRTUAL_BATCH_SIZE = BATCH_SIZE * ACCUMULATION_STEPS

EXPERIMENT_NAME = (
    f"EfficientNetV2S_{IMAGE_SIZE}px_batch{VIRTUAL_BATCH_SIZE}_without_NaturalBlur"
)
BASE_DIR = "datasets"


class BlurDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def main():
    set_seed(42)

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    paths, labels, weight_for_class_1 = prepare_dataset(BASE_DIR)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    train_transform = get_train_transforms(img_size=IMAGE_SIZE)
    val_transform = get_val_transforms(img_size=IMAGE_SIZE)

    train_dataset = BlurDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = BlurDataset(val_paths, val_labels, transform=val_transform)

    dataset_sizes = {"train": len(train_dataset), "val": len(val_dataset)}
    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True),
        "val": DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False),
    }

    model = get_efficientnet(DEVICE, eval_mode=False)

    weights = torch.tensor([1.0, weight_for_class_1], dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    metric_collection = torchmetrics.MetricCollection(
        [
            torchmetrics.Accuracy(task="binary"),
            torchmetrics.Precision(task="binary"),
            torchmetrics.Recall(task="binary"),
            torchmetrics.F1Score(task="binary"),
        ]
    ).to(DEVICE)

    history = []
    EPOCHS = 10
    best_val_f1 = 0.0

    checkpoints_path = calculate_path(__file__, "checkpoints")
    checkpoints_path.mkdir(parents=True, exist_ok=True)

    print("\nStarting training...")

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            metric_collection.reset()

            for i, (inputs, labels) in enumerate(
                tqdm(dataloaders[phase], desc=f"{phase.capitalize()} phase")
            ):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                if phase == "train" and i % ACCUMULATION_STEPS == 0:
                    optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    preds = torch.argmax(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss = loss / ACCUMULATION_STEPS
                        loss.backward()

                        if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(
                            dataloaders[phase]
                        ):
                            optimizer.step()

                running_loss += (
                    loss.item() * ACCUMULATION_STEPS
                    if phase == "train"
                    else loss.item()
                ) * inputs.size(0)
                metric_collection.update(preds, labels)

            epoch_loss = running_loss / dataset_sizes[phase]
            total_metrics = metric_collection.compute()

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f}")
            print(
                f"Acc: {total_metrics['BinaryAccuracy']:.4f} | F1: {total_metrics['BinaryF1Score']:.4f}"
            )
            print(
                f"Recall: {total_metrics['BinaryRecall']:.4f} | Prec: {total_metrics['BinaryPrecision']:.4f}"
            )

            if phase == "val":
                scheduler.step(epoch_loss)

                current_f1 = total_metrics["BinaryF1Score"].item()
                if current_f1 > best_val_f1:
                    best_val_f1 = current_f1
                    best_model_path = checkpoints_path / f"{EXPERIMENT_NAME}_BEST.pth"
                    torch.save(model.state_dict(), best_model_path)
                    print("New best F1-Score! Model saved.")

            history.append(
                {
                    "epoch": epoch + 1,
                    "phase": phase,
                    "loss": epoch_loss,
                    "accuracy": total_metrics["BinaryAccuracy"].item(),
                    "f1": total_metrics["BinaryF1Score"].item(),
                    "recall": total_metrics["BinaryRecall"].item(),
                    "precision": total_metrics["BinaryPrecision"].item(),
                }
            )

    print(f"\nBest val F1 score: {best_val_f1:.4f}")
    print("\nTraining complete! Saving results...")

    results_path = calculate_path(__file__)

    df = pd.DataFrame(history)
    csv_filename = results_path / f"{EXPERIMENT_NAME}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Metrics saved to {csv_filename}")

    plot_filename = results_path / f"{EXPERIMENT_NAME}_plot.png"
    plot_training_history(df, plot_filename)

    checkpoint_filename = checkpoints_path / f"{EXPERIMENT_NAME}_LAST.pth"
    torch.save(model.state_dict(), checkpoint_filename)
    print(f"Model weights saved to {checkpoint_filename}")


if __name__ == "__main__":
    main()
