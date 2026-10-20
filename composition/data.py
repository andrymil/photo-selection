import json

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from composition.config import IMAGES_DIR, SCORES_PATH, SPLIT_PATH, SEED, VAL_SIZE


class CompositionDataset(Dataset):
    def __init__(self, image_names, scores, transform=None):
        self.image_names = image_names
        self.scores = scores
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image = Image.open(IMAGES_DIR / image_name).convert("RGB")
        score = torch.tensor(self.scores[image_name], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, score


def load_scores():
    with open(SCORES_PATH) as file:
        annotations = json.load(file)

    return {name: annotation["mean"] for name, annotation in annotations.items()}


def load_splits():
    with open(SPLIT_PATH) as file:
        official_split = json.load(file)

    train_names, val_names = train_test_split(
        official_split["train"], test_size=VAL_SIZE, random_state=SEED
    )

    return {"train": train_names, "val": val_names, "test": official_split["test"]}


def print_summary():
    scores = load_scores()

    for split_name, image_names in load_splits().items():
        split_scores = np.array([scores[name] for name in image_names])
        print(
            f"{split_name:5s}: {len(image_names):4d} images | "
            f"mean score {split_scores.mean():.2f} | std {split_scores.std():.2f}"
        )


if __name__ == "__main__":
    print_summary()
