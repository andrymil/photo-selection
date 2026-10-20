import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms
from tqdm import tqdm

from composition.config import IMAGE_SIZE, KEEP_ASPECT_RATIO

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_composition_model(device, weights_path=None, initial_score=None):
    weights = None if weights_path else models.EfficientNet_V2_S_Weights.DEFAULT
    model = models.efficientnet_v2_s(weights=weights)

    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 1)

    if initial_score is not None:
        nn.init.constant_(model.classifier[1].bias, initial_score)

    if weights_path:
        model.load_state_dict(torch.load(weights_path, map_location=device))

    return model.to(device)


def predict_scores(model, loader, device):
    model.eval()
    predictions, targets = [], []

    with torch.no_grad():
        for images, batch_targets in tqdm(loader, desc="Predicting"):
            predictions.append(model(images.to(device)).squeeze(1).cpu())
            targets.append(batch_targets)

    return torch.cat(predictions).numpy(), torch.cat(targets).numpy()


def pad_to_square(image):
    size = (IMAGE_SIZE, IMAGE_SIZE)
    return ImageOps.pad(image, size, method=Image.Resampling.BILINEAR)


def get_resize():
    if KEEP_ASPECT_RATIO:
        return transforms.Lambda(pad_to_square)

    return transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))


def get_train_transforms():
    return transforms.Compose(
        [
            get_resize(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_eval_transforms():
    return transforms.Compose(
        [
            get_resize(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
