import torch
import torch.nn as nn
from torchvision import models, transforms

from composition.config import IMAGE_SIZE

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


def get_train_transforms():
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_eval_transforms():
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
