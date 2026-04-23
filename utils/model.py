import torch
import torch.nn as nn
from torchvision import models, transforms


def get_resnet18(device, weights_path=None, eval_mode=True):
    pretrained = weights_path is None and not eval_mode
    model = models.resnet18(pretrained=pretrained)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    if weights_path:
        model.load_state_dict(torch.load(weights_path, map_location=device))

    model = model.to(device)

    if eval_mode:
        model.eval()

    return model


def get_efficientnet(device, weights_path=None, eval_mode=True):
    pretrained = weights_path is None and not eval_mode
    weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None

    model = models.efficientnet_v2_s(weights=weights)

    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 2)

    if weights_path:
        model.load_state_dict(torch.load(weights_path, map_location=device))

    model = model.to(device)

    if eval_mode:
        model.eval()

    return model


def get_train_transforms(img_size=384):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def get_val_transforms(img_size=384):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def get_standard_transforms():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def predict_ai(model, image_pil, transform, device):
    img_t = transform(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_t)
        pred = torch.argmax(outputs, 1).item()
    return pred
