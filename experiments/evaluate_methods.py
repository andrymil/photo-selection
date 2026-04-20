import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from ultralytics import YOLO
from utils.seed import set_seed
from utils.data import prepare_dataset
from utils.path import calculate_path
from utils.vision import get_laplacian_score, get_largest_bbox

EXPERIMENT_GLOBAL = "ResNet18_224px_batch32"
EXPERIMENT_ROI = "ResNet18_ROI_224px_batch32"
BASE_DIR = "datasets"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LAPLACIAN_THRESHOLD = 390

set_seed(42)

paths, labels, _ = prepare_dataset(BASE_DIR)

_, val_paths, _, val_labels = train_test_split(
    paths, labels, test_size=0.2, random_state=42, stratify=labels
)

print("Loading global model...")
resnet_global = models.resnet18(pretrained=False)
resnet_global.fc = nn.Linear(resnet_global.fc.in_features, 2)
resnet_global.load_state_dict(
    torch.load(f"models/checkpoints/{EXPERIMENT_GLOBAL}.pth", map_location=DEVICE)
)
resnet_global = resnet_global.to(DEVICE)
resnet_global.eval()

print("Loading ROI model...")
resnet_roi = models.resnet18(pretrained=False)
resnet_roi.fc = nn.Linear(resnet_roi.fc.in_features, 2)
resnet_roi.load_state_dict(
    torch.load(f"models/checkpoints/{EXPERIMENT_ROI}.pth", map_location=DEVICE)
)
resnet_roi = resnet_roi.to(DEVICE)
resnet_roi.eval()

print("Loading YOLOv8...")
yolo = YOLO("models/pretrained/yolov8n.pt")

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def predict_ai(model, image_pil):
    img_t = transform(image_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_t)
        pred = torch.argmax(outputs, 1).item()
    return pred


results = {
    "true_labels": val_labels,
    "pred_global_ai": [],
    "pred_roi_ai": [],
    "pred_global_lap": [],
    "pred_roi_lap": [],
}

print(f"Starting testing with {len(val_paths)} images...")

for path in tqdm(val_paths):
    img_pil = Image.open(path).convert("RGB")
    img_cv2 = cv2.imread(path)

    if img_cv2 is None:
        continue

    results["pred_global_ai"].append(predict_ai(resnet_global, img_pil))

    lap_score = get_laplacian_score(img_cv2)
    results["pred_global_lap"].append(1 if lap_score < LAPLACIAN_THRESHOLD else 0)

    yolo_results = yolo(img_cv2, verbose=False)[0]
    bbox = get_largest_bbox(yolo_results, img_cv2.shape)

    if bbox:
        x1, y1, x2, y2 = bbox
        crop_cv2 = img_cv2[y1:y2, x1:x2]
        crop_pil = img_pil.crop((x1, y1, x2, y2))
    else:
        crop_cv2 = img_cv2
        crop_pil = img_pil

    results["pred_roi_ai"].append(predict_ai(resnet_roi, crop_pil))

    roi_lap_score = get_laplacian_score(crop_cv2)
    results["pred_roi_lap"].append(1 if roi_lap_score < LAPLACIAN_THRESHOLD else 0)


print("\nRESULTS")
print("=" * 50)


def print_metrics(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"--- {name} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}\n")
    return acc, prec, rec, f1


y_true = results["true_labels"]

methods_names = [
    "Global AI (ResNet18)",
    "ROI AI (YOLO + ResNet18)",
    "Global Laplacian",
    "ROI Laplacian (YOLO)",
]
method_preds = [
    results["pred_global_ai"],
    results["pred_roi_ai"],
    results["pred_global_lap"],
    results["pred_roi_lap"],
]

acc_list, prec_list, rec_list, f1_list = [], [], [], []

for name, preds in zip(methods_names, method_preds):
    a, p, r, f = print_metrics(name, y_true, preds)
    acc_list.append(a)
    prec_list.append(p)
    rec_list.append(r)
    f1_list.append(f)

x = np.arange(len(methods_names))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 7))

rects1 = ax.bar(x - 1.5 * width, acc_list, width, label="Accuracy", color="#4C72B0")
rects2 = ax.bar(x - 0.5 * width, prec_list, width, label="Precision", color="#55A868")
rects3 = ax.bar(x + 0.5 * width, rec_list, width, label="Recall", color="#C44E52")
rects4 = ax.bar(x + 1.5 * width, f1_list, width, label="F1-Score", color="#8172B3")

ax.set_ylabel("Metric value", fontsize=12)
ax.set_title("Methods comparison", fontsize=15, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(methods_names, fontsize=11, fontweight="bold")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=11)
ax.set_ylim(0, 1.1)
ax.grid(axis="y", linestyle="--", alpha=0.7)


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            rotation=90,
        )


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

fig.tight_layout()

results_path = calculate_path(__file__)
path = results_path / "methods_comparison.png"
plt.savefig(path, dpi=300, bbox_inches="tight")
print(f"Saved plot to: {path}")
