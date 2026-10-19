import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from common.paths import DATASETS_DIR, ensure_dir
from sharpness.config import IMAGE_SIZE, BEST_CHECKPOINT, EVALUATION_RESULTS_DIR
from sharpness.model import get_efficientnet, get_val_transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

EXPERIMENT = BEST_CHECKPOINT.stem
TEST_IMAGE_PATH = DATASETS_DIR / "CUHK" / "blurred" / "motion0284.jpg"


def main():
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    weights_path = BEST_CHECKPOINT

    model = get_efficientnet(DEVICE, weights_path=str(weights_path))
    model.eval()

    target_layers = [model.features[-1]]

    transform = get_val_transforms(img_size=IMAGE_SIZE)
    img_pil = Image.open(TEST_IMAGE_PATH).convert("RGB")
    input_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)

    rgb_img = img_pil.resize((IMAGE_SIZE, IMAGE_SIZE))
    rgb_img = np.float32(rgb_img) / 255.0

    print("Generating Heatmap...")
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)
        grayscale_cam = grayscale_cam[0, :]

    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(rgb_img)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    plt.title("Grad-CAM (Model Focus)")
    plt.axis("off")

    results_path = ensure_dir(EVALUATION_RESULTS_DIR)
    path = results_path / f"{EXPERIMENT}_heatmap.jpg"
    plt.savefig(path, bbox_inches="tight", dpi=300)
    print(f"Saved result to {path}")

    plt.show()


if __name__ == "__main__":
    main()
