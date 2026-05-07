import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from utils.path import calculate_path
from utils.model import get_efficientnet, get_val_transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

IMAGE_SIZE = 384
EXPERIMENT = "EfficientNetV2S_384px_batch16_with_EBB_BEST"
TEST_IMAGE_PATH = "datasets/CUHK/sharp/out_of_focus0245.jpg"


def main():
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    weights_path = f"models/checkpoints/{EXPERIMENT}.pth"

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

    results_path = calculate_path(__file__)
    path = results_path / f"{EXPERIMENT}_heatmap.jpg"
    plt.savefig(path, bbox_inches="tight", dpi=300)
    print(f"Saved result to {path}")

    plt.show()


if __name__ == "__main__":
    main()
