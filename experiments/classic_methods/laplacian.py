import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from utils.path import calculate_path

IMG_DIR = "datasets/CUHK/image/"
MASK_DIR = "datasets/CUHK/gt/"

RESULTS_PATH = calculate_path(__file__)


def calculate_sharpness(region):
    mask = region > 0

    if not np.any(mask):
        return 0.0

    laplacian = cv2.Laplacian(region, cv2.CV_64F)
    return np.var(laplacian[mask])


def analyze_image(filename, show_results=False):
    img_path = os.path.join(IMG_DIR, filename)
    mask_path = os.path.join(MASK_DIR, filename.replace(".jpg", ".png"))

    if not os.path.exists(mask_path):
        return None

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    mask_inv = cv2.bitwise_not(mask_binary)
    foreground = cv2.bitwise_and(img, img, mask=mask_inv)
    background = cv2.bitwise_and(img, img, mask=mask_binary)

    score_global = cv2.Laplacian(img, cv2.CV_64F).var()
    score_foreground = calculate_sharpness(foreground)
    score_background = calculate_sharpness(background)

    if show_results:
        print(f"--- Analysis of file: {os.path.basename(filename)} ---")
        print(f"Overall score: {score_global:.2f}")
        print(f"Foreground score: {score_foreground:.2f}")
        print(f"Background score: {score_background:.2f}")

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.title(f"Overall (Sharpness: {score_global:.0f})")
        plt.imshow(img, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title(f"Foreground (Sharpness: {score_foreground:.0f})")
        plt.imshow(foreground, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title(f"Background (Sharpness: {score_background:.0f})")
        plt.imshow(background, cmap="gray")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(RESULTS_PATH / "sharpness_of_example_image.png")
        plt.show()

    return {
        "file": filename,
        "overall": score_global,
        "foreground": score_foreground,
        "background": score_background,
        "ratio": score_foreground / score_global if score_global > 0 else 0,
    }


analyze_image("motion0001.jpg", show_results=True)

results = []
for filename in os.listdir(IMG_DIR):
    if not filename.endswith(".jpg"):
        continue

    result = analyze_image(filename, show_results=False)

    if result is not None:
        results.append(result)

print(f"Analyzed {len(results)} images. Generating plots...")

overalls = [r["overall"] for r in results]
foregrounds = [r["foreground"] for r in results]

max_val = np.percentile(overalls + foregrounds, 95)

plt.figure(figsize=(12, 6))
plt.hist(
    overalls,
    bins=50,
    range=(0, max_val),
    alpha=0.6,
    color="red",
    label="Całe zdjęcie",
)
plt.hist(
    foregrounds,
    bins=50,
    range=(0, max_val),
    alpha=0.6,
    color="blue",
    label="Tylko główny obiekt",
)

plt.title("Rozkład ostrości", fontsize=14)
plt.xlabel("Wartość wariancji Laplasjanu", fontsize=12)
plt.ylabel("Liczba zdjęć", fontsize=12)
plt.legend(fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(RESULTS_PATH / "sharpness_distribution.png")
plt.show()

plt.figure(figsize=(8, 8))
plt.scatter(overalls, foregrounds, alpha=0.5, color="purple", edgecolors="w", s=50)

plt.plot([0, max_val], [0, max_val], "k--", lw=2, label="Linia y=x (Brak różnicy)")

plt.title("Korelacja ostrości: Całe zdjęcie vs Obiekt", fontsize=14)
plt.xlabel("Ostrość całości zdjęcia", fontsize=12)
plt.ylabel("Ostrość wyciętego obiektu", fontsize=12)

plt.xlim(0, max_val)
plt.ylim(0, max_val)

plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(RESULTS_PATH / "sharpness_corelation.png")
plt.show()
