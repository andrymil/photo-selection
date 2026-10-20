"""Blind audit of CERTH NaturalBlurSet labels: the CERTH label and model prediction are hidden.

Left: whole photo, right: 100% crop of its most detailed region.
"""

import cv2
import numpy as np
import pandas as pd

from common.paths import DATASETS_DIR, ensure_dir
from sharpness.config import SEED, AUDIT_RESULTS_DIR, TEST_RESULTS_DIR, BEST_CHECKPOINT
from sharpness.evaluation.evaluate_test_set import NATURAL_BLUR

SCORES_PATH = TEST_RESULTS_DIR / f"{BEST_CHECKPOINT.stem}_scores.csv"
AUDIT_PATH = AUDIT_RESULTS_DIR / "certh_natural_blur_audit.csv"

SAMPLES_PER_GROUP = 25
OVERVIEW_SIZE = (1700, 1250)
DETAIL_SIZE = 800
TILE_SIZE = 400

WINDOW_NAME = "Label audit - s: sharp | b: blurred | u: unsure | q: quit"
KEY_LABELS = {ord("s"): "sharp", ord("b"): "blurred", ord("u"): "unsure"}
QUIT_KEY = ord("q")
CERTH_LABEL_NAMES = {0: "sharp", 1: "blurred"}


def load_test_images():
    scores = pd.read_csv(SCORES_PATH)
    test_images = scores[scores["dataset"] == NATURAL_BLUR]

    return test_images[~test_images["duplicate_in_training"].astype(bool)]


def sample_audit_images(test_images):
    blurred = test_images[test_images["label"] == 1]
    groups = {
        "certh_blurred_model_missed": blurred[blurred["blur_prob"] < 0.5],
        "certh_blurred_model_detected": blurred[blurred["blur_prob"] >= 0.5],
        "certh_sharp": test_images[test_images["label"] == 0],
    }

    samples = pd.concat(
        images.sample(SAMPLES_PER_GROUP, random_state=SEED).assign(group=name)
        for name, images in groups.items()
    )
    samples = samples.rename(columns={"label": "certh_label"})

    return samples.sample(frac=1, random_state=SEED)[["path", "certh_label", "group"]]


def fit_into(image, width, height):
    scale = min(width / image.shape[1], height / image.shape[0])
    size = (round(image.shape[1] * scale), round(image.shape[0] * scale))
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[: resized.shape[0], : resized.shape[1]] = resized

    return canvas


def find_most_detailed_center(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tiles = [
        (y, x)
        for y in range(0, max(1, gray.shape[0] - TILE_SIZE + 1), TILE_SIZE)
        for x in range(0, max(1, gray.shape[1] - TILE_SIZE + 1), TILE_SIZE)
    ]

    def detail(tile):
        y, x = tile
        return cv2.Laplacian(
            gray[y : y + TILE_SIZE, x : x + TILE_SIZE], cv2.CV_64F
        ).var()

    y, x = max(tiles, key=detail)

    return y + TILE_SIZE // 2, x + TILE_SIZE // 2


def crop_detail(image):
    center_y, center_x = find_most_detailed_center(image)
    top = int(
        np.clip(center_y - DETAIL_SIZE // 2, 0, max(0, image.shape[0] - DETAIL_SIZE))
    )
    left = int(
        np.clip(center_x - DETAIL_SIZE // 2, 0, max(0, image.shape[1] - DETAIL_SIZE))
    )

    return image[top : top + DETAIL_SIZE, left : left + DETAIL_SIZE]


def compose_view(image, progress):
    overview = fit_into(image, *OVERVIEW_SIZE)
    detail = fit_into(crop_detail(image), DETAIL_SIZE, OVERVIEW_SIZE[1])
    view = np.hstack([overview, detail])

    cv2.putText(view, progress, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    return view


def ask_user_label(view):
    cv2.imshow(WINDOW_NAME, view)

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == QUIT_KEY:
            return None
        if key in KEY_LABELS:
            return KEY_LABELS[key]


def load_audit_rows():
    if not AUDIT_PATH.exists():
        return []

    return pd.read_csv(AUDIT_PATH).to_dict("records")


def save_audit_rows(rows):
    ensure_dir(AUDIT_PATH.parent)
    pd.DataFrame(rows).to_csv(AUDIT_PATH, index=False)


def run_audit(samples, rows):
    audited_paths = {row["path"] for row in rows}
    pending = samples[~samples["path"].isin(audited_paths)]
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    for row in pending.to_dict("records"):
        image = cv2.imread(str(DATASETS_DIR / row["path"]))
        user_label = ask_user_label(
            compose_view(image, f"{len(rows) + 1}/{len(samples)}")
        )

        if user_label is None:
            break

        rows.append({**row, "user_label": user_label})
        save_audit_rows(rows)

    cv2.destroyAllWindows()

    return rows


def print_summary(rows):
    if not rows:
        print("No images audited yet.")
        return

    audit = pd.DataFrame(rows)
    print(pd.crosstab(audit["group"], audit["user_label"], margins=True))

    decided = audit[audit["user_label"] != "unsure"]
    agrees = decided["user_label"] == decided["certh_label"].map(CERTH_LABEL_NAMES)
    print(f"\nAgreement with CERTH labels: {agrees.mean():.0%} ({len(decided)} images)")


def main():
    samples = sample_audit_images(load_test_images())
    rows = run_audit(samples, load_audit_rows())
    print_summary(rows)


if __name__ == "__main__":
    main()
