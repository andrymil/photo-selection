from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATASETS_DIR = ROOT_DIR / "datasets"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
RESULTS_DIR = ROOT_DIR / "results"

YOLO_WEIGHTS = ARTIFACTS_DIR / "pretrained" / "yolov8n.pt"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)

    return path
