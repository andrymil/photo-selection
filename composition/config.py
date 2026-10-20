from common.paths import DATASETS_DIR

SEED = 42
VAL_SIZE = 0.1

IMAGE_SIZE = 224

CADB_DIR = DATASETS_DIR / "CADB" / "CADB_Dataset"
IMAGES_DIR = CADB_DIR / "images"
SCORES_PATH = CADB_DIR / "composition_scores.json"
SPLIT_PATH = CADB_DIR / "split.json"
