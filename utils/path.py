from pathlib import Path


def calculate_path(file: str | Path, path="results") -> Path:
    script_dir = Path(file).parent.resolve()
    results_dir = script_dir / path

    results_dir.mkdir(parents=True, exist_ok=True)

    return results_dir
