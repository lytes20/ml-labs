"""Download the Diabetes Health Indicators Dataset from Kaggle into project/data/."""

import shutil
from pathlib import Path

import kagglehub

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main():
    DATA_DIR.mkdir(exist_ok=True)
    cache_path = Path(kagglehub.dataset_download("alexteboul/diabetes-health-indicators-dataset"))

    for csv_file in cache_path.glob("*.csv"):
        dest = DATA_DIR / csv_file.name
        shutil.copy(csv_file, dest)
        print(f"Copied {csv_file.name} -> {dest}")


if __name__ == "__main__":
    main()
