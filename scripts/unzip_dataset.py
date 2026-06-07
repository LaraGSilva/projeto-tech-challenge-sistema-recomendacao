# scripts/unzip_dataset.py
import zipfile
from pathlib import Path

def unzip_dataset(zip_path: Path, output_dir: Path) -> None:
    """Extrai o dataset RetailRocket para data/interim/."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(output_dir)

if __name__ == "__main__":
    unzip_dataset(
        zip_path=Path("data/raw/ecommerce-dataset.zip"),
        output_dir=Path("data/interim"),
    )