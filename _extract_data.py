"""Extract data/zipped.zip into data/, flattening any internal folder structure."""

import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ZIP_PATH = BASE_DIR / "inputs_zipped.zip"
OUT_DIR = BASE_DIR / "data" / "inputs"


def extract_flat(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            dest = out_dir / Path(info.filename).name
            with zf.open(info) as src, open(dest, "wb") as dst:
                dst.write(src.read())


if __name__ == "__main__":
    extract_flat(ZIP_PATH, OUT_DIR)
