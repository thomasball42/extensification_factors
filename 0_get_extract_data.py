"""Extract data/zipped.zip into data/, flattening any internal folder structure,
and fetch the global cropping intensity dataset (Liu et al. 2021, Sci Data,
doi:10.1038/s41597-021-01065-9) from figshare into data/inputs/cropping_intensity,
plus the SASAM-GC cropland percentage map (Lu et al. 2020, ESSD) that its
authors used as the cropland extent, into data/inputs/cropland_extent."""

import hashlib
import zipfile
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent
ZIP_PATH = BASE_DIR / "inputs_zipped.zip"
OUT_DIR = BASE_DIR / "data" / "inputs"

# Harvard Dataverse 403s the default python-requests User-Agent
HTTP_HEADERS = {"User-Agent": "extensification-factors"}

CROPPING_INTENSITY_ARTICLE_ID = 14099402  # figshare, doi:10.6084/m9.figshare.14099402
CROPPING_INTENSITY_DIR = OUT_DIR / "cropping_intensity"

CROPLAND_EXTENT_DOI = "doi:10.7910/DVN/ZWSFAA"  # Harvard Dataverse, SASAM-GC (~2010, 1/240 deg)
CROPLAND_EXTENT_FILES = ["Global-cropland-percentage-map.tif"]
CROPLAND_EXTENT_DIR = OUT_DIR / "cropland_extent"


def extract_flat(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            dest = out_dir / Path(info.filename).name
            with zf.open(info) as src, open(dest, "wb") as dst:
                dst.write(src.read())


def _md5(path: Path, chunk_size: int = 1 << 24) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def _download_verified(url: str, dest: Path, size: int, md5: str | None) -> None:
    """Stream `url` to `dest` via a .part file, md5-checking it if `md5` is
    given. Skipped when `dest` already exists with the expected size."""
    if dest.exists() and dest.stat().st_size == size:
        print(f"{dest.name}: already downloaded")
        return
    print(f"{dest.name}: downloading {size / 1e9:.2f} GB")
    tmp = dest.with_name(dest.name + ".part")
    with requests.get(url, stream=True, timeout=60, headers=HTTP_HEADERS) as r:
        r.raise_for_status()
        with open(tmp, "wb") as dst:
            for chunk in r.iter_content(chunk_size=1 << 20):
                dst.write(chunk)
    if md5 and _md5(tmp) != md5:
        tmp.unlink()
        raise RuntimeError(f"{dest.name}: md5 mismatch after download")
    tmp.rename(dest)


def extract_archive(archive_path: Path, out_dir: Path) -> None:
    """Extract any libarchive-readable archive (.rar incl. RAR5, .zip, .7z,
    .tar.*) into `out_dir`, keeping its internal folder structure. libarchive
    verifies each entry's stored checksum as it reads."""
    # libarchive-c (reads RAR5, which zipfile/py7zr can't) is imported here so the
    # rest of this script runs without it. ImportError: package not installed;
    # AttributeError/OSError: package installed but the system libarchive is
    # missing, or LIBARCHIVE points at a bad path.
    try:
        import libarchive
    except (ImportError, AttributeError, OSError) as e:
        raise ImportError("libarchive is not installed on this machine. Either install it or more simply, manually extract the .rar in data/inputs/cropping_intensity") from e

    root = out_dir.resolve()
    with libarchive.file_reader(str(archive_path)) as arc:
        for entry in arc:
            dest = (root / entry.pathname).resolve()
            if not dest.is_relative_to(root):
                raise RuntimeError(f"{archive_path.name}: entry escapes out_dir: {entry.pathname}")
            if entry.isdir:
                dest.mkdir(parents=True, exist_ok=True)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as dst:
                for block in entry.get_blocks():
                    dst.write(block)


def fetch_cropping_intensity(article_id: int = CROPPING_INTENSITY_ARTICLE_ID,
                             out_dir: Path = CROPPING_INTENSITY_DIR) -> None:
    """Download every file in the figshare article into `out_dir`, then
    extract any .rar/.zip archives in place (see extract_archive).
    Both steps are skipped if already done: a download is skipped when a file
    of the expected size is present, and an extraction when its
    `.<archive>.extracted` marker exists. Downloads are md5-checked against
    figshare's computed_md5."""
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = requests.get(f"https://api.figshare.com/v2/articles/{article_id}", timeout=60, headers=HTTP_HEADERS)
    meta.raise_for_status()

    for f in meta.json()["files"]:
        dest = out_dir / f["name"]
        _download_verified(f["download_url"], dest, f["size"], f.get("computed_md5"))

        if dest.suffix.lower() not in (".rar", ".zip"):
            continue
        marker = out_dir / f".{dest.name}.extracted"
        if marker.exists():
            print(f"{f['name']}: already extracted")
            continue
        print(f"{f['name']}: extracting")
        extract_archive(dest, out_dir)
        marker.touch()


def fetch_cropland_extent(doi: str = CROPLAND_EXTENT_DOI,
                          filenames: list[str] = CROPLAND_EXTENT_FILES,
                          out_dir: Path = CROPLAND_EXTENT_DIR) -> None:
    """Download the named files of a Harvard Dataverse dataset into `out_dir`,
    md5-checked against Dataverse's stored checksum. Skipped per file when a
    file of the expected size is already present."""
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = requests.get("https://dataverse.harvard.edu/api/datasets/:persistentId/",
                        params={"persistentId": doi}, timeout=60, headers=HTTP_HEADERS)
    meta.raise_for_status()

    files = {f["dataFile"]["filename"]: f["dataFile"] for f in meta.json()["data"]["latestVersion"]["files"]}
    for name in filenames:
        df = files[name]
        _download_verified(f"https://dataverse.harvard.edu/api/access/datafile/{df['id']}",
                           out_dir / name, df["filesize"], df.get("md5"))


if __name__ == "__main__":
    extract_flat(ZIP_PATH, OUT_DIR)
    fetch_cropping_intensity()
    fetch_cropland_extent()
