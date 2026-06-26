from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = PROJECT_ROOT / "train.csv"
TRAIN_LABELS_CSV = PROJECT_ROOT / "train_labels.csv"
TEST_CSV = PROJECT_ROOT / "test.csv"
SAMPLE_SUBMISSION_CSV = PROJECT_ROOT / "sample_submission.csv"
TRAIN_IMAGE_DIR = PROJECT_ROOT / "train_images" / "train_images"
TEST_IMAGE_DIR = PROJECT_ROOT / "test_images" / "images"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


REQUIRED_FILES = (
    TRAIN_CSV,
    TRAIN_LABELS_CSV,
    TEST_CSV,
    SAMPLE_SUBMISSION_CSV,
)
REQUIRED_DIRS = (
    TRAIN_IMAGE_DIR,
    TEST_IMAGE_DIR,
)


def ensure_dirs() -> None:
    """Create local generated-output directories."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def image_path(image_id: str, split: str) -> Path:
    """Return the expected jpg path for a train or test image_id."""
    if split == "train":
        return TRAIN_IMAGE_DIR / f"{image_id}.jpg"
    if split == "test":
        return TEST_IMAGE_DIR / f"{image_id}.jpg"
    raise ValueError("split must be 'train' or 'test'")


def _status(path: Path, kind: str) -> str:
    exists = path.is_file() if kind == "file" else path.is_dir()
    return "OK" if exists else "MISSING"


def verify_required_paths() -> bool:
    """Print and verify all raw-data paths without modifying raw data."""
    ok = True
    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    for path in REQUIRED_FILES:
        status = _status(path, "file")
        print(f"{path.name} = {path} [{status}]")
        ok = ok and status == "OK"
    for path in REQUIRED_DIRS:
        status = _status(path, "dir")
        print(f"{path.name} = {path} [{status}]")
        ok = ok and status == "OK"
    print(f"OUTPUT_DIR = {OUTPUT_DIR}")
    print(f"ARTIFACT_DIR = {ARTIFACT_DIR}")
    return ok


def main() -> int:
    ensure_dirs()
    return 0 if verify_required_paths() else 1


if __name__ == "__main__":
    raise SystemExit(main())

