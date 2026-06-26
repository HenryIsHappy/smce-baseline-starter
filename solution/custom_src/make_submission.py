from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .normalize import clean_ocr_text, clean_product_text
from .paths import OUTPUT_DIR, TEST_CSV, ensure_dirs
from .validate_submission import print_report, validate_submission


DEFAULT_OUTPUT = OUTPUT_DIR / "submission.csv"
REQUIRED_COLUMNS = ["image_id", "ocr_text", "brand_name", "product_name"]
NULL_LIKE_VALUES = {
    "",
    "na",
    "n/a",
    "nan",
    "none",
    "null",
    "#n/a",
    "#na",
    "-nan",
    "1.#ind",
    "1.#qnan",
    "-1.#ind",
    "-1.#qnan",
}


@dataclass(frozen=True)
class SubmissionStats:
    total_rows: int
    matched_ocr_rows: int
    matched_brand_rows: int
    matched_product_rows: int
    empty_ocr_count: int
    empty_brand_count: int
    empty_product_count: int
    pandas_default_nan_count: int = 0
    validation_ok: bool = False


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")


def _require_columns(df: pd.DataFrame, path: Path, columns: set[str]) -> None:
    missing = columns - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")


def _dedupe_by_image_id(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    duplicate_count = int(df["image_id"].duplicated().sum())
    if duplicate_count:
        print(f"Warning: {path} has {duplicate_count} duplicate image_id rows; keeping first occurrence.")
    return df.drop_duplicates("image_id", keep="first")


def kaggle_safe_text(value: object, cleaner) -> str:
    cleaned = cleaner(value)
    if cleaned.strip().lower() in NULL_LIKE_VALUES:
        return " "
    return cleaned


def make_submission(
    test_path: Path = TEST_CSV,
    ocr_path: Path | None = None,
    brand_path: Path | None = None,
    product_path: Path | None = None,
) -> tuple[pd.DataFrame, SubmissionStats]:
    test = _read_csv(test_path)
    _require_columns(test, test_path, {"image_id"})
    sub = test[["image_id"]].copy()

    if ocr_path is None:
        sub["ocr_text"] = ""
        matched_ocr_rows = 0
    else:
        ocr = _read_csv(ocr_path)
        _require_columns(ocr, ocr_path, {"image_id", "ocr_text"})
        ocr = _dedupe_by_image_id(ocr[["image_id", "ocr_text"]], ocr_path)
        sub = sub.merge(ocr, on="image_id", how="left", indicator="_ocr_match")
        matched_ocr_rows = int(sub["_ocr_match"].eq("both").sum())
        sub = sub.drop(columns=["_ocr_match"])

    product_brand_available = False
    if product_path is None:
        product = None
        sub["brand_name"] = ""
        sub["product_name"] = ""
        matched_brand_rows = 0
        matched_product_rows = 0
    else:
        product = _read_csv(product_path)
        _require_columns(product, product_path, {"image_id", "product_name"})
        product_columns = ["image_id", "product_name"]
        if "brand_name" in product.columns:
            product_columns.append("brand_name")
            product_brand_available = True
        product = _dedupe_by_image_id(product[product_columns], product_path)
        sub = sub.merge(product, on="image_id", how="left", indicator="_product_match")
        matched_product_rows = int(sub["_product_match"].eq("both").sum())
        matched_brand_rows = matched_product_rows if product_brand_available else 0
        sub = sub.drop(columns=["_product_match"])
        if "brand_name" not in sub.columns:
            sub["brand_name"] = ""

    if brand_path is not None:
        brand = _read_csv(brand_path)
        _require_columns(brand, brand_path, {"image_id", "brand_name"})
        brand = _dedupe_by_image_id(brand[["image_id", "brand_name"]], brand_path)
        if "brand_name" in sub.columns:
            sub = sub.drop(columns=["brand_name"])
        sub = sub.merge(brand, on="image_id", how="left", indicator="_brand_match")
        matched_brand_rows = int(sub["_brand_match"].eq("both").sum())
        sub = sub.drop(columns=["_brand_match"])

    sub["ocr_text"] = sub["ocr_text"].map(lambda value: kaggle_safe_text(value, clean_ocr_text))
    sub["brand_name"] = sub["brand_name"].map(lambda value: kaggle_safe_text(value, clean_product_text))
    sub["product_name"] = sub["product_name"].map(lambda value: kaggle_safe_text(value, clean_product_text))
    sub = sub[REQUIRED_COLUMNS]

    stats = SubmissionStats(
        total_rows=len(sub),
        matched_ocr_rows=matched_ocr_rows,
        matched_brand_rows=matched_brand_rows,
        matched_product_rows=matched_product_rows,
        empty_ocr_count=int(sub["ocr_text"].eq(" ").sum()),
        empty_brand_count=int(sub["brand_name"].eq(" ").sum()),
        empty_product_count=int(sub["product_name"].eq(" ").sum()),
    )
    return sub, stats


def write_kaggle_safe_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df[REQUIRED_COLUMNS].to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_ALL,
        lineterminator="\n",
    )


def validate_kaggle_safe_output(path: Path, test_path: Path) -> tuple[bool, int, list[str]]:
    errors: list[str] = []
    test = pd.read_csv(test_path, dtype=str, keep_default_na=False, encoding="utf-8")
    sub = pd.read_csv(path)
    nan_count = int(sub.isna().sum().sum())

    expected_rows = len(test)
    if tuple(sub.shape) != (expected_rows, len(REQUIRED_COLUMNS)):
        errors.append(f"shape mismatch: got {sub.shape}, expected {(expected_rows, len(REQUIRED_COLUMNS))}")
    if list(sub.columns) != REQUIRED_COLUMNS:
        errors.append(f"columns mismatch: got {list(sub.columns)}, expected {REQUIRED_COLUMNS}")
    if nan_count:
        errors.append(f"pandas-default NaN count is {nan_count}")
    if sub["image_id"].duplicated().any():
        errors.append("duplicate image_id values found")

    expected_ids = set(test["image_id"].astype(str))
    got_ids = set(sub["image_id"].astype(str))
    missing = expected_ids - got_ids
    extra = got_ids - expected_ids
    if missing:
        errors.append(f"missing {len(missing)} image IDs")
    if extra:
        errors.append(f"found {len(extra)} extra image IDs")

    return not errors, nan_count, errors


def print_stats(stats: SubmissionStats) -> None:
    print("Submission build summary")
    print(f"Total rows          : {stats.total_rows:,}")
    print(f"Matched OCR rows    : {stats.matched_ocr_rows:,}")
    print(f"Matched brand rows  : {stats.matched_brand_rows:,}")
    print(f"Matched product rows: {stats.matched_product_rows:,}")
    print(f"Empty OCR count     : {stats.empty_ocr_count:,}")
    print(f"Empty brand count   : {stats.empty_brand_count:,}")
    print(f"Empty product count : {stats.empty_product_count:,}")
    print(f"Pandas-default NaN  : {stats.pandas_default_nan_count:,}")
    print(f"Validation status   : {'PASS' if stats.validation_ok else 'FAIL'}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=Path, default=TEST_CSV)
    parser.add_argument("--ocr", type=Path)
    parser.add_argument("--brand", type=Path, help="Optional CSV with image_id, brand_name")
    parser.add_argument("--product", type=Path)
    parser.add_argument("--ocr-cache", type=Path, help="Backward-compatible alias for --ocr")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    ensure_dirs()
    ocr_path = args.ocr or args.ocr_cache
    if ocr_path is None:
        raise SystemExit("--ocr is required, or use backward-compatible --ocr-cache")
    if args.product is None:
        raise SystemExit("--product is required for the current pipeline")

    submission, stats = make_submission(
        test_path=args.test,
        ocr_path=ocr_path,
        brand_path=args.brand,
        product_path=args.product,
    )
    write_kaggle_safe_csv(submission, args.out)

    report = validate_submission(args.out, args.test)
    print_report(report)
    kaggle_ok, nan_count, kaggle_errors = validate_kaggle_safe_output(args.out, args.test)
    stats = SubmissionStats(
        total_rows=stats.total_rows,
        matched_ocr_rows=stats.matched_ocr_rows,
        matched_brand_rows=stats.matched_brand_rows,
        matched_product_rows=stats.matched_product_rows,
        empty_ocr_count=stats.empty_ocr_count,
        empty_brand_count=stats.empty_brand_count,
        empty_product_count=stats.empty_product_count,
        pandas_default_nan_count=nan_count,
        validation_ok=report.ok and kaggle_ok,
    )
    print_stats(stats)
    print(f"Output path          : {args.out}")

    if kaggle_errors:
        print("Kaggle-safe validation errors:")
        for error in kaggle_errors:
            print(f"- {error}")

    if not report.ok or not kaggle_ok:
        return 1

    print(f"Wrote valid submission to {args.out}")
    if args.out.name == "submission_v2_kaggle_safe.csv":
        upload_path = args.out.parent / "UPLOAD_THIS_submission_v2.csv"
        shutil.copyfile(args.out, upload_path)
        print(f"Copied final v2 upload file to {upload_path}")
    if args.out.name == "submission_v3_kaggle_safe.csv":
        upload_root = args.out.parent.parent if args.out.parent.name.lower() == "v3" else args.out.parent
        upload_path = upload_root / "UPLOAD_THIS_submission_v3.csv"
        shutil.copyfile(args.out, upload_path)
        print(f"Copied final v3 upload file to {upload_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
