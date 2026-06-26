from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .normalize import clean_ocr_text, clean_product_text, export_empty_as_space
from .paths import TEST_CSV


REQUIRED_COLUMNS = ["image_id", "ocr_text", "brand_name", "product_name"]
TEXT_COLUMNS = ["ocr_text", "brand_name", "product_name"]
CONTROL_PATTERN = r"[\n\r\t]"


@dataclass
class ValidationReport:
    test_path: Path
    submission_path: Path
    test_rows: int
    submission_rows: int
    missing_ids: set[str] = field(default_factory=set)
    extra_ids: set[str] = field(default_factory=set)
    duplicate_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")


def prepare_for_export(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned submission dataframe with blank text fields as one space."""
    out = df[REQUIRED_COLUMNS].copy()
    out["image_id"] = out["image_id"].fillna("").astype(str).str.strip()
    out["ocr_text"] = out["ocr_text"].map(lambda value: export_empty_as_space(clean_ocr_text(value)))
    out["brand_name"] = out["brand_name"].map(lambda value: export_empty_as_space(clean_product_text(value)))
    out["product_name"] = out["product_name"].map(lambda value: export_empty_as_space(clean_product_text(value)))
    return out


def write_submission_csv(df: pd.DataFrame, path: Path) -> None:
    """Write UTF-8 CSV with all fields quoted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    prepare_for_export(df).to_csv(
        path,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_ALL,
        lineterminator="\n",
    )


def validate_submission(submission_path: Path, test_path: Path = TEST_CSV) -> ValidationReport:
    test = _read_csv(test_path)
    sub = _read_csv(submission_path)

    report = ValidationReport(
        test_path=test_path,
        submission_path=submission_path,
        test_rows=len(test),
        submission_rows=len(sub),
    )

    if "image_id" not in test.columns:
        report.errors.append(f"{test_path} must contain an image_id column")
        return report

    if list(sub.columns) != REQUIRED_COLUMNS:
        report.errors.append(f"columns must be exactly {REQUIRED_COLUMNS}; got {list(sub.columns)}")
        return report

    if len(sub) != len(test):
        report.errors.append(f"row count mismatch: got {len(sub)}, expected {len(test)}")

    duplicate_mask = sub["image_id"].duplicated(keep=False)
    if duplicate_mask.any():
        report.duplicate_ids = sorted(sub.loc[duplicate_mask, "image_id"].unique().tolist())
        preview = ", ".join(report.duplicate_ids[:10])
        report.errors.append(f"duplicate image_id values found: {preview}")

    expected_ids = set(test["image_id"].astype(str))
    got_ids = set(sub["image_id"].astype(str))
    report.missing_ids = expected_ids - got_ids
    report.extra_ids = got_ids - expected_ids

    if report.missing_ids:
        preview = ", ".join(sorted(report.missing_ids)[:10])
        report.errors.append(f"missing {len(report.missing_ids)} image_id values: {preview}")
    if report.extra_ids:
        preview = ", ".join(sorted(report.extra_ids)[:10])
        report.errors.append(f"found {len(report.extra_ids)} extra image_id values: {preview}")

    if sub.isna().any().any():
        bad_cols = [col for col in sub.columns if sub[col].isna().any()]
        report.errors.append(f"null/NaN values found in columns: {bad_cols}")

    for col in REQUIRED_COLUMNS:
        non_string_count = sum(not isinstance(value, str) for value in sub[col].tolist())
        if non_string_count:
            report.errors.append(f"{col} contains {non_string_count} non-string values")

    for col in TEXT_COLUMNS:
        bad_control = sub[col].str.contains(CONTROL_PATTERN, regex=True, na=False)
        if bad_control.any():
            report.errors.append(f"{col} contains newline, carriage return, or tab in {bad_control.sum()} rows")

        empty_count = sub[col].eq("").sum()
        if empty_count:
            report.warnings.append(
                f"{col} has {empty_count} empty strings; cleaned export writes these as single spaces"
            )

    return report


def print_report(report: ValidationReport) -> None:
    print("Submission validation report")
    print(f"Test file       : {report.test_path}")
    print(f"Submission file : {report.submission_path}")
    print(f"Rows            : submission={report.submission_rows:,} test={report.test_rows:,}")
    print(f"ID check        : missing={len(report.missing_ids):,} extra={len(report.extra_ids):,} duplicates={len(report.duplicate_ids):,}")
    print(f"Columns         : {', '.join(REQUIRED_COLUMNS)}")

    if report.warnings:
        print("\nWarnings:")
        for warning in report.warnings:
            print(f"- {warning}")

    if report.errors:
        print("\nErrors:")
        for error in report.errors:
            print(f"- {error}")
        print("\nValidation failed.")
    else:
        print("\nValidation passed.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=Path, default=TEST_CSV, help="Path to test.csv")
    parser.add_argument("--submission", type=Path, required=True, help="Path to submission CSV")
    parser.add_argument("--output", type=Path, help="Optional cleaned CSV output path")
    args = parser.parse_args()

    report = validate_submission(args.submission, args.test)
    print_report(report)
    if not report.ok:
        return 1

    if args.output:
        submission = _read_csv(args.submission)
        write_submission_csv(submission, args.output)
        print(f"\nWrote cleaned submission: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
