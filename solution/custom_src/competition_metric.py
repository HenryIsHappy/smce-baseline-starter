from __future__ import annotations

from collections import Counter
import math
import re
from typing import Iterable

import pandas as pd

from .normalize import clean_ocr_text, clean_product_text


FINAL_COLUMNS = ["image_id", "ocr_text", "brand_name", "product_name"]
TEXT_COLUMNS = ["ocr_text", "brand_name", "product_name"]

BRAND_WEIGHT = 0.40
OCR_WEIGHT = 0.35
PRODUCT_WEIGHT = 0.25

# Local organizer materials bundled with the project still include the older
# public starter metric. The phase-2 instructions define per-field empty-value
# behavior, so this package uses per-row component scores and averages them.
AGGREGATION_METHOD = "row_mean_weighted_components"

_SPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"\S+")


def clean_score_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = clean_ocr_text(value)
    return _SPACE_RE.sub(" ", text).strip()


def _tokenize(value: object) -> list[str]:
    text = clean_score_text(value).lower()
    if not text:
        return []
    return _TOKEN_RE.findall(text)


def token_f1(ground_truth: object, prediction: object) -> float:
    gt_tokens = _tokenize(ground_truth)
    pred_tokens = _tokenize(prediction)
    if not gt_tokens and not pred_tokens:
        return 1.0
    if not gt_tokens or not pred_tokens:
        return 0.0

    gt_counts = Counter(gt_tokens)
    pred_counts = Counter(pred_tokens)
    common = sum((gt_counts & pred_counts).values())
    if common == 0:
        return 0.0
    precision = common / sum(pred_counts.values())
    recall = common / sum(gt_counts.values())
    return 2.0 * precision * recall / (precision + recall)


def brand_f1(ground_truth_brand: object, prediction_brand: object) -> float:
    return token_f1(ground_truth_brand, prediction_brand)


def product_f1(ground_truth_product: object, prediction_product: object) -> float:
    return token_f1(ground_truth_product, prediction_product)


def char_edit_distance(left: object, right: object) -> int:
    a = clean_score_text(left)
    b = clean_score_text(right)
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            substitution = previous[j - 1] + (char_a != char_b)
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1]


def cer(ground_truth_ocr: object, prediction_ocr: object) -> float:
    gt = clean_score_text(ground_truth_ocr)
    pred = clean_score_text(prediction_ocr)
    if not gt and not pred:
        return 0.0
    if not gt and pred:
        return 1.0
    return char_edit_distance(gt, pred) / max(len(gt), 1)


def ocr_similarity(ground_truth_ocr: object, prediction_ocr: object) -> float:
    return max(0.0, 1.0 - cer(ground_truth_ocr, prediction_ocr))


def row_score(
    gt_brand: object,
    pred_brand: object,
    gt_ocr: object,
    pred_ocr: object,
    gt_product: object,
    pred_product: object,
) -> dict[str, float]:
    brand_score = brand_f1(gt_brand, pred_brand)
    ocr_score = ocr_similarity(gt_ocr, pred_ocr)
    product_score = product_f1(gt_product, pred_product)
    final = BRAND_WEIGHT * brand_score + OCR_WEIGHT * ocr_score + PRODUCT_WEIGHT * product_score
    return {
        "brand_f1": brand_score,
        "ocr_cer": cer(gt_ocr, pred_ocr),
        "ocr_score": ocr_score,
        "product_f1": product_score,
        "final_score": final,
    }


def _require_columns(df: pd.DataFrame, name: str, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} dataframe missing columns: {missing}")


def evaluate_dataframe(solution: pd.DataFrame, submission: pd.DataFrame) -> dict[str, object]:
    _require_columns(solution, "solution", FINAL_COLUMNS)
    _require_columns(submission, "submission", FINAL_COLUMNS)

    if submission["image_id"].duplicated().any():
        raise ValueError("submission contains duplicate image_id values")
    if solution["image_id"].duplicated().any():
        raise ValueError("solution contains duplicate image_id values")

    solution_ids = set(solution["image_id"].astype(str))
    submission_ids = set(submission["image_id"].astype(str))
    if solution_ids != submission_ids:
        missing = solution_ids - submission_ids
        extra = submission_ids - solution_ids
        raise ValueError(f"image_id mismatch: missing={len(missing)}, extra={len(extra)}")

    merged = solution[FINAL_COLUMNS].merge(
        submission[FINAL_COLUMNS],
        on="image_id",
        suffixes=("_gt", "_pred"),
        how="inner",
    )
    if merged.empty:
        raise ValueError("no matching image_id values between solution and submission")

    rows: list[dict[str, object]] = []
    for row in merged.itertuples(index=False):
        scores = row_score(
            row.brand_name_gt,
            row.brand_name_pred,
            row.ocr_text_gt,
            row.ocr_text_pred,
            row.product_name_gt,
            row.product_name_pred,
        )
        rows.append(
            {
                "image_id": row.image_id,
                "gt_brand": clean_product_text(row.brand_name_gt),
                "pred_brand": clean_product_text(row.brand_name_pred),
                "gt_ocr": clean_ocr_text(row.ocr_text_gt),
                "pred_ocr": clean_ocr_text(row.ocr_text_pred),
                "gt_product": clean_product_text(row.product_name_gt),
                "pred_product": clean_product_text(row.product_name_pred),
                **scores,
            }
        )

    per_row = pd.DataFrame(rows)
    return {
        "aggregation_method": AGGREGATION_METHOD,
        "matched_rows": int(len(per_row)),
        "brand_f1": float(per_row["brand_f1"].mean()),
        "average_cer": float(per_row["ocr_cer"].mean()),
        "ocr_score": float(per_row["ocr_score"].mean()),
        "product_f1": float(per_row["product_f1"].mean()),
        "final_score": float(per_row["final_score"].mean()),
        "per_row": per_row,
    }


def composite_score(solution: pd.DataFrame, submission: pd.DataFrame) -> float:
    return float(evaluate_dataframe(solution, submission)["final_score"])


product_token_f1 = product_f1

