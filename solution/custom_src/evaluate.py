from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .competition_metric import (
    AGGREGATION_METHOD,
    BRAND_WEIGHT,
    OCR_WEIGHT,
    PRODUCT_WEIGHT,
    cer,
    char_edit_distance,
    composite_score,
    evaluate_dataframe,
    ocr_similarity,
    product_f1,
    product_token_f1,
    row_score,
    token_f1,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate phase-2 OCR/brand/product submissions.")
    parser.add_argument("--gt", type=Path, required=True, help="Ground truth CSV")
    parser.add_argument("--pred", type=Path, required=True, help="Prediction CSV")
    parser.add_argument("--error-out", type=Path, help="Optional per-row score report CSV")
    args = parser.parse_args()

    gt_df = pd.read_csv(args.gt, dtype=str, keep_default_na=False, encoding="utf-8")
    pred_df = pd.read_csv(args.pred, dtype=str, keep_default_na=False, encoding="utf-8")
    result = evaluate_dataframe(gt_df, pred_df)

    print("Phase-2 competition metric")
    print(f"Aggregation   : {AGGREGATION_METHOD}")
    print(f"Weights       : brand={BRAND_WEIGHT:.2f} ocr={OCR_WEIGHT:.2f} product={PRODUCT_WEIGHT:.2f}")
    print(f"Matched rows  : {result['matched_rows']:,}")
    print(f"Brand F1      : {result['brand_f1']:.6f}")
    print(f"Average CER   : {result['average_cer']:.6f}")
    print(f"OCR score     : {result['ocr_score']:.6f}")
    print(f"Product F1    : {result['product_f1']:.6f}")
    print(f"Final score   : {result['final_score']:.6f}")

    if args.error_out:
        args.error_out.parent.mkdir(parents=True, exist_ok=True)
        per_row = result["per_row"]
        assert isinstance(per_row, pd.DataFrame)
        per_row.to_csv(args.error_out, index=False, encoding="utf-8-sig")
        print(f"Error report  : {args.error_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

