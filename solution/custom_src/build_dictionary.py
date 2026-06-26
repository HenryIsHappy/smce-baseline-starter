from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from .normalize import clean_product_text, normalize_for_match
from .paths import ARTIFACT_DIR, TRAIN_LABELS_CSV, ensure_dirs


DEFAULT_OUTPUT = ARTIFACT_DIR / "product_dictionary.csv"
DEFAULT_COUNTS_OUTPUT = ARTIFACT_DIR / "product_name_counts.csv"
DEFAULT_BRAND_OUTPUT = ARTIFACT_DIR / "brand_candidates.txt"


def _product_counts(labels_path: Path) -> pd.DataFrame:
    labels = pd.read_csv(labels_path, dtype=str, keep_default_na=False, encoding="utf-8")
    if "product_name" not in labels.columns:
        raise ValueError(f"{labels_path} must contain a product_name column")

    products = labels["product_name"].map(clean_product_text)
    products = products[products != ""]
    counts = products.value_counts().rename_axis("canonical_product_name").reset_index(name="count")
    counts["normalized_product_name"] = counts["canonical_product_name"].map(normalize_for_match)
    return counts[["canonical_product_name", "normalized_product_name", "count"]].sort_values(
        ["count", "canonical_product_name"],
        ascending=[False, True],
    )


def infer_brand_candidates(product_counts: pd.DataFrame, min_count: int = 2) -> list[tuple[str, int]]:
    """Infer frequent first 1-, 2-, and 3-token prefixes from product-name counts."""
    counter: Counter[str] = Counter()
    for row in product_counts.itertuples(index=False):
        tokens = str(row.canonical_product_name).split()
        for width in (1, 2, 3):
            if len(tokens) >= width:
                counter[" ".join(tokens[:width])] += int(row.count)
    candidates = [(name, count) for name, count in counter.items() if count >= min_count]
    return sorted(candidates, key=lambda item: (-item[1], item[0]))


def build_product_dictionary(labels_path: Path = TRAIN_LABELS_CSV) -> pd.DataFrame:
    """Build canonical product names with normalized match keys and counts."""
    return _product_counts(labels_path)


def write_dictionary_artifacts(
    labels_path: Path = TRAIN_LABELS_CSV,
    dictionary_out: Path = DEFAULT_OUTPUT,
    counts_out: Path = DEFAULT_COUNTS_OUTPUT,
    brand_out: Path = DEFAULT_BRAND_OUTPUT,
    min_brand_count: int = 2,
) -> pd.DataFrame:
    ensure_dirs()
    dictionary = build_product_dictionary(labels_path)
    dictionary_out.parent.mkdir(parents=True, exist_ok=True)
    counts_out.parent.mkdir(parents=True, exist_ok=True)
    brand_out.parent.mkdir(parents=True, exist_ok=True)

    dictionary.to_csv(dictionary_out, index=False, encoding="utf-8")
    dictionary.to_csv(counts_out, index=False, encoding="utf-8")

    candidates = infer_brand_candidates(dictionary, min_count=min_brand_count)
    with brand_out.open("w", encoding="utf-8", newline="\n") as handle:
        for name, count in candidates:
            handle.write(f"{name}\t{count}\n")

    return dictionary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=TRAIN_LABELS_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--counts-out", type=Path, default=DEFAULT_COUNTS_OUTPUT)
    parser.add_argument("--brand-out", type=Path, default=DEFAULT_BRAND_OUTPUT)
    parser.add_argument("--min-brand-count", type=int, default=2)
    args = parser.parse_args()

    dictionary = write_dictionary_artifacts(
        labels_path=args.labels,
        dictionary_out=args.out,
        counts_out=args.counts_out,
        brand_out=args.brand_out,
        min_brand_count=args.min_brand_count,
    )
    print(f"Wrote {len(dictionary):,} products to {args.out}")
    print(f"Wrote product counts to {args.counts_out}")
    print(f"Wrote brand candidates to {args.brand_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
