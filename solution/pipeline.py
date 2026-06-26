"""
Team solution pipeline — modified to use PaddleOCR and custom brand resolver.

The Streamlit demo and submission script import:
    predict_from_image(img) -> {"ocr_text", "brand_name", "product_name", "timing_ms"?}
    get_model_profile() -> see shared/benchmark.py (template-owned)
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Any
import numpy as np
from PIL import Image

from team_config import DEFAULT_MIN_CONF

@lru_cache(maxsize=1)
def get_ocr_reader():
    from solution.custom_src.run_ocr_paddle import _init_paddleocr
    # use_gpu=False for CPU inference
    return _init_paddleocr(lang="vi", use_gpu=False)

@lru_cache(maxsize=1)
def get_dictionary_and_candidates():
    from pathlib import Path
    import pandas as pd
    from solution.custom_src.extract_product import build_product_candidates, _build_token_index, load_dictionary
    
    repo_root = Path(__file__).resolve().parent.parent
    dict_path = repo_root / "data" / "product_dictionary.csv"
    if dict_path.exists():
        dictionary = pd.read_csv(dict_path, dtype=str, keep_default_na=False, encoding="utf-8")
    else:
        dictionary = load_dictionary()
        
    candidates = build_product_candidates(dictionary, min_match_tokens=2)
    token_index = _build_token_index(candidates)
    return candidates, token_index

@lru_cache(maxsize=1)
def get_brand_resolver():
    from pathlib import Path
    import pandas as pd
    from solution.custom_src.build_full_submission import BrandResolver
    
    repo_root = Path(__file__).resolve().parent.parent
    labels_path = repo_root / "data" / "train_labels.csv"
    if labels_path.exists():
        labels = pd.read_csv(labels_path, dtype=str, keep_default_na=False, encoding="utf-8")
        return BrandResolver(labels)
    return None

def run_ocr_on_image(img: Image.Image, reader, preprocess_mode="original") -> str:
    if reader is None:
        return ""
        
    from solution.custom_src.run_ocr_paddle import preprocess_image_array, _run_reader, sort_ocr_boxes, _flatten_ocr_result, clean_ocr_text
    
    # Convert PIL Image to BGR array for PaddleOCR
    rgb_array = np.asarray(img.convert("RGB"))
    bgr_array = rgb_array[:, :, ::-1].copy()
    
    image_array = preprocess_image_array(bgr_array, preprocess_mode)
    raw_result = _run_reader(reader, image_array)
    boxes = sort_ocr_boxes(_flatten_ocr_result(raw_result))
    parsed_text = clean_ocr_text(" ".join(box.text for box in boxes))
    if not parsed_text:
        parsed_text = " "
    return parsed_text

def predict_private(ocr_text: str) -> tuple[str, str]:
    from solution.custom_src.extract_product import predict_one
    from solution.custom_src.build_full_submission import _phase2_product_line
    
    candidates, token_index = get_dictionary_and_candidates()
    
    pred, scores, reason = predict_one(
        ocr_text,
        candidates,
        token_index,
        threshold=92.0,
        partial_threshold=95.0,
        min_match_tokens=2,
        disable_brand_fallback=True,
    )
    
    combined_product = str(pred["product_name"])
    
    brand_resolver = get_brand_resolver()
    if brand_resolver is not None:
        brand_name, brand_rule, brand_reason, matched_text = brand_resolver.resolve(
            evidence_text=ocr_text,
            product_name=combined_product,
        )
        phase2_product, split_rule, split_reason = _phase2_product_line(combined_product, brand_name)
    else:
        brand_name = " "
        phase2_product = combined_product
        
    brand_name = "" if brand_name.strip() == "" else brand_name
    phase2_product = "" if phase2_product.strip() == "" else phase2_product
    return brand_name, phase2_product

def predict_from_text(ocr_text: str) -> tuple[str, str]:
    """Extract brand + product from raw OCR text (no image)."""
    return predict_private(ocr_text)

def predict_from_image(
    img: Image.Image,
    min_conf: float = DEFAULT_MIN_CONF,
    *,
    include_timing: bool = True,
) -> dict[str, Any]:
    t0 = time.perf_counter()

    t_ocr = time.perf_counter()
    ocr_text = run_ocr_on_image(img, get_ocr_reader())
    ocr_ms = (time.perf_counter() - t_ocr) * 1000

    t_extract = time.perf_counter()
    brand, product = predict_private(ocr_text)
    extract_ms = (time.perf_counter() - t_extract) * 1000

    total_ms = (time.perf_counter() - t0) * 1000

    result: dict[str, Any] = {
        "ocr_text": ocr_text,
        "brand_name": brand,
        "product_name": product,
    }
    if include_timing:
        result["timing_ms"] = {
            "ocr": round(ocr_ms, 1),
            "extract": round(extract_ms, 1),
            "total": round(total_ms, 1),
        }
    return result
