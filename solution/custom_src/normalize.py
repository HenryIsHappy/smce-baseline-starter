from __future__ import annotations

import math
import re
import unicodedata


_SPACE_RE = re.compile(r"\s+")
_CONTROL_RE = re.compile(r"[\n\r\t]+")
_PUNCT_RE = re.compile(r"[^0-9a-zA-Z\s]+")


def _is_missing(text: object) -> bool:
    if text is None:
        return True
    if isinstance(text, float) and math.isnan(text):
        return True
    return False


def _clean_final_text(text: object) -> str:
    if _is_missing(text):
        return ""
    text = _CONTROL_RE.sub(" ", str(text))
    return _SPACE_RE.sub(" ", text).strip()


def clean_ocr_text(text: object) -> str:
    """Clean final OCR text while preserving Vietnamese diacritics and case."""
    return _clean_final_text(text)


def clean_product_text(text: object) -> str:
    """Clean final product text while preserving Vietnamese diacritics and case."""
    return _clean_final_text(text)


def clean_output_text(text: object) -> str:
    """Backward-compatible alias for final OCR text cleaning."""
    return clean_ocr_text(text)


def strip_diacritics(text: str) -> str:
    """Remove Vietnamese diacritics for internal matching only."""
    text = text.replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def normalize_for_match(text: object) -> str:
    """Normalize text for matching only; never use this for final output."""
    if _is_missing(text):
        return ""
    text = _CONTROL_RE.sub(" ", str(text))
    text = strip_diacritics(text).lower()
    text = _PUNCT_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def export_empty_as_space(text: object) -> str:
    """Clean final text and encode empty predictions as one space."""
    cleaned = _clean_final_text(text)
    return cleaned if cleaned else " "


if __name__ == "__main__":
    assert clean_ocr_text(" Sữa\n tươi\tVinamilk ") == "Sữa tươi Vinamilk"
    assert normalize_for_match("Sữa tươi Vinamilk") == "sua tuoi vinamilk"
    assert export_empty_as_space("") == " "
    print("normalize.py assertions passed")
