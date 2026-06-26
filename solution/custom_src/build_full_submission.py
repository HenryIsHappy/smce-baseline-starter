from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from .build_dictionary import write_dictionary_artifacts
from .extract_product import load_dictionary, predict_dataframe
from .normalize import clean_ocr_text, clean_product_text, normalize_for_match


BASE_FINAL_COLUMNS = ["image_id", "ocr_text", "product_name"]
PRODUCT_OUTPUT_COLUMNS = [
    "image_id",
    "product_name",
    "source",
    "base_product_name",
    "rule",
    "reason",
]
BRAND_OUTPUT_COLUMNS = [
    "image_id",
    "brand_name",
    "source",
    "rule",
    "reason",
    "matched_text",
    "product_name",
]

PRODUCT_SPLIT_OUTPUT_COLUMNS = [
    "image_id",
    "combined_product_name",
    "brand_name",
    "phase2_product_name",
    "rule",
    "reason",
]

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

SOURCE_MEDIA_PATTERNS = [
    "cafef",
    "cafe f",
    "op news",
    "vtv",
    "tap news",
    "tiktok",
    "thanh nien",
    "antv",
    "breaking news",
    "top club",
    "music vinyl",
    "logo finance x",
    "ometv",
    "ome tv",
    "laptop",
    "dramatic",
    "sconnect",
    "vanessa violin",
    "retro filtered",
    "raw rec",
    "epic inspiration",
    "epic dm production",
    "phuc tai chinh",
    "vietnamnet",
]

GENERIC_PRODUCT_PATTERNS = [
    "do hop",
    "thit hop",
    "san pham do hop",
    "pate",
    "pate cot den",
    "cot den",
    "sua",
    "nan",
    "milo",
    "flex",
    "grow",
    "grow+",
    "gold",
    "pate heo",
]

NOISE_TEXT_PATTERNS = [
    "giam",
    "sale",
    "freeship",
    "khuyen mai",
    "mua",
    "tang",
    "combo",
    "flash sale",
    "gia",
    "chi con",
    "hashtag",
    "follow",
    "livestream",
]

SPAN_BLOCK_TOKENS = {
    "giam",
    "sale",
    "freeship",
    "khuyen",
    "mai",
    "mua",
    "tang",
    "combo",
    "flash",
    "gia",
    "chi",
    "con",
    "hashtag",
    "follow",
    "livestream",
    "tiktok",
    "vtv",
    "cafef",
}

SPAN_UNIT_TOKENS = {
    "ml",
    "l",
    "g",
    "kg",
    "hop",
    "lon",
    "thung",
    "chai",
    "bich",
    "goi",
    "dong",
    "vnd",
    "k",
}

SPAN_GENERIC_TOKENS = {
    "san",
    "pham",
    "cong",
    "ty",
    "co",
    "phan",
    "chinh",
    "thuc",
    "thong",
    "bao",
    "review",
    "news",
    "tin",
    "moi",
}

BRAND_PREFIX_BLOCK_TOKENS = {
    "a",
    "an",
    "am",
    "bao",
    "banh",
    "binh",
    "cai",
    "coc",
    "dung",
    "hai",
    "kenh",
    "kem",
    "mat",
    "nhac",
    "nhan",
    "phong",
    "shop",
    "sua",
    "tra",
    "pate",
    "thit",
    "do",
    "hop",
    "san",
    "pham",
    "gan",
    "heo",
    "ga",
    "cot",
    "den",
    "nan",
    "optipro",
    "supremepro",
    "milo",
    "flex",
    "grow",
    "gold",
    "smoothie",
    "news",
    "tin",
    "nguon",
    "anh",
    "clip",
    "coffee",
    "carebiz",
    "darkzone",
    "dreamer",
    "finbiz",
    "fourjazz",
    "index",
    "mediaatp",
    "minitopia",
    "nct88",
    "net88",
    "quanh",
    "sconnet",
    "thdt",
    "the",
    "thong",
    "thuoc",
    "tam",
    "tuong",
    "viet",
    "vietnam",
    "vietnamindex",
    "vtvcab",
    "vuphongnews",
    "xland",
    "znews",
}

BRAND_PREFIX_SECOND_TOKEN_BLOCK = {
    "nan",
    "optipro",
    "supremepro",
    "milo",
    "flex",
    "grow",
    "gold",
    "pate",
    "smoothie",
    "gan",
    "heo",
    "ga",
}

BRAND_COMPOUND_FIRST_TOKENS = {
    "ha",
    "ba",
    "dutch",
    "th",
    "phuc",
    "the",
    "highlands",
}

PREFERRED_CANONICALS = {
    "halong_pate": "Ha Long Canfoco Pate C\u1ed9t \u0110\u00e8n",
    "halong_brand": "Ha Long Canfoco",
    "nestle_milo": "Nestl\u00e9 Milo",
    "nestle_nan": "Nestl\u00e9 NAN",
    "nestle_nan_optipro_plus": "Nestl\u00e9 NAN Optipro Plus",
    "nestle_nan_supremepro": "Nestl\u00e9 NAN Supremepro",
    "nestle_nan_optipro_2": "Nestl\u00e9 NAN Optipro 2",
    "vinamilk_flex": "Vinamilk Flex",
    "vissan_pate_heo": "Vissan Pate Heo",
    "dutch_lady_grow": "Dutch Lady Grow+",
    "ba_vi_gold": "Ba V\u00ec Gold",
}

KNOWN_BRAND_PREFIXES = [
    "Ha Long Canfoco",
    "Dutch Lady",
    "TH True Milk",
    "Nestlé",
    "Nestle",
    "Vinamilk",
    "Vissan",
    "Aptamil",
    "HiPP",
    "HIPP",
    "BEBA",
    "Ba Vì",
    "Ba Vi",
]

CPU_SURROGATE_SOURCE_TOKENS = {
    "cafef",
    "vtv",
    "tap",
    "news",
    "tiktok",
    "thanh",
    "nien",
    "antv",
    "breaking",
    "music",
    "vinyl",
    "dramatic",
    "epic",
    "ometv",
    "laptop",
    "virus",
    "dich",
    "ta",
    "duong",
    "tinh",
    "tieu",
    "huy",
    "thu",
    "hoi",
    "chau",
    "au",
    "xin",
    "lo",
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")


def _strip_diacritics(text: str) -> str:
    text = text.replace("\u0111", "d").replace("\u0110", "D")
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def _norm(text: object) -> str:
    return normalize_for_match(text)


def _compact(text: object) -> str:
    return _norm(text).replace(" ", "")


def _has_phrase(norm_text: str, phrase: str) -> bool:
    phrase_norm = _norm(phrase)
    if not phrase_norm:
        return False
    compact_text = norm_text.replace(" ", "")
    compact_phrase = phrase_norm.replace(" ", "")
    if " " in phrase_norm:
        return phrase_norm in norm_text or compact_phrase in compact_text
    if len(phrase_norm) <= 3:
        return phrase_norm in norm_text.split()
    return phrase_norm in norm_text.split() or compact_phrase in compact_text


def _has_any(norm_text: str, phrases: list[str]) -> bool:
    return any(_has_phrase(norm_text, phrase) for phrase in phrases)


def _product_is_empty(value: object) -> bool:
    return clean_product_text(value).strip() == ""


def _is_raw_exact_nan(value: object) -> bool:
    return clean_product_text(value).strip().lower() == "nan"


def _is_source_media(value: object) -> bool:
    normalized = _norm(value)
    if not normalized:
        return False
    return any(_has_phrase(normalized, phrase) for phrase in SOURCE_MEDIA_PATTERNS)


def _is_generic_or_underbranded(value: object) -> bool:
    normalized = _norm(value)
    if not normalized:
        return True
    if _is_source_media(value):
        return True
    if _is_raw_exact_nan(value):
        return True
    return any(_has_phrase(normalized, phrase) for phrase in GENERIC_PRODUCT_PATTERNS)


def _same_halong_family(value: object) -> bool:
    normalized = _norm(value)
    return _has_any(normalized, ["ha long", "halong", "canfoco", "canfoco", "do hop", "pate", "cot den", "cotden"])


def _same_nestle_family(value: object) -> bool:
    normalized = _norm(value)
    return _has_any(normalized, ["nestle", "nan", "milo", "sua"])


def _same_family_or_eligible(value: object, family: str) -> bool:
    if _is_generic_or_underbranded(value):
        return True
    if family == "halong":
        return _same_halong_family(value)
    if family == "nestle":
        return _same_nestle_family(value)
    normalized = _norm(value)
    family_tokens = {
        "vinamilk": ["vinamilk", "flex"],
        "vissan": ["vissan", "pate heo"],
        "dutch_lady": ["dutch lady", "dutchlady", "grow"],
        "ba_vi": ["ba vi", "bavi", "gold"],
    }.get(family, [])
    return _has_any(normalized, family_tokens)


def _safe_output_text(value: object) -> str:
    cleaned = clean_product_text(value)
    if cleaned.strip().lower() in NULL_LIKE_VALUES:
        return " "
    return cleaned if cleaned else " "


def _safe_brand_output_text(value: object) -> str:
    cleaned = clean_product_text(value)
    if cleaned.strip().lower() in NULL_LIKE_VALUES:
        return " "
    if _is_source_media(cleaned):
        return " "
    return cleaned if cleaned else " "


def _match_tokens(text: object) -> list[str]:
    return [token for token in _norm(text).split() if token]


def _cpu_surrogate_meaningful_tokens(text: object) -> list[str]:
    return [
        token
        for token in _match_tokens(text)
        if token not in SPAN_BLOCK_TOKENS
        and token not in SPAN_UNIT_TOKENS
        and token not in SPAN_GENERIC_TOKENS
        and token not in CPU_SURROGATE_SOURCE_TOKENS
        and len(token) >= 2
    ]


def _cpu_surrogate_clean_candidate(text: object) -> bool:
    tokens = _match_tokens(text)
    if not tokens:
        return False
    token_set = set(tokens)
    if token_set & CPU_SURROGATE_SOURCE_TOKENS:
        return False
    if token_set & SPAN_UNIT_TOKENS:
        return False
    if token_set & SPAN_BLOCK_TOKENS:
        return False
    return len(_cpu_surrogate_meaningful_tokens(text)) >= 2


class CpuSurrogateAliasLookup:
    """Deployable V44 gate: exact train labels or unique train-derived aliases visible in OCR."""

    def __init__(self, labels: pd.DataFrame) -> None:
        self.exact_map: dict[str, set[str]] = defaultdict(set)
        self.alias_map: dict[str, set[str]] = defaultdict(set)
        self.counts: Counter[str] = Counter()
        products = labels.get("product_name", pd.Series(dtype=str)).map(clean_product_text)
        for canonical in products:
            canonical = clean_product_text(canonical)
            canonical_norm = _norm(canonical)
            if not canonical_norm or canonical_norm == "nan":
                continue
            if len(_cpu_surrogate_meaningful_tokens(canonical)) < 2:
                continue
            self.counts[canonical] += 1
            self.exact_map[canonical_norm].add(canonical)
            self.alias_map[canonical_norm].add(canonical)
            if not self._aliasable_canonical(canonical_norm):
                continue
            tokens = canonical_norm.split()
            for width in range(2, min(6, len(tokens)) + 1):
                prefix = " ".join(tokens[:width])
                if len(_cpu_surrogate_meaningful_tokens(prefix)) >= 2:
                    self.alias_map[prefix].add(canonical)
                for start in range(0, len(tokens) - width + 1):
                    alias = " ".join(tokens[start : start + width])
                    if len(_cpu_surrogate_meaningful_tokens(alias)) >= 2:
                        self.alias_map[alias].add(canonical)

    def _best_canonical(self, canonicals: set[str]) -> str:
        return sorted(canonicals, key=lambda value: (-self.counts[value], value))[0]

    @staticmethod
    def _aliasable_canonical(canonical_norm: str) -> bool:
        tokens = canonical_norm.split()
        if len(tokens) > 8:
            return False
        token_set = set(tokens)
        if token_set & CPU_SURROGATE_SOURCE_TOKENS:
            return False
        if token_set & {"cong", "ty", "co", "phan", "thong", "bao", "chinh", "thuc"}:
            return False
        return True

    def resolve(self, evidence_text: object, current_product: object, allow_generic_upgrade: bool = False) -> tuple[str | None, str, str]:
        current_is_empty = _product_is_empty(current_product) or _is_raw_exact_nan(current_product)
        current_is_source = _is_source_media(current_product)
        if not current_is_empty and not current_is_source and not allow_generic_upgrade:
            return None, "current_nonempty_preserved", "CPU surrogate default fills only empty/raw-null/source-like base products"
        if allow_generic_upgrade and not _is_generic_or_underbranded(current_product):
            return None, "current_specific_preserved", "current product is non-empty and not generic/source-like"

        evidence_tokens = _match_tokens(evidence_text)
        if len(evidence_tokens) < 2:
            return None, "no_evidence_tokens", "not enough OCR tokens for deployable CPU gate"

        best: tuple[int, int, str, str, str] | None = None
        seen: set[str] = set()
        for width in range(2, min(6, len(evidence_tokens)) + 1):
            for start in range(0, len(evidence_tokens) - width + 1):
                phrase = " ".join(evidence_tokens[start : start + width])
                if not phrase or phrase in seen:
                    continue
                seen.add(phrase)
                if not _cpu_surrogate_clean_candidate(phrase):
                    continue
                meaningful_count = len(_cpu_surrogate_meaningful_tokens(phrase))
                exact_targets = self.exact_map.get(phrase, set())
                alias_targets = self.alias_map.get(phrase, set())
                if exact_targets:
                    canonical = self._best_canonical(exact_targets)
                    score_tuple = (2, meaningful_count, phrase, canonical, "cpu_surrogate_exact_canonical")
                elif len(alias_targets) == 1 and meaningful_count >= 3:
                    canonical = self._best_canonical(alias_targets)
                    score_tuple = (1, meaningful_count, phrase, canonical, "cpu_surrogate_unique_alias_min3")
                else:
                    continue
                if best is None or score_tuple[:3] > best[:3]:
                    best = score_tuple
        if best is None:
            return None, "no_deployable_cpu_surrogate_match", "no exact canonical or unique train alias matched OCR n-grams"
        _rank, _meaningful_count, matched_phrase, canonical, rule = best
        if _is_raw_exact_nan(canonical) or _is_source_media(canonical):
            return None, "cpu_surrogate_blocked_candidate", "candidate is raw NAN or source/media"
        return canonical, rule, f"matched OCR phrase '{matched_phrase}' to train-derived canonical '{canonical}'"


class BrandResolver:
    """V45 category-neutral brand resolver derived only from official train labels.

    If future train labels include a brand_name column, those labels are used
    directly. For the current data, brand candidates are conservative prefixes
    of official product_name labels. Independent brand fills require direct OCR
    evidence; product-prefix fallback is used only after product resolution.
    """

    def __init__(self, labels: pd.DataFrame) -> None:
        self.counts: Counter[str] = Counter()
        self.by_norm: dict[str, str] = {}
        self.source_by_norm: dict[str, str] = {}
        self.ocr_enabled_norms: set[str] = set()
        self.standalone_product_norms: set[str] = set()
        self._build_from_labels(labels)

    def _add(self, brand: object, source: str, count: int = 1, ocr_enabled: bool = False) -> None:
        brand_text = clean_product_text(brand)
        normalized = _norm(brand_text)
        if not self._valid_brand_candidate(brand_text):
            return
        self.counts[brand_text] += count
        if normalized not in self.by_norm or self.counts[brand_text] > self.counts[self.by_norm[normalized]]:
            self.by_norm[normalized] = brand_text
            self.source_by_norm[normalized] = source
        if ocr_enabled:
            self.ocr_enabled_norms.add(normalized)

    def _build_from_labels(self, labels: pd.DataFrame) -> None:
        if "brand_name" in labels.columns:
            for brand, count in labels["brand_name"].map(clean_product_text).value_counts().items():
                self._add(brand, "official_brand_name", int(count), ocr_enabled=True)

        products = labels.get("product_name", pd.Series(dtype=str)).map(clean_product_text)
        products = products[(products != "") & (products.str.lower() != "nan")]
        for product in products:
            tokens = _norm(product).split()
            if len(tokens) == 1 and self._valid_brand_candidate(product):
                self.standalone_product_norms.add(tokens[0])

        prefix_counts: Counter[str] = Counter()
        prefix_original: dict[str, str] = {}
        for product in products:
            if _is_source_media(product):
                continue
            tokens = product.split()
            norm_tokens = _norm(product).split()
            if not tokens or not norm_tokens:
                continue
            max_width = min(3, len(tokens))
            for width in range(1, max_width + 1):
                prefix = " ".join(tokens[:width])
                prefix_norm = " ".join(norm_tokens[:width])
                if not prefix_norm:
                    continue
                prefix_counts[prefix_norm] += 1
                prefix_original.setdefault(prefix_norm, prefix)

        for product in products:
            if _is_source_media(product):
                continue
            tokens = product.split()
            norm_tokens = _norm(product).split()
            if not tokens or not norm_tokens:
                continue
            first = tokens[0]
            first_norm = norm_tokens[0]
            self._add(
                first,
                "product_first_token_prefix",
                ocr_enabled=self._independent_brand_allowed(first, prefix_counts[first_norm]),
            )

            for width in range(2, min(3, len(tokens)) + 1):
                prefix = " ".join(tokens[:width])
                prefix_norm = " ".join(norm_tokens[:width])
                if not self._compound_prefix_allowed(prefix_norm, prefix_counts[prefix_norm]):
                    continue
                self._add(
                    prefix,
                    f"product_{width}_token_prefix",
                    prefix_counts[prefix_norm],
                    ocr_enabled=self._independent_brand_allowed(prefix, prefix_counts[prefix_norm]),
                )

    def _independent_brand_allowed(self, brand: object, count: int) -> bool:
        """Return whether a train-derived brand candidate may fill brand from OCR alone."""

        text = clean_product_text(brand)
        normalized = _norm(text)
        tokens = normalized.split()
        if not self._valid_brand_candidate(text) or not tokens:
            return False
        token_set = set(tokens)
        if token_set & CPU_SURROGATE_SOURCE_TOKENS:
            return False
        if len(tokens) == 1:
            token = tokens[0]
            if token in BRAND_PREFIX_BLOCK_TOKENS:
                return False
            if token in self.standalone_product_norms and count >= 2:
                return True
            # Allow train-derived ASCII/acronym brand-looking prefixes such as
            # ENA, Similac, Nutifood, Aptamil, but not Vietnamese category words.
            asciiish = _strip_diacritics(text) == text
            has_upper = any(ch.isupper() for ch in text)
            all_caps = text.isupper() and len(token) >= 3
            camel_or_title = has_upper and len(token) >= 4 and token not in BRAND_PREFIX_BLOCK_TOKENS
            return bool(count >= 2 and asciiish and (all_caps or camel_or_title))

        if "-" in text and count >= 2 and len(tokens) <= 2 and not (set(tokens) & BRAND_PREFIX_BLOCK_TOKENS):
            return True

        if tokens[0] not in BRAND_COMPOUND_FIRST_TOKENS:
            return False
        if tokens[1] in BRAND_PREFIX_SECOND_TOKEN_BLOCK:
            return False
        if any(token in BRAND_PREFIX_BLOCK_TOKENS for token in tokens[1:]):
            return False
        return count >= 2

    @staticmethod
    def _compound_prefix_allowed(prefix_norm: str, count: int) -> bool:
        tokens = prefix_norm.split()
        if len(tokens) < 2:
            return True
        if tokens[0] not in BRAND_COMPOUND_FIRST_TOKENS and count < 2:
            return False
        if len(tokens) >= 2 and tokens[1] in BRAND_PREFIX_SECOND_TOKEN_BLOCK:
            return False
        return count >= 2 or tokens[0] in BRAND_COMPOUND_FIRST_TOKENS

    @staticmethod
    def _valid_brand_candidate(brand: object) -> bool:
        text = clean_product_text(brand)
        normalized = _norm(text)
        if not normalized or normalized in NULL_LIKE_VALUES:
            return False
        if _is_source_media(text):
            return False
        tokens = normalized.split()
        if not tokens or len(tokens) > 3:
            return False
        token_set = set(tokens)
        if token_set & CPU_SURROGATE_SOURCE_TOKENS:
            return False
        if token_set <= BRAND_PREFIX_BLOCK_TOKENS:
            return False
        if tokens[0] in BRAND_PREFIX_BLOCK_TOKENS and tokens[0] not in BRAND_COMPOUND_FIRST_TOKENS:
            return False
        if all(len(token) <= 2 for token in tokens):
            return False
        return True

    def _best_ocr_brand(self, evidence_text: object) -> tuple[str | None, str, str]:
        normalized = _norm(evidence_text)
        if not normalized:
            return None, "", ""
        best: tuple[int, int, int, str, str] | None = None
        for brand_norm, brand in self.by_norm.items():
            if brand_norm not in self.ocr_enabled_norms:
                continue
            if not _has_phrase(normalized, brand_norm):
                continue
            token_count = len(brand_norm.split())
            source = self.source_by_norm.get(brand_norm, "train_label_prefix")
            # Prefer explicit official brands, then longer compound brands, then
            # higher train frequency. This keeps "Dutch Lady" over "Dutch" while
            # still avoiding arbitrary Brand+Line phrases.
            explicit = 1 if source == "official_brand_name" else 0
            score = (explicit, token_count, self.counts[brand], brand, brand_norm)
            if best is None or score > best:
                best = score
        if best is None:
            return None, "", ""
        _explicit, _token_count, _count, brand, brand_norm = best
        return brand, brand_norm, self.source_by_norm.get(brand_norm, "train_label_prefix")

    def _product_prefix_brand(self, product_name: object) -> tuple[str | None, str, str]:
        product = clean_product_text(product_name)
        product_norm = _norm(product)
        if not product_norm or _is_source_media(product):
            return None, "", ""
        best: tuple[int, int, str, str] | None = None
        for brand_norm, brand in self.by_norm.items():
            if brand_norm not in self.ocr_enabled_norms:
                continue
            if product_norm == brand_norm or product_norm.startswith(brand_norm + " "):
                token_count = len(brand_norm.split())
                score = (token_count, self.counts[brand], brand, brand_norm)
                if best is None or score > best:
                    best = score
        if best is None:
            return None, "", ""
        _token_count, _count, brand, brand_norm = best
        return brand, brand_norm, self.source_by_norm.get(brand_norm, "train_label_prefix")

    def _product_embedded_brand(self, product_name: object) -> tuple[str | None, str, str]:
        product = clean_product_text(product_name)
        product_norm = _norm(product)
        if not product_norm or _is_source_media(product):
            return None, "", ""
        best: tuple[int, int, str, str] | None = None
        for brand_norm, brand in self.by_norm.items():
            if brand_norm not in self.ocr_enabled_norms:
                continue
            if not _has_phrase(product_norm, brand_norm):
                continue
            token_count = len(brand_norm.split())
            score = (token_count, self.counts[brand], brand, brand_norm)
            if best is None or score > best:
                best = score
        if best is None:
            return None, "", ""
        _token_count, _count, brand, brand_norm = best
        return brand, brand_norm, self.source_by_norm.get(brand_norm, "train_label_prefix")

    def resolve(self, evidence_text: object, product_name: object) -> tuple[str, str, str, str]:
        product_brand, matched_norm, source = self._product_prefix_brand(product_name)
        if product_brand:
            return (
                _safe_brand_output_text(product_brand),
                "brand_from_product_prefix",
                f"brand prefix inferred from resolved product_name via {source}",
                matched_norm,
            )

        product_brand, matched_norm, source = self._product_embedded_brand(product_name)
        if product_brand:
            return (
                _safe_brand_output_text(product_brand),
                "brand_from_product_embedded",
                f"brand inferred from resolved product_name via {source}",
                matched_norm,
            )

        ocr_brand, matched_norm, source = self._best_ocr_brand(evidence_text)
        if ocr_brand:
            return (
                _safe_brand_output_text(ocr_brand),
                "brand_exact_ocr",
                f"train-derived brand '{ocr_brand}' visible in OCR evidence",
                matched_norm,
            )

        return " ", "brand_empty", "no safe train-derived brand evidence", ""

    def audit(self) -> pd.DataFrame:
        rows = [
            {
                "brand_name": brand,
                "normalized_brand_name": normalized,
                "count": self.counts[brand],
                "source": self.source_by_norm.get(normalized, ""),
                "ocr_enabled": normalized in self.ocr_enabled_norms,
            }
            for normalized, brand in sorted(self.by_norm.items(), key=lambda item: (item[1].lower(), item[0]))
        ]
        return pd.DataFrame(
            rows,
            columns=["brand_name", "normalized_brand_name", "count", "source", "ocr_enabled"],
        )


def _submission_columns(sample: pd.DataFrame) -> list[str]:
    columns = list(sample.columns)
    if columns == BASE_FINAL_COLUMNS:
        return BASE_FINAL_COLUMNS
    if columns == ["image_id", "ocr_text", "brand_name", "product_name"]:
        return columns
    raise ValueError(
        "sample submission must have columns "
        f"{BASE_FINAL_COLUMNS} or ['image_id', 'ocr_text', 'brand_name', 'product_name']; got {columns}"
    )


def _derive_brand_name(product_name: object) -> str:
    product = clean_product_text(product_name)
    if not product or product.strip().lower() in NULL_LIKE_VALUES:
        return " "
    product_norm = _norm(product)
    if _is_source_media(product):
        return " "
    for brand in KNOWN_BRAND_PREFIXES:
        brand_norm = _norm(brand)
        if product_norm == brand_norm or product_norm.startswith(brand_norm + " "):
            return _safe_output_text(brand)
    tokens = product.split()
    if not tokens:
        return " "
    if len(tokens) >= 2 and _norm(" ".join(tokens[:2])) in {"ha long", "ba vi", "dutch lady", "th true"}:
        return _safe_output_text(" ".join(tokens[:2]))
    first = tokens[0]
    if len(_norm(first)) < 3:
        return " "
    return _safe_output_text(first)


def _phase2_product_line(product_name: object, brand_name: object) -> tuple[str, str, str]:
    """Split old combined product labels into the phase-2 product target.

    This is deliberately prefix-only. It removes an already predicted brand
    from the start of an already predicted product label, but does not infer
    missing product lines or rewrite source/context labels.
    """

    product = clean_product_text(product_name)
    brand = clean_product_text(brand_name)
    if not product:
        return " ", "empty_product", "combined product prediction is empty"
    if not brand:
        return _safe_output_text(product), "no_brand_available", "no brand prediction available for product split"
    if _is_source_media(product):
        return _safe_output_text(product), "source_like_product_preserved", "source/context-like product label preserved"

    product_norm = _norm(product)
    brand_norm = _norm(brand)
    if brand_norm in {"ha long", "ha long canfoco", "halong", "halong canfoco"}:
        if _has_any(product_norm, ["pate", "patê", "cot den", "cotden"]):
            if _has_any(product_norm, ["cot den", "cotden"]):
                return _safe_output_text("Pate Cột Đèn"), "halong_pate_line", "mapped Ha Long/CANFOCO wrapper to product line Pate Cột Đèn"
            return _safe_output_text("Pate"), "halong_pate_line", "mapped Ha Long/CANFOCO wrapper to product line Pate"
        if _has_any(product_norm, ["do hop", "ha long", "halong", "canfoco", "canfood", "cong ty"]):
            return " ", "halong_company_only", "Ha Long/CANFOCO company/category label has no visible product line"

    if brand_norm == "nestle" and product_norm in {"sua cong thuc nestle", "sua cong thuc cua nestle"}:
        return _safe_output_text("SỮA CÔNG THỨC"), "embedded_brand_removed", "removed trailing Nestlé brand from formula product line"

    product_tokens = product.split()
    product_norm_tokens = product_norm.split()
    brand_norm_tokens = brand_norm.split()
    if not product_tokens or not product_norm_tokens or not brand_norm_tokens:
        return _safe_output_text(product), "split_not_applicable", "missing normalized product or brand tokens"
    if len(product_norm_tokens) < len(brand_norm_tokens):
        return _safe_output_text(product), "split_not_applicable", "brand has more tokens than product"
    if product_norm_tokens[: len(brand_norm_tokens)] != brand_norm_tokens:
        return _safe_output_text(product), "brand_not_prefix", "brand is not a product-name prefix"

    remaining = " ".join(product_tokens[len(brand_norm_tokens) :]).strip()
    if not remaining:
        return " ", "product_equals_brand", "combined product is brand-only after prefix split"
    return _safe_output_text(remaining), "brand_prefix_removed", f"removed brand prefix '{brand}' from combined product"


def _ocr_tokens_preserving_text(text: object) -> list[str]:
    cleaned = clean_ocr_text(text)
    return re.findall(r"[0-9A-Za-zÀ-ỹĐđ+]+", cleaned)


def _span_is_blocked(tokens: list[str]) -> bool:
    normalized_tokens = [_norm(token) for token in tokens if _norm(token)]
    if not normalized_tokens:
        return True
    token_set = set(normalized_tokens)
    if token_set & SPAN_BLOCK_TOKENS:
        return True
    if all(token in SPAN_UNIT_TOKENS or token in SPAN_GENERIC_TOKENS for token in normalized_tokens):
        return True
    if sum(any(ch.isalpha() for ch in token) for token in normalized_tokens) < 2:
        return True
    if len(normalized_tokens) == 1 and len(normalized_tokens[0]) < 4:
        return True
    return False


def _score_ocr_span(tokens: list[str]) -> tuple[float, str]:
    normalized_tokens = [_norm(token) for token in tokens if _norm(token)]
    if _span_is_blocked(tokens):
        return 0.0, "blocked_noise_or_too_weak"

    token_count = len(normalized_tokens)
    long_token_count = sum(len(token) >= 5 for token in normalized_tokens)
    has_model_like = any(any(ch.isdigit() for ch in token) or "+" in token for token in tokens)
    has_repeated = len(set(normalized_tokens)) < len(normalized_tokens)
    generic_count = sum(token in SPAN_GENERIC_TOKENS or token in SPAN_UNIT_TOKENS for token in normalized_tokens)

    score = 48.0 + token_count * 8.0 + long_token_count * 4.0
    if has_model_like:
        score += 6.0
    score -= generic_count * 8.0
    if has_repeated:
        score -= 10.0
    if token_count > 5:
        score -= (token_count - 5) * 12.0
    return max(0.0, min(100.0, score)), "open_vocab_span_score"


def _open_vocab_fallback(evidence_text: object, threshold: float) -> tuple[str | None, float, str]:
    tokens = _ocr_tokens_preserving_text(evidence_text)
    normalized_evidence = _norm(evidence_text)
    normalized_tokens = normalized_evidence.split()
    if len(normalized_tokens) > 18:
        return None, 0.0, "blocked_long_context_ocr"
    if _is_source_media(evidence_text):
        return None, 0.0, "blocked_source_media_context"
    if _has_any(normalized_evidence, NOISE_TEXT_PATTERNS):
        return None, 0.0, "blocked_price_promo_noise_context"

    best_span = ""
    best_score = 0.0
    best_reason = "no_candidate_span"
    seen: set[str] = set()

    for width in range(2, 6):
        for start in range(0, max(0, len(tokens) - width + 1)):
            span_tokens = tokens[start : start + width]
            span_text = " ".join(span_tokens)
            span_norm = _norm(span_text)
            if not span_norm or span_norm in seen:
                continue
            seen.add(span_norm)
            if _is_source_media(span_text):
                continue
            if _has_any(span_norm, NOISE_TEXT_PATTERNS):
                continue
            score, reason = _score_ocr_span(span_tokens)
            if score > best_score:
                best_span = clean_product_text(span_text)
                best_score = score
                best_reason = reason

    if best_span and best_score >= threshold:
        return best_span, best_score, best_reason
    return None, best_score, best_reason


class CanonicalLookup:
    def __init__(self, labels: pd.DataFrame) -> None:
        products = labels.get("product_name", pd.Series(dtype=str)).map(clean_product_text)
        products = products[products != ""]
        self.counts = Counter(products.tolist())
        self.by_norm: dict[str, str] = {}
        for name, _count in self.counts.most_common():
            normalized = _norm(name)
            if normalized and normalized not in self.by_norm:
                self.by_norm[normalized] = name

    def get(self, preferred_key: str) -> str | None:
        preferred = PREFERRED_CANONICALS[preferred_key]
        normalized = _norm(preferred)
        if normalized in self.by_norm:
            return self.by_norm[normalized]
        # Product labels in future organizer data may use casing variants.
        # If the exact preferred canonical is unseen, do not invent it.
        return None


def _detect_brandline(evidence_text: str, current_product: object, lookup: CanonicalLookup) -> tuple[str | None, str, str, str]:
    normalized = _norm(evidence_text)

    halong_brand = _has_any(normalized, ["ha long", "halong", "canfoco", "canfoco", "canfood", "do hop ha long"])
    halong_line = _has_any(normalized, ["pate", "pate gan", "cot den", "cotden"])
    if halong_brand and halong_line and _same_family_or_eligible(current_product, "halong"):
        candidate = lookup.get("halong_pate")
        if candidate:
            return candidate, "halong_pate_brandline", "halong", "brand and pate/cot-den evidence"

    if halong_brand and _same_family_or_eligible(current_product, "halong"):
        candidate = lookup.get("halong_brand")
        if candidate and _product_is_empty(current_product):
            return candidate, "halong_brand_only", "halong", "brand evidence without product-line evidence"

    has_nestle = _has_any(normalized, ["nestle", "nestl"])
    has_nan = _has_phrase(normalized, "nan")
    has_milo = _has_phrase(normalized, "milo")

    if has_nestle and has_nan and _same_family_or_eligible(current_product, "nestle"):
        if _has_any(normalized, ["optipro plus", "opti pro plus", "optiproplus", "optipro+"]):
            candidate = lookup.get("nestle_nan_optipro_plus")
            if candidate:
                return candidate, "nestle_nan_optipro_plus", "nestle", "Nestle + NAN + Optipro Plus evidence"
        if _has_any(normalized, ["supremepro", "supreme pro"]):
            candidate = lookup.get("nestle_nan_supremepro")
            if candidate:
                return candidate, "nestle_nan_supremepro", "nestle", "Nestle + NAN + Supremepro evidence"
        if _has_any(normalized, ["optipro 2", "opti pro 2", "optipro2"]):
            candidate = lookup.get("nestle_nan_optipro_2")
            if candidate:
                return candidate, "nestle_nan_optipro_2", "nestle", "Nestle + NAN + Optipro 2 evidence"
        candidate = lookup.get("nestle_nan")
        if candidate:
            return candidate, "nestle_nan_brandline", "nestle", "Nestle + NAN evidence"

    if has_nestle and has_milo and _same_family_or_eligible(current_product, "nestle"):
        candidate = lookup.get("nestle_milo")
        if candidate:
            return candidate, "nestle_milo_brandline", "nestle", "Nestle + Milo evidence"

    if _has_phrase(normalized, "vinamilk") and _has_phrase(normalized, "flex") and _same_family_or_eligible(current_product, "vinamilk"):
        candidate = lookup.get("vinamilk_flex")
        if candidate:
            return candidate, "vinamilk_flex_brandline", "vinamilk", "Vinamilk + Flex evidence"

    if _has_phrase(normalized, "vissan") and _has_any(normalized, ["pate heo", "pateheo"]) and _same_family_or_eligible(current_product, "vissan"):
        candidate = lookup.get("vissan_pate_heo")
        if candidate:
            return candidate, "vissan_pate_heo_brandline", "vissan", "Vissan + Pate Heo evidence"

    if _has_any(normalized, ["dutch lady", "dutchlady"]) and _has_phrase(normalized, "grow") and _same_family_or_eligible(current_product, "dutch_lady"):
        candidate = lookup.get("dutch_lady_grow")
        if candidate:
            return candidate, "dutch_lady_grow_brandline", "dutch_lady", "Dutch Lady + Grow evidence"

    if _has_any(normalized, ["ba vi", "bavi"]) and _has_phrase(normalized, "gold") and _same_family_or_eligible(current_product, "ba_vi"):
        candidate = lookup.get("ba_vi_gold")
        if candidate:
            return candidate, "ba_vi_gold_brandline", "ba_vi", "Ba Vi + Gold evidence"

    return None, "", "", ""


def build_submission(
    train_labels_path: Path,
    test_csv_path: Path,
    sample_submission_path: Path,
    test_en_ocr_path: Path,
    test_vi_ocr_path: Path,
    out_dir: Path,
    threshold: float = 92.0,
    use_known_family_adapters: bool = False,
    enable_cpu_surrogate_gate: bool = False,
    cpu_surrogate_upgrade_generic: bool = False,
    open_vocab_fallback: bool = False,
    open_vocab_threshold: float = 84.0,
    mode: str = "private_default",
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = out_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    labels = _read_csv(train_labels_path)
    test = _read_csv(test_csv_path)
    sample = _read_csv(sample_submission_path)
    test_en = _read_csv(test_en_ocr_path)
    test_vi = _read_csv(test_vi_ocr_path)

    if "image_id" not in test.columns:
        raise ValueError(f"{test_csv_path} must contain image_id")
    final_columns = _submission_columns(sample)

    dictionary_path = artifacts_dir / "product_dictionary.csv"
    write_dictionary_artifacts(labels_path=train_labels_path, dictionary_out=dictionary_path)
    dictionary = load_dictionary(dictionary_path)
    base_products, base_debug, base_stats = predict_dataframe(
        ocr=test_en[["image_id", "ocr_text"]],
        dictionary=dictionary,
        threshold=threshold,
        partial_threshold=95.0,
        min_match_tokens=2,
        disable_brand_fallback=True,
        limit=None,
    )
    base_products.to_csv(out_dir / "product_base_dictionary.csv", index=False, encoding="utf-8", lineterminator="\n")
    base_debug.to_csv(out_dir / "product_base_debug.csv", index=False, encoding="utf-8", lineterminator="\n")

    lookup = CanonicalLookup(labels)
    cpu_surrogate_lookup = CpuSurrogateAliasLookup(labels) if enable_cpu_surrogate_gate else None
    brand_resolver = BrandResolver(labels)
    merged = test[["image_id"]].merge(
        test_en[["image_id", "ocr_text"]].rename(columns={"ocr_text": "ocr_text_en"}),
        on="image_id",
        how="left",
    )
    merged = merged.merge(
        test_vi[["image_id", "ocr_text"]].rename(columns={"ocr_text": "ocr_text_vi"}),
        on="image_id",
        how="left",
    )
    merged = merged.merge(base_products[["image_id", "product_name"]].rename(columns={"product_name": "base_product_name"}), on="image_id", how="left")

    product_rows: list[dict[str, Any]] = []
    change_rows: list[dict[str, Any]] = []
    cpu_surrogate_audit_rows: list[dict[str, Any]] = []
    for row in merged.itertuples(index=False):
        image_id = str(row.image_id)
        en_text = clean_ocr_text(getattr(row, "ocr_text_en", ""))
        vi_text = clean_ocr_text(getattr(row, "ocr_text_vi", ""))
        evidence_text = f"{en_text} {vi_text}".strip()
        base_product = clean_product_text(getattr(row, "base_product_name", ""))
        new_product = base_product
        source = "dictionary_base"
        rule = "preserve_base"
        reason = "base dictionary/fuzzy prediction preserved"

        candidate: str | None = None
        candidate_rule = ""
        candidate_reason = ""
        if use_known_family_adapters:
            candidate, candidate_rule, _family, candidate_reason = _detect_brandline(evidence_text, base_product, lookup)

        if candidate:
            new_product = candidate
            source = "known_family_adapter"
            rule = candidate_rule
            reason = candidate_reason
        elif cpu_surrogate_lookup is not None:
            surrogate_candidate, surrogate_rule, surrogate_reason = cpu_surrogate_lookup.resolve(
                evidence_text,
                base_product,
                allow_generic_upgrade=cpu_surrogate_upgrade_generic,
            )
            cpu_surrogate_audit_rows.append(
                {
                    "image_id": image_id,
                    "base_product_name": _safe_output_text(base_product),
                    "candidate_product_name": _safe_output_text(surrogate_candidate or ""),
                    "rule": surrogate_rule,
                    "applied": bool(surrogate_candidate),
                    "reason": surrogate_reason,
                    "evidence_text": evidence_text,
                }
            )
            if surrogate_candidate:
                new_product = surrogate_candidate
                source = "cpu_surrogate_gate"
                rule = surrogate_rule
                reason = surrogate_reason

        if source == "dictionary_base" and _is_raw_exact_nan(base_product):
            new_product = ""
            source = "nan_safety"
            rule = "raw_nan_blocked"
            reason = "raw exact NAN blocked to avoid CSV null ambiguity"
        elif source == "dictionary_base" and _product_is_empty(base_product) and open_vocab_fallback:
            span, span_score, span_reason = _open_vocab_fallback(evidence_text, threshold=open_vocab_threshold)
            if span:
                new_product = span
                source = "open_vocab_fallback"
                rule = "ocr_span_fallback"
                reason = f"{span_reason}; score={span_score:.2f}; no strong train-label match"

        product_rows.append(
            {
                "image_id": image_id,
                "product_name": _safe_output_text(new_product),
                "source": source,
                "base_product_name": _safe_output_text(base_product),
                "rule": rule,
                "reason": reason,
            }
        )
        if clean_product_text(base_product) != clean_product_text(new_product):
            change_rows.append(
                {
                    "image_id": image_id,
                    "old_product_name": _safe_output_text(base_product),
                    "new_product_name": _safe_output_text(new_product),
                    "rule": rule,
                    "reason": reason,
                    "evidence_text": evidence_text,
                }
            )

    product_df = pd.DataFrame(product_rows, columns=PRODUCT_OUTPUT_COLUMNS)
    changes_df = pd.DataFrame(change_rows)
    cpu_surrogate_audit_df = pd.DataFrame(cpu_surrogate_audit_rows)
    brand_candidate_audit_df = brand_resolver.audit()

    final = test[["image_id"]].merge(
        test_en[["image_id", "ocr_text"]],
        on="image_id",
        how="left",
    )
    final = final.merge(product_df[["image_id", "product_name"]], on="image_id", how="left")
    final = final.merge(
        test_en[["image_id", "ocr_text"]].rename(columns={"ocr_text": "ocr_text_en"}),
        on="image_id",
        how="left",
    )
    final = final.merge(
        test_vi[["image_id", "ocr_text"]].rename(columns={"ocr_text": "ocr_text_vi"}),
        on="image_id",
        how="left",
    )
    final["ocr_text"] = final["ocr_text"].map(lambda value: clean_ocr_text(value) or " ")
    final["product_name"] = final["product_name"].map(_safe_output_text)
    brand_rows: list[dict[str, Any]] = []
    product_split_rows: list[dict[str, Any]] = []
    if "brand_name" in final_columns:
        resolved_brands: list[str] = []
        phase2_products: list[str] = []
        for row in final.itertuples(index=False):
            evidence_text = f"{clean_ocr_text(getattr(row, 'ocr_text_en', ''))} {clean_ocr_text(getattr(row, 'ocr_text_vi', ''))}".strip()
            combined_product = getattr(row, "product_name", "")
            brand_name, brand_rule, brand_reason, matched_text = brand_resolver.resolve(
                evidence_text=evidence_text,
                product_name=combined_product,
            )
            phase2_product, split_rule, split_reason = _phase2_product_line(combined_product, brand_name)
            resolved_brands.append(brand_name)
            phase2_products.append(phase2_product)
            brand_rows.append(
                {
                    "image_id": str(row.image_id),
                    "brand_name": brand_name,
                    "source": "v45_brand_resolver" if clean_product_text(brand_name) else "brand_empty",
                    "rule": brand_rule,
                    "reason": brand_reason,
                    "matched_text": matched_text,
                    "product_name": _safe_output_text(combined_product),
                }
            )
            product_split_rows.append(
                {
                    "image_id": str(row.image_id),
                    "combined_product_name": _safe_output_text(combined_product),
                    "brand_name": brand_name,
                    "phase2_product_name": phase2_product,
                    "rule": split_rule,
                    "reason": split_reason,
                }
            )
        final["brand_name"] = resolved_brands
        final["product_name"] = phase2_products
    final = final.drop(columns=[column for column in ["ocr_text_en", "ocr_text_vi"] if column in final.columns])
    final = final[final_columns]
    brand_df = pd.DataFrame(brand_rows, columns=BRAND_OUTPUT_COLUMNS)
    product_split_df = pd.DataFrame(product_split_rows, columns=PRODUCT_SPLIT_OUTPUT_COLUMNS)

    submission_path = out_dir / "submission.csv"
    final_path = out_dir / "UPLOAD_THIS_submission.csv"
    product_path = out_dir / "product_predictions.csv"
    brand_path = out_dir / "brand_predictions.csv"
    product_split_path = out_dir / "phase2_product_split_debug.csv"
    brand_candidate_audit_path = out_dir / "brand_candidate_audit.csv"
    changes_path = out_dir / "brandline_changes.csv"
    cpu_surrogate_audit_path = out_dir / "cpu_surrogate_gate_audit.csv"
    summary_path = out_dir / "summary.json"

    product_df.to_csv(product_path, index=False, encoding="utf-8", lineterminator="\n")
    brand_df.to_csv(brand_path, index=False, encoding="utf-8", lineterminator="\n")
    product_split_df.to_csv(product_split_path, index=False, encoding="utf-8", lineterminator="\n")
    brand_candidate_audit_df.to_csv(brand_candidate_audit_path, index=False, encoding="utf-8", lineterminator="\n")
    changes_df.to_csv(changes_path, index=False, encoding="utf-8", lineterminator="\n")
    cpu_surrogate_audit_df.to_csv(cpu_surrogate_audit_path, index=False, encoding="utf-8", lineterminator="\n")
    final.to_csv(submission_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL, lineterminator="\n")
    final.to_csv(final_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL, lineterminator="\n")

    read_back = pd.read_csv(final_path)
    errors: list[str] = []
    if list(read_back.columns) != final_columns:
        errors.append(f"bad columns: {list(read_back.columns)}")
    if len(read_back) != len(sample):
        errors.append(f"row count mismatch: got {len(read_back)}, expected {len(sample)}")
    if read_back["image_id"].astype(str).tolist() != sample["image_id"].astype(str).tolist():
        errors.append("image_id order does not match sample_submission.csv")
    nan_count = int(read_back.isna().sum().sum())
    if nan_count:
        errors.append(f"pandas NaN count is {nan_count}")
    raw_nan_products = int(read_back["product_name"].astype(str).str.strip().str.lower().eq("nan").sum())
    if raw_nan_products:
        errors.append(f"raw exact NAN product_name count is {raw_nan_products}")
    raw_nan_brands = 0
    if "brand_name" in read_back.columns:
        raw_nan_brands = int(read_back["brand_name"].astype(str).str.strip().str.lower().eq("nan").sum())
        if raw_nan_brands:
            errors.append(f"raw exact NAN brand_name count is {raw_nan_brands}")

    rule_counts = Counter(product_df["rule"].tolist())
    brand_rule_counts = Counter(brand_df["rule"].tolist()) if not brand_df.empty else Counter()
    product_split_rule_counts = Counter(product_split_df["rule"].tolist()) if not product_split_df.empty else Counter()
    summary = {
        "mode": mode,
        "rows": len(final),
        "final_columns": final_columns,
        "use_known_family_adapters": use_known_family_adapters,
        "enable_cpu_surrogate_gate": enable_cpu_surrogate_gate,
        "cpu_surrogate_upgrade_generic": cpu_surrogate_upgrade_generic,
        "open_vocab_fallback": open_vocab_fallback,
        "open_vocab_threshold": open_vocab_threshold,
        "base_empty_product_count": int(base_products["product_name"].astype(str).str.strip().eq("").sum()),
        "combined_empty_product_count": int(product_df["product_name"].astype(str).str.strip().eq("").sum()),
        "final_empty_product_count": int(final["product_name"].astype(str).str.strip().eq("").sum()),
        "final_nonempty_product_count": int(final["product_name"].astype(str).str.strip().ne("").sum()),
        "final_empty_brand_count": int(final["brand_name"].astype(str).str.strip().eq("").sum()) if "brand_name" in final.columns else None,
        "final_nonempty_brand_count": int(final["brand_name"].astype(str).str.strip().ne("").sum()) if "brand_name" in final.columns else None,
        "changed_product_rows": len(changes_df),
        "cpu_surrogate_applied_rows": int(cpu_surrogate_audit_df["applied"].astype(bool).sum()) if not cpu_surrogate_audit_df.empty else 0,
        "rule_counts": dict(rule_counts),
        "brand_rule_counts": dict(brand_rule_counts),
        "product_split_rule_counts": dict(product_split_rule_counts),
        "brand_candidate_count": int(len(brand_candidate_audit_df)),
        "base_stats": base_stats,
        "pandas_nan_count": nan_count,
        "raw_exact_nan_product_count": raw_nan_products,
        "raw_exact_nan_brand_count": raw_nan_brands,
        "submission_valid": not errors,
        "validation_errors": errors,
        "submission_path": str(final_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a full hidden-test submission from official train labels and local OCR outputs.")
    parser.add_argument("--train-labels", type=Path, default=Path("train_labels.csv"))
    parser.add_argument("--test-csv", type=Path, default=Path("test.csv"))
    parser.add_argument("--sample-submission", type=Path, default=Path("sample_submission.csv"))
    parser.add_argument("--test-en-ocr", type=Path, default=Path("outputs/ocr_test_raw.csv"))
    parser.add_argument("--test-vi-ocr", type=Path, default=Path("outputs/ocr_test_vi_raw.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/full_inference"))
    parser.add_argument("--threshold", type=float, default=92.0)
    parser.add_argument("--mode", choices=["private_default", "private_hybrid", "reproduce_public_style"], default="private_default")
    parser.add_argument("--use-known-family-adapters", action="store_true", help="Enable public-family Brand+Line adapters as an explicit ablation.")
    parser.add_argument("--enable-cpu-surrogate-gate", action="store_true", help="Enable V44 deployable CPU-only exact/unique-alias OCR n-gram gate.")
    parser.add_argument("--cpu-surrogate-upgrade-generic", action="store_true", help="Allow V44 CPU surrogate to upgrade generic non-empty base products. Default fills only empty/raw-null/source-like rows.")
    parser.add_argument("--enable-open-vocab-fallback", action="store_true", help="Enable conservative OCR-span fallback for unmatched rows.")
    parser.add_argument("--no-open-vocab-fallback", action="store_true", help="Backward-compatible explicit disable flag.")
    parser.add_argument("--open-vocab-threshold", type=float, default=84.0)
    args = parser.parse_args()

    use_known_family_adapters = args.use_known_family_adapters or args.mode in {"private_hybrid", "reproduce_public_style"}
    open_vocab_fallback = bool(args.enable_open_vocab_fallback and not args.no_open_vocab_fallback)
    if args.mode == "reproduce_public_style":
        open_vocab_fallback = False

    summary = build_submission(
        train_labels_path=args.train_labels,
        test_csv_path=args.test_csv,
        sample_submission_path=args.sample_submission,
        test_en_ocr_path=args.test_en_ocr,
        test_vi_ocr_path=args.test_vi_ocr,
        out_dir=args.out_dir,
        threshold=args.threshold,
        use_known_family_adapters=use_known_family_adapters,
        enable_cpu_surrogate_gate=args.enable_cpu_surrogate_gate,
        cpu_surrogate_upgrade_generic=args.cpu_surrogate_upgrade_generic,
        open_vocab_fallback=open_vocab_fallback,
        open_vocab_threshold=args.open_vocab_threshold,
        mode=args.mode,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["submission_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
