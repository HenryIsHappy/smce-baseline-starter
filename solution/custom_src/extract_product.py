from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

from .build_dictionary import DEFAULT_OUTPUT, build_product_dictionary
from .normalize import clean_ocr_text, normalize_for_match


OUTPUT_COLUMNS = ["image_id", "product_name", "product_score", "matched_canonical", "match_type", "matched_text"]
DEBUG_COLUMNS = [
    "image_id",
    "rank",
    "canonical",
    "normalized",
    "matched_text",
    "decision_rule",
    "accepted",
    "score",
    "token_set_score",
    "partial_score",
    "meaningful_overlap",
    "meaningful_required",
    "candidate_meaningful_tokens",
    "frequency_bonus",
    "reject_reason",
]
MISSED_COLUMNS = ["image_id", "ocr_text", "normalized_ocr", "top_candidates", "top_scores", "reason"]

NOISE_TOKENS = {
    "giam",
    "sale",
    "freeship",
    "khuyen",
    "mai",
    "mua",
    "tang",
    "combo",
    "flash",
    "k",
    "vnd",
    "dong",
    "chi",
    "con",
    "gia",
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
    "hashtag",
    "follow",
    "livestream",
}
GENERIC_TOKENS = {
    "sua",
    "do",
    "hop",
    "pate",
    "review",
    "tra",
    "ca",
    "gao",
    "banh",
    "nuoc",
    "san",
    "pham",
}

try:
    from rapidfuzz import fuzz as _rapidfuzz_fuzz
except ImportError:  # pragma: no cover - optional dependency
    _rapidfuzz_fuzz = None


@dataclass(frozen=True)
class ProductCandidate:
    candidate_id: int
    canonical: str
    normalized: str
    count: int
    tokens: tuple[str, ...]
    meaningful_tokens: frozenset[str]
    aliases: tuple[str, ...]
    is_known_one_token: bool
    frequency_bonus: float


@dataclass(frozen=True)
class CandidateScore:
    candidate: ProductCandidate
    score: float
    token_set_score: float
    partial_score: float
    meaningful_overlap: int
    matched_text: str
    decision_rule: str
    accepted: bool
    reject_reason: str


def load_dictionary(path: Path = DEFAULT_OUTPUT) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    return build_product_dictionary()


def _validate_dictionary(dictionary: pd.DataFrame) -> None:
    required = {"canonical_product_name", "normalized_product_name", "count"}
    missing = required - set(dictionary.columns)
    if missing:
        raise ValueError(f"product dictionary missing required columns: {sorted(missing)}")


def _validate_ocr(ocr: pd.DataFrame) -> None:
    missing = {"image_id", "ocr_text"} - set(ocr.columns)
    if missing:
        raise ValueError(f"OCR CSV missing required columns: {sorted(missing)}")


def _tokenize(text: str) -> list[str]:
    return [token for token in text.split() if token]


def _meaningful(tokens: list[str] | tuple[str, ...]) -> list[str]:
    return [token for token in tokens if token not in NOISE_TOKENS and token not in GENERIC_TOKENS and len(token) >= 2]


def _safe_count(value: object) -> int:
    try:
        return max(1, int(float(value)))
    except (TypeError, ValueError):
        return 1


def _frequency_bonus(count: int, max_count: int) -> float:
    if max_count <= 1:
        return 0.0
    return min(3.0, math.log1p(count) / math.log1p(max_count) * 3.0)


def _ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio() * 100.0


def _fallback_partial_ratio(needle: str, haystack: str) -> float:
    if not needle or not haystack:
        return 0.0
    if needle in haystack:
        return 100.0
    if len(needle) > len(haystack):
        needle, haystack = haystack, needle
    width = len(needle)
    if width == 0 or len(haystack) < width:
        return 0.0
    return max(_ratio(needle, haystack[start : start + width]) for start in range(len(haystack) - width + 1))


def _fallback_token_set_ratio(a: str, b: str) -> float:
    a_tokens = set(_tokenize(a))
    b_tokens = set(_tokenize(b))
    if not a_tokens or not b_tokens:
        return 0.0
    common = a_tokens & b_tokens
    left = " ".join(sorted(common | (a_tokens - common)))
    right = " ".join(sorted(common | (b_tokens - common)))
    common_text = " ".join(sorted(common))
    return max(_ratio(left, right), _ratio(common_text, left), _ratio(common_text, right))


def token_set_ratio(a: str, b: str) -> float:
    if _rapidfuzz_fuzz is not None:
        return float(_rapidfuzz_fuzz.token_set_ratio(a, b))
    return _fallback_token_set_ratio(a, b)


def partial_ratio(a: str, b: str) -> float:
    if _rapidfuzz_fuzz is not None:
        return float(_rapidfuzz_fuzz.partial_ratio(a, b))
    return _fallback_partial_ratio(a, b)


def _is_unit_or_noise_only(tokens: tuple[str, ...]) -> bool:
    return bool(tokens) and all(token in NOISE_TOKENS or token in GENERIC_TOKENS for token in tokens)


def _strong_aliases(normalized: str, min_match_tokens: int) -> tuple[str, ...]:
    tokens = _tokenize(normalized)
    aliases: list[str] = [normalized]
    if len(tokens) == 2:
        alias = " ".join(tokens)
        if len(_meaningful(tokens)) >= min_match_tokens and not _is_unit_or_noise_only(tuple(tokens)):
            aliases.append(alias)
    if len(tokens) >= 3:
        alias_tokens = tokens[:3]
        alias = " ".join(alias_tokens)
        if len(_meaningful(alias_tokens)) >= min_match_tokens and not _is_unit_or_noise_only(tuple(alias_tokens)):
            aliases.append(alias)
    # Preserve order while deduplicating.
    return tuple(dict.fromkeys(aliases))


def build_product_candidates(dictionary: pd.DataFrame, min_match_tokens: int) -> list[ProductCandidate]:
    _validate_dictionary(dictionary)
    groups: dict[str, dict[str, object]] = {}
    for row in dictionary.itertuples(index=False):
        canonical = clean_ocr_text(getattr(row, "canonical_product_name"))
        normalized = normalize_for_match(getattr(row, "normalized_product_name")) or normalize_for_match(canonical)
        count = _safe_count(getattr(row, "count"))
        if not canonical or not normalized:
            continue
        current = groups.get(normalized)
        if current is None:
            groups[normalized] = {"canonical": canonical, "count": count, "best_count": count}
        else:
            current["count"] = int(current["count"]) + count
            if count > int(current["best_count"]):
                current["canonical"] = canonical
                current["best_count"] = count

    max_count = max((int(data["count"]) for data in groups.values()), default=1)
    candidates: list[ProductCandidate] = []
    for candidate_id, (normalized, data) in enumerate(groups.items()):
        tokens = tuple(_tokenize(normalized))
        meaningful = frozenset(_meaningful(tokens))
        is_known_one_token = (
            len(tokens) == 1
            and len(tokens[0]) >= 4
            and tokens[0] not in NOISE_TOKENS
            and tokens[0] not in GENERIC_TOKENS
            and int(data["count"]) >= 2
        )
        if _is_unit_or_noise_only(tokens):
            continue
        candidates.append(
            ProductCandidate(
                candidate_id=candidate_id,
                canonical=str(data["canonical"]),
                normalized=normalized,
                count=int(data["count"]),
                tokens=tokens,
                meaningful_tokens=meaningful,
                aliases=_strong_aliases(normalized, min_match_tokens=min_match_tokens),
                is_known_one_token=is_known_one_token,
                frequency_bonus=_frequency_bonus(int(data["count"]), max_count),
            )
        )
    return candidates


def _build_token_index(candidates: list[ProductCandidate]) -> dict[str, set[int]]:
    token_index: dict[str, set[int]] = defaultdict(set)
    for idx, candidate in enumerate(candidates):
        for token in candidate.meaningful_tokens:
            token_index[token].add(idx)
    return dict(token_index)


def _shortlist(normalized_ocr: str, candidates: list[ProductCandidate], token_index: dict[str, set[int]]) -> list[ProductCandidate]:
    ocr_tokens = set(_meaningful(_tokenize(normalized_ocr)))
    ids: set[int] = set()
    for token in ocr_tokens:
        ids.update(token_index.get(token, set()))
    for idx, candidate in enumerate(candidates):
        if candidate.normalized in normalized_ocr:
            ids.add(idx)
            continue
        for alias in candidate.aliases:
            if alias in normalized_ocr:
                ids.add(idx)
                break
    return [candidates[idx] for idx in ids]


def _meaningful_overlap(candidate: ProductCandidate, normalized_ocr: str) -> int:
    ocr_tokens = set(_meaningful(_tokenize(normalized_ocr)))
    return len(candidate.meaningful_tokens & ocr_tokens)


def _has_min_tokens(candidate: ProductCandidate, overlap: int, min_match_tokens: int) -> bool:
    if candidate.is_known_one_token:
        return overlap >= 1
    return overlap >= min_match_tokens


def _best_exact_alias(candidate: ProductCandidate, normalized_ocr: str, min_match_tokens: int) -> str:
    aliases = sorted(candidate.aliases, key=lambda alias: (len(_tokenize(alias)), len(alias)), reverse=True)
    for alias in aliases:
        alias_tokens = _tokenize(alias)
        if alias in normalized_ocr:
            if candidate.is_known_one_token or len(_meaningful(alias_tokens)) >= min_match_tokens:
                return alias
    return ""


def score_candidate(
    candidate: ProductCandidate,
    normalized_ocr: str,
    threshold: float,
    partial_threshold: float,
    min_match_tokens: int,
) -> CandidateScore:
    overlap = _meaningful_overlap(candidate, normalized_ocr)
    token_score = token_set_ratio(candidate.normalized, normalized_ocr)
    partial_score_value = partial_ratio(candidate.normalized, normalized_ocr)
    exact_alias = _best_exact_alias(candidate, normalized_ocr, min_match_tokens)

    accepted = False
    decision_rule = ""
    matched_text = ""
    reject_reason = ""
    base_score = 0.0

    if exact_alias:
        accepted = True
        decision_rule = "exact_full_or_alias"
        matched_text = exact_alias
        base_score = 100.0 if exact_alias == candidate.normalized else 96.0
    elif not _has_min_tokens(candidate, overlap, min_match_tokens):
        reject_reason = "insufficient_meaningful_token_overlap"
        base_score = max(token_score, partial_score_value)
    elif token_score >= threshold:
        accepted = True
        decision_rule = "token_set_threshold"
        matched_text = candidate.normalized
        base_score = token_score
    elif partial_score_value >= partial_threshold and len(candidate.normalized) >= 10:
        accepted = True
        decision_rule = "partial_threshold_long_span"
        matched_text = candidate.normalized
        base_score = partial_score_value
    else:
        reject_reason = "below_threshold"
        base_score = max(token_score, partial_score_value)

    if candidate.is_known_one_token and decision_rule != "exact_full_or_alias":
        accepted = False
        reject_reason = "one_token_requires_exact_match"

    score = min(100.0, base_score + (candidate.frequency_bonus if accepted else 0.0))
    return CandidateScore(
        candidate=candidate,
        score=round(score, 6),
        token_set_score=round(token_score, 6),
        partial_score=round(partial_score_value, 6),
        meaningful_overlap=overlap,
        matched_text=matched_text,
        decision_rule=decision_rule,
        accepted=accepted,
        reject_reason=reject_reason,
    )


def _sort_scores(scores: list[CandidateScore]) -> list[CandidateScore]:
    return sorted(
        scores,
        key=lambda item: (
            int(item.accepted),
            item.score,
            item.meaningful_overlap,
            len(item.candidate.tokens),
            item.candidate.count,
        ),
        reverse=True,
    )


def predict_one(
    ocr_text: object,
    candidates: list[ProductCandidate],
    token_index: dict[str, set[int]],
    threshold: float,
    partial_threshold: float,
    min_match_tokens: int,
    disable_brand_fallback: bool,
) -> tuple[dict[str, object], list[CandidateScore], str]:
    cleaned = clean_ocr_text(ocr_text)
    normalized_ocr = normalize_for_match(cleaned)
    if not normalized_ocr:
        return {
            "product_name": " ",
            "product_score": 0.0,
            "matched_canonical": " ",
            "match_type": "empty",
            "matched_text": " ",
        }, [], "empty_ocr"

    shortlist = _shortlist(normalized_ocr, candidates, token_index)
    scores = _sort_scores(
        [
            score_candidate(
                candidate,
                normalized_ocr,
                threshold=threshold,
                partial_threshold=partial_threshold,
                min_match_tokens=min_match_tokens,
            )
            for candidate in shortlist
        ]
    )
    accepted = [score for score in scores if score.accepted]
    if accepted:
        best = accepted[0]
        return {
            "product_name": best.candidate.canonical,
            "product_score": best.score,
            "matched_canonical": best.candidate.canonical,
            "match_type": "full_product",
            "matched_text": best.matched_text,
        }, scores, best.decision_rule

    # Explicitly keep this off by default. It exists only for backwards experiments.
    if not disable_brand_fallback:
        return {
            "product_name": " ",
            "product_score": scores[0].score if scores else 0.0,
            "matched_canonical": " ",
            "match_type": "empty",
            "matched_text": " ",
        }, scores, "brand_fallback_disabled_for_precision"

    reason = "no_shortlist" if not scores else "no_high_confidence_match"
    return {
        "product_name": " ",
        "product_score": scores[0].score if scores else 0.0,
        "matched_canonical": " ",
        "match_type": "empty",
        "matched_text": " ",
    }, scores, reason


def extract_product_name(
    ocr_text: object,
    dictionary: pd.DataFrame | None = None,
    threshold: float = 92.0,
) -> str:
    dictionary = load_dictionary() if dictionary is None else dictionary
    candidates = build_product_candidates(dictionary, min_match_tokens=2)
    token_index = _build_token_index(candidates)
    row, _, _ = predict_one(
        ocr_text,
        candidates,
        token_index,
        threshold=threshold,
        partial_threshold=95.0,
        min_match_tokens=2,
        disable_brand_fallback=True,
    )
    product_name = str(row["product_name"])
    return "" if product_name == " " else product_name


def predict_dataframe(
    ocr: pd.DataFrame,
    dictionary: pd.DataFrame,
    threshold: float,
    partial_threshold: float,
    min_match_tokens: int,
    disable_brand_fallback: bool,
    limit: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    _validate_ocr(ocr)
    if limit is not None:
        ocr = ocr.head(limit).copy()

    candidates = build_product_candidates(dictionary, min_match_tokens=min_match_tokens)
    token_index = _build_token_index(candidates)
    rows: list[dict[str, object]] = []
    debug_rows: list[dict[str, object]] = []
    exact_count = 0
    fuzzy_count = 0

    for record in ocr.itertuples(index=False):
        image_id = str(getattr(record, "image_id"))
        pred, scores, reason = predict_one(
            getattr(record, "ocr_text"),
            candidates,
            token_index,
            threshold=threshold,
            partial_threshold=partial_threshold,
            min_match_tokens=min_match_tokens,
            disable_brand_fallback=disable_brand_fallback,
        )
        rows.append({"image_id": image_id, **pred})
        if pred["match_type"] == "full_product":
            if reason == "exact_full_or_alias":
                exact_count += 1
            else:
                fuzzy_count += 1

        for rank, score in enumerate(scores[:10], start=1):
            candidate = score.candidate
            debug_rows.append(
                {
                    "image_id": image_id,
                    "rank": rank,
                    "canonical": candidate.canonical,
                    "normalized": candidate.normalized,
                    "matched_text": score.matched_text,
                    "decision_rule": score.decision_rule or reason,
                    "accepted": score.accepted,
                    "score": score.score,
                    "token_set_score": score.token_set_score,
                    "partial_score": score.partial_score,
                    "meaningful_overlap": score.meaningful_overlap,
                    "meaningful_required": min_match_tokens,
                    "candidate_meaningful_tokens": " ".join(sorted(candidate.meaningful_tokens)),
                    "frequency_bonus": round(candidate.frequency_bonus, 6),
                    "reject_reason": score.reject_reason,
                }
            )
        if not scores:
            debug_rows.append(
                {
                    "image_id": image_id,
                    "rank": 0,
                    "canonical": "",
                    "normalized": "",
                    "matched_text": "",
                    "decision_rule": reason,
                    "accepted": False,
                    "score": 0.0,
                    "token_set_score": 0.0,
                    "partial_score": 0.0,
                    "meaningful_overlap": 0,
                    "meaningful_required": min_match_tokens,
                    "candidate_meaningful_tokens": "",
                    "frequency_bonus": 0.0,
                    "reject_reason": reason,
                }
            )

    pred_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS).fillna(" ")
    debug_df = pd.DataFrame(debug_rows, columns=DEBUG_COLUMNS)
    non_empty_scores = pred_df.loc[pred_df["product_name"].astype(str).str.strip() != "", "product_score"].astype(float)
    stats = {
        "total_rows": float(len(pred_df)),
        "empty_product_count": float(pred_df["product_name"].astype(str).str.strip().eq("").sum()),
        "full_product_count": float(pred_df["match_type"].eq("full_product").sum()),
        "exact_match_count": float(exact_count),
        "fuzzy_match_count": float(fuzzy_count),
        "brand_fallback_count": float(pred_df["match_type"].eq("brand_fallback").sum()),
        "avg_non_empty_score": float(non_empty_scores.mean()) if len(non_empty_scores) else 0.0,
        "candidate_count": float(len(candidates)),
    }
    return pred_df, debug_df, stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr", type=Path, required=True)
    parser.add_argument("--dict", dest="dictionary", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=92.0)
    parser.add_argument("--partial-threshold", type=float, default=95.0)
    parser.add_argument("--min-match-tokens", type=int, default=2)
    parser.add_argument("--disable-brand-fallback", action="store_true", default=True)
    parser.add_argument("--enable-brand-fallback", dest="disable_brand_fallback", action="store_false")
    parser.add_argument("--save-debug", type=Path)
    parser.add_argument("--save-missed", type=Path)
    parser.add_argument("--limit", type=int)
    # Backward-compatible ignored options from v2 scripts.
    parser.add_argument("--brand-threshold", type=float, default=82.0)
    parser.add_argument("--line-threshold", type=float, default=70.0)
    parser.add_argument("--max-candidates-per-row", type=int, default=80)
    parser.add_argument("--topk-debug", type=int, default=0)
    parser.add_argument("--debug-out", type=Path)
    args = parser.parse_args()

    ocr = pd.read_csv(args.ocr, dtype=str, keep_default_na=False, encoding="utf-8")
    dictionary = load_dictionary(args.dictionary)
    pred, debug, stats = predict_dataframe(
        ocr=ocr,
        dictionary=dictionary,
        threshold=args.threshold,
        partial_threshold=args.partial_threshold,
        min_match_tokens=args.min_match_tokens,
        disable_brand_fallback=args.disable_brand_fallback,
        limit=args.limit,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(args.out, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL, lineterminator="\n")

    debug_out = args.save_debug or args.debug_out
    if debug_out is None and args.topk_debug > 0:
        debug_out = args.out.with_name(f"{args.out.stem}_debug.csv")
    if debug_out is not None:
        debug_out.parent.mkdir(parents=True, exist_ok=True)
        debug.to_csv(debug_out, index=False, encoding="utf-8", lineterminator="\n")
        print(f"Wrote debug rows to {debug_out}")

    if args.save_missed is not None:
        missed = pred[pred["product_name"].astype(str).str.strip() == ""].copy()
        missed.to_csv(args.save_missed, index=False, encoding="utf-8", lineterminator="\n")
        print(f"Wrote missed rows to {args.save_missed}")

    print(f"Total rows: {int(stats['total_rows']):,}")
    print(f"Empty product count: {int(stats['empty_product_count']):,}")
    print(f"full_product match count: {int(stats['full_product_count']):,}")
    print(f"exact match count: {int(stats['exact_match_count']):,}")
    print(f"fuzzy match count: {int(stats['fuzzy_match_count']):,}")
    print(f"brand fallback count: {int(stats['brand_fallback_count']):,}")
    print(f"Average score for non-empty predictions: {stats['avg_non_empty_score']:.2f}")
    print(f"Candidate count: {int(stats['candidate_count']):,}")
    print(f"Wrote {len(pred):,} product predictions to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
