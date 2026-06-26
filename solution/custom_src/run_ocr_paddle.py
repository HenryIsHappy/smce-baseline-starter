import os

os.environ["FLAGS_use_onednn"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import argparse
import csv
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .normalize import clean_ocr_text
from .paths import OUTPUT_DIR, TEST_CSV, TRAIN_CSV, ensure_dirs, image_path


OUTPUT_COLUMNS = ["image_id", "ocr_text", "avg_confidence", "num_boxes", "preprocess_mode"]
PREPROCESS_MODES = ["original", "upscale_2x", "sharpen", "contrast"]
DEFAULT_OUTPUT = OUTPUT_DIR / "ocr_paddle_test.csv"


@dataclass
class OCRBox:
    box: list[list[float]]
    text: str
    confidence: float

    @property
    def center_x(self) -> float:
        return sum(point[0] for point in self.box) / len(self.box)

    @property
    def center_y(self) -> float:
        return sum(point[1] for point in self.box) / len(self.box)

    @property
    def height(self) -> float:
        ys = [point[1] for point in self.box]
        return max(ys) - min(ys)


@dataclass
class ImageDebugInfo:
    image_id: str
    image_path: Path
    path_exists: bool
    pil_format: str
    pil_mode: str
    is_animated: bool
    n_frames: int | None
    numpy_shape: tuple[int, ...] | None


@dataclass
class RunSummary:
    total_images: int = 0
    missing_images: int = 0
    ocr_failures: int = 0
    images_with_boxes: int = 0
    images_with_text: int = 0


def _get_tqdm():
    try:
        from tqdm import tqdm

        return tqdm
    except ImportError:
        return lambda iterable, **_: iterable


def _blank_row(image_id: str, preprocess_mode: str) -> dict[str, object]:
    return {
        "image_id": image_id,
        "ocr_text": " ",
        "avg_confidence": 0.0,
        "num_boxes": 0,
        "preprocess_mode": preprocess_mode,
    }


def _load_ids(split: str) -> list[str]:
    csv_path = TRAIN_CSV if split == "train" else TEST_CSV
    return pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding="utf-8")["image_id"].tolist()


def _processed_ids(out_path: Path) -> set[str]:
    if not out_path.exists() or out_path.stat().st_size == 0:
        return set()
    try:
        existing = pd.read_csv(out_path, dtype=str, keep_default_na=False, encoding="utf-8")
    except pd.errors.EmptyDataError:
        return set()
    if "image_id" not in existing.columns:
        return set()
    return set(existing["image_id"].astype(str))


def _validate_existing_output(out_path: Path) -> None:
    if not out_path.exists() or out_path.stat().st_size == 0:
        return
    existing = pd.read_csv(out_path, nrows=0, keep_default_na=False, encoding="utf-8")
    if list(existing.columns) != OUTPUT_COLUMNS:
        raise ValueError(f"{out_path} has columns {list(existing.columns)}; expected {OUTPUT_COLUMNS}")


def _append_rows(out_path: Path, rows: list[dict[str, object]], append: bool = True) -> None:
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not append or not out_path.exists() or out_path.stat().st_size == 0
    pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(
        out_path,
        mode="a" if append else "w",
        header=write_header,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_MINIMAL,
        lineterminator="\n",
    )


def _init_paddleocr(lang: str, use_gpu: bool):
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise RuntimeError(f"PaddleOCR is not installed: {exc}")
    except OSError as exc:
        if "torch" not in sys.modules:
            torch_stub = types.ModuleType("torch")
            torch_stub.Tensor = object
            sys.modules["torch"] = torch_stub
        try:
            from paddleocr import PaddleOCR
            print(f"PaddleOCR import recovered with torch shim after OSError: {exc}")
        except Exception as retry_exc:
            raise RuntimeError(f"PaddleOCR could not be imported. Error: {retry_exc}")

    lang_candidates = [lang]
    if lang != "en":
        lang_candidates.append("en")

    last_error: Exception | None = None
    for candidate in lang_candidates:
        kwargs_candidates = [
            {"use_angle_cls": True, "lang": candidate, "use_gpu": use_gpu, "show_log": False},
            {"use_angle_cls": True, "lang": candidate, "show_log": False},
            {"lang": candidate, "use_gpu": use_gpu},
            {"lang": candidate},
        ]
        for kwargs in kwargs_candidates:
            try:
                return PaddleOCR(**kwargs)
            except Exception as exc:  # pragma: no cover - depends on local PaddleOCR version
                last_error = exc

    raise RuntimeError(f"PaddleOCR could not be initialized. Error: {last_error}")


def _pil_image_info(path: Path) -> tuple[str, str, bool, int | None]:
    try:
        from PIL import Image

        with Image.open(path) as img:
            return (
                str(img.format or ""),
                str(img.mode or ""),
                bool(getattr(img, "is_animated", False)),
                int(getattr(img, "n_frames", 1)),
            )
    except Exception:
        return "", "", False, None


def load_image_for_ocr(path: Path) -> np.ndarray:
    """Load an image as a BGR numpy array, using PIL frame 0 before cv2 fallback."""
    pil_error: Exception | None = None
    try:
        from PIL import Image, ImageSequence

        with Image.open(path) as img:
            if bool(getattr(img, "is_animated", False)) or int(getattr(img, "n_frames", 1)) > 1:
                frame = next(ImageSequence.Iterator(img))
            else:
                frame = img
            rgb = frame.convert("RGB")
            rgb_array = np.asarray(rgb)
            return rgb_array[:, :, ::-1].copy()
    except Exception as exc:
        pil_error = exc

    try:
        import cv2

        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is not None:
            return bgr
    except Exception as cv2_error:
        raise ValueError(f"failed to load image with PIL ({pil_error}) or cv2 ({cv2_error}): {path}") from cv2_error

    raise ValueError(f"failed to load image with PIL ({pil_error}) or cv2.imread returned None: {path}")


def preprocess_image_array(image: np.ndarray, mode: str) -> np.ndarray:
    """Apply one preprocessing mode to a BGR numpy image."""
    if mode == "original":
        return image

    try:
        import cv2

        if mode == "upscale_2x":
            return cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        if mode == "sharpen":
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            return cv2.filter2D(image, -1, kernel)
        if mode == "contrast":
            return cv2.convertScaleAbs(image, alpha=1.35, beta=0)
    except ImportError:
        pass

    from PIL import Image, ImageEnhance, ImageFilter

    rgb = image[:, :, ::-1]
    pil_img = Image.fromarray(rgb)
    if mode == "upscale_2x":
        pil_img = pil_img.resize((pil_img.width * 2, pil_img.height * 2), Image.Resampling.LANCZOS)
    elif mode == "sharpen":
        pil_img = pil_img.filter(ImageFilter.SHARPEN)
    elif mode == "contrast":
        pil_img = ImageEnhance.Contrast(pil_img).enhance(1.5)
    else:
        raise ValueError(f"unknown preprocess mode: {mode}")
    return np.asarray(pil_img)[:, :, ::-1].copy()


def _is_box(value: object) -> bool:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return False
    try:
        for point in value[:4]:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                return False
            float(point[0])
            float(point[1])
    except (TypeError, ValueError):
        return False
    return True


def _coerce_box(value: object, index: int = 0) -> list[list[float]]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if _is_box(value):
        return [[float(point[0]), float(point[1])] for point in value[:4]]  # type: ignore[index]
    y = float(index * 20)
    return [[0.0, y], [1.0, y], [1.0, y + 1.0], [0.0, y + 1.0]]


def _parse_ocr_item(item: object) -> OCRBox | None:
    if not isinstance(item, (list, tuple)) or len(item) < 2 or not _is_box(item[0]):
        return None

    text = ""
    confidence = 0.0
    payload = item[1]
    if isinstance(payload, (list, tuple)) and payload:
        text = "" if payload[0] is None else str(payload[0])
        if len(payload) > 1:
            try:
                confidence = float(payload[1])
            except (TypeError, ValueError):
                confidence = 0.0
    elif payload is not None:
        text = str(payload)

    box = _coerce_box(item[0])
    text = clean_ocr_text(text)
    if not text:
        return None
    return OCRBox(box=box, text=text, confidence=confidence)


def _extract_dict_boxes(result: dict[str, Any]) -> list[OCRBox]:
    boxes: list[OCRBox] = []
    text_values = (
        result.get("rec_texts")
        or result.get("texts")
        or result.get("text")
        or result.get("transcription")
        or result.get("label")
    )
    score_values = result.get("rec_scores") or result.get("scores") or result.get("confidence") or result.get("score")
    box_values = (
        result.get("rec_boxes")
        or result.get("dt_polys")
        or result.get("rec_polys")
        or result.get("boxes")
        or result.get("points")
        or result.get("box")
    )

    if isinstance(text_values, str):
        texts = [text_values]
    elif isinstance(text_values, (list, tuple)):
        texts = ["" if text is None else str(text) for text in text_values]
    else:
        texts = []

    if not texts:
        return boxes

    if isinstance(score_values, (int, float, str)):
        scores = [score_values] * len(texts)
    elif isinstance(score_values, (list, tuple)):
        scores = list(score_values)
    else:
        scores = [0.0] * len(texts)

    if isinstance(box_values, np.ndarray):
        if box_values.ndim == 2:
            box_list = [box_values]
        elif box_values.ndim >= 3:
            box_list = list(box_values)
        else:
            box_list = []
    elif _is_box(box_values):
        box_list: list[object] = [box_values] * len(texts)
    elif isinstance(box_values, (list, tuple)):
        box_list = list(box_values)
    else:
        box_list = []

    for index, text in enumerate(texts):
        clean_text = clean_ocr_text(text)
        if not clean_text:
            continue
        try:
            confidence = float(scores[index])
        except (IndexError, TypeError, ValueError):
            confidence = 0.0
        box_value = box_list[index] if index < len(box_list) else None
        boxes.append(OCRBox(box=_coerce_box(box_value, index=index), text=clean_text, confidence=confidence))
    return boxes


def _flatten_ocr_result(result: object) -> list[OCRBox]:
    boxes: list[OCRBox] = []
    if result is None:
        return boxes
    if isinstance(result, dict):
        boxes.extend(_extract_dict_boxes(result))
        for value in result.values():
            if isinstance(value, (dict, list, tuple)):
                boxes.extend(_flatten_ocr_result(value))
        return boxes
    if hasattr(result, "json"):
        try:
            json_value = result.json
            if callable(json_value):
                json_value = json_value()
            boxes.extend(_flatten_ocr_result(json_value))
            if boxes:
                return boxes
        except Exception:
            pass
    if hasattr(result, "to_dict"):
        try:
            boxes.extend(_flatten_ocr_result(result.to_dict()))
            if boxes:
                return boxes
        except Exception:
            pass
    parsed = _parse_ocr_item(result)
    if parsed is not None:
        return [parsed]
    if isinstance(result, (list, tuple)):
        for item in result:
            boxes.extend(_flatten_ocr_result(item))
    return boxes


def sort_ocr_boxes(boxes: list[OCRBox]) -> list[OCRBox]:
    """Sort boxes top-to-bottom, grouping near rows, then left-to-right inside rows."""
    if not boxes:
        return []

    heights = sorted(max(box.height, 1.0) for box in boxes)
    median_height = heights[len(heights) // 2]
    row_threshold = max(10.0, median_height * 0.60)

    rows: list[list[OCRBox]] = []
    for box in sorted(boxes, key=lambda item: (item.center_y, item.center_x)):
        placed = False
        for row in rows:
            row_y = sum(item.center_y for item in row) / len(row)
            if abs(box.center_y - row_y) <= row_threshold:
                row.append(box)
                placed = True
                break
        if not placed:
            rows.append([box])

    rows.sort(key=lambda row: sum(item.center_y for item in row) / len(row))
    ordered: list[OCRBox] = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda item: item.center_x))
    return ordered


def _run_reader(reader, ocr_input: Any) -> object:
    errors: list[str] = []
    
    for attempt in range(4):
        safe_input = ocr_input.copy() if hasattr(ocr_input, "copy") else ocr_input
        
        try:
            return reader.ocr(safe_input, cls=True)
        except Exception as exc:
            errors.append(f"[Attempt {attempt}] ocr(cls=True): {type(exc).__name__}: {exc}")

        try:
            return reader.ocr(safe_input)
        except Exception as exc:
            errors.append(f"[Attempt {attempt}] ocr(): {type(exc).__name__}: {exc}")

        if hasattr(reader, "predict"):
            try:
                return reader.predict(safe_input)
            except Exception as exc:
                errors.append(f"[Attempt {attempt}] predict(): {type(exc).__name__}: {exc}")
        else:
            errors.append("reader.predict(image): unavailable")
            
        import time
        time.sleep(0.2) # Let C++ memory flush before retrying

    raise RuntimeError("All PaddleOCR inference calls failed. " + " | ".join(errors))


def _short_repr(value: object, limit: int = 1000) -> str:
    text = repr(value)
    if len(text) > limit:
        return text[:limit] + "... <truncated>"
    return text


def _parsed_items_repr(boxes: list[OCRBox], limit: int = 10) -> str:
    items = [
        {"text": box.text, "confidence": round(float(box.confidence), 6), "box": box.box}
        for box in boxes[:limit]
    ]
    suffix = "" if len(boxes) <= limit else f" ... +{len(boxes) - limit} more"
    return f"{items!r}{suffix}"


def _debug_print(
    debug_info: ImageDebugInfo,
    raw_result: object | None,
    boxes: list[OCRBox],
    parsed_text: str,
    error: Exception | None = None,
) -> None:
    print("\nDEBUG OCR IMAGE")
    print(f"image_id       : {debug_info.image_id}")
    print(f"path           : {debug_info.image_path}")
    print(f"path exists    : {debug_info.path_exists}")
    print(f"PIL format     : {debug_info.pil_format}")
    print(f"PIL mode       : {debug_info.pil_mode}")
    print(f"is_animated    : {debug_info.is_animated}")
    print(f"n_frames       : {debug_info.n_frames}")
    print(f"numpy shape    : {debug_info.numpy_shape}")
    if error is not None:
        print(f"error          : {type(error).__name__}: {error}")
    print(f"raw result type: {type(raw_result).__name__}")
    print(f"raw result repr: {_short_repr(raw_result)}")
    print(f"parsed items   : {_parsed_items_repr(boxes)}")
    print(f"parsed OCR text: {parsed_text}")


def run_single_image(
    reader,
    image_id: str,
    split: str,
    preprocess_mode: str,
    pass_path_directly: bool = False,
    debug: bool = False,
) -> tuple[dict[str, object], bool, bool]:
    image_file = image_path(image_id, split)
    if not image_file.exists() or reader is None:
        if debug:
            pil_format, pil_mode, is_animated, n_frames = _pil_image_info(image_file)
            numpy_shape = None
            if not image_file.exists():
                debug_error: Exception | None = FileNotFoundError(image_file)
            elif reader is None:
                debug_error = RuntimeError("PaddleOCR reader is unavailable")
            else:
                debug_error = None
            if image_file.exists():
                try:
                    numpy_shape = tuple(load_image_for_ocr(image_file).shape)
                except Exception as exc:
                    debug_error = exc
            _debug_print(
                ImageDebugInfo(
                    image_id=image_id,
                    image_path=image_file,
                    path_exists=image_file.exists(),
                    pil_format=pil_format,
                    pil_mode=pil_mode,
                    is_animated=is_animated,
                    n_frames=n_frames,
                    numpy_shape=numpy_shape,
                ),
                raw_result=None,
                boxes=[],
                parsed_text=" ",
                error=debug_error,
            )
        return _blank_row(image_id, preprocess_mode), reader is None or image_file.exists(), not image_file.exists()

    raw_result: object | None = None
    parsed_text = " "
    boxes: list[OCRBox] = []
    pil_format, pil_mode, is_animated, n_frames = _pil_image_info(image_file)
    debug_info = ImageDebugInfo(image_id, image_file, True, pil_format, pil_mode, is_animated, n_frames, None)
    try:
        if pass_path_directly and preprocess_mode == "original":
            ocr_input: Any = str(image_file)
            numpy_shape: tuple[int, ...] | None = None
        else:
            image_array = load_image_for_ocr(image_file)
            image_array = preprocess_image_array(image_array, preprocess_mode)
            ocr_input = image_array
            numpy_shape = tuple(image_array.shape)
        debug_info = ImageDebugInfo(image_id, image_file, True, pil_format, pil_mode, is_animated, n_frames, numpy_shape)
        raw_result = _run_reader(reader, ocr_input)
        boxes = sort_ocr_boxes(_flatten_ocr_result(raw_result))
        parsed_text = clean_ocr_text(" ".join(box.text for box in boxes))
        if not parsed_text:
            parsed_text = " "
        avg_confidence = sum(box.confidence for box in boxes) / len(boxes) if boxes else 0.0
        row = {
            "image_id": image_id,
            "ocr_text": parsed_text,
            "avg_confidence": round(float(avg_confidence), 6),
            "num_boxes": len(boxes),
            "preprocess_mode": preprocess_mode,
        }
        if debug:
            _debug_print(debug_info, raw_result=raw_result, boxes=boxes, parsed_text=parsed_text)
        return row, False, False
    except Exception as exc:
        print(f"Warning: OCR failed for {image_id}: {exc}")
        if debug:
            _debug_print(debug_info, raw_result=raw_result, boxes=boxes, parsed_text=parsed_text, error=exc)
        return _blank_row(image_id, preprocess_mode), True, False


def run_ocr(
    split: str,
    out_path: Path,
    limit: int | None = None,
    resume: bool = True,
    lang: str = "vi",
    use_gpu: bool = False,
    preprocess_mode: str = "original",
    pass_path_directly: bool = False,
    debug_first: int = 0,
) -> None:
    ensure_dirs()
    if resume:
        _validate_existing_output(out_path)
    ids = _load_ids(split)
    if limit is not None:
        ids = ids[:limit]

    processed = _processed_ids(out_path) if resume else set()
    ids_to_process = [image_id for image_id in ids if image_id not in processed]
    print(f"Split: {split}")
    print(f"Total requested: {len(ids):,}")
    print(f"Already processed: {len(processed):,}")
    print(f"Remaining: {len(ids_to_process):,}")

    if not ids_to_process:
        if not out_path.exists():
            _append_rows(out_path, [])
        print(f"No new images to process. Output: {out_path}")
        return

    reader = _init_paddleocr(lang=lang, use_gpu=use_gpu)
    tqdm = _get_tqdm()
    buffer: list[dict[str, object]] = []
    summary = RunSummary(total_images=len(ids_to_process))
    first_write = True
    for index, image_id in enumerate(tqdm(ids_to_process, desc=f"OCR {split}"), start=1):
        row, failed, missing = run_single_image(
            reader,
            image_id,
            split,
            preprocess_mode,
            pass_path_directly=pass_path_directly,
            debug=index <= debug_first,
        )
        summary.missing_images += int(missing)
        summary.ocr_failures += int(failed)
        summary.images_with_boxes += int(int(row["num_boxes"]) > 0)
        summary.images_with_text += int(str(row["ocr_text"]).strip() != "")
        buffer.append(row)
        if len(buffer) >= 25:
            _append_rows(out_path, buffer, append=resume or not first_write)
            first_write = False
            buffer.clear()
    _append_rows(out_path, buffer, append=resume or not first_write)
    print(f"Wrote OCR rows to {out_path}")
    print("OCR summary")
    print(f"Total images processed : {summary.total_images:,}")
    print(f"Missing images         : {summary.missing_images:,}")
    print(f"OCR failures           : {summary.ocr_failures:,}")
    print(f"Images with boxes      : {summary.images_with_boxes:,}")
    print(f"Images with OCR text   : {summary.images_with_text:,}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lang", default="vi")
    parser.add_argument("--use-gpu", action="store_true", default=False)
    parser.add_argument("--preprocess-mode", choices=PREPROCESS_MODES, default="original")
    parser.add_argument("--pass-path-directly", action="store_true", default=False)
    parser.add_argument("--debug-first", type=int, default=0)
    args = parser.parse_args()

    run_ocr(
        split=args.split,
        out_path=args.out,
        limit=args.limit,
        resume=args.resume,
        lang=args.lang,
        use_gpu=args.use_gpu,
        preprocess_mode=args.preprocess_mode,
        pass_path_directly=args.pass_path_directly,
        debug_first=args.debug_first,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
