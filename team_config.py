"""Team-facing configuration — edit this file after forking the template."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


# Team identity (required after fork)

TEAM_NAME = "Team 24 - 2URA4"
TEAM_MEMBERS = "Nguyễn Hoàng Thắng, Nguyễn Cảnh Hưng, Lê Bảo Nguyên, Dương Minh Nhật"
GITHUB_REPO = "https://github.com/HenryIsHappy/smce-baseline-starter"
OTHER_RESOURCE = "https://example.com/other-resource"
STREAMLIT_APP_URL = "https://smce-baseline-starter-etbnbnzpthaqxq8x36xav6.streamlit.app/"  # e.g. "https://ura-team-abc.streamlit.app" after deploy


# Streamlit page copy

SUBTITLE = (
    "OCR & Product Name Extraction from Social Media Images "
    "by HCMUT URA Research Group"
)
PAGE_TITLE = f"The 2nd URA Hackathon - {TEAM_NAME}"
BROWSER_TITLE = PAGE_TITLE


# Branding assets (replace files under assets/ if needed)

ASSETS_DIR = REPO_ROOT / "assets"
FAVICON = ASSETS_DIR / "kaggle_144224_logos_thumb76_76.png"
LOGO = ASSETS_DIR / "bk_name_en.png"
LOGO_WIDTH = 280


# UI theme

THEME_PRIMARY = "#1565C0"
THEME_PRIMARY_DARK = "#0D47A1"
THEME_BG = "#FFFFFF"
THEME_TEXT = "#1A2B4A"
THEME_MUTED = "#5C6B8A"


# Default inference settings (override inside solution/pipeline.py if needed)

DEFAULT_MIN_CONF = 0.35


# Model footprint (edit when you change OCR / models — benchmark layer reads this)

MODEL_PROFILE: dict[str, str | float | None] = {
    "pipeline": "PaddleOCR (vi) + custom fuzzy dictionary matching + brand resolver",
    "runtime_device": "CPU",
    "product_head_mb": None,  # auto-estimate when None
    "ocr_backend_note": "PaddleOCR models",
    "lightweight_notes": (
        "Updated to use the new V45 architecture with PaddleOCR."
    ),
}
