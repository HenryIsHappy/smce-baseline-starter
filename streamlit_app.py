#!/usr/bin/env python3
"""Streamlit demo shell for URA Hackathon teams — customize team_config.py + solution/."""

from __future__ import annotations

import io

import streamlit as st
from PIL import Image

import team_config as cfg
from shared.benchmark import (
    get_deploy_smoke_benchmark,
    get_model_profile,
    run_predict_with_metrics,
)

APP_CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&display=swap');

:root {{
    --ura-blue: {cfg.THEME_PRIMARY};
    --ura-blue-dark: {cfg.THEME_PRIMARY_DARK};
    --ura-bg: {cfg.THEME_BG};
    --ura-text: {cfg.THEME_TEXT};
    --ura-muted: {cfg.THEME_MUTED};
}}

html, body, .stApp {{
    font-family: 'Montserrat', sans-serif !important;
    font-size: 14px !important;
    background-color: var(--ura-bg) !important;
    color: var(--ura-text) !important;
}}

[data-testid="stSidebar"] {{ display: none; }}
[data-testid="collapsedControl"] {{ display: none; }}

[data-testid="stAppViewContainer"] > section > div {{
    padding-top: 1rem;
}}

[data-testid="stImage"]:first-of-type {{
    margin-bottom: 1rem;
}}

[data-testid="stImage"]:first-of-type img {{
    max-height: 72px;
    width: auto;
}}

.app-title,
[data-testid="stMarkdownContainer"] p.app-title {{
    display: block;
    font-family: 'Montserrat', sans-serif !important;
    font-size: 32px !important;
    font-weight: 700 !important;
    color: var(--ura-blue) !important;
    margin: 0 0 0.5rem 0 !important;
    line-height: 1.25 !important;
}}

.app-subtitle {{
    display: block;
    font-family: 'Montserrat', sans-serif !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    color: var(--ura-muted) !important;
    margin: 0 0 0.75rem 0 !important;
    line-height: 1.5 !important;
    max-width: 100%;
}}

.app-team-info {{
    margin: 0 0 1.25rem 0;
    padding: 0;
    list-style: none;
}}

.app-team-info li {{
    font-family: 'Montserrat', sans-serif !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
    margin: 0 0 0.35rem 0 !important;
    color: var(--ura-text) !important;
}}

.app-team-info li strong {{
    color: var(--ura-blue);
    font-weight: 600;
}}

.app-team-info a {{
    color: var(--ura-blue);
    text-decoration: none;
    font-weight: 500;
}}

.app-team-info a:hover {{
    text-decoration: underline;
}}

[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4 {{
    font-family: 'Montserrat', sans-serif !important;
    color: var(--ura-blue) !important;
}}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stCaptionContainer"] {{
    font-family: 'Montserrat', sans-serif !important;
    font-size: 14px !important;
}}

.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
    color: var(--ura-blue) !important;
    border-bottom-color: var(--ura-blue) !important;
}}

.stTabs [data-baseweb="tab-list"] button {{
    font-family: 'Montserrat', sans-serif !important;
    font-size: 14px !important;
    font-weight: 600 !important;
}}

.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {{
    background-color: var(--ura-blue) !important;
    border-color: var(--ura-blue) !important;
    color: #FFFFFF !important;
    font-family: 'Montserrat', sans-serif !important;
    font-size: 14px !important;
    font-weight: 600 !important;
}}

.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {{
    background-color: var(--ura-blue-dark) !important;
    border-color: var(--ura-blue-dark) !important;
}}

.stTextInput input,
.stTextArea textarea,
.stTextInput label,
.stTextArea label {{
    font-family: 'Montserrat', sans-serif !important;
    font-size: 14px !important;
}}

[data-testid="stFileUploader"] label {{
    font-family: 'Montserrat', sans-serif !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    color: var(--ura-text) !important;
}}

[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] {{
    font-family: 'Montserrat', sans-serif !important;
    font-size: 14px !important;
}}

[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] button {{
    font-family: 'Montserrat', sans-serif !important;
    font-size: 14px !important;
}}
"""

st.set_page_config(
    page_title=cfg.BROWSER_TITLE,
    page_icon=str(cfg.FAVICON),
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(f"<style>{APP_CSS}</style>", unsafe_allow_html=True)

st.image(str(cfg.LOGO), width=cfg.LOGO_WIDTH)

st.markdown(
    f'<p class="app-title">{cfg.PAGE_TITLE}</p>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<p class="app-subtitle">{cfg.SUBTITLE}</p>',
    unsafe_allow_html=True,
)
st.markdown(
    f"""
    <ul class="app-team-info">
        <li><strong>Team Member:</strong> {cfg.TEAM_MEMBERS}</li>
        <li><strong>Github Repo link:</strong> <a href="{cfg.GITHUB_REPO}" target="_blank">{cfg.GITHUB_REPO}</a></li>
        <li><strong>Other resource link:</strong> <a href="{cfg.OTHER_RESOURCE}" target="_blank">{cfg.OTHER_RESOURCE}</a></li>
    </ul>
    """,
    unsafe_allow_html=True,
)


def _init_live_state() -> None:
    defaults = {
        "ocr_text_live": "",
        "brand_name_live": "",
        "product_name_live": "",
        "upload_file_id": None,
        "timing_ms": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def _load_uploaded_image(uploaded) -> Image.Image:
    return Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")


def _clear_live_results() -> None:
    st.session_state["ocr_text_live"] = ""
    st.session_state["brand_name_live"] = ""
    st.session_state["product_name_live"] = ""
    st.session_state["timing_ms"] = None


@st.cache_data(show_spinner=False)
def _cached_model_profile() -> dict:
    return get_model_profile()


@st.cache_resource(show_spinner="Running deploy smoke benchmark (1 image)...")
def _cached_deploy_smoke() -> dict:
    return get_deploy_smoke_benchmark()


def _render_about_tab() -> None:
    st.header("About")
    st.markdown(
        """
        Tab này dành cho **mỗi team** trình bày giải pháp OCR + trích xuất
        **brand_name** và **product_name** cho cuộc thi. Hãy thay các placeholder
        bên dưới bằng nội dung thật của team bạn (hoặc chỉnh trực tiếp trong
        [`streamlit_app.py`](streamlit_app.py) hàm `_render_about_tab`).
        """
    )

    st.subheader("1. Thông tin team")
    st.markdown(
        f"""
        | Trường | Nội dung |
        |--------|----------|
        | **Tên team** | {cfg.TEAM_NAME} |
        | **Thành viên** | {cfg.TEAM_MEMBERS} |
        | **GitHub** | [{cfg.GITHUB_REPO}]({cfg.GITHUB_REPO}) |
        """
    )

    st.subheader("2. Bài toán")
    st.markdown(
        """
        Từ **ảnh sản phẩm trên kệ hàng / social media**, hệ thống cần trích xuất:

        - **`ocr_text`** — toàn bộ văn bản đọc được từ ảnh
        - **`brand_name`** — tên thương hiệu
        - **`product_name`** — tên / mô tả sản phẩm

        **Điểm private round:**

        `0.4 × F1_brand + 0.35 × (1 − CER) + 0.25 × F1_product`
        """
    )

    st.subheader("3. Ý tưởng & pipeline giải pháp")
    st.markdown(
        """
        > **Mô tả pipeline của Team 24 - 2URA4**

        1. **Tiền xử lý ảnh** — Chuyển đổi định dạng ảnh, giữ nguyên cấu trúc gốc hoặc upscale nhẹ để tăng cường độ nét cho OCR.
        2. **OCR** — Sử dụng mô hình **PaddleOCR** để nhận diện chữ tiếng Việt với độ chính xác cao trên CPU.
        3. **Hậu xử lý OCR** — Sắp xếp các text box từ trên xuống dưới, trái qua phải; làm sạch các ký tự rác và khoảng trắng thừa.
        4. **Trích xuất brand** — Sử dụng `BrandResolver` đối chiếu với tập dữ liệu `train_labels.csv`, kết hợp thuật toán fuzzy matching (RapidFuzz) để sửa lỗi chính tả.
        5. **Trích xuất product** — Xây dựng từ điển sản phẩm `product_dictionary.csv`, sử dụng thuật toán nối token N-gram và chấm điểm độ tương đồng để tìm ra tên sản phẩm chuẩn xác nhất.
        6. **Hậu kiểm / ensemble** — Chuẩn hóa định dạng đầu ra, đảm bảo brand không bị lặp lại trong product name.
        """
    )

    st.subheader("4. Điểm khác biệt & đóng góp chính")
    st.markdown(
        """
        - **Tối ưu hóa tốc độ & bộ nhớ**: Tích hợp PaddleOCR siêu nhẹ, chạy ổn định và mượt mà trên môi trường CPU 100%.
        - **Thuật toán Fuzzy Matching**: Sử dụng thư viện `rapidfuzz` giúp hệ thống chống chịu tốt với các lỗi nhận diện sai chính tả của mô hình OCR.
        - **Cơ chế Pipeline Module**: Tách biệt rõ ràng phần OCR, giải quyết Brand và Product thành các module riêng biệt.
        """
    )

    st.subheader("5. Công nghệ sử dụng")
    st.markdown(
        """
        | Thành phần | Công nghệ |
        |------------|-----------|
        | OCR | **PaddleOCR** |
        | Brand extraction | **Fuzzy Matching (RapidFuzz) + Brand Rules** |
        | Product extraction | **Dictionary Token Matching** |
        | Runtime | **CPU, Python 3.12** |
        | Demo UI | **Streamlit Cloud** |
        """
    )

    st.subheader("6. Kết quả & đánh giá")
    st.markdown(
        """
        | Metric | Giá trị |
        |--------|---------|
        | F1 brand (local) | `Đang cập nhật` |
        | 1 − CER (local) | `Đang cập nhật` |
        | F1 product (local) | `Đang cập nhật` |
        | **Private score** | `Đang cập nhật` |
        | Latency (avg / image) | `~0.1` ms |
        | Product head size | `0.0` MB (Dictionary-based) |
        """
    )
    st.markdown(
        """
        **Đo lightweight model (latency + footprint):**

        ```bash
        python scripts/benchmark_solution.py --limit 6
        ```

        Cập nhật `MODEL_PROFILE` trong [`team_config.py`](team_config.py)
        khi đổi OCR / model. Benchmark luôn chạy qua [`shared/benchmark.py`](shared/benchmark.py).
        """
    )

    st.subheader("7. Hạn chế & hướng phát triển")
    st.markdown(
        """
        **Hạn chế hiện tại**
        - Thuật toán fuzzy matching phụ thuộc nhiều vào chất lượng của bộ từ điển `product_dictionary.csv`. Nếu có sản phẩm hoặc thương hiệu quá mới, hệ thống có thể ghép sai.
        - Khả năng xử lý ảnh chụp ở góc quá nghiêng hoặc mờ nhòe còn hạn chế nếu OCR đọc thiếu quá nhiều chữ.

        **Hướng phát triển**
        - Fine-tune lại trọng số của mô hình PaddleOCR trên tập dữ liệu đặc thù về các thương hiệu tại Việt Nam.
        - Kết hợp các mô hình phân loại (Classification) hoặc NLP nhẹ như PhởBERT để trích xuất ngữ nghĩa tốt hơn thay vì chỉ dùng fuzzy matching.
        """
    )

    st.subheader("8. Liên kết")
    links = [
        f"- **Repository:** [{cfg.GITHUB_REPO}]({cfg.GITHUB_REPO})",
        "- **Setup & deploy:** [README.md](README.md)",
        f"- **Other resource:** [{cfg.OTHER_RESOURCE}]({cfg.OTHER_RESOURCE})",
    ]
    streamlit_url = getattr(cfg, "STREAMLIT_APP_URL", "")
    if streamlit_url:
        links.insert(
            1,
            f"- **Live demo (Streamlit Cloud):** [{streamlit_url}]({streamlit_url})",
        )
    st.markdown("\n".join(links))


tab_live, tab_about = st.tabs(["Live test", "About"])

with tab_live:
    _init_live_state()
    st.subheader("Live test")

    profile = _cached_model_profile()
    smoke = _cached_deploy_smoke()
    with st.expander("Model footprint (lightweight check)", expanded=False):
        st.markdown(
            f"- **Pipeline:** {profile.get('pipeline', '—')}\n"
            f"- **Runtime:** {profile.get('runtime_device', '—')}\n"
            f"- **Product head:** {profile.get('product_head_mb', 0)} MB\n"
            f"- **OCR note:** {profile.get('ocr_backend_note', '—')}\n\n"
            f"{profile.get('lightweight_notes', '')}"
        )
        if smoke.get("latency_ms"):
            lat = smoke["latency_ms"]
            st.markdown(
                f"**Deploy smoke benchmark (1 image):** "
                f"total **{lat.get('total_avg', '—')} ms** "
                f"(ocr {lat.get('ocr_avg', '—')} · extract {lat.get('extract_avg', '—')})"
            )
        elif smoke.get("error"):
            st.caption(f"Deploy smoke benchmark skipped: {smoke['error']}")
        st.caption("Full report: `python scripts/benchmark_solution.py --limit 6`")

    uploaded = st.file_uploader(
        "Ảnh sản phẩm",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        key="live_upload",
    )

    if uploaded:
        file_id = f"{uploaded.name}:{uploaded.size}"
        if st.session_state["upload_file_id"] != file_id:
            st.session_state["upload_file_id"] = file_id
            _clear_live_results()

        img = _load_uploaded_image(uploaded)
        col_img, col_result = st.columns(2)

        with col_img:
            st.image(img, use_container_width=True)

        with col_result:
            if st.button("Chạy OCR", type="primary", key="run_ocr_live"):
                with st.spinner("Đang chạy OCR..."):
                    pred = run_predict_with_metrics(img)
                    st.session_state["ocr_text_live"] = pred["ocr_text"]
                    st.session_state["brand_name_live"] = pred["brand_name"]
                    st.session_state["product_name_live"] = pred["product_name"]
                    st.session_state["timing_ms"] = pred.get("timing_ms")

            timing = st.session_state.get("timing_ms")
            if timing:
                t1, t2, t3 = st.columns(3)
                t1.metric("Total (ms)", f"{timing['total']:.1f}")
                t2.metric("OCR (ms)", f"{timing['ocr']:.1f}")
                t3.metric("Extract (ms)", f"{timing['extract']:.1f}")

            st.text_area("ocr_text", height=140, key="ocr_text_live")
            st.text_input("brand_name", key="brand_name_live")
            st.text_input("product_name", key="product_name_live")
    else:
        st.session_state["upload_file_id"] = None
        _clear_live_results()

with tab_about:
    _render_about_tab()
