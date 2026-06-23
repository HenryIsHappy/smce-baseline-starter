<div align="center">

# The 2nd URA Hackathon — Team Submission Template

**Fork-friendly starter: Streamlit demo, batch submission, and baseline OCR pipeline**

<p>
  <img src="https://img.shields.io/badge/python-3.11+-blue?style=for-the-badge" alt="Python 3.11+"/>
  <img src="https://img.shields.io/badge/Type-Team%20Template-7c3aed?style=for-the-badge" alt="Team Template"/>
  <img src="https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge" alt="MIT"/>
</p>

<p>
  <img src="https://skillicons.dev/icons?i=python,github" alt="Python, GitHub"/>
</p>

</div>

## Overview

Template repository for **The 2nd URA Hackathon 2026** (SMCE private round). Teams **fork** this repo, customize branding, and **replace the code under `solution/`** with their own OCR + brand/product extraction pipeline.

Included out of the box:

- **Streamlit demo** — Live test (image upload) + About tab for team presentation
- **Baseline pipeline** — EasyOCR + regex brand rules + optional sklearn product head
- **Batch script** — writes `outputs/submission_private.csv` (Kaggle-ready)
- **Benchmark layer** — [`shared/benchmark.py`](shared/benchmark.py) measures latency + footprint on deploy and Live test (see [Benchmark](#benchmark-lightweight-model))
- **Sample data** — 6 private-test JPEGs + 1,202 IDs; full images installed separately
- **Official metric** — [`private_test/metric.py`](private_test/metric.py) (do not edit)

**Private score:** `0.4 × F1_brand + 0.35 × (1 − CER) + 0.25 × F1_product`

**Fork in 3 steps:**

1. **Fork** → edit [`team_config.py`](team_config.py) (team name, members, links, logo).
2. **Replace** [`solution/`](solution/) — keep `predict_from_image()` working.
3. **Run** `streamlit run streamlit_app.py` and `python scripts/run_submission.py`.

---

## Repository layout

```text
ura-hackathon-template/
├── team_config.py           # Edit: branding & team info
├── streamlit_app.py         # Demo UI (Live test + About)
├── solution/                # Replace: your ML pipeline
│   ├── pipeline.py          # predict_from_image() entry point
│   ├── brand_rules.py
│   ├── product_model.py
│   ├── baseline_notebook.ipynb
│   └── README.md
├── shared/                  # Template helpers (do not replace)
│   ├── data_utils.py
│   └── benchmark.py         # timing + footprint (always used by Streamlit)
├── scripts/
│   ├── setup_private_images.py
│   ├── run_submission.py
│   └── benchmark_solution.py
├── data/
│   ├── train_labels.csv
│   └── private_test/
├── assets/                  # Logos & favicon
├── outputs/                 # Generated submissions (gitignored)
└── private_test/metric.py   # Official scoring — do not edit
```

| Customize | Keep as-is (template-owned) |
|-----------|----------------------------|
| `team_config.py` + `MODEL_PROFILE` | `shared/benchmark.py`, `shared/data_utils.py` |
| `solution/*.py` + notebook | `streamlit_app.py` core, `scripts/benchmark_solution.py` |
| `assets/` logos | `private_test/metric.py` |
| About tab content | Submission column format |

---

## Team setup

Follow this checklist after forking.

### 1. Fork & clone

```bash
git clone https://github.com/YOUR_ORG/ura-hackathon-team-abc.git
cd ura-hackathon-team-abc
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Prerequisites:** Python **3.11+**, ~**1 GB RAM** (EasyOCR downloads ~200 MB of weights on first run).

### 2. Customize team branding (~5 min)

Edit [`team_config.py`](team_config.py):

| Field | Example |
|-------|---------|
| `TEAM_NAME` | `"Team Phoenix"` |
| `TEAM_MEMBERS` | `"Alice, Bob, Carol"` |
| `GITHUB_REPO` | Your public fork URL |
| `OTHER_RESOURCE` | Slide deck, report, Kaggle notebook, … |
| `SUBTITLE` | One-line project description |
| `LOGO` / `FAVICON` | Paths under [`assets/`](assets/) |
| `STREAMLIT_APP_URL` | Set after Cloud deploy |
| `MODEL_PROFILE` | Pipeline description + footprint for benchmark UI (see [Benchmark](#benchmark-lightweight-model)) |

Optional: replace PNG files in [`assets/`](assets/).

Verify locally:

```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501`.

### 3. Replace the solution (main work)

All inference code lives in [`solution/`](solution/).

**Minimum change** — keep [`solution/pipeline.py`](solution/pipeline.py) but improve:

- [`brand_rules.py`](solution/brand_rules.py) — add regex / brand dictionary entries
- [`product_model.py`](solution/product_model.py) — train a better product classifier

**Full replacement** — swap OCR, NER, LLM, or end-to-end model. **Keep this signature:**

```python
# solution/pipeline.py
def predict_from_image(img: PIL.Image.Image, min_conf: float = 0.35) -> dict[str, str]:
    return {
        "ocr_text": "...",
        "brand_name": "...",
        "product_name": "...",
    }
```

[`streamlit_app.py`](streamlit_app.py) and [`scripts/run_submission.py`](scripts/run_submission.py) import only this API.

**Notebook** — replace [`solution/baseline_notebook.ipynb`](solution/baseline_notebook.ipynb) with your experiments. It is not wired into Streamlit automatically; port logic into `solution/pipeline.py`.

See also [`solution/README.md`](solution/README.md).

### 4. Install full private images (recommended)

Repo ships **6 sample JPEGs** in `data/private_test/images_sample/`.

For all **1,202** images (from BTC bundle or hackathon monorepo):

```bash
python scripts/setup_private_images.py /path/to/private_test
# e.g.
python scripts/setup_private_images.py ../private_test
```

See [`data/private_test/README.md`](data/private_test/README.md).

> **Benchmark:** Deploy smoke and CLI benchmark need at least one `priv_*.jpg` under `images_sample/` (bundled) or `images/` (after setup). See [Benchmark prerequisites](#prerequisites--what-must-be-present-for-benchmark-to-run).

### 5. About tab (presentation)

Edit `_render_about_tab()` in [`streamlit_app.py`](streamlit_app.py) with your pipeline description, results, and links for judges / reviewers.

### Troubleshooting

| Issue | Fix |
|-------|-----|
| `private_test not found` | Ensure `data/private_test/private_test.csv` + `images_sample/` exist |
| EasyOCR slow first run | Downloads ~200 MB weights once |
| Streamlit upload shows wrong OCR | Confirm `predict_from_image` returns updated dict keys |
| Empty submission rows | Run `setup_private_images.py` or use `--limit` on sample IDs only |

---

## Submission

### CSV columns

| Column | Description |
|--------|-------------|
| `image_id` | `priv_h_*` or `priv_d_*` |
| `ocr_text` | Raw OCR output |
| `brand_name` | Brand entity |
| `product_name` | Product entity |

### Empty values

Use a **single space** `" "` in CSV for empty fields (Kaggle convention).

### Generate submission

```bash
python scripts/run_submission.py --limit 6               # smoke test
python scripts/run_submission.py                         # all images on disk
python scripts/run_submission.py -o outputs/my.csv       # custom path
```

Default output: [`outputs/submission_private.csv`](outputs/submission_private.csv)

Upload that file to Kaggle.

### Scoring formula

```
0.4 × F1_brand + 0.35 × (1 − CER) + 0.25 × F1_product
```

Implementation: [`private_test/metric.py`](private_test/metric.py) (do not edit).

### Local evaluation (optional, BTC only)

1. Copy `solution_private.csv` to `data/private_test/` (gitignored).
2. Score:

```python
import pandas as pd
import sys
sys.path.insert(0, "private_test")
from metric import score

sol = pd.read_csv("data/private_test/solution_private.csv", keep_default_na=False)
sub = pd.read_csv("outputs/submission_private.csv", keep_default_na=False)
print(score(sol, sub, "image_id"))
```

Never commit or publish ground-truth files.

---

## Deploy Streamlit Cloud

Deploy the **Live test + About** demo to [Streamlit Community Cloud](https://share.streamlit.io) (free, GitHub-connected).

> **Note:** Deployment is done on the Streamlit website (there is no `streamlit deploy` CLI). Each team: fork → edit `team_config.py` → push to GitHub → create an app on Cloud.

### Pre-deploy checklist

| Item | File / action |
|------|----------------|
| Team name, links, logo | [`team_config.py`](team_config.py) |
| `MODEL_PROFILE` filled in | [`team_config.py`](team_config.py) |
| Solution runs locally | `streamlit run streamlit_app.py` |
| Sample images for deploy smoke | `data/private_test/images_sample/` (6 JPGs — bundled in repo) |
| Python dependencies | [`requirements.txt`](requirements.txt) |
| System libs (OpenCV/EasyOCR) | [`packages.txt`](packages.txt) |
| Python version on Cloud | [`.python-version`](.python-version) → `3.11` |
| **Public** GitHub repo | Settings → Change visibility |
| Do **not** replace benchmark layer | Keep [`shared/benchmark.py`](shared/benchmark.py) + [`streamlit_app.py`](streamlit_app.py) imports |
| No committed secrets | `.streamlit/secrets.toml` is gitignored |

### 1. Push code to GitHub

```bash
git add .
git commit -m "Team ABC: ready for Streamlit Cloud"
git push origin main
```

### 2. Create a Streamlit Cloud account

1. Open [share.streamlit.io](https://share.streamlit.io)
2. **Sign in with GitHub**
3. **Authorize** Streamlit to access your repositories

### 3. Create a new app

1. Click **Create app**
2. Fill in the form:

| Field | Value |
|-------|-------|
| **Repository** | `YOUR_USER/ura-hackathon-team-abc` |
| **Branch** | `main` |
| **Main file path** | `streamlit_app.py` |
| **App URL** (optional) | `ura-team-abc` → `https://ura-team-abc.streamlit.app` |

3. Click **Deploy**

Streamlit installs `requirements.txt`, apt packages from `packages.txt`, and runs `streamlit run streamlit_app.py`.

**First deploy:** allow 5–15 minutes (PyTorch + EasyOCR weights). Watch the build log in the app dashboard.

### 4. Verify after deploy

1. Open `https://<app-name>.streamlit.app`
2. Confirm header: logo, title, subtitle, team links from `team_config.py`
3. **Live test** tab → expand **Model footprint** → deploy smoke latency should appear (needs `images_sample/` in repo)
4. **Live test** → upload JPG/PNG → **Chạy OCR** → check Total / OCR / Extract ms
5. **About** tab → team content

If the build fails → **Manage app → Logs**.

### 5. Redeploy on push

Each `git push` to the linked branch (usually `main`) triggers an automatic rebuild. Track progress under **Manage app → Activity**.

### 6. Common issues

**Build fail — memory / timeout**

Baseline uses **PyTorch + EasyOCR** (~1 GB+ RAM at load). Free Community Cloud has RAM limits.

- Use **Live test** (single image upload) — avoid large batches on Cloud
- Slim down `requirements.txt` if you switch to a lighter OCR stack
- Consider [Streamlit Teams](https://streamlit.io/cloud) for more RAM

**Apt build fails (`libffi7`, `held broken packages`)**

Remove `libglib2.0-0` from [`packages.txt`](packages.txt). Streamlit Cloud’s Debian image cannot install it. Keep only `libgl1`, commit, push, and reboot the app.

**`libGL.so` / OpenCV import error**

Ensure [`packages.txt`](packages.txt) contains **only**:

```text
libgl1
```

Do **not** add `libglib2.0-0` — it breaks on Streamlit Cloud (missing `libffi7` on the base image). Use `opencv-python-headless` in `requirements.txt` (already included).

**App crashes on OCR**

- First EasyOCR run downloads ~200 MB — wait 1–2 minutes
- Check **Logs** when OCR is triggered
- Test locally first: `streamlit run streamlit_app.py`

**Stale team name / links on Cloud**

Edit [`team_config.py`](team_config.py) → push → wait for rebuild (or **Reboot app**).

**Private images (1,202 JPEGs)**

Cloud does **not** bundle `data/private_test/images/` (gitignored, ~100 MB). Use **Live test → Upload** on Cloud for ad-hoc tests. Run batch submission **locally**:

```bash
python scripts/run_submission.py
```

**Deploy smoke benchmark** still works on Cloud using the **6 bundled** files in `data/private_test/images_sample/` — keep that folder committed. See [Benchmark](#benchmark-lightweight-model).

### 7. Secrets (optional)

If your solution needs API keys (OpenAI, Gemini, …):

**Local:** copy [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example) → `.streamlit/secrets.toml`

**Cloud:** **Manage app → Settings → Secrets**

```toml
OPENAI_API_KEY = "sk-..."
```

```python
import streamlit as st
api_key = st.secrets.get("OPENAI_API_KEY", "")
```

Never commit secrets files.

### 8. Record your app URL

After deploy, set in [`team_config.py`](team_config.py):

```python
STREAMLIT_APP_URL = "https://ura-team-abc.streamlit.app"
```

**One-line summary:**

```text
Fork → team_config.py → push GitHub (public) → share.streamlit.io → Create app → streamlit_app.py → Deploy
```

External reference: [Streamlit Cloud docs](https://docs.streamlit.io/streamlit-community-cloud)

---

## Solution API (contract)

```python
from PIL import Image
from shared.benchmark import run_predict_with_metrics, get_model_profile

result = run_predict_with_metrics(Image.open("path/to.jpg"))
# timing_ms always present (measured by template layer)

profile = get_model_profile()  # reads team_config.MODEL_PROFILE
```

Teams implement **`solution.pipeline.predict_from_image()`** only. Streamlit and CLI benchmark call **`shared/benchmark.py`** — do not replace that module.

## Benchmark (lightweight model)

Teams demonstrate a **lightweight** pipeline with two signals:

| Signal | Where | What to report |
|--------|-------|----------------|
| **Latency** | [`shared/benchmark.py`](shared/benchmark.py) → `timing_ms` / deploy smoke | ms per image (avg / p50 / p95) |
| **Footprint** | [`team_config.py`](team_config.py) → `MODEL_PROFILE` | Pipeline description, product-head MB, CPU/GPU |

Benchmark logic lives in the **template layer** — teams replace [`solution/`](solution/) only. Do **not** delete or overwrite [`shared/benchmark.py`](shared/benchmark.py); Streamlit and CLI always import from there.

### What runs automatically

| When | What | Where in UI / code |
|------|------|---------------------|
| Streamlit **first load** (deploy / reboot) | Deploy smoke benchmark — **1** sample image, cached | Live test → **Model footprint** expander |
| User clicks **Chạy OCR** | Per-image timing via `run_predict_with_metrics()` | Live test → Total / OCR / Extract metrics |
| Any time | Footprint text from `MODEL_PROFILE` | Same expander + About tab |

On Streamlit Cloud, smoke benchmark runs on **first app load** after deploy (spinner: *"Running deploy smoke benchmark (1 image)..."*). Result is cached with `@st.cache_resource` until reboot.

### Prerequisites — what must be present for benchmark to run

**1. Template files (keep as-is)**

| File | Role |
|------|------|
| [`shared/benchmark.py`](shared/benchmark.py) | Wraps `predict_from_image`, measures `timing_ms`, runs smoke/full benchmark |
| [`streamlit_app.py`](streamlit_app.py) | Imports `get_deploy_smoke_benchmark`, `run_predict_with_metrics`, `get_model_profile` |
| [`scripts/benchmark_solution.py`](scripts/benchmark_solution.py) | Local CLI report (6 images by default) |
| [`shared/data_utils.py`](shared/data_utils.py) | Resolves `data/private_test/` and image folder |

**2. Team contract (you customize)**

| Requirement | Details |
|-------------|---------|
| [`solution/pipeline.py`](solution/pipeline.py) | Must expose `predict_from_image(img, min_conf=...)` and return `ocr_text`, `brand_name`, `product_name` |
| Optional `timing_ms` in return dict | If omitted, [`shared/benchmark.py`](shared/benchmark.py) still measures wall-clock time |
| [`team_config.MODEL_PROFILE`](team_config.py) | Update when you swap OCR / models (pipeline string, device, notes) |

**3. Image data (required for smoke + CLI; not for upload-only Live test)**

Benchmark reads JPGs from private-test data via [`shared/data_utils.find_private_root()`](shared/data_utils.py):

```text
data/private_test/
├── private_test.csv          # catalog (recommended)
└── images_sample/            # bundled — 6 sample JPGs (enough for deploy smoke)
    └── priv_*.jpg
# or, after setup script:
└── images/                   # full 1,202 JPGs (for CLI --limit N)
    └── priv_*.jpg
```

Alternative root: set env var `SMCE_PRIVATE_TEST_DIR=/path/to/private_test`.

| Scenario | Minimum data | Benchmark behavior |
|----------|--------------|-------------------|
| **Streamlit Cloud deploy** | Keep repo’s `data/private_test/images_sample/` (do not gitignore) | Smoke runs on first load |
| **Local Streamlit** | Same sample folder, or run `setup_private_images.py` | Smoke + Live upload both work |
| **CLI full report** | At least `--limit` JPGs under `images/` or `images_sample/` | `python scripts/benchmark_solution.py --limit 6` |

If no JPGs are found, the app still works (**Live test → Upload**), but the expander shows *"Deploy smoke benchmark skipped: …"* and CLI exits with `private_test not found` / `No JPG files`.

**4. Runtime dependencies**

- Python **3.11+**, [`requirements.txt`](requirements.txt) installed
- Streamlit Cloud: [`packages.txt`](packages.txt) → `libgl1` only (see [Deploy](#deploy-streamlit-cloud))
- First OCR run downloads EasyOCR weights (~200 MB) — allow warmup time

### CLI benchmark (local)

```bash
python scripts/benchmark_solution.py --limit 6
python scripts/benchmark_solution.py --limit 6 --warmup 1   # default warmup
```

Prints per-image `timing_ms`, then JSON summary (avg / p50 / p95 + `model_profile`). Use this before/after model changes; copy numbers into your About tab if needed.

Update [`team_config.MODEL_PROFILE`](team_config.py) whenever you change the stack, for example:

```python
MODEL_PROFILE = {
    "pipeline": "PaddleOCR + custom NER",
    "runtime_device": "CPU",
    "product_head_mb": 2.5,
    "ocr_backend_note": "Paddle weights cached on first run",
    "lightweight_notes": "Replaced EasyOCR for faster Cloud cold start.",
}
```

### Troubleshooting benchmark

| Issue | Fix |
|-------|-----|
| Smoke skipped on Cloud | Ensure `data/private_test/images_sample/` with `priv_*.jpg` is **committed** (not gitignored) |
| `private_test not found` (CLI) | Run from repo root; check `data/private_test/private_test.csv` + sample/full images |
| No timing on Live test | Do not bypass `run_predict_with_metrics` in a forked `streamlit_app.py` — keep template imports |
| Smoke very slow first time | Expected — OCR weights download + cold start; cached until app reboot |
| Wrong footprint in UI | Edit `MODEL_PROFILE` in `team_config.py`, not `shared/benchmark.py` |

### Tips for a lighter stack

- Prefer `opencv-python-headless` + smaller OCR (or API) over full PyTorch stacks when possible
- Keep trainable weights in-repo small (sklearn, regex, quantized models)
- Report **warmup** separately — first run loads OCR weights (~200 MB download)

---

## Current status

| Component | Status |
|-----------|--------|
| Team template structure | Ready |
| Streamlit demo (Live test + About) | Ready |
| Baseline solution in `solution/` | Reference implementation |
| Sample private images (6) | Included |
| Full 1,202 images | Install via script |

CI workflow is not configured yet.

---

## License

MIT — see [LICENSE](LICENSE).
