"""
SteelVision — Streamlit frontend

Place at: src/ui/app.py
Run from project root:
    streamlit run src/ui/app.py
"""

import base64
import html
import mimetypes
import sys
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
import streamlit as st
from PIL import Image, ImageDraw, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inpainting.common import (
    attention_center,
    class_flip as inpainting_class_flip,
    confidence_change as inpainting_confidence_change,
    entropy_change as inpainting_entropy_change,
    generate_gradcam,
    get_device as get_inpainting_device,
    load_classifier as load_inpainting_classifier,
    predict_image as predict_inpainting_image,
)
from src.inpainting.inpainters import (
    load_unet_inpainter,
    run_inpainter,
)
from src.inpainting.masks import make_mask, mask_coverage
from src.inpainting.metrics import evaluate_reconstruction


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

API_URL = "http://127.0.0.1:8000"

HEALTH_TIMEOUT = 3
DOMAIN_TIMEOUT = 30
PREDICT_TIMEOUT = 30
EXPLAIN_TIMEOUT = 60
LOCALIZE_TIMEOUT = 60

CLASS_DISPLAY_NAMES = {
    "crazing": "Crazing",
    "inclusion": "Inclusion",
    "patches": "Patches",
    "pitted_surface": "Pitted Surface",
    "rolled_in_scale": "Rolled-in Scale",
    "scratches": "Scratches",
}

ENTROPY_LOW = 0.5
ENTROPY_HIGH = 1.0

HERO_FEATURES = [
    "Domain Validation",
    "Classification",
    "Uncertainty Estimation",
    "Grad-CAM Explainability",
    "Defect Localization",
    "Coordinate Mapping",
    "Inpainting Reliability",
]

MODE_STANDARD = "Standard Analysis"
MODE_EXPERIMENT = "Resolution Experiment"
MODE_INPAINTING = "Inpainting Reliability"

MODE_DESCRIPTIONS = {
    MODE_STANDARD: (
        "Validate and analyze the uploaded image using classification, "
        "explainability, and defect localization."
    ),
    MODE_EXPERIMENT: (
        "Evaluate how changing image resolution affects defect localization."
    ),
    MODE_INPAINTING: (
        "Remove controlled image regions, reconstruct them, and measure whether "
        "the classifier decision remains stable."
    ),
}

ENTROPY_HELP = (
    "Entropy is used as a model uncertainty indicator. Lower values generally "
    "indicate more concentrated model confidence, while higher values indicate "
    "greater uncertainty.\n\n"
    "UI thresholds: < 0.5 low, 0.5–1.0 moderate, > 1.0 elevated. These are "
    "interface-level interpretation thresholds and are not universal scientific "
    "thresholds."
)

# --- YOLO ground-truth evaluation configuration ---

YOLO_VAL_IMAGES_DIR = "data/raw/NEU-CLS-YOLO/valid/images"
YOLO_VAL_LABELS_DIR = "data/raw/NEU-CLS-YOLO/valid/labels"
YOLO_WEIGHTS_PATH = "runs/detect/steel_defect_yolo/weights/best.pt"

YOLO_CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches",
]

YOLO_DEFAULT_CONF = 0.25
YOLO_DEFAULT_IOU = 0.50

# Minimum overlap required before an unmatched GT/prediction pair is
# considered a *diagnostic* pair (used only to explain failures; it never
# changes TP / FP / FN counts).
DIAGNOSTIC_MIN_IOU = 0.10

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

REASON_POOR_LOC = "Poor Localization"
REASON_WRONG_CLASS = "Wrong Class"
REASON_WRONG_CLASS_POOR_LOC = "Wrong Class + Poor Localization"

GT_COLOR = "cyan"
PRED_COLOR = "red"

# --- Inpainting reliability configuration ---

INPAINTING_UNET_WEIGHTS = "models/inpainting_unet.pth"
INPAINTING_IMAGE_SIZE = 224
INPAINTING_MASK_TYPES = ["block", "line", "blob"]
INPAINTING_MASK_RATIOS = [0.10, 0.20, 0.40]
INPAINTING_PLACEMENTS = ["random", "high_attention", "low_attention"]
INPAINTING_METHODS = ["mean_fill", "telea", "navier_stokes", "unet"]

INPAINTING_METHOD_LABELS = {
    "mean_fill": "Mean Fill",
    "telea": "Telea",
    "navier_stokes": "Navier–Stokes",
    "unet": "U-Net",
}

INPAINTING_PLACEMENT_LABELS = {
    "random": "Random",
    "high_attention": "High Attention",
    "low_attention": "Low Attention",
}

INPAINTING_FINAL_PLACEMENT_RESULTS = [
    {"Placement": "High Attention", "Flip Rate": "22.61%", "Accuracy": "77.32%"},
    {"Placement": "Random", "Flip Rate": "13.93%", "Accuracy": "85.85%"},
    {"Placement": "Low Attention", "Flip Rate": "9.72%", "Accuracy": "90.13%"},
]

INPAINTING_FINAL_METHOD_RESULTS = [
    {"Method": "Mean Fill", "Flip Rate": "11.13%", "Accuracy": "88.95%"},
    {"Method": "U-Net", "Flip Rate": "15.98%", "Accuracy": "83.81%"},
    {"Method": "Navier–Stokes", "Flip Rate": "17.23%", "Accuracy": "82.59%"},
    {"Method": "Telea", "Flip Rate": "17.36%", "Accuracy": "82.40%"},
]

st.set_page_config(
    page_title="SteelVision",
    page_icon="🔩",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
:root {
  --bg: #0a111f;
  --panel: #111a2d;
  --panel-2: #16213a;
  --border: rgba(148, 170, 205, 0.16);
  --text: #e6ecf6;
  --muted: #8fa0ba;
  --accent: #38bdf8;
  --accent-2: #6366f1;
  --good: #34d399;
  --warn: #fbbf24;
  --danger: #fb923c;
}

.stApp {
  background:
    radial-gradient(
      1100px 500px at 85% -10%,
      rgba(56,189,248,0.10),
      transparent 60%
    ),
    radial-gradient(
      900px 500px at -10% 0%,
      rgba(99,102,241,0.10),
      transparent 55%
    ),
    var(--bg);
  color: var(--text);
}

header[data-testid="stHeader"] {
  background: transparent;
}

.block-container {
  padding-top: 2rem;
  padding-bottom: 3rem;
  max-width: 1180px;
}

section[data-testid="stSidebar"] {
  background: linear-gradient(
    180deg,
    #0c1424 0%,
    #0a101d 100%
  );
  border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] * {
  color: var(--text);
}

.hero {
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 2.1rem 2.3rem;
  margin-bottom: 1.4rem;
  background:
    linear-gradient(
      135deg,
      rgba(56,189,248,0.10),
      rgba(99,102,241,0.08) 55%,
      rgba(15,23,42,0.4)
    ),
    repeating-linear-gradient(
      115deg,
      rgba(255,255,255,0.025) 0 2px,
      transparent 2px 14px
    ),
    #0f1a30;
  box-shadow: 0 18px 50px rgba(0,0,0,0.35);
}

.hero-badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 5px 12px;
  border-radius: 999px;
  font-size: 0.74rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  font-weight: 600;
  color: var(--accent);
  background: rgba(56,189,248,0.10);
  border: 1px solid rgba(56,189,248,0.30);
}

.hero-badge .dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--accent);
  box-shadow: 0 0 10px var(--accent);
}

.hero h1 {
  margin: 0.85rem 0 0.15rem 0;
  font-size: 2.9rem;
  font-weight: 800;
  letter-spacing: -0.02em;
  background: linear-gradient(
    90deg,
    #f1f5f9 0%,
    #7dd3fc 60%,
    #a5b4fc 100%
  );
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent !important;
}

.hero .subtitle {
  font-size: 1.15rem;
  font-weight: 500;
  color: #cbd5e1;
}

.hero-feat-label {
  margin-top: 1.2rem;
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--muted);
  font-weight: 700;
}

.chips {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 0.6rem;
}

.chip {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 15px;
  border-radius: 10px;
  font-size: 0.88rem;
  font-weight: 600;
  color: #e2e8f0;
  background: rgba(56,189,248,0.07);
  border: 1px solid rgba(56,189,248,0.22);
}

.chip::before {
  content: "";
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--accent);
  box-shadow: 0 0 8px var(--accent);
}

.section-title {
  display: flex;
  align-items: center;
  gap: 10px;
  margin: 1.9rem 0 0.7rem 0;
  font-size: 1.15rem;
  font-weight: 700;
}

.section-title .bar {
  width: 4px;
  height: 20px;
  border-radius: 3px;
  background: linear-gradient(
    180deg,
    var(--accent),
    var(--accent-2)
  );
}

.section-sub {
  color: var(--muted);
  font-size: 0.88rem;
  margin: -0.35rem 0 0.9rem 0;
}

[data-testid="stRadio"] label p {
  font-size: 1rem;
  line-height: 1.55;
  color: #e2e8f0;
}

[data-testid="stRadio"] [role="radiogroup"] {
  gap: 0.9rem;
}

[data-testid="stMetric"] {
  background: linear-gradient(
    180deg,
    var(--panel-2),
    var(--panel)
  );
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1rem 1.15rem;
  box-shadow: 0 8px 28px rgba(0,0,0,0.28);
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] div,
[data-testid="stMetricLabel"] label {
  color: var(--muted) !important;
  font-weight: 600;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div,
[data-testid="stMetricValue"] p {
  color: var(--text) !important;
  font-weight: 800;
  letter-spacing: -0.02em;
}

[data-testid="stMetricDelta"] {
  color: var(--muted) !important;
}

.card {
  background: linear-gradient(
    180deg,
    var(--panel-2),
    var(--panel)
  );
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1.15rem 1.25rem;
  box-shadow: 0 8px 28px rgba(0,0,0,0.28);
}

.metric-label {
  font-size: 0.73rem;
  text-transform: uppercase;
  letter-spacing: 0.09em;
  color: var(--muted);
  font-weight: 600;
}

.metric-value {
  font-size: 1.8rem;
  font-weight: 800;
  margin-top: 0.3rem;
  letter-spacing: -0.02em;
  color: var(--text);
}

.metric-value.compact {
  font-size: 1.3rem;
  line-height: 1.35;
}

.metric-note {
  font-size: 0.8rem;
  color: var(--muted);
  margin-top: 0.25rem;
}

.accent-text {
  color: var(--accent);
}

.pbar {
  height: 9px;
  background: rgba(148,170,205,0.14);
  border-radius: 999px;
  overflow: hidden;
}

.pbar > div {
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(
    90deg,
    var(--accent-2),
    var(--accent)
  );
}

.pbar.thin {
  height: 7px;
}

.callout {
  border-radius: 14px;
  padding: 1rem 1.2rem;
  border: 1px solid var(--border);
  border-left-width: 4px;
  background: var(--panel);
}

.callout.low {
  border-left-color: var(--good);
}

.callout.mid {
  border-left-color: var(--warn);
}

.callout.high {
  border-left-color: var(--danger);
}

.callout .title {
  font-weight: 700;
  margin-bottom: 0.2rem;
}

.callout .body {
  color: #b4c0d4;
  font-size: 0.92rem;
  line-height: 1.5;
}

.callout .fine {
  color: var(--muted);
  font-size: 0.76rem;
  margin-top: 0.5rem;
}

.img-head {
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.09em;
  color: var(--muted);
  font-weight: 600;
  margin-bottom: 0.5rem;
}

[data-testid="stImage"] img {
  border-radius: 12px;
  border: 1px solid var(--border);
}

.note {
  color: var(--muted);
  font-size: 0.88rem;
  line-height: 1.5;
  margin-top: 0.6rem;
}

.disclaimer {
  margin-top: 1rem;
  padding: 0.8rem 1rem;
  border-radius: 12px;
  background: rgba(251,191,36,0.07);
  border: 1px solid rgba(251,191,36,0.25);
  color: #e5d3a0;
  font-size: 0.86rem;
  line-height: 1.5;
}

.rank-row {
  display: grid;
  grid-template-columns: 44px 1fr 90px;
  gap: 14px;
  align-items: center;
  padding: 0.82rem 0;
  border-bottom: 1px solid var(--border);
}

.rank-row:last-child {
  border-bottom: none;
}

.rank-badge {
  width: 34px;
  height: 34px;
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 800;
  background: rgba(148,170,205,0.10);
  color: var(--muted);
}

.rank-badge.first {
  background: linear-gradient(
    135deg,
    var(--accent-2),
    var(--accent)
  );
  color: #06101d;
}

.rank-name {
  font-weight: 600;
  margin-bottom: 6px;
  color: var(--text);
}

.rank-prob {
  text-align: right;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
  color: var(--text);
}

.side-title {
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--muted);
  font-weight: 700;
  margin: 1.2rem 0 0.5rem 0;
}

.kv {
  display: flex;
  justify-content: space-between;
  padding: 6px 0;
  font-size: 0.88rem;
  border-bottom: 1px dashed var(--border);
}

.kv span:first-child {
  color: var(--muted);
}

.status-pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 6px 12px;
  border-radius: 999px;
  font-size: 0.85rem;
  font-weight: 600;
}

.status-pill.on {
  background: rgba(52,211,153,0.10);
  color: var(--good);
  border: 1px solid rgba(52,211,153,0.3);
}

.status-pill.off {
  background: rgba(248,113,113,0.10);
  color: #f87171;
  border: 1px solid rgba(248,113,113,0.3);
}

.defect-item {
  padding: 5px 0;
  font-size: 0.88rem;
  color: #cbd5e1;
}

.defect-item::before {
  content: "▸ ";
  color: var(--accent);
}

[data-testid="stFileUploader"] section {
  background: rgba(17,26,45,0.7);
  border: 1.5px dashed rgba(56,189,248,0.35);
  border-radius: 14px;
  padding: 1.2rem;
}

[data-testid="stFileUploader"] section:hover {
  border-color: var(--accent);
}

.stButton > button,
.stButton > button[kind="primary"] {
  width: 100%;
  border: none;
  border-radius: 12px;
  padding: 0.75rem 1.2rem;
  font-weight: 700;
  font-size: 1rem;
  color: #05101d;
  background: linear-gradient(
    90deg,
    #38bdf8,
    #818cf8
  );
  box-shadow: 0 8px 24px rgba(56,189,248,0.25);
}

.stButton > button:hover {
  transform: translateY(-1px);
  color: #05101d;
  box-shadow: 0 12px 30px rgba(56,189,248,0.35);
}

.stButton > button:disabled {
  opacity: 0.45;
  box-shadow: none;
}

[data-testid="stExpander"] {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 14px;
}

.footer {
  margin-top: 3rem;
  padding-top: 1.2rem;
  border-top: 1px solid var(--border);
  text-align: center;
  color: var(--muted);
  font-size: 0.82rem;
  line-height: 1.7;
}

.footer b {
  color: #cbd5e1;
}

#MainMenu,
footer {
  visibility: hidden;
}
</style>
"""


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def render_html(
    markup: str,
) -> None:
    cleaned = "\n".join(
        line.strip()
        for line in markup.strip().splitlines()
        if line.strip()
    )

    st.markdown(
        cleaned,
        unsafe_allow_html=True,
    )


def section_title(
    text: str,
    subtitle: str | None = None,
) -> None:
    render_html(
        '<div class="section-title">'
        '<span class="bar"></span>'
        f'{html.escape(text)}'
        '</div>'
    )

    if subtitle:
        render_html(
            f'<div class="section-sub">'
            f'{html.escape(subtitle)}'
            f'</div>'
        )


def format_class_name(
    raw: str,
) -> str:
    key = (
        str(raw)
        .strip()
        .lower()
    )

    return CLASS_DISPLAY_NAMES.get(
        key,
        key.replace(
            "_",
            " ",
        ).title(),
    )


def pct(
    value: float,
    digits: int = 2,
) -> str:
    return (
        f"{value * 100:.{digits}f}%"
    )


def clamp_pct_width(
    value: float,
) -> float:
    return max(
        0.0,
        min(
            100.0,
            value * 100,
        ),
    )


def guess_mime(
    filename: str,
    declared: str | None,
) -> str:
    if (
        declared
        and declared.startswith(
            "image/"
        )
    ):
        return declared

    guessed, _ = (
        mimetypes.guess_type(
            filename
        )
    )

    return (
        guessed
        or "image/jpeg"
    )


def validate_image(
    image_bytes: bytes,
) -> Image.Image:
    try:
        probe = Image.open(
            BytesIO(
                image_bytes
            )
        )

        probe.verify()

        image = Image.open(
            BytesIO(
                image_bytes
            )
        )

        image.load()

        return image.convert(
            "RGB"
        )

    except (
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        ValueError,
    ) as exc:
        raise ValueError(
            "The uploaded file could not be opened as a valid image."
        ) from exc


def decode_base64_image(
    value: str,
) -> Image.Image:
    try:
        return Image.open(
            BytesIO(
                base64.b64decode(
                    value
                )
            )
        ).convert(
            "RGB"
        )

    except Exception as exc:
        raise RuntimeError(
            "The backend returned an invalid encoded image."
        ) from exc


@st.cache_data(
    ttl=3,
    show_spinner=False,
)
def check_backend() -> tuple[
    bool,
    str,
]:
    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=HEALTH_TIMEOUT,
        )

        if response.status_code != 200:
            return (
                False,
                (
                    "Health check returned "
                    f"HTTP {response.status_code}."
                ),
            )

        payload = response.json()

        if (
            str(
                payload.get(
                    "status",
                    "",
                )
            ).lower()
            == "ok"
        ):
            return (
                True,
                "Backend is online.",
            )

        return (
            False,
            (
                "Backend responded but did not "
                "report status 'ok'."
            ),
        )

    except requests.exceptions.ConnectionError:
        return (
            False,
            (
                "Cannot connect to the "
                "FastAPI backend."
            ),
        )

    except requests.exceptions.Timeout:
        return (
            False,
            "Health check timed out.",
        )

    except requests.exceptions.RequestException as exc:
        return (
            False,
            f"Health check failed: {exc}",
        )


def _post_file(
    endpoint: str,
    image_bytes: bytes,
    filename: str,
    mime: str,
    timeout: int,
    data=None,
):
    try:
        response = requests.post(
            f"{API_URL}/{endpoint}",
            files={
                "file": (
                    filename,
                    image_bytes,
                    mime,
                )
            },
            data=data,
            timeout=timeout,
        )

    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"Could not connect to /{endpoint}. "
            "Make sure the FastAPI backend is running."
        ) from exc

    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            f"The /{endpoint} request timed out "
            f"after {timeout} seconds."
        ) from exc

    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            f"The /{endpoint} request failed: "
            f"{exc}"
        ) from exc

    if response.status_code != 200:
        detail = (
            response.text[
                :400
            ].strip()
        )

        raise RuntimeError(
            f"/{endpoint} returned "
            f"HTTP {response.status_code}. "
            f"{detail}"
        )

    return response


# -----------------------------------------------------------------------------
# API calls
# -----------------------------------------------------------------------------

def call_validate_domain(
    image_bytes: bytes,
    filename: str,
    mime: str,
) -> dict:
    response = _post_file(
        "validate-domain",
        image_bytes,
        filename,
        mime,
        DOMAIN_TIMEOUT,
    )

    try:
        data = response.json()

        return {
            "filename":
                str(
                    data.get(
                        "filename",
                        filename,
                    )
                ),

            "supported":
                bool(
                    data[
                        "supported"
                    ]
                ),

            "supported_probability":
                float(
                    data[
                        "supported_probability"
                    ]
                ),

            "unsupported_probability":
                float(
                    data[
                        "unsupported_probability"
                    ]
                ),

            "threshold":
                float(
                    data[
                        "threshold"
                    ]
                ),
        }

    except (
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
    ) as exc:
        raise RuntimeError(
            "/validate-domain returned a response "
            "in an unexpected format."
        ) from exc


def call_predict(
    image_bytes: bytes,
    filename: str,
    mime: str,
) -> dict:
    response = _post_file(
        "predict",
        image_bytes,
        filename,
        mime,
        PREDICT_TIMEOUT,
    )

    try:
        data = response.json()

        return {
            "filename":
                str(
                    data.get(
                        "filename",
                        filename,
                    )
                ),

            "predicted_class":
                str(
                    data[
                        "predicted_class"
                    ]
                ),

            "confidence":
                float(
                    data[
                        "confidence"
                    ]
                ),

            "entropy":
                float(
                    data[
                        "entropy"
                    ]
                ),

            "top_3": [
                {
                    "class_name":
                        str(
                            item[
                                "class_name"
                            ]
                        ),

                    "probability":
                        float(
                            item[
                                "probability"
                            ]
                        ),
                }
                for item
                in data[
                    "top_3"
                ]
            ],
        }

    except (
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
    ) as exc:
        raise RuntimeError(
            "/predict returned a response "
            "in an unexpected format."
        ) from exc


def call_explain(
    image_bytes: bytes,
    filename: str,
    mime: str,
) -> bytes:
    response = _post_file(
        "explain",
        image_bytes,
        filename,
        mime,
        EXPLAIN_TIMEOUT,
    )

    try:
        Image.open(
            BytesIO(
                response.content
            )
        ).verify()

    except (
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        ValueError,
    ) as exc:
        raise RuntimeError(
            "/explain did not return "
            "a valid image."
        ) from exc

    return response.content


def call_localize(
    image_bytes: bytes,
    filename: str,
    mime: str,
    resolution: str,
) -> dict:
    response = _post_file(
        "localize",
        image_bytes,
        filename,
        mime,
        LOCALIZE_TIMEOUT,
        data={
            "resolution":
                resolution
        },
    )

    try:
        data = response.json()

        return {
            "filename":
                str(
                    data.get(
                        "filename",
                        filename,
                    )
                ),

            "resolution":
                str(
                    data[
                        "resolution"
                    ]
                ),

            "original_size":
                list(
                    data[
                        "original_size"
                    ]
                ),

            "processed_size":
                list(
                    data[
                        "processed_size"
                    ]
                ),

            "detection_count":
                int(
                    data[
                        "detection_count"
                    ]
                ),

            "detections":
                list(
                    data[
                        "detections"
                    ]
                ),

            "low_res_image":
                decode_base64_image(
                    data[
                        "low_res_image"
                    ]
                ),

            "processed_image":
                decode_base64_image(
                    data[
                        "processed_image"
                    ]
                ),

            "mapped_image":
                decode_base64_image(
                    data[
                        "mapped_image"
                    ]
                ),
        }

    except (
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
    ) as exc:
        raise RuntimeError(
            "/localize returned a response "
            "in an unexpected format."
        ) from exc


# -----------------------------------------------------------------------------
# Analysis helpers
# -----------------------------------------------------------------------------

def uncertainty_level(
    entropy: float,
) -> tuple[
    str,
    str,
    str,
]:
    if entropy < ENTROPY_LOW:
        return (
            "low",
            "Low Uncertainty",
            (
                "Model uncertainty is relatively "
                "low for this prediction."
            ),
        )

    if entropy <= ENTROPY_HIGH:
        return (
            "mid",
            "Moderate Uncertainty",
            (
                "Moderate model uncertainty. "
                "Review the secondary predictions "
                "alongside the primary result."
            ),
        )

    return (
        "high",
        "Elevated Uncertainty",
        (
            "Model uncertainty is elevated. "
            "Interpret this prediction cautiously."
        ),
    )


def highest_detection_confidence(
    result: dict,
) -> float | None:
    detections = result.get(
        "detections",
        [],
    )

    if not detections:
        return None

    try:
        return max(
            float(
                item[
                    "confidence"
                ]
            )
            for item
            in detections
        )

    except (
        KeyError,
        TypeError,
        ValueError,
    ):
        return None


def format_detection_summary(
    result: dict,
) -> str:
    count = int(
        result.get(
            "detection_count",
            0,
        )
    )

    noun = (
        "detection"
        if count == 1
        else "detections"
    )

    top_conf = (
        highest_detection_confidence(
            result
        )
    )

    if (
        count == 0
        or top_conf is None
    ):
        return (
            f"{count} {noun} · "
            "No confidence available"
        )

    return (
        f"{count} {noun} · "
        f"{pct(top_conf)} confidence"
    )


def split_detections_by_prediction(
    detections: list,
    predicted_class: str,
) -> tuple[
    list,
    list,
]:
    matching = []
    secondary = []

    for detection in detections:
        if (
            detection[
                "class_name"
            ]
            == predicted_class
        ):
            matching.append(
                detection
            )

        else:
            secondary.append(
                detection
            )

    matching.sort(
        key=lambda item:
            float(
                item[
                    "confidence"
                ]
            ),
        reverse=True,
    )

    secondary.sort(
        key=lambda item:
            float(
                item[
                    "confidence"
                ]
            ),
        reverse=True,
    )

    return (
        matching,
        secondary,
    )


def draw_detection_subset(
    image: Image.Image,
    detections: list,
    color: str = PRED_COLOR,
) -> Image.Image:
    output = (
        image
        .copy()
        .convert(
            "RGB"
        )
    )

    draw = ImageDraw.Draw(
        output
    )

    line_width = max(
        2,
        int(
            min(
                output.size
            )
            * 0.01
        ),
    )

    for detection in detections:
        (
            x1,
            y1,
            x2,
            y2,
        ) = detection[
            "mapped_box"
        ]

        draw.rectangle(
            [
                x1,
                y1,
                x2,
                y2,
            ],
            outline=color,
            width=line_width,
        )

        label = (
            f"{format_class_name(detection['class_name'])} "
            f"{float(detection['confidence']) * 100:.1f}%"
        )

        draw.text(
            (
                x1 + 3,
                y1 + 3,
            ),
            label,
            fill=color,
        )

    return output


def build_detection_rows(
    detections: list,
) -> list:
    rows = []

    for idx, detection in enumerate(
        detections,
        start=1,
    ):
        rows.append(
            {
                "#":
                    idx,

                "Defect":
                    format_class_name(
                        detection[
                            "class_name"
                        ]
                    ),

                "Confidence":
                    (
                        f'{float(detection["confidence"]) * 100:.2f}%'
                    ),

                "Processed Box":
                    str(
                        detection[
                            "processed_box"
                        ]
                    ),

                "Mapped Box":
                    str(
                        detection[
                            "mapped_box"
                        ]
                    ),
            }
        )

    return rows


# -----------------------------------------------------------------------------
# Header / sidebar
# -----------------------------------------------------------------------------

def render_hero() -> None:
    chips = "".join(
        f'<span class="chip">'
        f'{html.escape(name)}'
        f'</span>'
        for name
        in HERO_FEATURES
    )

    render_html(
        f"""
        <div class="hero">
          <span class="hero-badge">
            <span class="dot"></span>
            Research Prototype
          </span>

          <h1>SteelVision</h1>

          <div class="subtitle">
            AI-Powered Steel Surface Defect Analysis
          </div>

          <div class="hero-feat-label">
            Capabilities
          </div>

          <div class="chips">
            {chips}
          </div>
        </div>
        """
    )


def render_sidebar() -> bool:
    (
        online,
        message,
    ) = check_backend()

    with st.sidebar:
        render_html(
            """
            <div style="font-size:1.25rem;font-weight:800;letter-spacing:-0.01em;">
              SteelVision
            </div>

            <div style="color:#8fa0ba;font-size:0.82rem;margin-top:2px;">
              Steel Surface Defect Analysis
            </div>

            <div class="side-title">
              Backend Status
            </div>
            """
        )

        render_html(
            (
                '<span class="status-pill on">'
                '● Online'
                '</span>'
            )
            if online
            else (
                '<span class="status-pill off">'
                '● Offline'
                '</span>'
            )
        )

        if not online:
            st.caption(
                message
            )

        if st.button(
            "Refresh status",
            key="refresh_status",
        ):
            check_backend.clear()
            st.rerun()

        render_html(
            """
            <div class="side-title">
              System
            </div>

            <div class="kv">
              <span>Domain Gate</span>
              <span>MobileNetV3</span>
            </div>

            <div class="kv">
              <span>Classifier</span>
              <span>ResNet-50</span>
            </div>

            <div class="kv">
              <span>Runtime</span>
              <span>ONNX</span>
            </div>

            <div class="kv">
              <span>Detector</span>
              <span>YOLO</span>
            </div>

            <div class="kv">
              <span>Explainability</span>
              <span>Grad-CAM</span>
            </div>

            <div class="kv">
              <span>Inpainter</span>
              <span>U-Net</span>
            </div>

            <div class="kv">
              <span>Classes</span>
              <span>6</span>
            </div>

            <div class="side-title">
              Supported Defects
            </div>
            """
        )

        render_html(
            "".join(
                (
                    '<div class="defect-item">'
                    f'{html.escape(name)}'
                    '</div>'
                )
                for name
                in CLASS_DISPLAY_NAMES.values()
            )
        )

    return online


# -----------------------------------------------------------------------------
# Shared upload UI
# -----------------------------------------------------------------------------

def render_upload(
    backend_online: bool,
    key: str,
):
    section_title(
        "Upload Steel Surface Image",
        (
            "Accepted formats: JPG, JPEG, PNG. "
            "The input is validated before defect analysis."
        ),
    )

    if not backend_online:
        st.error(
            "The FastAPI backend appears to be offline. "
            "Start it in a separate terminal with "
            "`uvicorn src.api.main:app --reload`, "
            "then press **Refresh status** in the sidebar."
        )

    uploaded = st.file_uploader(
        "Upload Steel Surface Image",
        type=[
            "jpg",
            "jpeg",
            "png",
        ],
        label_visibility="collapsed",
        key=key,
    )

    if uploaded is None:
        render_html(
            '<div class="note">'
            'Select an image to begin. '
            'No analysis is performed until you click the analysis button.'
            '</div>'
        )

        return (
            None,
            None,
        )

    try:
        image = validate_image(
            uploaded.getvalue()
        )

    except ValueError as exc:
        st.error(
            str(
                exc
            )
        )

        return (
            None,
            None,
        )

    left, right = st.columns(
        [
            1.15,
            1,
        ],
        gap="large",
    )

    with left:
        render_html(
            '<div class="img-head">'
            'Image Preview'
            '</div>'
        )

        st.image(
            image,
            width="stretch",
        )

    with right:
        (
            width,
            height,
        ) = image.size

        render_html(
            f"""
            <div class="card">

              <div class="metric-label">
                File
              </div>

              <div style="font-weight:600;margin:4px 0 14px 0;word-break:break-all;">
                {html.escape(uploaded.name)}
              </div>

              <div class="metric-label">
                Uploaded Dimensions
              </div>

              <div style="font-weight:600;margin:4px 0 14px 0;">
                {width} × {height} px
              </div>

              <div class="metric-label">
                Size
              </div>

              <div style="font-weight:600;margin:4px 0 0 0;">
                {uploaded.size / 1024:.1f} KB
              </div>

            </div>
            """
        )

    return (
        uploaded,
        image,
    )


# -----------------------------------------------------------------------------
# Domain validation rendering
# -----------------------------------------------------------------------------

def render_domain_validation(
    result: dict,
) -> None:
    section_title(
        "Input Domain Validation"
    )

    supported_probability = result[
        "supported_probability"
    ]

    threshold = result[
        "threshold"
    ]

    if result[
        "supported"
    ]:
        render_html(
            f"""
            <div class="callout low">

              <div class="title">
                Supported Input
              </div>

              <div class="body">
                The uploaded image passed the SteelVision
                supported-domain validator and downstream
                defect analysis was allowed to run.
              </div>

              <div class="fine">
                Supported-domain probability:
                {pct(supported_probability)}
                · Acceptance threshold:
                {pct(threshold)}
              </div>

            </div>
            """
        )

    else:
        render_html(
            f"""
            <div class="callout high">

              <div class="title">
                Unsupported Input — Analysis Stopped
              </div>

              <div class="body">
                The uploaded image did not pass the
                supported-domain validation threshold.
                SteelVision therefore did not run
                defect classification, Grad-CAM,
                or YOLO localization.
              </div>

              <div class="fine">
                Supported-domain probability:
                {pct(supported_probability)}
                · Required threshold:
                {pct(threshold)}
              </div>

            </div>
            """
        )

        render_html(
            '<div class="disclaimer">'
            '<b>Scope:</b> The domain validator was trained to distinguish '
            'the six-class steel-defect image domain from generic and '
            'industrial unsupported examples. Rejection does not prove '
            'that an image is physically non-steel; it means the input '
            'does not sufficiently resemble the supported analysis domain.'
            '</div>'
        )


# -----------------------------------------------------------------------------
# Standard analysis
# -----------------------------------------------------------------------------

def run_standard_analysis(
    uploaded,
) -> None:
    image_bytes = (
        uploaded.getvalue()
    )

    filename = (
        uploaded.name
    )

    mime = guess_mime(
        filename,
        getattr(
            uploaded,
            "type",
            None,
        ),
    )

    st.session_state.standard_domain_validation = None
    st.session_state.standard_result = None
    st.session_state.standard_gradcam = None
    st.session_state.standard_gradcam_error = None
    st.session_state.standard_localization = None
    st.session_state.standard_original_bytes = image_bytes
    st.session_state.standard_error = None

    with st.spinner(
        "Validating input domain..."
    ):
        try:
            domain_result = (
                call_validate_domain(
                    image_bytes,
                    filename,
                    mime,
                )
            )

        except RuntimeError as exc:
            st.session_state.standard_error = (
                str(
                    exc
                )
            )

            return

    st.session_state.standard_domain_validation = (
        domain_result
    )

    if not domain_result[
        "supported"
    ]:
        return

    with st.spinner(
        "Running classification, "
        "explainability, and localization..."
    ):
        try:
            prediction = call_predict(
                image_bytes,
                filename,
                mime,
            )

            localization = call_localize(
                image_bytes,
                filename,
                mime,
                "original",
            )

        except RuntimeError as exc:
            st.session_state.standard_error = (
                str(
                    exc
                )
            )

            return

        st.session_state.standard_result = (
            prediction
        )

        st.session_state.standard_localization = (
            localization
        )

        try:
            st.session_state.standard_gradcam = (
                call_explain(
                    image_bytes,
                    filename,
                    mime,
                )
            )

        except RuntimeError as exc:
            st.session_state.standard_gradcam_error = (
                str(
                    exc
                )
            )


def render_prediction_summary(
    result: dict,
) -> None:
    section_title(
        "Prediction"
    )

    predicted = format_class_name(
        result[
            "predicted_class"
        ]
    )

    confidence = result[
        "confidence"
    ]

    entropy = result[
        "entropy"
    ]

    c1, c2, c3 = st.columns(
        3,
        gap="medium",
    )

    cards = [
        (
            c1,
            "Defect",
            predicted,
            "Primary Image-Level Classification",
            True,
        ),
        (
            c2,
            "Confidence",
            pct(
                confidence
            ),
            "Classifier Softmax Confidence",
            False,
        ),
        (
            c3,
            "Entropy",
            f"{entropy:.4f}",
            "Model Uncertainty Indicator",
            False,
        ),
    ]

    for (
        col,
        label,
        value,
        note,
        accent,
    ) in cards:
        with col:
            cls = (
                "metric-value accent-text"
                if accent
                else "metric-value"
            )

            render_html(
                f'<div class="card">'
                f'<div class="metric-label">'
                f'{html.escape(label)}'
                f'</div>'
                f'<div class="{cls}">'
                f'{html.escape(str(value))}'
                f'</div>'
                f'<div class="metric-note">'
                f'{html.escape(note)}'
                f'</div>'
                f'</div>'
            )

            if label == "Entropy":
                st.caption(
                    "ⓘ About entropy",
                    help=ENTROPY_HELP,
                )

    render_html(
        f'<div style="margin-top:1rem;">'
        f'<div style="display:flex;justify-content:space-between;'
        f'font-size:0.82rem;color:#8fa0ba;margin-bottom:6px;">'
        f'<span>Classification Confidence</span>'
        f'<span>{pct(confidence)}</span>'
        f'</div>'
        f'<div class="pbar">'
        f'<div style="width:{clamp_pct_width(confidence):.2f}%;">'
        f'</div>'
        f'</div>'
        f'</div>'
    )


def render_uncertainty(
    entropy: float,
) -> None:
    section_title(
        "Uncertainty Interpretation"
    )

    (
        css_class,
        title,
        message,
    ) = uncertainty_level(
        entropy
    )

    render_html(
        f"""
        <div class="callout {css_class}">

          <div class="title">
            {html.escape(title)}
          </div>

          <div class="body">
            {html.escape(message)}
            Entropy is an uncertainty indicator
            and does not guarantee that a prediction is correct.
          </div>

          <div class="fine">
            UI thresholds:
            &lt; {ENTROPY_LOW} low,
            {ENTROPY_LOW}–{ENTROPY_HIGH} moderate,
            &gt; {ENTROPY_HIGH} elevated.
            These are interface-level thresholds,
            not universal scientific thresholds.
          </div>

        </div>
        """
    )


def render_gradcam(
    original_bytes: bytes,
    gradcam_bytes: bytes | None,
    gradcam_error: str | None,
) -> None:
    section_title(
        "Explainability",
        (
            "Grad-CAM highlights image regions "
            "that influenced the ResNet-50 classifier."
        ),
    )

    left, right = st.columns(
        2,
        gap="large",
    )

    with left:
        render_html(
            '<div class="img-head">'
            'Original Image'
            '</div>'
        )

        st.image(
            Image.open(
                BytesIO(
                    original_bytes
                )
            ).convert(
                "RGB"
            ),
            width="stretch",
        )

    with right:
        render_html(
            '<div class="img-head">'
            'Grad-CAM Explanation'
            '</div>'
        )

        if gradcam_bytes:
            st.image(
                Image.open(
                    BytesIO(
                        gradcam_bytes
                    )
                ).convert(
                    "RGB"
                ),
                width="stretch",
            )

        else:
            st.warning(
                gradcam_error
                or (
                    "Grad-CAM is not "
                    "available for this image."
                )
            )

    render_html(
        '<div class="disclaimer">'
        '<b>Interpretation:</b> '
        'highlighted regions show model influence, '
        'not proof of a physical defect mechanism.'
        '</div>'
    )


def render_top3(
    top_3: list,
) -> None:
    section_title(
        "Top Defect Predictions"
    )

    row_parts = []

    for idx, item in enumerate(
        top_3[
            :3
        ],
        start=1,
    ):
        name = html.escape(
            format_class_name(
                item[
                    "class_name"
                ]
            )
        )

        prob = float(
            item[
                "probability"
            ]
        )

        badge_class = (
            "rank-badge first"
            if idx == 1
            else "rank-badge"
        )

        row_parts.append(
            '<div class="rank-row">'
            f'<div class="{badge_class}">'
            f'{idx}'
            f'</div>'
            '<div>'
            f'<div class="rank-name">'
            f'{name}'
            f'</div>'
            f'<div class="pbar thin">'
            f'<div style="width:{clamp_pct_width(prob):.2f}%;">'
            f'</div>'
            f'</div>'
            '</div>'
            f'<div class="rank-prob">'
            f'{pct(prob)}'
            f'</div>'
            '</div>'
        )

    card_html = (
        '<div class="card" '
        'style="padding-top:0.4rem;'
        'padding-bottom:0.4rem;">'
        + "".join(
            row_parts
        )
        + "</div>"
    )

    render_html(
        card_html
    )


def load_standard_ground_truth(result: dict) -> tuple[Image.Image | None, list]:
    """Load labelled YOLO ground truth only when the uploaded filename matches a validation image."""
    filename = Path(str(result.get("filename", ""))).name
    if not filename:
        return None, []

    label_dir = resolve_project_path(YOLO_VAL_LABELS_DIR)
    label_path = label_dir / f"{Path(filename).stem}.txt"

    if not label_path.is_file():
        return None, []

    image = result["low_res_image"].copy().convert("RGB")
    gt_boxes = read_yolo_labels(
        label_path,
        image.width,
        image.height,
    )

    if not gt_boxes:
        return None, []

    gt_image = draw_labelled_boxes(
        image,
        gt_boxes,
        GT_COLOR,
        lambda item: yolo_class_label(item["class_id"]),
    )

    return gt_image, gt_boxes


def render_localization(
    result: dict,
    predicted_class: str,
) -> None:
    section_title(
        "Defect Localization & Coordinate Mapping",
        (
            "Compare annotated ground truth with YOLO localization. "
            "Ground truth is shown only when the uploaded filename matches "
            "a labelled validation image."
        ),
    )

    detections = result.get(
        "detections",
        [],
    )

    (
        matching,
        secondary,
    ) = split_detections_by_prediction(
        detections,
        predicted_class,
    )

    primary_image = draw_detection_subset(
        result["low_res_image"],
        matching,
        PRED_COLOR,
    )

    all_detection_image = draw_detection_subset(
        result["low_res_image"],
        detections,
        PRED_COLOR,
    )

    gt_image, gt_boxes = load_standard_ground_truth(
        result
    )

    # Keep the presentation view focused: only the two images that matter
    # most for evaluation are visible by default.
    gt_col, pred_col = st.columns(
        2,
        gap="large",
    )

    with gt_col:
        render_html(
            '<div class="img-head">Ground Truth · Cyan</div>'
        )

        if gt_image is not None:
            st.image(
                gt_image,
                width="stretch",
            )
        else:
            render_html(
                '<div class="card" '
                'style="min-height:250px;display:flex;align-items:center;'
                'justify-content:center;text-align:center;">'
                '<div>'
                '<div class="metric-label">Ground Truth Unavailable</div>'
                '<div class="metric-note" '
                'style="margin-top:10px;line-height:1.6;max-width:360px;">'
                'Ground-truth boxes are available only for labelled YOLO '
                'validation images. No annotation is created for arbitrary uploads.'
                '</div>'
                '</div>'
                '</div>'
            )

    with pred_col:
        render_html(
            '<div class="img-head">YOLO Prediction · Red</div>'
        )
        st.image(
            all_detection_image,
            width="stretch",
        )

    render_html(
        '<div class="note" style="margin-top:0.75rem;">'
        '<b style="color:#22d3ee;">Cyan</b> = Ground Truth'
        ' &nbsp;&nbsp;•&nbsp;&nbsp; '
        '<b style="color:#ff5c5c;">Red</b> = YOLO Prediction'
        '</div>'
    )

    # Compact textual summary instead of four large metric cards.
    processing_size = (
        f'{result["processed_size"][0]} × '
        f'{result["processed_size"][1]}'
    )

    gt_value = (
        str(len(gt_boxes))
        if gt_image is not None
        else "N/A"
    )

    render_html(
        '<div class="card" style="margin-top:1rem;padding:0.9rem 1.1rem;">'
        '<div style="display:flex;flex-wrap:wrap;gap:10px 26px;'
        'align-items:center;">'
        f'<span><span class="metric-label">Matching Regions</span> '
        f'<b>{len(matching)}</b></span>'
        f'<span><span class="metric-label">YOLO Regions</span> '
        f'<b>{len(detections)}</b></span>'
        f'<span><span class="metric-label">Ground-Truth Boxes</span> '
        f'<b>{gt_value}</b></span>'
        f'<span><span class="metric-label">Processing Size</span> '
        f'<b>{html.escape(processing_size)}</b></span>'
        '</div>'
        '</div>'
    )

    render_html(
        '<div class="img-head" '
        'style="margin-top:1rem;">'
        'Localization Status'
        '</div>'
    )

    if matching:
        st.success(
            f'YOLO localized '
            f'{len(matching)} region'
            f'{"s" if len(matching) != 1 else ""} '
            f'matching the primary ResNet-50 prediction: '
            f'{format_class_name(predicted_class)}.'
        )

    elif not detections:
        st.info(
            "ResNet-50 produced an image-level classification, "
            "but YOLO did not localize any defect region above "
            "the current detection threshold. Localization is therefore "
            "unavailable for this image. This does not automatically "
            "invalidate the image-level classification."
        )

    else:
        detected_names = ", ".join(
            sorted(
                {
                    format_class_name(
                        item["class_name"]
                    )
                    for item
                    in detections
                }
            )
        )

        st.warning(
            f'ResNet-50 predicts '
            f'{format_class_name(predicted_class)}, '
            f'but YOLO did not localize a matching region. '
            f'YOLO instead detected: {detected_names}. '
            "The models are providing inconsistent evidence, "
            "so this result should be interpreted cautiously."
        )

    # Secondary visuals and coordinate tables remain available without
    # cluttering the presentation view.
    with st.expander(
        "View localization details",
        expanded=False,
    ):
        detail_left, detail_right = st.columns(
            2,
            gap="large",
        )

        with detail_left:
            render_html(
                '<div class="img-head">Uploaded Image</div>'
            )
            st.image(
                result["low_res_image"],
                width="stretch",
            )

        with detail_right:
            render_html(
                '<div class="img-head">'
                'Primary Localization Evidence · Red'
                '</div>'
            )
            st.image(
                primary_image,
                width="stretch",
            )

        if matching:
            render_html(
                '<div class="img-head" style="margin-top:1rem;">'
                'Matching Localization Coordinates'
                '</div>'
            )
            st.dataframe(
                build_detection_rows(
                    matching
                ),
                width="stretch",
                hide_index=True,
            )

        if secondary:
            render_html(
                '<div class="img-head" style="margin-top:1rem;">'
                'Additional Detector Observations'
                '</div>'
            )
            st.dataframe(
                build_detection_rows(
                    secondary
                ),
                width="stretch",
                hide_index=True,
            )

    render_html(
        '<div class="note">'
        '<b>Model roles:</b> ResNet-50 performs image-level classification, '
        'while YOLO performs bounding-box localization. Ground-truth annotations '
        'are shown only when they genuinely exist for a labelled validation image. '
        'A missing YOLO box means the detector did not provide localization above '
        'its threshold; it does not by itself prove that the classifier is wrong.'
        '</div>'
    )

def render_standard_mode(
    backend_online: bool,
) -> None:
    section_title(
        "Standard Analysis"
    )

    (
        uploaded,
        _,
    ) = render_upload(
        backend_online,
        "standard_uploader",
    )

    if uploaded is not None:
        if st.button(
            "Run SteelVision Analysis",
            type="primary",
            key="standard_analyze",
            disabled=not backend_online,
        ):
            run_standard_analysis(
                uploaded
            )

    if st.session_state.standard_error:
        st.error(
            st.session_state.standard_error
        )

    domain_result = (
        st.session_state
        .standard_domain_validation
    )

    if domain_result:
        render_domain_validation(
            domain_result
        )

        if not domain_result[
            "supported"
        ]:
            return

    result = (
        st.session_state
        .standard_result
    )

    localization = (
        st.session_state
        .standard_localization
    )

    original_bytes = (
        st.session_state
        .standard_original_bytes
    )

    if (
        result
        and localization
        and original_bytes
    ):
        render_prediction_summary(
            result
        )

        render_uncertainty(
            result[
                "entropy"
            ]
        )

        render_gradcam(
            original_bytes,
            st.session_state.standard_gradcam,
            st.session_state.standard_gradcam_error,
        )

        render_localization(
            localization,
            result[
                "predicted_class"
            ],
        )

        render_top3(
            result[
                "top_3"
            ]
        )


# -----------------------------------------------------------------------------
# Resolution experiment
# -----------------------------------------------------------------------------

def run_resolution_experiment(
    uploaded,
    resolution: str,
) -> None:
    image_bytes = (
        uploaded.getvalue()
    )

    filename = (
        uploaded.name
    )

    mime = guess_mime(
        filename,
        getattr(
            uploaded,
            "type",
            None,
        ),
    )

    st.session_state.experiment_domain_validation = None
    st.session_state.experiment_original = None
    st.session_state.experiment_simulated = None
    st.session_state.experiment_error = None

    with st.spinner(
        "Validating input domain..."
    ):
        try:
            domain_result = (
                call_validate_domain(
                    image_bytes,
                    filename,
                    mime,
                )
            )

        except RuntimeError as exc:
            st.session_state.experiment_error = (
                str(
                    exc
                )
            )

            return

    st.session_state.experiment_domain_validation = (
        domain_result
    )

    if not domain_result[
        "supported"
    ]:
        return

    with st.spinner(
        f"Comparing original input "
        f"with simulated {resolution} px input..."
    ):
        try:
            st.session_state.experiment_original = (
                call_localize(
                    image_bytes,
                    filename,
                    mime,
                    "original",
                )
            )

            st.session_state.experiment_simulated = (
                call_localize(
                    image_bytes,
                    filename,
                    mime,
                    resolution,
                )
            )

        except RuntimeError as exc:
            st.session_state.experiment_error = (
                str(
                    exc
                )
            )


def render_detection_group(
    label: str,
    result: dict,
) -> None:
    summary = html.escape(
        format_detection_summary(
            result
        )
    )

    render_html(
        '<div class="card">'
        f'<div class="metric-label">'
        f'{html.escape(label)}'
        f'</div>'
        f'<div class="metric-value compact">'
        f'{summary}'
        f'</div>'
        '</div>'
    )


def render_resolution_comparison(
    original: dict,
    simulated: dict,
) -> None:
    section_title(
        "Resolution Comparison",
        (
            "Compare the original image with "
            f'a simulated {simulated["resolution"]}-pixel '
            "version of the same image."
        ),
    )

    c1, c2, c3 = st.columns(
        3,
        gap="medium",
    )

    with c1:
        render_html(
            '<div class="img-head">'
            'Uploaded Image · Baseline'
            '</div>'
        )

        st.image(
            original[
                "low_res_image"
            ],
            width="stretch",
        )

    with c2:
        render_html(
            f'<div class="img-head">'
            f'Simulated '
            f'{html.escape(simulated["resolution"])} '
            f'px Source'
            f'</div>'
        )

        st.image(
            simulated[
                "low_res_image"
            ],
            width="stretch",
        )

    with c3:
        render_html(
            '<div class="img-head">'
            'Mapped Detection After Simulation'
            '</div>'
        )

        st.image(
            simulated[
                "mapped_image"
            ],
            width="stretch",
        )

    st.write(
        ""
    )

    g1, g2 = st.columns(
        2,
        gap="medium",
    )

    with g1:
        render_detection_group(
            "Baseline Defect Detections",
            original,
        )

    with g2:
        render_detection_group(
            "Simulated Defect Detections",
            simulated,
        )

    st.write(
        ""
    )

    left, right = st.columns(
        2,
        gap="large",
    )

    with left:
        render_html(
            '<div class="img-head">'
            'Baseline Defects'
            '</div>'
        )

        st.image(
            original[
                "mapped_image"
            ],
            width="stretch",
        )

    with right:
        render_html(
            '<div class="img-head">'
            'Simulated Defects'
            '</div>'
        )

        st.image(
            simulated[
                "processed_image"
            ],
            width="stretch",
        )

    if simulated[
        "detections"
    ]:
        st.dataframe(
            build_detection_rows(
                simulated[
                    "detections"
                ]
            ),
            width="stretch",
            hide_index=True,
        )

    else:
        st.info(
            "No defect regions were detected "
            "in the simulated input."
        )

    render_html(
        '<div class="note">'
        'The simulation adjusts the image to the resolution selected '
        'by the user and then resizes it to the detector processing '
        'dimensions for inference. Changing the resolution may alter '
        'or reduce available visual detail. Therefore, a higher detector '
        'confidence after resolution adjustment should not necessarily '
        'be interpreted as improved image quality or detection reliability.'
        '</div>'
    )


def render_experiment_mode(
    backend_online: bool,
) -> None:
    section_title(
        "Resolution Experiment"
    )

    render_html(
        '<div class="disclaimer" style="margin-top:0;">'
        '<b>Research Mode:</b> '
        'This mode intentionally adjusts the uploaded image '
        'to study how the selected resolution changes localization.'
        '</div>'
    )

    (
        uploaded,
        _,
    ) = render_upload(
        backend_online,
        "experiment_uploader",
    )

    if uploaded is None:
        return

    resolution = st.selectbox(
        "Resolution to Experiment With",
        options=[
            "224",
            "128",
            "64",
            "32",
        ],
        index=1,
        help=(
            "The uploaded image is intentionally adjusted "
            "to this square resolution before detector processing."
        ),
    )

    if st.button(
        "Run Experiment",
        type="primary",
        key="experiment_analyze",
        disabled=not backend_online,
    ):
        run_resolution_experiment(
            uploaded,
            resolution,
        )

    if st.session_state.experiment_error:
        st.error(
            st.session_state.experiment_error
        )

    domain_result = (
        st.session_state
        .experiment_domain_validation
    )

    if domain_result:
        render_domain_validation(
            domain_result
        )

        if not domain_result[
            "supported"
        ]:
            return

    if (
        st.session_state.experiment_original
        and st.session_state.experiment_simulated
    ):
        render_resolution_comparison(
            st.session_state.experiment_original,
            st.session_state.experiment_simulated,
        )


# -----------------------------------------------------------------------------
# YOLO ground-truth evaluation — self-contained helpers
# -----------------------------------------------------------------------------

def resolve_project_path(
    relative_path: str,
) -> Path:
    """
    Resolve a project-relative path. Tries the current working directory first
    (the app is normally run from the project root), then the project root
    derived from this file's location (src/ui/app.py -> project root).
    """
    candidate = Path(
        relative_path
    )

    if candidate.exists():
        return candidate

    project_root = (
        Path(
            __file__
        )
        .resolve()
        .parents[2]
    )

    alternative = (
        project_root
        / relative_path
    )

    if alternative.exists():
        return alternative

    return candidate


@st.cache_resource(
    show_spinner="Loading YOLO evaluation model..."
)
def load_yolo_eval_model(
    weights_path: str,
):
    # Imported lazily so the rest of SteelVision still starts even if
    # ultralytics is unavailable.
    from ultralytics import YOLO

    return YOLO(
        weights_path
    )


def list_validation_images() -> list:
    images_dir = resolve_project_path(
        YOLO_VAL_IMAGES_DIR
    )

    if not images_dir.is_dir():
        return []

    return sorted(
        path
        for path in images_dir.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_SUFFIXES
        )
    )


def read_yolo_labels(
    label_path: Path,
    image_width: int,
    image_height: int,
) -> list:
    """
    Read YOLO-format labels (class_id cx cy w h, normalized) and convert
    each box to pixel xyxy using the actual image size.
    """
    boxes = []

    if not label_path.is_file():
        return boxes

    with open(
        label_path,
        "r",
        encoding="utf-8",
    ) as handle:
        for raw_line in handle:
            parts = raw_line.strip().split()

            if len(parts) < 5:
                continue

            try:
                class_id = int(
                    float(
                        parts[0]
                    )
                )

                cx, cy, bw, bh = (
                    float(
                        value
                    )
                    for value
                    in parts[1:5]
                )

            except ValueError:
                continue

            x1 = (cx - bw / 2) * image_width
            y1 = (cy - bh / 2) * image_height
            x2 = (cx + bw / 2) * image_width
            y2 = (cy + bh / 2) * image_height

            boxes.append(
                {
                    "class_id":
                        class_id,

                    "box":
                        [
                            x1,
                            y1,
                            x2,
                            y2,
                        ],
                }
            )

    return boxes


def compute_iou(
    box_a: list,
    box_b: list,
) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_w = max(
        0.0,
        min(ax2, bx2) - max(ax1, bx1),
    )

    inter_h = max(
        0.0,
        min(ay2, by2) - max(ay1, by1),
    )

    intersection = inter_w * inter_h

    area_a = max(
        0.0,
        ax2 - ax1,
    ) * max(
        0.0,
        ay2 - ay1,
    )

    area_b = max(
        0.0,
        bx2 - bx1,
    ) * max(
        0.0,
        by2 - by1,
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


def predict_yolo_boxes(
    model,
    image: Image.Image,
    confidence_threshold: float,
) -> list:
    results = model.predict(
        image,
        conf=confidence_threshold,
        verbose=False,
    )

    predictions = []

    if not results:
        return predictions

    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return predictions

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy()

    for idx in range(len(xyxy)):
        predictions.append(
            {
                "class_id":
                    int(
                        classes[idx]
                    ),

                "confidence":
                    float(
                        confs[idx]
                    ),

                "box":
                    [
                        float(value)
                        for value
                        in xyxy[idx]
                    ],
            }
        )

    return predictions


def safe_divide(
    numerator: float,
    denominator: float,
) -> float:
    if denominator <= 0:
        return 0.0

    return numerator / denominator


def evaluate_detections(
    gt_boxes: list,
    pred_boxes: list,
    iou_threshold: float,
) -> dict:
    """
    Standard one-to-one detection matching plus diagnostic matching.

    True positives: same class, IoU >= threshold, greedy by IoU descending,
    each GT and each prediction used at most once.

    Diagnostic pairs are built only from the remaining unmatched GT and
    predictions to explain failures. They never become true positives.
    """
    candidate_pairs = []

    for gi, gt in enumerate(gt_boxes):
        for pi, pred in enumerate(pred_boxes):
            if gt["class_id"] != pred["class_id"]:
                continue

            iou = compute_iou(
                gt["box"],
                pred["box"],
            )

            if iou >= iou_threshold:
                candidate_pairs.append(
                    (
                        iou,
                        gi,
                        pi,
                    )
                )

    candidate_pairs.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    used_gt = set()
    used_pred = set()
    matches = []

    for iou, gi, pi in candidate_pairs:
        if gi in used_gt or pi in used_pred:
            continue

        used_gt.add(gi)
        used_pred.add(pi)

        matches.append(
            {
                "gt_index": gi,
                "pred_index": pi,
                "class_id": gt_boxes[gi]["class_id"],
                "confidence": pred_boxes[pi]["confidence"],
                "iou": iou,
            }
        )

    # --- Diagnostic matching on the remaining boxes ---

    diagnostic_candidates = []

    for gi, gt in enumerate(gt_boxes):
        if gi in used_gt:
            continue

        for pi, pred in enumerate(pred_boxes):
            if pi in used_pred:
                continue

            iou = compute_iou(
                gt["box"],
                pred["box"],
            )

            if iou < DIAGNOSTIC_MIN_IOU:
                continue

            same_class = (
                gt["class_id"]
                == pred["class_id"]
            )

            if same_class and iou < iou_threshold:
                reason = REASON_POOR_LOC

            elif (
                not same_class
                and iou >= iou_threshold
            ):
                reason = REASON_WRONG_CLASS

            elif (
                not same_class
                and iou < iou_threshold
            ):
                reason = REASON_WRONG_CLASS_POOR_LOC

            else:
                continue

            diagnostic_candidates.append(
                (
                    iou,
                    gi,
                    pi,
                    reason,
                )
            )

    diagnostic_candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    diag_gt = set()
    diag_pred = set()
    diagnostics = []

    for iou, gi, pi, reason in diagnostic_candidates:
        if gi in diag_gt or pi in diag_pred:
            continue

        diag_gt.add(gi)
        diag_pred.add(pi)

        diagnostics.append(
            {
                "gt_index": gi,
                "pred_index": pi,
                "gt_class_id": gt_boxes[gi]["class_id"],
                "pred_class_id": pred_boxes[pi]["class_id"],
                "confidence": pred_boxes[pi]["confidence"],
                "iou": iou,
                "reason": reason,
            }
        )

    missed_gt = [
        gi
        for gi in range(len(gt_boxes))
        if gi not in used_gt
        and gi not in diag_gt
    ]

    extra_pred = [
        pi
        for pi in range(len(pred_boxes))
        if pi not in used_pred
        and pi not in diag_pred
    ]

    tp = len(matches)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp

    mean_iou = (
        sum(
            item["iou"]
            for item in matches
        )
        / tp
        if tp
        else 0.0
    )

    return {
        "gt_count": len(gt_boxes),
        "pred_count": len(pred_boxes),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": safe_divide(tp, tp + fp),
        "recall": safe_divide(tp, tp + fn),
        "mean_iou": mean_iou,
        "matches": matches,
        "diagnostics": diagnostics,
        "missed_gt": missed_gt,
        "extra_pred": extra_pred,
    }


def yolo_class_label(
    class_id: int,
) -> str:
    if 0 <= class_id < len(YOLO_CLASS_NAMES):
        return format_class_name(
            YOLO_CLASS_NAMES[class_id]
        )

    return f"Class {class_id}"


def load_validation_sample(
    image_path: Path,
) -> tuple:
    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )

    label_path = (
        resolve_project_path(
            YOLO_VAL_LABELS_DIR
        )
        / f"{image_path.stem}.txt"
    )

    gt_boxes = read_yolo_labels(
        label_path,
        image.width,
        image.height,
    )

    return (
        image,
        gt_boxes,
    )


def evaluate_validation_image(
    model,
    image_path: Path,
    confidence_threshold: float,
    iou_threshold: float,
) -> tuple:
    (
        image,
        gt_boxes,
    ) = load_validation_sample(
        image_path
    )

    pred_boxes = predict_yolo_boxes(
        model,
        image,
        confidence_threshold,
    )

    evaluation = evaluate_detections(
        gt_boxes,
        pred_boxes,
        iou_threshold,
    )

    return (
        image,
        gt_boxes,
        pred_boxes,
        evaluation,
    )


def run_full_yolo_evaluation(
    confidence_threshold: float,
    iou_threshold: float,
) -> dict:
    image_paths = list_validation_images()

    model = load_yolo_eval_model(
        str(
            resolve_project_path(
                YOLO_WEIGHTS_PATH
            )
        )
    )

    class_gt = defaultdict(int)
    class_pred = defaultdict(int)
    class_tp = defaultdict(int)
    class_tp_iou = defaultdict(float)

    failure_counts = {
        REASON_POOR_LOC: 0,
        REASON_WRONG_CLASS: 0,
        REASON_WRONG_CLASS_POOR_LOC: 0,
    }

    per_image_rows = []

    total_gt = 0
    total_pred = 0
    total_tp = 0
    total_iou_sum = 0.0

    progress = st.progress(
        0.0,
        text="Evaluating validation images...",
    )

    for idx, image_path in enumerate(
        image_paths,
        start=1,
    ):
        (
            _,
            gt_boxes,
            pred_boxes,
            evaluation,
        ) = evaluate_validation_image(
            model,
            image_path,
            confidence_threshold,
            iou_threshold,
        )

        for gt in gt_boxes:
            class_gt[gt["class_id"]] += 1

        for pred in pred_boxes:
            class_pred[pred["class_id"]] += 1

        for match in evaluation["matches"]:
            class_tp[match["class_id"]] += 1
            class_tp_iou[match["class_id"]] += match["iou"]
            total_iou_sum += match["iou"]

        for diag in evaluation["diagnostics"]:
            failure_counts[diag["reason"]] += 1

        total_gt += evaluation["gt_count"]
        total_pred += evaluation["pred_count"]
        total_tp += evaluation["tp"]

        per_image_rows.append(
            {
                "Image": image_path.name,
                "GT": evaluation["gt_count"],
                "Pred": evaluation["pred_count"],
                "TP": evaluation["tp"],
                "FP": evaluation["fp"],
                "FN": evaluation["fn"],
            }
        )

        progress.progress(
            idx / len(image_paths),
            text=(
                f"Evaluated {idx} of "
                f"{len(image_paths)} images..."
            ),
        )

    progress.empty()

    per_class_rows = []

    for class_id, class_name in enumerate(
        YOLO_CLASS_NAMES
    ):
        gt_n = class_gt[class_id]
        pred_n = class_pred[class_id]
        tp_n = class_tp[class_id]
        fp_n = pred_n - tp_n
        fn_n = gt_n - tp_n

        per_class_rows.append(
            {
                "Class": format_class_name(class_name),
                "GT": gt_n,
                "Predictions": pred_n,
                "TP": tp_n,
                "FP": fp_n,
                "FN": fn_n,
                "Precision": round(
                    safe_divide(tp_n, tp_n + fp_n),
                    4,
                ),
                "Recall": round(
                    safe_divide(tp_n, tp_n + fn_n),
                    4,
                ),
                "Mean IoU": round(
                    safe_divide(
                        class_tp_iou[class_id],
                        tp_n,
                    ),
                    4,
                ),
            }
        )

    total_fp = total_pred - total_tp
    total_fn = total_gt - total_tp

    return {
        "confidence_threshold": confidence_threshold,
        "iou_threshold": iou_threshold,
        "images_evaluated": len(image_paths),
        "gt_boxes": total_gt,
        "pred_boxes": total_pred,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": safe_divide(
            total_tp,
            total_tp + total_fp,
        ),
        "recall": safe_divide(
            total_tp,
            total_tp + total_fn,
        ),
        "mean_iou": safe_divide(
            total_iou_sum,
            total_tp,
        ),
        "failure_counts": failure_counts,
        "per_class_rows": per_class_rows,
        "per_image_rows": per_image_rows,
    }


def draw_labelled_boxes(
    image: Image.Image,
    items: list,
    color: str,
    label_fn,
) -> Image.Image:
    output = (
        image
        .copy()
        .convert(
            "RGB"
        )
    )

    draw = ImageDraw.Draw(
        output
    )

    line_width = max(
        2,
        int(
            min(
                output.size
            )
            * 0.01
        ),
    )

    for item in items:
        x1, y1, x2, y2 = item["box"]

        draw.rectangle(
            [
                x1,
                y1,
                x2,
                y2,
            ],
            outline=color,
            width=line_width,
        )

        draw.text(
            (
                x1 + 3,
                y1 + 3,
            ),
            label_fn(
                item
            ),
            fill=color,
        )

    return output


def render_eval_metric_card(
    label: str,
    value: str,
    note: str = "",
    accent: bool = False,
) -> None:
    cls = (
        "metric-value accent-text"
        if accent
        else "metric-value"
    )

    note_html = (
        f'<div class="metric-note">'
        f'{html.escape(note)}'
        f'</div>'
        if note
        else ""
    )

    render_html(
        f'<div class="card">'
        f'<div class="metric-label">'
        f'{html.escape(label)}'
        f'</div>'
        f'<div class="{cls}">'
        f'{html.escape(str(value))}'
        f'</div>'
        f'{note_html}'
        f'</div>'
    )


def render_eval_metric_row(
    cards: list,
) -> None:
    columns = st.columns(
        len(
            cards
        ),
        gap="medium",
    )

    for column, card in zip(
        columns,
        cards,
    ):
        with column:
            render_eval_metric_card(
                *card
            )


def render_full_yolo_results(
    result: dict,
    current_conf: float,
    current_iou: float,
) -> None:
    section_title(
        "Full Validation Results",
        (
            f"Confidence ≥ {result['confidence_threshold']:.2f} · "
            f"IoU ≥ {result['iou_threshold']:.2f} · "
            "one-to-one matching"
        ),
    )

    if (
        abs(
            result["confidence_threshold"]
            - current_conf
        )
        > 1e-9
        or abs(
            result["iou_threshold"]
            - current_iou
        )
        > 1e-9
    ):
        st.warning(
            "The thresholds above differ from the thresholds used for the "
            "results shown below. Press **Run Full Validation Evaluation** "
            "again to refresh the full-validation results."
        )

    render_eval_metric_row(
        [
            (
                "Precision",
                f"{result['precision']:.4f}",
                "TP / (TP + FP)",
                True,
            ),
            (
                "Recall",
                f"{result['recall']:.4f}",
                "TP / (TP + FN)",
                True,
            ),
            (
                "Mean TP IoU",
                f"{result['mean_iou']:.4f}",
                "Across true positives only",
                False,
            ),
        ]
    )

    st.write("")

    render_eval_metric_row(
        [
            (
                "True Positives",
                str(result["tp"]),
                "Valid one-to-one matches",
                False,
            ),
            (
                "False Positives",
                str(result["fp"]),
                "Predictions − TP",
                False,
            ),
            (
                "False Negatives",
                str(result["fn"]),
                "Ground-truth boxes − TP",
                False,
            ),
        ]
    )

    st.write("")

    render_eval_metric_row(
        [
            (
                "Images Evaluated",
                str(result["images_evaluated"]),
                "",
                False,
            ),
            (
                "Ground-Truth Boxes",
                str(result["gt_boxes"]),
                "",
                False,
            ),
            (
                "Predicted Boxes",
                str(result["pred_boxes"]),
                "",
                False,
            ),
        ]
    )

    render_html(
        '<div class="img-head" style="margin-top:1.4rem;">'
        'Failure Breakdown'
        '</div>'
    )

    failures = result["failure_counts"]

    render_eval_metric_row(
        [
            (
                REASON_POOR_LOC,
                str(failures[REASON_POOR_LOC]),
                "Correct class, IoU below threshold",
                False,
            ),
            (
                REASON_WRONG_CLASS,
                str(failures[REASON_WRONG_CLASS]),
                "Overlapping box, incorrect class",
                False,
            ),
            (
                REASON_WRONG_CLASS_POOR_LOC,
                str(failures[REASON_WRONG_CLASS_POOR_LOC]),
                "Incorrect class and IoU below threshold",
                False,
            ),
        ]
    )

    render_html(
        '<div class="img-head" style="margin-top:1.4rem;">'
        'Per-Class Results'
        '</div>'
    )

    st.dataframe(
        result["per_class_rows"],
        width="stretch",
        hide_index=True,
    )

    render_html(
        '<div class="img-head" style="margin-top:1.4rem;">'
        'Per-Image Summary'
        '</div>'
    )

    st.dataframe(
        result["per_image_rows"],
        width="stretch",
        hide_index=True,
    )

    render_html(
        '<div class="note">'
        'Failure categories are diagnostic explanations only. '
        'Under standard detection counting, a poorly localized prediction '
        'is a false positive and its ground-truth box is a false negative.'
        '</div>'
    )


def render_per_image_explorer(
    model,
    image_paths: list,
    confidence_threshold: float,
    iou_threshold: float,
) -> None:
    section_title(
        "Per-Image Ground-Truth Explorer",
        (
            "Inspect annotated ground-truth boxes against YOLO "
            "predictions for one labelled validation image."
        ),
    )

    names = [
        path.name
        for path in image_paths
    ]

    selected_name = st.selectbox(
        "Validation Image",
        options=names,
        key="yolo_eval_selected_image",
    )

    selected_path = image_paths[
        names.index(
            selected_name
        )
    ]

    with st.spinner(
        "Evaluating selected image..."
    ):
        (
            image,
            gt_boxes,
            pred_boxes,
            evaluation,
        ) = evaluate_validation_image(
            model,
            selected_path,
            confidence_threshold,
            iou_threshold,
        )

    render_eval_metric_row(
        [
            (
                "True Positives",
                str(evaluation["tp"]),
                "",
                False,
            ),
            (
                "False Positives",
                str(evaluation["fp"]),
                "",
                False,
            ),
            (
                "False Negatives",
                str(evaluation["fn"]),
                "",
                False,
            ),
        ]
    )

    st.write("")

    render_eval_metric_row(
        [
            (
                "Precision",
                f"{evaluation['precision']:.4f}",
                "",
                True,
            ),
            (
                "Recall",
                f"{evaluation['recall']:.4f}",
                "",
                True,
            ),
            (
                "Mean TP IoU",
                f"{evaluation['mean_iou']:.4f}",
                "",
                False,
            ),
        ]
    )

    def gt_label(
        item: dict,
    ) -> str:
        return yolo_class_label(
            item["class_id"]
        )

    def pred_label(
        item: dict,
    ) -> str:
        return (
            f"{yolo_class_label(item['class_id'])} "
            f"{item['confidence'] * 100:.1f}%"
        )

    gt_image = draw_labelled_boxes(
        image,
        gt_boxes,
        GT_COLOR,
        gt_label,
    )

    pred_image = draw_labelled_boxes(
        image,
        pred_boxes,
        PRED_COLOR,
        pred_label,
    )

    overlay_image = draw_labelled_boxes(
        gt_image,
        pred_boxes,
        PRED_COLOR,
        pred_label,
    )

    st.write("")

    c1, c2, c3 = st.columns(
        3,
        gap="medium",
    )

    with c1:
        render_html(
            '<div class="img-head">'
            'Original Image'
            '</div>'
        )

        st.image(
            image,
            width="stretch",
        )

    with c2:
        render_html(
            '<div class="img-head">'
            'Ground Truth'
            '</div>'
        )

        st.image(
            gt_image,
            width="stretch",
        )

    with c3:
        render_html(
            '<div class="img-head">'
            'YOLO Predictions'
            '</div>'
        )

        st.image(
            pred_image,
            width="stretch",
        )

    render_html(
        '<div class="img-head" style="margin-top:1.2rem;">'
        'Ground Truth + Predictions'
        '</div>'
    )

    st.image(
        overlay_image,
        width="stretch",
    )

    render_html(
        '<div class="note">'
        '<b style="color:#ff5c5c;">Red</b> = Ground Truth · '
        '<b style="color:#22d3ee;">Cyan</b> = YOLO Prediction'
        '</div>'
    )

    # --- Valid detection matches ---

    section_title(
        "Valid Detection Matches"
    )

    if evaluation["matches"]:
        rows = [
            {
                "Match #": idx,
                "Ground Truth": (
                    f"GT {m['gt_index'] + 1} · "
                    f"{yolo_class_label(m['class_id'])}"
                ),
                "Prediction": (
                    f"Pred {m['pred_index'] + 1} · "
                    f"{yolo_class_label(m['class_id'])}"
                ),
                "Confidence": f"{m['confidence'] * 100:.2f}%",
                "IoU": f"{m['iou']:.4f}",
                "Result": "True Positive",
            }
            for idx, m in enumerate(
                sorted(
                    evaluation["matches"],
                    key=lambda item: item["gt_index"],
                ),
                start=1,
            )
        ]

        st.dataframe(
            rows,
            width="stretch",
            hide_index=True,
        )

    else:
        st.info(
            "No valid true-positive matches at the selected "
            "confidence and IoU thresholds for this image."
        )

    # --- Failure diagnostics ---

    section_title(
        "Failure Diagnostics"
    )

    if evaluation["diagnostics"]:
        rows = [
            {
                "Failure #": idx,
                "Ground Truth": (
                    f"GT {d['gt_index'] + 1} · "
                    f"{yolo_class_label(d['gt_class_id'])}"
                ),
                "Prediction": (
                    f"Pred {d['pred_index'] + 1} · "
                    f"{yolo_class_label(d['pred_class_id'])}"
                ),
                "Confidence": f"{d['confidence'] * 100:.2f}%",
                "IoU": f"{d['iou']:.4f}",
                "Reason": d["reason"],
            }
            for idx, d in enumerate(
                evaluation["diagnostics"],
                start=1,
            )
        ]

        st.dataframe(
            rows,
            width="stretch",
            hide_index=True,
        )

        render_html(
            '<div class="note">'
            'These are not true positives. The ground-truth box '
            'contributes to FN and the failed prediction contributes to FP.'
            '</div>'
        )

    else:
        st.info(
            "No failure-diagnostic pairs for this image at the "
            "selected thresholds."
        )

    # --- Completely missed ground truth ---

    section_title(
        "Completely Missed Ground-Truth Defects"
    )

    if evaluation["missed_gt"]:
        rows = [
            {
                "GT index": gi + 1,
                "Class": yolo_class_label(
                    gt_boxes[gi]["class_id"]
                ),
                "Result": "False Negative",
            }
            for gi in evaluation["missed_gt"]
        ]

        st.dataframe(
            rows,
            width="stretch",
            hide_index=True,
        )

    else:
        st.success(
            "No completely missed ground-truth defects in this image."
        )

    # --- Extra predictions ---

    section_title(
        "Extra YOLO Predictions"
    )

    if evaluation["extra_pred"]:
        rows = [
            {
                "Prediction index": pi + 1,
                "Class": yolo_class_label(
                    pred_boxes[pi]["class_id"]
                ),
                "Confidence": (
                    f"{pred_boxes[pi]['confidence'] * 100:.2f}%"
                ),
                "Result": "False Positive",
            }
            for pi in evaluation["extra_pred"]
        ]

        st.dataframe(
            rows,
            width="stretch",
            hide_index=True,
        )

    else:
        st.success(
            "No extra YOLO predictions in this image."
        )


def render_yolo_evaluation_mode() -> None:
    section_title(
        "YOLO Ground-Truth Evaluation",
        (
            "Compare detector predictions with annotated bounding-box "
            "ground truth from the labelled validation set."
        ),
    )

    render_html(
        '<div class="callout mid">'
        '<div class="title">Research Scope</div>'
        '<div class="body">'
        'Ground truth exists only for the labelled YOLO validation images. '
        'This page evaluates bounding-box detection against the dataset '
        'annotations. It does not generate ground truth for arbitrary '
        'uploaded images.'
        '</div>'
        '</div>'
    )

    image_paths = list_validation_images()

    weights_path = resolve_project_path(
        YOLO_WEIGHTS_PATH
    )

    if not image_paths:
        st.error(
            "No validation images were found at "
            f"`{YOLO_VAL_IMAGES_DIR}`. Run SteelVision from the project "
            "root and verify the dataset path."
        )
        return

    if not weights_path.is_file():
        st.error(
            "YOLO weights were not found at "
            f"`{YOLO_WEIGHTS_PATH}`."
        )
        return

    section_title(
        "Evaluation Settings"
    )

    s1, s2 = st.columns(
        2,
        gap="large",
    )

    with s1:
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.05,
            max_value=0.95,
            value=YOLO_DEFAULT_CONF,
            step=0.05,
            key="yolo_eval_conf",
            help=(
                "Minimum detector confidence for a prediction "
                "to be kept."
            ),
        )

    with s2:
        iou_threshold = st.slider(
            "IoU Threshold",
            min_value=0.10,
            max_value=0.95,
            value=YOLO_DEFAULT_IOU,
            step=0.05,
            key="yolo_eval_iou",
            help=(
                "Minimum localization overlap (IoU) between a prediction "
                "and a same-class ground-truth box for a true positive."
            ),
        )

    render_html(
        '<div class="note">'
        f'Default thresholds: confidence {YOLO_DEFAULT_CONF:.2f}, '
        f'IoU {YOLO_DEFAULT_IOU:.2f}. '
        'A prediction is a true positive only when its class matches the '
        'ground-truth class and the localization overlap reaches the '
        'selected IoU threshold, using one-to-one matching.'
        '</div>'
    )

    try:
        model = load_yolo_eval_model(
            str(
                weights_path
            )
        )

    except Exception as exc:
        st.error(
            f"Could not load the YOLO model: {exc}"
        )
        return

    # --- Full validation evaluation ---

    section_title(
        "Full Validation Evaluation",
        (
            f"Evaluate all {len(image_paths)} labelled validation images "
            "at the selected thresholds."
        ),
    )

    if st.button(
        "Run Full Validation Evaluation",
        type="primary",
        key="yolo_eval_run_full",
    ):
        try:
            st.session_state.yolo_eval_full_result = (
                run_full_yolo_evaluation(
                    confidence_threshold,
                    iou_threshold,
                )
            )

        except Exception as exc:
            st.session_state.yolo_eval_full_result = None
            st.error(
                f"Full validation evaluation failed: {exc}"
            )

    full_result = (
        st.session_state
        .yolo_eval_full_result
    )

    if full_result:
        render_full_yolo_results(
            full_result,
            confidence_threshold,
            iou_threshold,
        )

    else:
        render_html(
            '<div class="note">'
            'No full-validation result yet. Press '
            '<b>Run Full Validation Evaluation</b> to evaluate the '
            'labelled validation set.'
            '</div>'
        )

    # --- Per-image explorer ---

    render_per_image_explorer(
        model,
        image_paths,
        confidence_threshold,
        iou_threshold,
    )

    render_html(
        '<div class="disclaimer">'
        '<b>Interpretation:</b> '
        'YOLO performs bounding-box object detection, not segmentation. '
        'Results reflect agreement with the annotated bounding boxes of '
        'this validation set at the selected confidence and IoU thresholds '
        'and should not be generalized beyond it.'
        '</div>'
    )


# -----------------------------------------------------------------------------
# Inpainting reliability
# -----------------------------------------------------------------------------

@st.cache_resource(
    show_spinner="Loading inpainting models..."
)
def load_inpainting_runtime():
    """Load the PyTorch classifier and trained U-Net used by EXP-016."""
    device = get_inpainting_device()

    classifier = load_inpainting_classifier(
        device=device
    )

    unet_path = resolve_project_path(
        INPAINTING_UNET_WEIGHTS
    )

    if not unet_path.is_file():
        raise FileNotFoundError(
            "Inpainting U-Net weights were not found at "
            f"{INPAINTING_UNET_WEIGHTS}."
        )

    unet_model = load_unet_inpainter(
        checkpoint_path=unet_path,
        device=device,
    )

    return (
        device,
        classifier,
        unet_model,
    )


def prepare_inpainting_image(
    image_bytes: bytes,
) -> np.ndarray:
    """Convert an uploaded image to the 224×224 grayscale EXP-016 format."""
    image = Image.open(
        BytesIO(image_bytes)
    ).convert(
        "L"
    ).resize(
        (
            INPAINTING_IMAGE_SIZE,
            INPAINTING_IMAGE_SIZE,
        ),
        Image.Resampling.BICUBIC,
    )

    return np.asarray(
        image,
        dtype=np.uint8,
    )


def inpainting_mask_preview(
    image: np.ndarray,
    mask: np.ndarray,
) -> Image.Image:
    """Show the controlled missing region as a dark masked area."""
    preview = image.copy()
    preview[
        mask > 0
    ] = 0

    return Image.fromarray(
        preview,
        mode="L",
    ).convert(
        "RGB"
    )


def inpainting_gradcam_preview(
    image: np.ndarray,
    heatmap: np.ndarray,
    center: tuple[int, int] | None,
) -> Image.Image:
    """Create a lightweight Grad-CAM overlay for the dashboard."""
    base = np.stack(
        [image, image, image],
        axis=-1,
    ).astype(
        np.float32
    )

    heat = np.clip(
        heatmap,
        0.0,
        1.0,
    )

    overlay = np.zeros_like(
        base
    )

    overlay[..., 0] = 255.0 * heat
    overlay[..., 1] = 180.0 * np.sqrt(heat)
    overlay[..., 2] = 70.0 * (1.0 - heat)

    blended = np.clip(
        0.58 * base
        + 0.42 * overlay,
        0,
        255,
    ).astype(
        np.uint8
    )

    output = Image.fromarray(
        blended,
        mode="RGB",
    )

    if center is not None:
        draw = ImageDraw.Draw(
            output
        )

        x, y = center
        radius = 7

        draw.ellipse(
            [
                x - radius,
                y - radius,
                x + radius,
                y + radius,
            ],
            outline="white",
            width=2,
        )

        draw.line(
            [
                x - 10,
                y,
                x + 10,
                y,
            ],
            fill="white",
            width=2,
        )

        draw.line(
            [
                x,
                y - 10,
                x,
                y + 10,
            ],
            fill="white",
            width=2,
        )

    return output


def build_inpainting_comparison_rows(
    original_prediction: dict,
    method_results: dict,
) -> list:
    """Create the compare-all-methods dashboard table."""
    rows = []

    for method in INPAINTING_METHODS:
        result = method_results[
            method
        ]

        prediction = result[
            "prediction"
        ]

        metrics = result[
            "metrics"
        ]

        flipped = inpainting_class_flip(
            original_prediction,
            prediction,
        )

        rows.append(
            {
                "Method": INPAINTING_METHOD_LABELS[
                    method
                ],
                "Prediction": format_class_name(
                    prediction[
                        "predicted_class"
                    ]
                ),
                "Class Flip": (
                    "Yes"
                    if flipped
                    else "No"
                ),
                "Confidence": (
                    f"{prediction['confidence'] * 100:.2f}%"
                ),
                "Δ Confidence": (
                    f"{inpainting_confidence_change(original_prediction, prediction):+.4f}"
                ),
                "Entropy": (
                    f"{prediction['entropy']:.4f}"
                ),
                "Δ Entropy": (
                    f"{inpainting_entropy_change(original_prediction, prediction):+.4f}"
                ),
                "PSNR": (
                    f"{metrics['psnr']:.3f}"
                ),
                "SSIM": (
                    f"{metrics['ssim']:.4f}"
                ),
                "MAE": (
                    f"{metrics['mae']:.4f}"
                ),
            }
        )

    return rows


def run_inpainting_analysis(
    uploaded,
    mask_type: str,
    mask_ratio: float,
    placement: str,
    selected_method: str,
) -> None:
    """Run one fair-mask inpainting comparison on the uploaded image."""
    image_bytes = uploaded.getvalue()
    filename = uploaded.name
    mime = guess_mime(
        filename,
        getattr(
            uploaded,
            "type",
            None,
        ),
    )

    st.session_state.inpainting_domain_validation = None
    st.session_state.inpainting_result = None
    st.session_state.inpainting_error = None

    with st.spinner(
        "Validating input domain..."
    ):
        try:
            domain_result = call_validate_domain(
                image_bytes,
                filename,
                mime,
            )

        except RuntimeError as exc:
            st.session_state.inpainting_error = str(
                exc
            )
            return

    st.session_state.inpainting_domain_validation = (
        domain_result
    )

    if not domain_result[
        "supported"
    ]:
        return

    with st.spinner(
        "Running controlled masking, Grad-CAM placement, and inpainting comparison..."
    ):
        try:
            (
                device,
                classifier,
                unet_model,
            ) = load_inpainting_runtime()

            original_image = prepare_inpainting_image(
                image_bytes
            )

            original_prediction = predict_inpainting_image(
                model=classifier,
                image=original_image,
                device=device,
            )

            heatmap = generate_gradcam(
                model=classifier,
                image=original_image,
                device=device,
                target_class=original_prediction[
                    "predicted_index"
                ],
            )

            center = None
            if placement in (
                "high_attention",
                "low_attention",
            ):
                center = attention_center(
                    heatmap,
                    placement,
                )

            rng = np.random.default_rng(
                42
            )

            mask = make_mask(
                kind=mask_type,
                ratio=mask_ratio,
                rng=rng,
                size=INPAINTING_IMAGE_SIZE,
                center=center,
            )

            method_results = {}

            for method in INPAINTING_METHODS:
                reconstructed = run_inpainter(
                    method=method,
                    image=original_image,
                    mask=mask,
                    unet_model=unet_model,
                    device=device,
                )

                prediction = predict_inpainting_image(
                    model=classifier,
                    image=reconstructed,
                    device=device,
                )

                metrics = evaluate_reconstruction(
                    original=original_image,
                    reconstructed=reconstructed,
                    mask=mask,
                )

                method_results[
                    method
                ] = {
                    "reconstructed": reconstructed,
                    "prediction": prediction,
                    "metrics": metrics,
                }

            st.session_state.inpainting_result = {
                "original_image": original_image,
                "original_prediction": original_prediction,
                "heatmap": heatmap,
                "center": center,
                "mask": mask,
                "mask_type": mask_type,
                "mask_ratio_target": mask_ratio,
                "mask_ratio_actual": mask_coverage(
                    mask
                ),
                "placement": placement,
                "selected_method": selected_method,
                "method_results": method_results,
            }

        except Exception as exc:
            st.session_state.inpainting_error = (
                f"Inpainting analysis failed: {exc}"
            )


def render_inpainting_result(
    result: dict,
) -> None:
    """Render one interactive EXP-016 result and all-method comparison."""
    original_image = result[
        "original_image"
    ]

    original_prediction = result[
        "original_prediction"
    ]

    selected_method = result[
        "selected_method"
    ]

    selected = result[
        "method_results"
    ][
        selected_method
    ]

    reconstructed = selected[
        "reconstructed"
    ]

    reconstructed_prediction = selected[
        "prediction"
    ]

    metrics = selected[
        "metrics"
    ]

    flipped = inpainting_class_flip(
        original_prediction,
        reconstructed_prediction,
    )

    confidence_delta = inpainting_confidence_change(
        original_prediction,
        reconstructed_prediction,
    )

    entropy_delta = inpainting_entropy_change(
        original_prediction,
        reconstructed_prediction,
    )

    section_title(
        "Controlled Missing-Information Experiment",
        (
            "The same synthetic mask is reused across all reconstruction methods "
            "so their outputs can be compared fairly."
        ),
    )

    # A full-width summary strip avoids narrow cards and long-value overlap.
    placement_label = INPAINTING_PLACEMENT_LABELS[
        result["placement"]
    ]

    render_html(
        '<div class="card" style="padding:0.95rem 1.1rem;">'
        '<div style="display:flex;flex-wrap:wrap;gap:12px 28px;'
        'align-items:center;">'
        f'<span><span class="metric-label">Mask Shape</span> '
        f'<b>{html.escape(result["mask_type"].title())}</b></span>'
        f'<span><span class="metric-label">Placement</span> '
        f'<b>{html.escape(placement_label)}</b></span>'
        f'<span><span class="metric-label">Target Missing Area</span> '
        f'<b>{result["mask_ratio_target"] * 100:.0f}%</b></span>'
        f'<span><span class="metric-label">Actual Missing Area</span> '
        f'<b>{result["mask_ratio_actual"] * 100:.1f}%</b></span>'
        '</div>'
        '</div>'
    )

    st.write("")

    original_pil = Image.fromarray(
        original_image,
        mode="L",
    ).convert(
        "RGB"
    )

    masked_preview = inpainting_mask_preview(
        original_image,
        result[
            "mask"
        ],
    )

    reconstructed_pil = Image.fromarray(
        reconstructed,
        mode="L",
    ).convert(
        "RGB"
    )

    cam_preview = inpainting_gradcam_preview(
        original_image,
        result[
            "heatmap"
        ],
        result[
            "center"
        ],
    )

    # Use a 2 × 2 image grid. Four columns are too narrow on laptop screens.
    image_top_left, image_top_right = st.columns(
        2,
        gap="large",
    )

    with image_top_left:
        render_html(
            '<div class="img-head">Original</div>'
        )
        st.image(
            original_pil,
            width="stretch",
        )

    with image_top_right:
        render_html(
            '<div class="img-head">Masked Input</div>'
        )
        st.image(
            masked_preview,
            width="stretch",
        )

    image_bottom_left, image_bottom_right = st.columns(
        2,
        gap="large",
    )

    with image_bottom_left:
        render_html(
            f'<div class="img-head">'
            f'{html.escape(INPAINTING_METHOD_LABELS[selected_method])} '
            f'Reconstruction</div>'
        )
        st.image(
            reconstructed_pil,
            width="stretch",
        )

    with image_bottom_right:
        render_html(
            '<div class="img-head">Grad-CAM Attention</div>'
        )
        st.image(
            cam_preview,
            width="stretch",
        )

    render_html(
        '<div class="disclaimer">'
        '<b>Attention interpretation:</b> High- and low-attention placements are '
        'defined from Grad-CAM model influence. They are not physical defect '
        'ground truth and do not establish causal defect location.'
        '</div>'
    )

    section_title(
        "Decision Preservation",
        (
            "Compare the original ResNet-50 decision with the selected "
            "reconstruction method."
        ),
    )

    # Stack the before/after decisions vertically so nothing competes for width.
    original_class = format_class_name(
        original_prediction[
            "predicted_class"
        ]
    )

    reconstructed_class = format_class_name(
        reconstructed_prediction[
            "predicted_class"
        ]
    )

    selected_method_label = INPAINTING_METHOD_LABELS[
        selected_method
    ]

    render_html(
        '<div class="card" style="margin-bottom:0.75rem;">'
        '<div class="metric-label">Before Reconstruction</div>'
        f'<div class="metric-value compact" style="margin-top:0.45rem;">'
        f'{html.escape(original_class)}</div>'
        f'<div class="metric-note" style="margin-top:0.55rem;">'
        f'Confidence <b>{pct(original_prediction["confidence"])}</b>'
        f' &nbsp;&nbsp;•&nbsp;&nbsp; '
        f'Entropy <b>{original_prediction["entropy"]:.4f}</b>'
        '</div>'
        '</div>'
    )

    render_html(
        '<div style="text-align:center;color:#8fa0ba;'
        'font-size:1.25rem;margin:0.2rem 0 0.75rem 0;">↓</div>'
    )

    render_html(
        '<div class="card" style="margin-bottom:0.9rem;">'
        f'<div class="metric-label">After {html.escape(selected_method_label)}</div>'
        f'<div class="metric-value compact" style="margin-top:0.45rem;">'
        f'{html.escape(reconstructed_class)}</div>'
        f'<div class="metric-note" style="margin-top:0.55rem;">'
        f'Confidence <b>{pct(reconstructed_prediction["confidence"])}</b>'
        f' &nbsp;&nbsp;•&nbsp;&nbsp; '
        f'Entropy <b>{reconstructed_prediction["entropy"]:.4f}</b>'
        '</div>'
        '</div>'
    )

    if flipped:
        render_html(
            '<div class="callout high" style="margin-top:0.4rem;">'
            '<div class="title">Class Flip Detected</div>'
            '<div class="body">'
            'The predicted defect class changed after reconstruction, '
            'indicating decision instability for this condition.'
            '</div>'
            f'<div class="fine">Selected method: '
            f'{html.escape(selected_method_label)}</div>'
            '</div>'
        )
    else:
        render_html(
            '<div class="callout low" style="margin-top:0.4rem;">'
            '<div class="title">Class Preserved</div>'
            '<div class="body">'
            'The predicted defect class remained unchanged after '
            'reconstruction for this condition.'
            '</div>'
            f'<div class="fine">Selected method: '
            f'{html.escape(selected_method_label)}</div>'
            '</div>'
        )

    # Keep change metrics on separate rows rather than small cards.
    render_html(
        '<div class="card" style="margin-top:0.9rem;padding:0.95rem 1.1rem;">'
        '<div style="display:flex;flex-wrap:wrap;gap:12px 32px;">'
        f'<span><span class="metric-label">Confidence Change</span> '
        f'<b>{confidence_delta:+.4f}</b></span>'
        f'<span><span class="metric-label">Entropy Change</span> '
        f'<b>{entropy_delta:+.4f}</b></span>'
        '</div>'
        '</div>'
    )

    section_title(
        "Reconstruction Quality",
        (
            "Higher PSNR/SSIM and lower MAE indicate closer pixel reconstruction. "
            "These metrics do not guarantee classifier stability."
        ),
    )

    # Three short numeric metrics fit safely here.
    r1, r2, r3 = st.columns(
        3,
        gap="medium",
    )

    with r1:
        st.metric(
            "PSNR",
            f"{metrics['psnr']:.3f} dB",
        )

    with r2:
        st.metric(
            "SSIM",
            f"{metrics['ssim']:.4f}",
        )

    with r3:
        st.metric(
            "MAE",
            f"{metrics['mae']:.4f}",
        )

    section_title(
        "Compare All Reconstruction Methods",
        (
            "All four methods receive the exact same synthetic missing-region mask."
        ),
    )

    st.dataframe(
        build_inpainting_comparison_rows(
            original_prediction,
            result[
                "method_results"
            ],
        ),
        width="stretch",
        hide_index=True,
    )

    render_html(
        '<div class="note">'
        '<b>Key interpretation:</b> A reconstruction can score well on PSNR, SSIM, '
        'or MAE while still changing the classifier decision. Pixel reconstruction '
        'quality and downstream decision reliability are therefore evaluated separately.'
        '</div>'
    )

def render_inpainting_research_findings() -> None:
    section_title(
        "EXP-016 Full-Test Findings",
        (
            "Aggregated results from 27,540 cases across 255 held-out test images, "
            "three mask shapes, three mask sizes, three placements, and four reconstruction methods."
        ),
    )

    left, right = st.columns(
        2,
        gap="large",
    )

    with left:
        render_html(
            '<div class="img-head">Placement Effect</div>'
        )

        st.dataframe(
            INPAINTING_FINAL_PLACEMENT_RESULTS,
            width="stretch",
            hide_index=True,
        )

    with right:
        render_html(
            '<div class="img-head">Method Effect</div>'
        )

        st.dataframe(
            INPAINTING_FINAL_METHOD_RESULTS,
            width="stretch",
            hide_index=True,
        )

    render_html(
        '<div class="callout mid">'
        '<div class="title">Main Reliability Finding</div>'
        '<div class="body">'
        'High-attention masking produced substantially more class flips than '
        'random or low-attention masking. At the same time, the simple mean-fill '
        'baseline preserved classifier decisions more often than the visually '
        'stronger reconstruction methods. This supports the broader SteelVision '
        'finding that visual reconstruction quality and downstream AI reliability '
        'are not the same objective.'
        '</div>'
        '</div>'
    )


def render_inpainting_mode(
    backend_online: bool,
) -> None:
    section_title(
        "Inpainting Reliability",
        (
            "Remove controlled regions from a supported steel-defect image, reconstruct "
            "the missing pixels, and test whether the classifier decision remains stable."
        ),
    )

    render_html(
        '<div class="disclaimer" style="margin-top:0;">'
        '<b>Research Mode:</b> Missing regions are created synthetically, so the original '
        'image remains available as reconstruction reference. This experiment measures '
        'robustness under controlled information loss; it does not prove that genuinely '
        'missing real-world defect information can be recovered.'
        '</div>'
    )

    uploaded, _ = render_upload(
        backend_online,
        "inpainting_uploader",
    )

    if uploaded is None:
        render_inpainting_research_findings()
        return

    section_title(
        "Experiment Settings"
    )

    c1, c2 = st.columns(
        2,
        gap="large",
    )

    with c1:
        mask_type = st.selectbox(
            "Mask Shape",
            options=INPAINTING_MASK_TYPES,
            format_func=lambda value: value.title(),
            key="inpainting_mask_type",
            help=(
                "Block removes a rectangular region, line creates scratch-like missing "
                "regions, and blob creates irregular missing areas."
            ),
        )

        mask_ratio = st.select_slider(
            "Missing Area",
            options=INPAINTING_MASK_RATIOS,
            value=0.20,
            format_func=lambda value: f"{value * 100:.0f}%",
            key="inpainting_mask_ratio",
            help="Approximate fraction of the 224×224 experimental image removed.",
        )

    with c2:
        placement = st.selectbox(
            "Mask Placement",
            options=INPAINTING_PLACEMENTS,
            format_func=lambda value: INPAINTING_PLACEMENT_LABELS[
                value
            ],
            key="inpainting_placement",
            help=(
                "High/low attention uses the original ResNet-50 Grad-CAM heatmap. "
                "Attention means model influence, not confirmed physical defect location."
            ),
        )

        selected_method = st.selectbox(
            "Primary Reconstruction Method",
            options=INPAINTING_METHODS,
            index=3,
            format_func=lambda value: INPAINTING_METHOD_LABELS[
                value
            ],
            key="inpainting_method",
            help=(
                "The selected method is shown in the primary reconstruction panel. "
                "All four methods are still evaluated on the same mask for comparison."
            ),
        )

    if st.button(
        "Run Inpainting Reliability Analysis",
        type="primary",
        key="inpainting_analyze",
        disabled=not backend_online,
    ):
        run_inpainting_analysis(
            uploaded,
            mask_type,
            mask_ratio,
            placement,
            selected_method,
        )

    if st.session_state.inpainting_error:
        st.error(
            st.session_state.inpainting_error
        )

    domain_result = (
        st.session_state
        .inpainting_domain_validation
    )

    if domain_result:
        render_domain_validation(
            domain_result
        )

        if not domain_result[
            "supported"
        ]:
            render_inpainting_research_findings()
            return

    if st.session_state.inpainting_result:
        render_inpainting_result(
            st.session_state.inpainting_result
        )

    render_inpainting_research_findings()


# -----------------------------------------------------------------------------
# Model details / footer / state
# -----------------------------------------------------------------------------

def render_about() -> None:
    st.write(
        ""
    )

    with st.expander(
        "Model Details"
    ):
        st.markdown(
            """
**Input Domain Validator:** MobileNetV3-Small binary supported-vs-unsupported model  
**Domain Acceptance Threshold:** 0.80 supported-domain probability  
**Classifier:** Fine-tuned ResNet-50  
**Deployment Runtime:** FP32 ONNX Runtime  
**Detector:** YOLO steel-defect detector  
**Explainability:** Grad-CAM using the final ResNet convolutional block  
**Localization:** Bounding boxes mapped back to uploaded-image coordinates  
**Supported Classes:** 6 steel surface defect categories

**Domain Validation:** The input validator is evaluated before downstream analysis. Inputs below the supported-domain threshold are rejected before ResNet-50 classification, Grad-CAM generation, or YOLO localization. The validator is a learned research gate and should not be interpreted as a universal steel/non-steel detector.

**Model Integration:** ResNet-50 provides the primary image-level defect classification. YOLO provides supporting bounding-box localization. Matching YOLO detections are shown as primary localization evidence, while non-matching detections remain visible as additional detector observations.

**Resolution Experiment:** Simulation modes intentionally adjust the image to the selected experimental resolution before detection in order to evaluate model behavior under changed visual detail. These modes do not estimate camera distance and do not guarantee recovery of information that may be missing from the original image.

**YOLO Ground Truth:** When a Standard Analysis upload matches a labelled YOLO validation image, the dashboard shows its annotated ground-truth bounding box beside the YOLO localization output. Ground truth is never invented for arbitrary uploaded images. YOLO performs bounding-box detection, not segmentation.

**Inpainting Reliability:** EXP-016 synthetically removes 10%, 20%, or 40% of the image using block, line, or blob masks placed randomly or according to ResNet-50 Grad-CAM attention. Mean Fill, Telea, Navier–Stokes, and a domain-specific U-Net reconstruct the same missing region. Reconstruction quality (PSNR, SSIM, MAE) is evaluated separately from downstream decision preservation (class flip, confidence change, entropy change). Grad-CAM placement indicates model influence, not physical defect ground truth, and inpainting does not guarantee recovery of genuinely missing real-world information.
            """
        )


def render_footer() -> None:
    render_html(
        """
        <div class="footer">
          <b>SteelVision</b><br>
          Research Prototype for Steel Surface Defect Analysis<br>
          MobileNetV3 • ResNet-50 • YOLO • U-Net • ONNX • FastAPI • Grad-CAM
        </div>
        """
    )


def init_state() -> None:
    # Reset the old standalone YOLO workflow selection after removing that page.
    if st.session_state.get("analysis_mode") == "YOLO Evaluation":
        st.session_state.pop("analysis_mode", None)

    defaults = {
        "standard_domain_validation":
            None,

        "standard_result":
            None,

        "standard_gradcam":
            None,

        "standard_gradcam_error":
            None,

        "standard_localization":
            None,

        "standard_original_bytes":
            None,

        "standard_error":
            None,

        "experiment_domain_validation":
            None,

        "experiment_original":
            None,

        "experiment_simulated":
            None,

        "experiment_error":
            None,

        "inpainting_domain_validation":
            None,

        "inpainting_result":
            None,

        "inpainting_error":
            None,
    }

    for (
        key,
        value,
    ) in defaults.items():
        st.session_state.setdefault(
            key,
            value,
        )


def format_mode_label(
    mode: str,
) -> str:
    return (
        f"**{mode}**  \n"
        f"{MODE_DESCRIPTIONS[mode]}"
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    init_state()

    st.markdown(
        CUSTOM_CSS,
        unsafe_allow_html=True,
    )

    backend_online = (
        render_sidebar()
    )

    render_hero()

    mode = st.radio(
        "Workflow",
        [
            MODE_STANDARD,
            MODE_EXPERIMENT,
            MODE_INPAINTING,
        ],
        format_func=format_mode_label,
        horizontal=False,
        label_visibility="collapsed",
        key="analysis_mode",
    )

    if mode == MODE_STANDARD:
        render_standard_mode(
            backend_online
        )

    elif mode == MODE_EXPERIMENT:
        render_experiment_mode(
            backend_online
        )

    else:
        render_inpainting_mode(
            backend_online
        )

    render_about()
    render_footer()


main()