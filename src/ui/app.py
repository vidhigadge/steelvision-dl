"""
SteelVision — Streamlit frontend

Place at: src/ui/app.py
Run from project root:
    streamlit run src/ui/app.py
"""

import base64
import html
import mimetypes
from io import BytesIO

import requests
import streamlit as st
from PIL import Image, UnidentifiedImageError


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

API_URL = "http://127.0.0.1:8000"

HEALTH_TIMEOUT = 3
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
    "Classification",
    "Uncertainty Estimation",
    "Grad-CAM Explainability",
    "Defect Localization",
    "Coordinate Mapping",
]

MODE_STANDARD = "Standard Analysis"
MODE_EXPERIMENT = "Resolution Experiment"

MODE_DESCRIPTIONS = {
    MODE_STANDARD: (
        "Analyze the uploaded image using the standard classification, "
        "explainability, and defect localization workflow."
    ),
    MODE_EXPERIMENT: "Evaluate how changing image resolution affects defect localization.",
}

ENTROPY_HELP = (
    "Entropy is used as a model uncertainty indicator. Lower values generally indicate "
    "more concentrated model confidence, while higher values indicate greater uncertainty."
    "\n\n"
    "UI thresholds: < 0.5 low, 0.5–1.0 moderate, > 1.0 elevated. These are "
    "interface-level interpretation thresholds and are not universal scientific thresholds."
)

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
    radial-gradient(1100px 500px at 85% -10%, rgba(56,189,248,0.10), transparent 60%),
    radial-gradient(900px 500px at -10% 0%, rgba(99,102,241,0.10), transparent 55%),
    var(--bg);
  color: var(--text);
}

header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1180px; }
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0c1424 0%, #0a101d 100%);
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text); }

.hero {
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 2.1rem 2.3rem;
  margin-bottom: 1.4rem;
  background:
    linear-gradient(135deg, rgba(56,189,248,0.10), rgba(99,102,241,0.08) 55%, rgba(15,23,42,0.4)),
    repeating-linear-gradient(115deg, rgba(255,255,255,0.025) 0 2px, transparent 2px 14px),
    #0f1a30;
  box-shadow: 0 18px 50px rgba(0,0,0,0.35);
}
.hero-badge {
  display: inline-flex; align-items: center; gap: 8px;
  padding: 5px 12px; border-radius: 999px;
  font-size: 0.74rem; letter-spacing: 0.08em; text-transform: uppercase;
  font-weight: 600; color: var(--accent);
  background: rgba(56,189,248,0.10);
  border: 1px solid rgba(56,189,248,0.30);
}
.hero-badge .dot {
  width: 7px; height: 7px; border-radius: 50%;
  background: var(--accent); box-shadow: 0 0 10px var(--accent);
}
.hero h1 {
  margin: 0.85rem 0 0.15rem 0;
  font-size: 2.9rem; font-weight: 800; letter-spacing: -0.02em;
  background: linear-gradient(90deg, #f1f5f9 0%, #7dd3fc 60%, #a5b4fc 100%);
  -webkit-background-clip: text; background-clip: text;
  color: transparent !important;
}
.hero .subtitle { font-size: 1.15rem; font-weight: 500; color: #cbd5e1; }
.hero-feat-label {
  margin-top: 1.2rem; font-size: 0.72rem; text-transform: uppercase;
  letter-spacing: 0.1em; color: var(--muted); font-weight: 700;
}
.chips { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 0.6rem; }
.chip {
  display: inline-flex; align-items: center; gap: 8px;
  padding: 8px 15px; border-radius: 10px; font-size: 0.88rem; font-weight: 600;
  color: #e2e8f0; background: rgba(56,189,248,0.07);
  border: 1px solid rgba(56,189,248,0.22);
}
.chip::before {
  content: ""; width: 6px; height: 6px; border-radius: 50%;
  background: var(--accent); box-shadow: 0 0 8px var(--accent);
}

.section-title {
  display: flex; align-items: center; gap: 10px;
  margin: 1.9rem 0 0.7rem 0; font-size: 1.15rem; font-weight: 700;
}
.section-title .bar {
  width: 4px; height: 20px; border-radius: 3px;
  background: linear-gradient(180deg, var(--accent), var(--accent-2));
}
.section-sub { color: var(--muted); font-size: 0.88rem; margin: -0.35rem 0 0.9rem 0; }

[data-testid="stRadio"] label p { font-size: 1rem; line-height: 1.55; color: #e2e8f0; }
[data-testid="stRadio"] [role="radiogroup"] { gap: 0.9rem; }

/* Streamlit metrics: keep readable on the dark theme */
[data-testid="stMetric"] {
  background: linear-gradient(180deg, var(--panel-2), var(--panel));
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
[data-testid="stMetricDelta"] { color: var(--muted) !important; }

.card {
  background: linear-gradient(180deg, var(--panel-2), var(--panel));
  border: 1px solid var(--border); border-radius: 16px;
  padding: 1.15rem 1.25rem; box-shadow: 0 8px 28px rgba(0,0,0,0.28);
}
.metric-label { font-size: 0.73rem; text-transform: uppercase; letter-spacing: 0.09em; color: var(--muted); font-weight: 600; }
.metric-value { font-size: 1.8rem; font-weight: 800; margin-top: 0.3rem; letter-spacing: -0.02em; color: var(--text); }
.metric-value.compact { font-size: 1.3rem; line-height: 1.35; }
.metric-note { font-size: 0.8rem; color: var(--muted); margin-top: 0.25rem; }
.accent-text { color: var(--accent); }

.pbar { height: 9px; background: rgba(148,170,205,0.14); border-radius: 999px; overflow: hidden; }
.pbar > div { height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--accent-2), var(--accent)); }
.pbar.thin { height: 7px; }

.callout {
  border-radius: 14px; padding: 1rem 1.2rem;
  border: 1px solid var(--border); border-left-width: 4px; background: var(--panel);
}
.callout.low { border-left-color: var(--good); }
.callout.mid { border-left-color: var(--warn); }
.callout.high { border-left-color: var(--danger); }
.callout .title { font-weight: 700; margin-bottom: 0.2rem; }
.callout .body { color: #b4c0d4; font-size: 0.92rem; line-height: 1.5; }
.callout .fine { color: var(--muted); font-size: 0.76rem; margin-top: 0.5rem; }

.img-head {
  font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.09em;
  color: var(--muted); font-weight: 600; margin-bottom: 0.5rem;
}
[data-testid="stImage"] img { border-radius: 12px; border: 1px solid var(--border); }
.note { color: var(--muted); font-size: 0.88rem; line-height: 1.5; margin-top: 0.6rem; }
.disclaimer {
  margin-top: 1rem; padding: 0.8rem 1rem; border-radius: 12px;
  background: rgba(251,191,36,0.07); border: 1px solid rgba(251,191,36,0.25);
  color: #e5d3a0; font-size: 0.86rem; line-height: 1.5;
}

.rank-row {
  display: grid; grid-template-columns: 44px 1fr 90px; gap: 14px;
  align-items: center; padding: 0.82rem 0; border-bottom: 1px solid var(--border);
}
.rank-row:last-child { border-bottom: none; }
.rank-badge {
  width: 34px; height: 34px; border-radius: 10px; display: flex;
  align-items: center; justify-content: center; font-weight: 800;
  background: rgba(148,170,205,0.10); color: var(--muted);
}
.rank-badge.first { background: linear-gradient(135deg, var(--accent-2), var(--accent)); color: #06101d; }
.rank-name { font-weight: 600; margin-bottom: 6px; color: var(--text); }
.rank-prob { text-align: right; font-weight: 700; font-variant-numeric: tabular-nums; color: var(--text); }

.side-title {
  font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.1em;
  color: var(--muted); font-weight: 700; margin: 1.2rem 0 0.5rem 0;
}
.kv { display: flex; justify-content: space-between; padding: 6px 0; font-size: 0.88rem; border-bottom: 1px dashed var(--border); }
.kv span:first-child { color: var(--muted); }
.status-pill {
  display: inline-flex; align-items: center; gap: 8px; padding: 6px 12px;
  border-radius: 999px; font-size: 0.85rem; font-weight: 600;
}
.status-pill.on { background: rgba(52,211,153,0.10); color: var(--good); border: 1px solid rgba(52,211,153,0.3); }
.status-pill.off { background: rgba(248,113,113,0.10); color: #f87171; border: 1px solid rgba(248,113,113,0.3); }
.defect-item { padding: 5px 0; font-size: 0.88rem; color: #cbd5e1; }
.defect-item::before { content: "▸ "; color: var(--accent); }

[data-testid="stFileUploader"] section {
  background: rgba(17,26,45,0.7); border: 1.5px dashed rgba(56,189,248,0.35);
  border-radius: 14px; padding: 1.2rem;
}
[data-testid="stFileUploader"] section:hover { border-color: var(--accent); }
.stButton > button, .stButton > button[kind="primary"] {
  width: 100%; border: none; border-radius: 12px; padding: 0.75rem 1.2rem;
  font-weight: 700; font-size: 1rem; color: #05101d;
  background: linear-gradient(90deg, #38bdf8, #818cf8);
  box-shadow: 0 8px 24px rgba(56,189,248,0.25);
}
.stButton > button:hover { transform: translateY(-1px); color: #05101d; box-shadow: 0 12px 30px rgba(56,189,248,0.35); }
.stButton > button:disabled { opacity: 0.45; box-shadow: none; }
[data-testid="stExpander"] { background: var(--panel); border: 1px solid var(--border); border-radius: 14px; }

.footer {
  margin-top: 3rem; padding-top: 1.2rem; border-top: 1px solid var(--border);
  text-align: center; color: var(--muted); font-size: 0.82rem; line-height: 1.7;
}
.footer b { color: #cbd5e1; }
#MainMenu, footer { visibility: hidden; }
</style>
"""


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def render_html(markup: str) -> None:
    """Render HTML through Markdown. Indentation and blank lines are stripped so
    Markdown never interprets any part of the markup as a code block."""
    cleaned = "\n".join(line.strip() for line in markup.strip().splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


def section_title(text: str, subtitle: str | None = None) -> None:
    render_html(
        '<div class="section-title">'
        '<span class="bar"></span>'
        f'{html.escape(text)}'
        '</div>'
    )
    if subtitle:
        render_html(f'<div class="section-sub">{html.escape(subtitle)}</div>')


def format_class_name(raw: str) -> str:
    key = str(raw).strip().lower()
    return CLASS_DISPLAY_NAMES.get(key, key.replace("_", " ").title())


def pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def clamp_pct_width(value: float) -> float:
    return max(0.0, min(100.0, value * 100))


def guess_mime(filename: str, declared: str | None) -> str:
    if declared and declared.startswith("image/"):
        return declared
    guessed, _ = mimetypes.guess_type(filename)
    return guessed or "image/jpeg"


def validate_image(image_bytes: bytes) -> Image.Image:
    try:
        probe = Image.open(BytesIO(image_bytes))
        probe.verify()
        image = Image.open(BytesIO(image_bytes))
        image.load()
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ValueError("The uploaded file could not be opened as a valid image.") from exc


def decode_base64_image(value: str) -> Image.Image:
    try:
        return Image.open(BytesIO(base64.b64decode(value))).convert("RGB")
    except Exception as exc:
        raise RuntimeError("The backend returned an invalid encoded image.") from exc


@st.cache_data(ttl=3, show_spinner=False)
def check_backend() -> tuple[bool, str]:
    try:
        response = requests.get(f"{API_URL}/health", timeout=HEALTH_TIMEOUT)
        if response.status_code != 200:
            return False, f"Health check returned HTTP {response.status_code}."
        payload = response.json()
        if str(payload.get("status", "")).lower() == "ok":
            return True, "Backend is online."
        return False, "Backend responded but did not report status 'ok'."
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to the FastAPI backend."
    except requests.exceptions.Timeout:
        return False, "Health check timed out."
    except requests.exceptions.RequestException as exc:
        return False, f"Health check failed: {exc}"


def _post_file(endpoint: str, image_bytes: bytes, filename: str, mime: str, timeout: int, data=None):
    try:
        response = requests.post(
            f"{API_URL}/{endpoint}",
            files={"file": (filename, image_bytes, mime)},
            data=data,
            timeout=timeout,
        )
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(f"Could not connect to /{endpoint}. Make sure the FastAPI backend is running.") from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(f"The /{endpoint} request timed out after {timeout} seconds.") from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"The /{endpoint} request failed: {exc}") from exc

    if response.status_code != 200:
        detail = response.text[:400].strip()
        raise RuntimeError(f"/{endpoint} returned HTTP {response.status_code}. {detail}")

    return response


def call_predict(image_bytes: bytes, filename: str, mime: str) -> dict:
    response = _post_file("predict", image_bytes, filename, mime, PREDICT_TIMEOUT)
    try:
        data = response.json()
        return {
            "filename": str(data.get("filename", filename)),
            "predicted_class": str(data["predicted_class"]),
            "confidence": float(data["confidence"]),
            "entropy": float(data["entropy"]),
            "top_3": [
                {
                    "class_name": str(item["class_name"]),
                    "probability": float(item["probability"]),
                }
                for item in data["top_3"]
            ],
        }
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise RuntimeError("/predict returned a response in an unexpected format.") from exc


def call_explain(image_bytes: bytes, filename: str, mime: str) -> bytes:
    response = _post_file("explain", image_bytes, filename, mime, EXPLAIN_TIMEOUT)
    try:
        Image.open(BytesIO(response.content)).verify()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise RuntimeError("/explain did not return a valid image.") from exc
    return response.content


def call_localize(image_bytes: bytes, filename: str, mime: str, resolution: str) -> dict:
    response = _post_file(
        "localize",
        image_bytes,
        filename,
        mime,
        LOCALIZE_TIMEOUT,
        data={"resolution": resolution},
    )
    try:
        data = response.json()
        return {
            "filename": str(data.get("filename", filename)),
            "resolution": str(data["resolution"]),
            "original_size": list(data["original_size"]),
            "processed_size": list(data["processed_size"]),
            "detection_count": int(data["detection_count"]),
            "detections": list(data["detections"]),
            "low_res_image": decode_base64_image(data["low_res_image"]),
            "processed_image": decode_base64_image(data["processed_image"]),
            "mapped_image": decode_base64_image(data["mapped_image"]),
        }
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise RuntimeError("/localize returned a response in an unexpected format.") from exc


def uncertainty_level(entropy: float) -> tuple[str, str, str]:
    if entropy < ENTROPY_LOW:
        return (
            "low",
            "Low Uncertainty",
            "Model uncertainty is relatively low for this prediction.",
        )
    if entropy <= ENTROPY_HIGH:
        return (
            "mid",
            "Moderate Uncertainty",
            "Moderate model uncertainty. Review the secondary predictions alongside the primary result.",
        )
    return (
        "high",
        "Elevated Uncertainty",
        "Model uncertainty is elevated. Interpret this prediction cautiously.",
    )


def highest_detection_confidence(result: dict) -> float | None:
    detections = result.get("detections", [])
    if not detections:
        return None
    try:
        return max(float(item["confidence"]) for item in detections)
    except (KeyError, TypeError, ValueError):
        return None


def format_detection_summary(result: dict) -> str:
    """Return e.g. '1 detection · 81.72% confidence' or '0 detections · No confidence available'."""
    count = int(result.get("detection_count", 0))
    noun = "detection" if count == 1 else "detections"
    top_conf = highest_detection_confidence(result)
    if count == 0 or top_conf is None:
        return f"{count} {noun} · No confidence available"
    return f"{count} {noun} · {pct(top_conf)} confidence"


def build_detection_rows(detections: list) -> list:
    rows = []
    for idx, detection in enumerate(detections, start=1):
        rows.append(
            {
                "#": idx,
                "Defect": format_class_name(detection["class_name"]),
                "Confidence": f'{float(detection["confidence"]) * 100:.2f}%',
                "Processed Box": str(detection["processed_box"]),
                "Mapped Box": str(detection["mapped_box"]),
            }
        )
    return rows


# -----------------------------------------------------------------------------
# Header / sidebar
# -----------------------------------------------------------------------------

def render_hero() -> None:
    chips = "".join(f'<span class="chip">{html.escape(name)}</span>' for name in HERO_FEATURES)
    render_html(
        f"""
        <div class="hero">
          <span class="hero-badge"><span class="dot"></span>Research Prototype</span>
          <h1>SteelVision</h1>
          <div class="subtitle">AI-Powered Steel Surface Defect Analysis</div>
          <div class="hero-feat-label">Capabilities</div>
          <div class="chips">{chips}</div>
        </div>
        """
    )


def render_sidebar() -> bool:
    online, message = check_backend()
    with st.sidebar:
        render_html(
            """
            <div style="font-size:1.25rem;font-weight:800;letter-spacing:-0.01em;">SteelVision</div>
            <div style="color:#8fa0ba;font-size:0.82rem;margin-top:2px;">Steel Surface Defect Analysis</div>
            <div class="side-title">Backend Status</div>
            """
        )
        render_html(
            '<span class="status-pill on">● Online</span>'
            if online
            else '<span class="status-pill off">● Offline</span>'
        )
        if not online:
            st.caption(message)

        if st.button("Refresh status", key="refresh_status"):
            check_backend.clear()
            st.rerun()

        render_html(
            """
            <div class="side-title">System</div>
            <div class="kv"><span>Classifier</span><span>ResNet-50</span></div>
            <div class="kv"><span>Runtime</span><span>ONNX</span></div>
            <div class="kv"><span>Detector</span><span>YOLO</span></div>
            <div class="kv"><span>Explainability</span><span>Grad-CAM</span></div>
            <div class="kv"><span>Classes</span><span>6</span></div>
            <div class="side-title">Supported Defects</div>
            """
        )
        render_html(
            "".join(
                f'<div class="defect-item">{html.escape(name)}</div>'
                for name in CLASS_DISPLAY_NAMES.values()
            )
        )
    return online


# -----------------------------------------------------------------------------
# Shared upload UI
# -----------------------------------------------------------------------------

def render_upload(backend_online: bool, key: str):
    section_title("Upload Steel Surface Image", "Accepted formats: JPG, JPEG, PNG")

    if not backend_online:
        st.error(
            "The FastAPI backend appears to be offline. Start it in a separate terminal with "
            "`uvicorn src.api.main:app --reload`, then press **Refresh status** in the sidebar."
        )

    uploaded = st.file_uploader(
        "Upload Steel Surface Image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        key=key,
    )

    if uploaded is None:
        render_html('<div class="note">Select an image to begin. No request is sent until you click the analysis button.</div>')
        return None, None

    try:
        image = validate_image(uploaded.getvalue())
    except ValueError as exc:
        st.error(str(exc))
        return None, None

    left, right = st.columns([1.15, 1], gap="large")
    with left:
        render_html('<div class="img-head">Image Preview</div>')
        st.image(image, use_container_width=True)

    with right:
        width, height = image.size
        render_html(
            f"""
            <div class="card">
              <div class="metric-label">File</div>
              <div style="font-weight:600;margin:4px 0 14px 0;word-break:break-all;">{html.escape(uploaded.name)}</div>
              <div class="metric-label">Uploaded Dimensions</div>
              <div style="font-weight:600;margin:4px 0 14px 0;">{width} × {height} px</div>
              <div class="metric-label">Size</div>
              <div style="font-weight:600;margin:4px 0 0 0;">{uploaded.size / 1024:.1f} KB</div>
            </div>
            """
        )

    return uploaded, image


# -----------------------------------------------------------------------------
# Standard analysis
# -----------------------------------------------------------------------------

def run_standard_analysis(uploaded) -> None:
    image_bytes = uploaded.getvalue()
    filename = uploaded.name
    mime = guess_mime(filename, getattr(uploaded, "type", None))

    st.session_state.standard_result = None
    st.session_state.standard_gradcam = None
    st.session_state.standard_gradcam_error = None
    st.session_state.standard_localization = None
    st.session_state.standard_error = None

    with st.spinner("Running classification, explainability, and localization..."):
        try:
            prediction = call_predict(image_bytes, filename, mime)
            localization = call_localize(image_bytes, filename, mime, "original")
        except RuntimeError as exc:
            st.session_state.standard_error = str(exc)
            return

        st.session_state.standard_result = prediction
        st.session_state.standard_localization = localization
        st.session_state.standard_original_bytes = image_bytes

        try:
            st.session_state.standard_gradcam = call_explain(image_bytes, filename, mime)
        except RuntimeError as exc:
            st.session_state.standard_gradcam_error = str(exc)


def render_prediction_summary(result: dict) -> None:
    section_title("Prediction")
    predicted = format_class_name(result["predicted_class"])
    confidence = result["confidence"]
    entropy = result["entropy"]

    c1, c2, c3 = st.columns(3, gap="medium")
    cards = [
        (c1, "Defect", predicted, "Primary Classification", True),
        (c2, "Confidence", pct(confidence), "Probability of Defect", False),
        (c3, "Entropy", f"{entropy:.4f}", "Model Uncertainty Indicator", False),
    ]
    for col, label, value, note, accent in cards:
        with col:
            cls = "metric-value accent-text" if accent else "metric-value"
            render_html(
                f'<div class="card"><div class="metric-label">{html.escape(label)}</div>'
                f'<div class="{cls}">{html.escape(str(value))}</div>'
                f'<div class="metric-note">{html.escape(note)}</div></div>'
            )
            if label == "Entropy":
                st.caption("ⓘ About entropy", help=ENTROPY_HELP)

    render_html(
        f'<div style="margin-top:1rem;"><div style="display:flex;justify-content:space-between;font-size:0.82rem;color:#8fa0ba;margin-bottom:6px;">'
        f'<span>Confidence</span><span>{pct(confidence)}</span></div>'
        f'<div class="pbar"><div style="width:{clamp_pct_width(confidence):.2f}%;"></div></div></div>'
    )


def render_uncertainty(entropy: float) -> None:
    section_title("Uncertainty Interpretation")
    css_class, title, message = uncertainty_level(entropy)
    render_html(
        f"""
        <div class="callout {css_class}">
          <div class="title">{html.escape(title)}</div>
          <div class="body">{html.escape(message)} Entropy is an uncertainty indicator and does not guarantee that a prediction is correct.</div>
          <div class="fine">UI thresholds: &lt; {ENTROPY_LOW} low, {ENTROPY_LOW}–{ENTROPY_HIGH} moderate, &gt; {ENTROPY_HIGH} elevated. These are interface-level thresholds, not universal scientific thresholds.</div>
        </div>
        """
    )


def render_gradcam(original_bytes: bytes, gradcam_bytes: bytes | None, gradcam_error: str | None) -> None:
    section_title("Explainability", "Grad-CAM highlights image regions that influenced the classifier.")
    left, right = st.columns(2, gap="large")
    with left:
        render_html('<div class="img-head">Original Image</div>')
        st.image(Image.open(BytesIO(original_bytes)).convert("RGB"), use_container_width=True)
    with right:
        render_html('<div class="img-head">Grad-CAM Explanation</div>')
        if gradcam_bytes:
            st.image(Image.open(BytesIO(gradcam_bytes)).convert("RGB"), use_container_width=True)
        else:
            st.warning(gradcam_error or "Grad-CAM is not available for this image.")

    render_html(
        '<div class="disclaimer"><b>Interpretation:</b> highlighted regions show model influence, not proof of a physical defect mechanism.</div>'
    )


def render_top3(top_3: list) -> None:
    section_title("Top Defect Predictions")

    row_parts = []
    for idx, item in enumerate(top_3[:3], start=1):
        name = html.escape(format_class_name(item["class_name"]))
        prob = float(item["probability"])
        badge_class = "rank-badge first" if idx == 1 else "rank-badge"
        row_parts.append(
            '<div class="rank-row">'
            f'<div class="{badge_class}">{idx}</div>'
            '<div>'
            f'<div class="rank-name">{name}</div>'
            f'<div class="pbar thin"><div style="width:{clamp_pct_width(prob):.2f}%;"></div></div>'
            '</div>'
            f'<div class="rank-prob">{pct(prob)}</div>'
            '</div>'
        )

    card_html = (
        '<div class="card" style="padding-top:0.4rem;padding-bottom:0.4rem;">'
        + "".join(row_parts)
        + "</div>"
    )
    render_html(card_html)


def render_localization(result: dict) -> None:
    section_title("Defect Localization & Coordinate Mapping (Original Image)")

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        render_html('<div class="img-head">Input Image</div>')
        st.image(result["low_res_image"], use_container_width=True)
    with c2:
        render_html('<div class="img-head">Resized Input</div>')
        st.image(result["processed_image"], use_container_width=True)
    with c3:
        render_html('<div class="img-head">Detection Mapped</div>')
        st.image(result["mapped_image"], use_container_width=True)

    m1, m2, m3 = st.columns(3, gap="medium")
    with m1:
        st.metric("Identified Defect Regions", result["detection_count"])
    with m2:
        st.metric("Original Image Size", f'{result["original_size"][0]} × {result["original_size"][1]}')
    with m3:
        st.metric("Model Input Size", f'{result["processed_size"][0]} × {result["processed_size"][1]}')

    render_html('<div class="img-head" style="margin-top:1rem;">Defect Details</div>')
    if result["detections"]:
        st.dataframe(build_detection_rows(result["detections"]), use_container_width=True, hide_index=True)
    else:
        st.info("No defect regions were detected above the current confidence threshold.")

    render_html(
        '<div class="note">The processed image is resized to meet detector input requirements. This increases image dimensions without guaranteeing recovery of missing detail. Detected bounding boxes are then mapped back to the coordinate space of the uploaded image.</div>'
    )


def render_standard_mode(backend_online: bool) -> None:
    section_title("Standard Analysis")

    uploaded, _ = render_upload(backend_online, "standard_uploader")

    if uploaded is not None:
        if st.button(
            "Run SteelVision Analysis",
            type="primary",
            key="standard_analyze",
            disabled=not backend_online,
        ):
            run_standard_analysis(uploaded)

    if st.session_state.standard_error:
        st.error(st.session_state.standard_error)

    result = st.session_state.standard_result
    localization = st.session_state.standard_localization
    original_bytes = st.session_state.standard_original_bytes

    if result and localization and original_bytes:
        render_prediction_summary(result)
        render_uncertainty(result["entropy"])
        render_gradcam(
            original_bytes,
            st.session_state.standard_gradcam,
            st.session_state.standard_gradcam_error,
        )
        render_localization(localization)
        render_top3(result["top_3"])


# -----------------------------------------------------------------------------
# Resolution experiment
# -----------------------------------------------------------------------------

def run_resolution_experiment(uploaded, resolution: str) -> None:
    image_bytes = uploaded.getvalue()
    filename = uploaded.name
    mime = guess_mime(filename, getattr(uploaded, "type", None))

    st.session_state.experiment_original = None
    st.session_state.experiment_simulated = None
    st.session_state.experiment_error = None

    with st.spinner(f"Comparing original input with simulated {resolution} px input..."):
        try:
            st.session_state.experiment_original = call_localize(
                image_bytes, filename, mime, "original"
            )
            st.session_state.experiment_simulated = call_localize(
                image_bytes, filename, mime, resolution
            )
        except RuntimeError as exc:
            st.session_state.experiment_error = str(exc)


def render_detection_group(label: str, result: dict) -> None:
    summary = html.escape(format_detection_summary(result))
    render_html(
        '<div class="card">'
        f'<div class="metric-label">{html.escape(label)}</div>'
        f'<div class="metric-value compact">{summary}</div>'
        '</div>'
    )


def render_resolution_comparison(original: dict, simulated: dict) -> None:
    section_title(
        "Resolution Comparison",
        f'Compare the original image with a simulated {simulated["resolution"]}-pixel version of the same image.',
    )

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        render_html('<div class="img-head">Uploaded Image · Baseline</div>')
        st.image(original["low_res_image"], use_container_width=True)
    with c2:
        render_html(f'<div class="img-head">Simulated {html.escape(simulated["resolution"])} px Source</div>')
        st.image(simulated["low_res_image"], use_container_width=True)
    with c3:
        render_html('<div class="img-head">Mapped Detection After Simulation</div>')
        st.image(simulated["mapped_image"], use_container_width=True)

    st.write("")
    g1, g2 = st.columns(2, gap="medium")
    with g1:
        render_detection_group("Baseline Defect Detections", original)
    with g2:
        render_detection_group("Simulated Defect Detections", simulated)

    st.write("")
    left, right = st.columns(2, gap="large")
    with left:
        render_html('<div class="img-head">Baseline Defects</div>')
        st.image(original["mapped_image"], use_container_width=True)
    with right:
        render_html('<div class="img-head">Simulated Defects</div>')
        st.image(simulated["processed_image"], use_container_width=True)

    if simulated["detections"]:
        st.dataframe(build_detection_rows(simulated["detections"]), use_container_width=True, hide_index=True)
    else:
        st.info("No defect regions were detected in the simulated input.")

    render_html(
        '<div class="note">The simulation adjusts the image to the resolution selected by the user and then resizes it to the detector’s required input dimensions for inference. Changing the resolution may alter or reduce available visual detail. Therefore, a higher detector confidence after resolution adjustment should not necessarily be interpreted as improved image quality or detection reliability.</div>'
    )


def render_experiment_mode(backend_online: bool) -> None:
    section_title("Resolution Experiment")

    render_html(
        '<div class="disclaimer" style="margin-top:0;"><b>Research Mode:</b> This mode intentionally adjusts the uploaded image to study how the selected resolution changes localization.</div>'
    )

    uploaded, _ = render_upload(backend_online, "experiment_uploader")

    if uploaded is None:
        return

    resolution = st.selectbox(
        "Resolution to Experiment With",
        options=["224", "128", "64", "32"],
        index=1,
        help="The uploaded image is intentionally adjusted to this square resolution before detector processing.",
    )

    if st.button(
        "Run Experiment",
        type="primary",
        key="experiment_analyze",
        disabled=not backend_online,
    ):
        run_resolution_experiment(uploaded, resolution)

    if st.session_state.experiment_error:
        st.error(st.session_state.experiment_error)

    if st.session_state.experiment_original and st.session_state.experiment_simulated:
        render_resolution_comparison(
            st.session_state.experiment_original,
            st.session_state.experiment_simulated,
        )


# -----------------------------------------------------------------------------
# Model details / footer / state
# -----------------------------------------------------------------------------

def render_about() -> None:
    st.write("")
    with st.expander("Model Details"):
        st.markdown(
            """
**Classifier:** Fine-tuned ResNet-50  
**Deployment Runtime:** FP32 ONNX Runtime  
**Detector:** YOLO steel-defect detector  
**Explainability:** Grad-CAM using the final ResNet convolutional block  
**Localization:** Bounding boxes mapped back to uploaded-image coordinates  
**Supported Classes:** 6 steel surface defect categories

**Note:** Simulation modes are designed for controlled experimental analysis. They intentionally adjust the image to the selected experimental resolution before detection in order to evaluate model behavior under changed visual detail. These modes do not estimate camera distance and do not guarantee recovery of information that may be missing from the original image.
            """
        )


def render_footer() -> None:
    render_html(
        """
        <div class="footer">
          <b>SteelVision</b><br>
          Research Prototype for Steel Surface Defect Analysis<br>
          ResNet-50 • YOLO • ONNX • FastAPI • Grad-CAM
        </div>
        """
    )


def init_state() -> None:
    defaults = {
        "standard_result": None,
        "standard_gradcam": None,
        "standard_gradcam_error": None,
        "standard_localization": None,
        "standard_original_bytes": None,
        "standard_error": None,
        "experiment_original": None,
        "experiment_simulated": None,
        "experiment_error": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def format_mode_label(mode: str) -> str:
    return f"**{mode}**  \n{MODE_DESCRIPTIONS[mode]}"


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    init_state()
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    backend_online = render_sidebar()
    render_hero()

    mode = st.radio(
        "Workflow",
        [MODE_STANDARD, MODE_EXPERIMENT],
        format_func=format_mode_label,
        horizontal=False,
        label_visibility="collapsed",
        key="analysis_mode",
    )

    if mode == MODE_STANDARD:
        render_standard_mode(backend_online)
    else:
        render_experiment_mode(backend_online)

    render_about()
    render_footer()


main()