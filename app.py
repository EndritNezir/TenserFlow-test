import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ---------------- CONFIG ---------------- #

MODEL_PATH = "model/best_model.keras"
CLASS_NAMES_PATH = "model/class_names.txt"
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.35

st.set_page_config(
    page_title="Visual Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- STYLING ---------------- #

st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", sans-serif;
}

.stApp {
    background: linear-gradient(180deg, #f5f5f7 0%, #ffffff 100%);
    color: #111111;
}

.block-container {
    max-width: 1240px;
    padding-top: 1.8rem;
    padding-bottom: 2rem;
}

.hero {
    padding: 2.4rem;
    border-radius: 30px;
    background: rgba(255, 255, 255, 0.95);
    border: 1px solid rgba(0,0,0,0.06);
    box-shadow: 0 20px 55px rgba(0,0,0,0.08);
    margin-bottom: 1.4rem;
}

.badge {
    display: inline-block;
    padding: 0.42rem 0.95rem;
    border-radius: 999px;
    background: #f2f2f2;
    color: #444;
    font-size: 0.82rem;
    margin-bottom: 1rem;
    border: 1px solid rgba(0,0,0,0.05);
}

.hero-title {
    font-size: 3.15rem;
    font-weight: 700;
    color: #111111;
    line-height: 1.02;
    margin-bottom: 0.8rem;
    letter-spacing: -0.04em;
}

.hero-subtitle {
    font-size: 1.08rem;
    color: #4b4b4b;
    max-width: 760px;
    line-height: 1.7;
}

.card {
    background: rgba(255, 255, 255, 0.97);
    border-radius: 26px;
    padding: 1.35rem;
    border: 1px solid rgba(0,0,0,0.06);
    box-shadow: 0 10px 30px rgba(0,0,0,0.06);
    margin-bottom: 1rem;
}

.section-title {
    font-size: 1.03rem;
    font-weight: 650;
    margin-bottom: 0.8rem;
    color: #161616;
}

.kpi {
    background: rgba(255,255,255,0.98);
    border-radius: 22px;
    padding: 1rem 1.1rem;
    border: 1px solid rgba(0,0,0,0.05);
    box-shadow: 0 8px 22px rgba(0,0,0,0.04);
    min-height: 108px;
}

.kpi-label {
    color: #6b6b6b;
    font-size: 0.88rem;
    margin-bottom: 0.35rem;
}

.kpi-value {
    color: #111111;
    font-size: 1.85rem;
    font-weight: 700;
    letter-spacing: -0.03em;
}

.kpi-sub {
    color: #666;
    font-size: 0.86rem;
    margin-top: 0.2rem;
}

.main-prediction {
    font-size: 2.3rem;
    font-weight: 700;
    color: #111111;
    letter-spacing: -0.03em;
    margin-bottom: 0.3rem;
}

.secondary-text {
    color: #666;
    font-size: 0.97rem;
    margin-bottom: 1rem;
}

.footer {
    text-align: center;
    color: #777;
    font-size: 0.9rem;
    margin-top: 2rem;
}

/* Upload box */

[data-testid="stFileUploader"] {
    border: 1px dashed rgba(0,0,0,0.16);
    border-radius: 18px;
    background: #fafafa;
    padding: 0.5rem;
}

/* Drag & drop text */

[data-testid="stFileUploaderDropzone"] * {
    color: #111111 !important;
}

/* All uploader text */

[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] div,
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] p {
    color: #111111 !important;
}

/* Browse button */

[data-testid="stFileUploader"] section button,
[data-testid="stBaseButton-secondary"] {
    background: #ffffff !important;
    color: #111111 !important;
    border: 1px solid rgba(0,0,0,0.14) !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
}

/* Hover effect */

[data-testid="stFileUploader"] section button:hover,
[data-testid="stBaseButton-secondary"]:hover {
    background: #f3f4f6 !important;
    color: #000000 !important;
    border: 1px solid rgba(0,0,0,0.18) !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HELPERS ---------------- #

def format_label(label: str) -> str:
    mapping = {
        "telephone": "Phone"
    }
    return mapping.get(label, label.replace("_", " ").title())


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_names():
    with open(CLASS_NAMES_PATH, "r") as f:
        return [line.strip() for line in f.readlines()]


def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = tf.keras.applications.efficientnet.preprocess_input(image_array)
    return image_array

# ---------------- HEADER ---------------- #

st.markdown("""
<div class="hero">
    <div class="badge">Machine Learning • Computer Vision • Premium Demo</div>
    <div class="hero-title">Visual Intelligence</div>
    <div class="hero-subtitle">
        A premium image classification experience powered by deep learning.
        Upload a photo to inspect top predictions, confidence scores, and class probabilities
        in a polished Apple-style interface.
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------- MODEL CHECK ---------------- #

if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_NAMES_PATH):
    st.error("Model files are missing.")
    st.info("Run `python3 train.py` first.")
    st.stop()

model = load_model()
class_names = load_class_names()

# ---------------- KPI ROW ---------------- #

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown("""
    <div class="kpi">
        <div class="kpi-label">Model</div>
        <div class="kpi-value">Ready</div>
        <div class="kpi-sub">Loaded successfully</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="kpi">
        <div class="kpi-label">Classes</div>
        <div class="kpi-value">{len(class_names)}</div>
        <div class="kpi-sub">Supported categories</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown("""
    <div class="kpi">
        <div class="kpi-label">Input</div>
        <div class="kpi-value">224×224</div>
        <div class="kpi-sub">Image resolution</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown("""
    <div class="kpi">
        <div class="kpi-label">Threshold</div>
        <div class="kpi-value">0.35</div>
        <div class="kpi-sub">Low-confidence guard</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- UPLOAD SECTION ---------------- #

left_top, right_top = st.columns([1.2, 0.8])

with left_top:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Upload Image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Select an image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    st.caption("Accepted formats: JPG, JPEG, PNG")
    st.markdown('</div>', unsafe_allow_html=True)

with right_top:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Supported Categories</div>', unsafe_allow_html=True)
    st.write(", ".join(format_label(name) for name in class_names))
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- MAIN LOGIC ---------------- #

if uploaded_file is None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">How it works</div>', unsafe_allow_html=True)
    st.write(
        "Upload an image and the model will analyse it, produce a top prediction, "
        "rank the most likely categories, and show a confidence distribution."
    )
    st.markdown('</div>', unsafe_allow_html=True)
else:
    image = Image.open(uploaded_file)

    with st.spinner("Analysing image..."):
        processed = preprocess_image(image)
        predictions = model.predict(processed, verbose=0)[0]

    predicted_index = int(np.argmax(predictions))
    best_prob = float(np.max(predictions))
    predicted_label = format_label(class_names[predicted_index])

    sorted_predictions = sorted(
        zip(class_names, predictions),
        key=lambda x: x[1],
        reverse=True
    )

    col1, col2 = st.columns([1.05, 0.95])

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Preview</div>', unsafe_allow_html=True)
        st.image(image, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Prediction</div>', unsafe_allow_html=True)

        if best_prob < CONFIDENCE_THRESHOLD:
            st.markdown('<div class="main-prediction">Low Confidence Result</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="secondary-text">The model is not fully confident. Review the top predictions below.</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(f'<div class="main-prediction">{predicted_label}</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="secondary-text">Most likely class predicted by the model.</div>',
                unsafe_allow_html=True
            )

        st.progress(min(best_prob, 1.0))
        st.write(f"Confidence: **{best_prob * 100:.2f}%**")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Top 3 Predictions</div>', unsafe_allow_html=True)
        for i, (label, prob) in enumerate(sorted_predictions[:3], start=1):
            st.write(f"**{i}. {format_label(label)}** — {prob * 100:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)

    lower_left, lower_right = st.columns([1, 1])

    with lower_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Probability Distribution</div>', unsafe_allow_html=True)
        chart_data = {
            format_label(label): float(prob)
            for label, prob in zip(class_names, predictions)
        }
        st.bar_chart(chart_data)
        st.markdown('</div>', unsafe_allow_html=True)

    with lower_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Confidence Breakdown</div>', unsafe_allow_html=True)
        for label, prob in sorted_predictions:
            nice_label = format_label(label)
            st.write(f"**{nice_label}**")
            st.progress(min(float(prob), 1.0))
            st.caption(f"{prob * 100:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ---------------- #

st.markdown("""
<div class="footer">
Built by Endrit Nezir • Visual Intelligence • TensorFlow + Streamlit
</div>
""", unsafe_allow_html=True)