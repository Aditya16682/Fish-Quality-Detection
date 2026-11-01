
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
import hashlib

st.set_page_config(
    page_title="FishIQ | AI Fish Quality Detection",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = Path("models/fish_quality.keras")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #081525, #10283b);
    color: #f2f7fc;
}

[data-testid="stSidebar"] {
    background: #0b1d30;
    border-right: 1px solid #1d4055;
}

[data-testid="stSidebar"] * {
    color: #e6f3fa !important;
}

.main .block-container {
    padding: 2rem 3rem;
    max-width: 1500px;
}

.hero {
    background: linear-gradient(120deg, #103c52, #087e8b);
    padding: 32px;
    border-radius: 22px;
    border: 1px solid #2b9eaa;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 42px;
    margin: 0;
    font-weight: 800;
}

.hero p {
    color: #d5f5f5;
    font-size: 16px;
    margin-top: 10px;
}

.section-title {
    color: #58d8d2;
    font-size: 23px;
    font-weight: 750;
    margin-top: 15px;
    margin-bottom: 15px;
}

.metric-card {
    background: linear-gradient(145deg, #142d43, #1b4053);
    border: 1px solid #2c6072;
    padding: 22px;
    border-radius: 17px;
    min-height: 120px;
}

.metric-label {
    color: #b8d4e1;
    font-size: 13px;
    font-weight: 600;
}

.metric-value {
    color: #ffffff;
    font-size: 27px;
    font-weight: 800;
    margin-top: 12px;
}

.upload-card {
    background: #10283b;
    border: 1px solid #286077;
    padding: 20px;
    border-radius: 18px;
}

.result-fresh {
    background: linear-gradient(135deg, #075b49, #0c886b);
    border: 1px solid #31d6a5;
    padding: 25px;
    border-radius: 18px;
    text-align: center;
}

.result-spoiled {
    background: linear-gradient(135deg, #702d3b, #b43e50);
    border: 1px solid #ff8291;
    padding: 25px;
    border-radius: 18px;
    text-align: center;
}

.result-title {
    font-size: 32px;
    color: white;
    font-weight: 800;
}

.result-subtitle {
    color: #ffffff;
    font-size: 15px;
}

.info-card {
    background: #122c40;
    border: 1px solid #28536a;
    border-radius: 15px;
    padding: 18px;
}

div.stButton > button {
    background: linear-gradient(90deg, #087e8b, #13b8a6);
    color: white;
    border: none;
    border-radius: 12px;
    height: 48px;
    font-weight: 750;
    width: 100%;
}

div.stButton > button:hover {
    background: linear-gradient(90deg, #0b9da8, #28d6bd);
    color: white;
    border: none;
}

[data-testid="stFileUploader"] {
    background: #f8fafc !important;
    padding: 15px !important;
    border-radius: 12px !important;
    border: 1px solid #94a3b8 !important;
}

[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] label p,
[data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] section div,
[data-testid="stFileUploader"] section span,
[data-testid="stFileUploader"] small {
    color: #111827 !important;
}

[data-testid="stFileUploader"] section {
    background: #f8fafc !important;
}

[data-testid="stFileUploader"] section button {
    background: #e2e8f0 !important;
    color: #111827 !important;
    border: 1px solid #64748b !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
}

[data-testid="stFileUploader"] section button:hover {
    background: #cbd5e1 !important;
    color: #000000 !important;
}

[data-testid="stFileUploader"] small {
    color: #334155 !important;
}

.stProgress > div > div > div > div {
    background: #21c9b3;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error("Trained model not found!")
    st.info("Run: python src/train.py")
    st.stop()

try:
    model = load_model()
except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.stop()


with st.sidebar:
    st.markdown("# 🐟 FishIQ")
    st.caption("SMART FISH QUALITY ANALYTICS")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Dashboard",
            "🔬 About Model",
            "📊 Project Information"
        ]
    )

    st.divider()

    st.markdown("### 🧠 Active Model")
    st.success("MobileNetV2")

    st.markdown("### ⚙️ Configuration")
    st.write("Image Size: 224 × 224")
    st.write("Classes: Fresh / Spoiled")
    st.write("Output: Binary Classification")

    st.divider()

    st.caption("Developed as an academic deep learning project.")


st.markdown("""
<div class="hero">
    <h1>🐟 FishIQ Analytics</h1>
    <p>
        Intelligent Fish Freshness Detection
        using Deep Learning and Computer Vision.
    </p>
    <span style="
        background:#0a5968;
        padding:7px 13px;
        border-radius:20px;
        color:#ffffff;
        font-size:12px;">
        AI POWERED • IMAGE CLASSIFICATION
    </span>
</div>
""", unsafe_allow_html=True)


if page == "🏠 Dashboard":

    st.markdown(
        '<div class="section-title">📈 System Overview</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    cards = [
        (c1, "🧠 MODEL", "MobileNetV2"),
        (c2, "🖼️ INPUT SIZE", "224 × 224"),
        (c3, "🏷️ CLASSES", "2 Classes"),
        (c4, "⚡ TASK", "Classification"),
    ]

    for col, label, value in cards:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">🔍 Fish Image Analysis</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns([1.05, 0.95], gap="large")

    with left:

        st.markdown("""
        <div class="upload-card">
            <h3>📤 Upload Fish Image</h3>
            <p style="color:#b8d4e1;">
                Upload a JPG, JPEG or PNG image
                to analyze its visual freshness category.
            </p>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Choose your fish image",
            type=["jpg", "jpeg", "png"],
            key="fish_upload"
        )

        if uploaded_file is not None:

            image_bytes = uploaded_file.getvalue()
            image_id = hashlib.sha256(image_bytes).hexdigest()

            try:
                image = Image.open(uploaded_file).convert("RGB")

                st.image(
                    image,
                    caption="Uploaded Fish Image",
                    use_container_width=True
                )

                if st.button(
                    "🔍 ANALYZE FISH QUALITY",
                    use_container_width=True
                ):

                    with st.spinner("AI model is analyzing the image..."):

                        resized = image.resize((224, 224))

                        image_array = np.asarray(
                            resized,
                            dtype=np.float32
                        )

                        image_array = np.expand_dims(
                            image_array,
                            axis=0
                        )

                        image_array = (
                            tf.keras.applications.mobilenet_v2
                            .preprocess_input(image_array)
                        )

                        prediction = float(
                            model.predict(
                                image_array,
                                verbose=0
                            )[0][0]
                        )

                        if prediction >= 0.5:
                            label = "Spoiled"
                            confidence = prediction * 100
                        else:
                            label = "Fresh"
                            confidence = (1 - prediction) * 100

                        st.session_state["fish_result"] = {
                            "label": label,
                            "confidence": confidence,
                            "image_id": image_id
                        }

            except Exception as error:
                st.error(f"Image processing failed: {error}")

    with right:

        st.markdown(
            '<div class="section-title">📊 Prediction Results</div>',
            unsafe_allow_html=True
        )

        result = st.session_state.get("fish_result")

        current_image_id = (
            hashlib.sha256(uploaded_file.getvalue()).hexdigest()
            if uploaded_file is not None else None
        )

        if result is not None and result["image_id"] == current_image_id:

            label = result["label"]
            confidence = result["confidence"]

            if label == "Fresh":

                st.markdown("""
                <div class="result-fresh">
                    <div style="font-size:45px;">✅</div>
                    <div class="result-title">FRESH</div>
                    <div class="result-subtitle">
                        Model prediction: Fresh
                    </div>
                </div>
                """, unsafe_allow_html=True)

            else:

                st.markdown("""
                <div class="result-spoiled">
                    <div style="font-size:45px;">⚠️</div>
                    <div class="result-title">SPOILED</div>
                    <div class="result-subtitle">
                        Model prediction: Spoiled
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.write("### 🎯 Prediction Confidence")
            st.markdown(f"**{confidence:.2f}%**")

            st.progress(
                min(1.0, max(0.0, confidence / 100))
            )

            st.caption(
                "Confidence is the model's output estimate, "
                "not a guarantee of correctness."
            )

        else:

            st.markdown("""
            <div class="info-card">
                <div style="font-size:38px;">🔬</div>
                <h3 style="color:#58d8d2;">
                    Waiting for Analysis
                </h3>
                <p style="color:#b8d4e1;">
                    Upload an image and click Analyze Fish Quality
                    to see the prediction here.
                </p>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### ⚠️ Food Safety Disclaimer")

    st.warning(
        "This is an experimental academic image-classification "
        "project. Visual predictions cannot establish whether "
        "fish is safe to consume. Do not use this application "
        "as the sole basis for food-safety decisions."
    )


elif page == "🔬 About Model":

    st.markdown(
        '<div class="section-title">🧠 About the AI Model</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-card">
        <h2 style="color:#58d8d2;">MobileNetV2</h2>
        <p>
            MobileNetV2 is a lightweight convolutional
            neural network architecture used for efficient
            image feature extraction.
        </p>
        <p>
            This project uses transfer learning with pretrained
            ImageNet weights and a binary classification head.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("### ⚙️ Model Pipeline")

    st.markdown("""
    1. Input fish image
    2. Resize to 224 × 224
    3. MobileNetV2 preprocessing
    4. Feature extraction
    5. Global average pooling
    6. Dropout layer
    7. Sigmoid classification output
    """)

    st.info(
        "The application loads models/fish_quality.keras. "
        "This is your existing MobileNetV2 model."
    )


elif page == "📊 Project Information":

    st.markdown(
        '<div class="section-title">📚 Project Information</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-card">
        <h3 style="color:#58d8d2;">
            Fish Freshness Detection Using Deep Learning
        </h3>
        <p>
            The project investigates image-based classification
            of fish samples into fresh and spoiled categories.
        </p>
        <p><b>Architecture:</b> MobileNetV2</p>
        <p><b>Framework:</b> TensorFlow / Keras</p>
        <p><b>Interface:</b> Streamlit</p>
        <p><b>Image Processing:</b> Pillow and NumPy</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("### 🎯 Project Objectives")

    st.markdown("""
    - Develop an image-based fish classification system.
    - Apply transfer learning.
    - Compare deep learning architectures.
    - Evaluate model performance using suitable metrics.
    - Build an interactive prediction interface.
    """)

    st.write("### 📌 Academic Project")
    st.caption(
        "Deep Learning | Computer Vision | Image Classification"
    )


st.divider()

st.markdown("""
<div style="
    text-align:center;
    color:#91b5c5;
    padding:15px;">
    <b>🐟 FishIQ</b> | Fish Quality Detection<br>
    Academic Deep Learning Project | MobileNetV2
</div>
""", unsafe_allow_html=True)
