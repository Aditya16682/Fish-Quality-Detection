
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Fish Quality Detection",
    page_icon="🐟",
    layout="centered",
)

MODEL_PATH = Path("models/fish_quality.keras")

# ---------------- CUSTOM UI ----------------
st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #777777;
        margin-bottom: 28px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">🐟 Fish Quality Detection</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered experimental fish image classification'
    '</div>',
    unsafe_allow_html=True,
)

st.divider()

# ---------------- SESSION STATE ----------------
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "uploaded_image_id" not in st.session_state:
    st.session_state.uploaded_image_id = None

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error("Trained model not found.")
    st.info("Run: python src/train.py")
    st.stop()

try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()

# ---------------- IMAGE UPLOAD ----------------
st.subheader("Upload Fish Image")

uploaded_file = st.file_uploader(
    "Choose a fish image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:

    # Reset old result if a different image is uploaded
    image_id = (
        uploaded_file.name,
        uploaded_file.size,
    )

    if st.session_state.uploaded_image_id != image_id:
        st.session_state.prediction_result = None
        st.session_state.uploaded_image_id = image_id

    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Fish Image",
            width=450,
        )

        # ---------------- PREDICTION BUTTON ----------------
        if st.button(
            "🔍 Predict Fish Quality",
            type="primary",
            use_container_width=True,
        ):
            try:
                with st.spinner("Analyzing fish image..."):

                    resized = image.resize((224, 224))

                    image_array = np.asarray(
                        resized,
                        dtype=np.float32,
                    )

                    image_array = np.expand_dims(
                        image_array,
                        axis=0,
                    )

                    # Same preprocessing as MobileNetV2 training
                    image_array = (
                        tf.keras.applications.mobilenet_v2
                        .preprocess_input(image_array)
                    )

                    prediction = float(
                        model.predict(
                            image_array,
                            verbose=0,
                        )[0][0]
                    )

                    # Class order: fresh = 0, spoiled = 1
                    if prediction >= 0.5:
                        label = "Spoiled"
                        confidence = prediction * 100
                    else:
                        label = "Fresh"
                        confidence = (1 - prediction) * 100

                    # Save result so it survives Streamlit reruns
                    st.session_state.prediction_result = {
                        "label": label,
                        "confidence": confidence,
                    }

            except Exception as error:
                st.session_state.prediction_result = None
                st.error(f"Prediction failed: {error}")

        # ---------------- DISPLAY RESULT ----------------
        result = st.session_state.prediction_result

        if result is not None:
            st.divider()
            st.subheader("📊 Prediction Result")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    label="Predicted Class",
                    value=result["label"],
                )

            with col2:
                st.metric(
                    label="Model Confidence",
                    value=f'{result["confidence"]:.2f}%',
                )

            st.progress(
                min(
                    100,
                    max(
                        0,
                        int(round(result["confidence"])),
                    ),
                )
            )

            if result["label"] == "Fresh":
                st.success("Model prediction: Fresh")
            else:
                st.error("Model prediction: Spoiled")

            st.caption(
                "Confidence is the model's output estimate, "
                "not a guarantee that the prediction is correct."
            )

    except Exception as error:
        st.error(f"Could not process this image: {error}")

else:
    # Clear result when no image is selected
    st.session_state.prediction_result = None
    st.session_state.uploaded_image_id = None

    st.info("Upload a fish image to get started.")

# ---------------- DISCLAIMER ----------------
st.divider()

st.warning(
    "Food safety notice: This is an experimental image "
    "classification project. The prediction cannot establish "
    "whether fish is safe to eat. Do not rely on this app alone "
    "to make food-safety decisions."
)

st.caption(
    "Academic Project | Deep Learning | MobileNetV2"
)
