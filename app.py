from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "final_fixed_green_clahe_80_10_10_runtime_results" / "models" / "best_phase2.keras"
TEST_METRICS_PATH = (
        BASE_DIR
        / "final_fixed_green_clahe_80_10_10_runtime_results"
        / "evaluation"
        / "reports"
)

IMG_SIZE = 300
BATCH_SIZE = 1
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
GRADE_DESCRIPTIONS = {
    "No DR": "No visible signs of diabetic retinopathy are indicated by the model in this image.",
    "Mild": (
        "The image shows patterns that may be consistent with mild diabetic retinopathy. "
        "Mild changes can include early retinal abnormalities and may require monitoring "
        "and professional evaluation."
    ),
    "Moderate": (
        "The image shows patterns that may be consistent with moderate diabetic retinopathy. "
        "More noticeable retinal changes may be present and professional eye evaluation is recommended."
    ),
    "Severe": (
        "The image shows patterns that may be consistent with severe diabetic retinopathy. "
        "Significant retinal changes may be present and prompt professional eye evaluation is recommended."
    ),
    "Proliferative DR": (
        "The image shows patterns that may be consistent with proliferative diabetic retinopathy, "
        "an advanced stage in which abnormal blood vessel growth may occur. Professional eye "
        "evaluation is strongly recommended."
    ),
}
DEFAULT_GRADCAM_TARGET_LAYER = "top_activation"


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner="Preparing analysis model...")
def load_model() -> tf.keras.Model:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")
    return tf.keras.models.load_model(str(MODEL_PATH), compile=False)


@st.cache_data
def load_test_metrics() -> dict:
    return load_json(TEST_METRICS_PATH)


def apply_fixed_green_channel_clahe(img_rgb: np.ndarray) -> np.ndarray:
    r, g, b = cv2.split(img_rgb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g_clahe = clahe.apply(g)
    return cv2.merge([r, g_clahe, b])


def decode_uploaded_image(uploaded_bytes: bytes) -> np.ndarray:
    encoded = np.frombuffer(uploaded_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode the uploaded image. Use JPG, PNG, BMP, TIFF, or WEBP.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def preprocess_uploaded_image(img_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    resized_rgb = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    preprocessed_rgb = apply_fixed_green_channel_clahe(resized_rgb)
    batch = np.expand_dims(preprocessed_rgb.astype(np.float32), axis=0)
    return resized_rgb, batch


def unpack_predictions(preds) -> np.ndarray:
    if isinstance(preds, dict):
        grade_probs = preds["grade_output"]
    else:
        grade_probs = preds
    grade_probs = np.asarray(grade_probs)
    if grade_probs.ndim != 2 or grade_probs.shape[1] != len(CLASS_NAMES):
        raise ValueError(
            f"Expected grade probabilities with shape (batch, {len(CLASS_NAMES)}), "
            f"got {grade_probs.shape}."
        )
    return grade_probs


def get_nested_efficientnet(model: tf.keras.Model) -> tf.keras.Model:
    base_model = model.get_layer("efficientnetb3")
    if not isinstance(base_model, tf.keras.Model):
        raise TypeError("Layer 'efficientnetb3' exists but is not a nested Keras model.")
    return base_model


@st.cache_resource(show_spinner=False)
def build_gradcam_models(_model: tf.keras.Model, target_layer_name: str = DEFAULT_GRADCAM_TARGET_LAYER):
    base_model = get_nested_efficientnet(_model)
    target_layer = base_model.get_layer(target_layer_name)
    conv_feature_model = tf.keras.Model(base_model.inputs, target_layer.output, name="gradcam_feature_model")

    feature_input = tf.keras.Input(shape=target_layer.output.shape[1:], name="gradcam_feature_input")
    x = feature_input
    start_forwarding = False
    for layer in base_model.layers:
        if start_forwarding:
            x = layer(x, training=False)
        if layer.name == target_layer_name:
            start_forwarding = True

    for layer_name in [
        "global_average_pooling2d",
        "batch_normalization",
        "dropout",
        "dense",
        "batch_normalization_1",
        "dropout_1",
        "grade_output",
    ]:
        layer = _model.get_layer(layer_name)
        if isinstance(layer, (tf.keras.layers.BatchNormalization, tf.keras.layers.Dropout)):
            x = layer(x, training=False)
        else:
            x = layer(x)

    grade_from_features_model = tf.keras.Model(
        feature_input,
        x,
        name="gradcam_grade_from_features_model",
    )
    return conv_feature_model, grade_from_features_model


def compute_gradcam_overlay(
        display_rgb: np.ndarray,
        image_batch: np.ndarray,
        conv_feature_model: tf.keras.Model,
        grade_from_features_model: tf.keras.Model,
        target_class_idx: int,
        alpha: float = 0.4,
) -> np.ndarray:
    with tf.GradientTape() as tape:
        conv_outputs = conv_feature_model(image_batch, training=False)
        tape.watch(conv_outputs)
        grade_probs = grade_from_features_model(conv_outputs, training=False)
        target_score = grade_probs[:, target_class_idx]

    grads = tape.gradient(target_score, conv_outputs)
    if grads is None:
        raise RuntimeError("Grad-CAM gradient is None. Check target layer connectivity.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)
    heatmap = tf.where(max_value > 0, heatmap / max_value, heatmap).numpy()

    heatmap_resized = cv2.resize(heatmap, (display_rgb.shape[1], display_rgb.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    return np.uint8(np.clip((1.0 - alpha) * display_rgb + alpha * heatmap_color, 0, 255))


def prediction_summary(grade_probs: np.ndarray) -> dict:
    pred_label = int(np.argmax(grade_probs[0]))
    pred_class = CLASS_NAMES[pred_label]
    return {
        "display_label": pred_label,
        "display_class": pred_class,
        "confidence": float(grade_probs[0, pred_label]),
        "referable_probability": float(grade_probs[0, 2:].sum()),
        "referable_prediction": bool(grade_probs[0, 2:].sum() >= 0.5),
    }


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
            :root {
                --dr-primary: #0f766e;
                --dr-primary-dark: #115e59;
                --dr-accent: #2563eb;
                --dr-bg: #f7fafc;
                --dr-card: #ffffff;
                --dr-border: #e2e8f0;
                --dr-muted: #64748b;
                --dr-text: #0f172a;
            }

            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(15, 118, 110, 0.08), transparent 34rem),
                    linear-gradient(180deg, #f8fbfd 0%, #f7fafc 42%, #ffffff 100%);
                color: var(--dr-text);
            }

            .block-container {
                max-width: 1180px;
                padding-top: 2.2rem;
                padding-bottom: 4rem;
            }

            h1, h2, h3 {
                letter-spacing: -0.03em;
            }

            h2 {
                margin-top: 0.4rem;
            }

            [data-testid="stMetric"] {
                background: #ffffff;
                border: 1px solid var(--dr-border);
                border-radius: 18px;
                padding: 1rem 1.1rem;
                box-shadow: 0 10px 28px rgba(15, 23, 42, 0.04);
            }

            [data-testid="stMetricLabel"] {
                color: var(--dr-muted);
            }

            [data-testid="stMetricValue"] {
                color: var(--dr-primary-dark);
                font-weight: 760;
            }


            [data-testid="stFileUploader"] {
                background: #ffffff;
                border: 1px dashed #94a3b8;
                border-radius: 18px;
                padding: 1.1rem;
            }

            [data-testid="stFileUploader"] section {
                background: #f8fafc;
                border-radius: 14px;
            }

            .dr-hero {
                background: linear-gradient(135deg, #ffffff 0%, #eefbf8 100%);
                border: 1px solid var(--dr-border);
                border-radius: 28px;
                padding: 2.35rem;
                box-shadow: 0 24px 60px rgba(15, 23, 42, 0.07);
                margin-bottom: 1.4rem;
            }

            .dr-eyebrow {
                display: inline-flex;
                align-items: center;
                gap: 0.45rem;
                color: var(--dr-primary-dark);
                background: rgba(15, 118, 110, 0.10);
                border: 1px solid rgba(15, 118, 110, 0.18);
                border-radius: 999px;
                padding: 0.35rem 0.75rem;
                font-size: 0.82rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.07em;
                margin-bottom: 1rem;
            }

            .dr-hero h1 {
                font-size: clamp(2.1rem, 5vw, 4.1rem);
                line-height: 1.02;
                margin: 0;
                color: #0f172a;
            }

            .dr-hero p {
                max-width: 760px;
                color: #475569;
                font-size: 1.08rem;
                line-height: 1.72;
                margin: 1rem 0 0;
            }

            .dr-pill-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.65rem;
                margin-top: 1.35rem;
            }

            .dr-pill {
                background: #ffffff;
                color: #334155;
                border: 1px solid var(--dr-border);
                border-radius: 999px;
                padding: 0.55rem 0.8rem;
                font-size: 0.9rem;
                box-shadow: 0 8px 18px rgba(15, 23, 42, 0.04);
            }

            .dr-card {
                background: var(--dr-card);
                border: 1px solid var(--dr-border);
                border-radius: 22px;
                padding: 1.25rem;
                box-shadow: 0 18px 44px rgba(15, 23, 42, 0.055);
                margin-bottom: 1rem;
            }

            .dr-section-label {
                color: var(--dr-primary-dark);
                font-size: 0.82rem;
                font-weight: 800;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                margin-bottom: 0.35rem;
            }

            .dr-result {
                background: linear-gradient(135deg, rgba(15, 118, 110, 0.10), rgba(37, 99, 235, 0.07));
                border: 1px solid rgba(15, 118, 110, 0.18);
                border-radius: 22px;
                padding: 1.3rem;
                margin-bottom: 1rem;
            }

            .dr-result .label {
                color: #475569;
                font-size: 0.86rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.08em;
            }

            .dr-result .grade {
                color: #0f172a;
                font-size: 2.1rem;
                font-weight: 800;
                letter-spacing: -0.04em;
                margin-top: 0.25rem;
            }

            .dr-note {
                color: #475569;
                font-size: 0.96rem;
                line-height: 1.62;
            }

            .dr-small {
                color: var(--dr-muted);
                font-size: 0.88rem;
                line-height: 1.55;
            }

            /* File uploader button */
            [data-testid="stFileUploader"] button {
                background-color: #ffffff !important;
                color: #1f2937 !important;
                border: 1px solid #cbd5e1 !important;
                border-radius: 8px !important;
                font-weight: 600 !important;
            }

            [data-testid="stFileUploader"] button:hover {
                background-color: #f8fafc !important;
                color: #111827 !important;
                border-color: #94a3b8 !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <section class="dr-hero">
            <div class="dr-eyebrow">retinal image screening</div>
            <h1>Diabetic retinopathy detection from fundus images</h1>
            <p>
                Upload a retinal fundus image to receive a diabetic retinopathy grade estimate,
                probability breakdown, and a visual model explanation for the predicted class.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="dr-card">
            <div class="dr-section-label">Ready for analysis</div>
            <h3 style="margin-top:0;">Upload a retinal image to begin</h3>
            <p class="dr-note">
                The app will show the predicted DR category, confidence level, class probabilities,
                and an attention view after one image is uploaded.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="RetinaDR Screening", page_icon="🩺", layout="wide")
    inject_custom_css()
    render_hero()

    st.markdown('<div class="dr-section-label">Image analysis</div>', unsafe_allow_html=True)
    st.markdown("### Upload a retinal fundus image")
    st.caption("Supported formats: JPG, PNG, BMP, TIFF, and WEBP. One image is analyzed at a time.")

    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )

    try:
        model = load_model()
        test_metrics = load_test_metrics()
    except Exception as exc:
        st.error(f"Could not initialize app: {exc}")
        st.stop()

    if uploaded_file is None:
        render_empty_state()
        return

    try:
        original_rgb = decode_uploaded_image(uploaded_file.getvalue())
        display_rgb, image_batch = preprocess_uploaded_image(original_rgb)
        preds = model.predict(image_batch, batch_size=BATCH_SIZE, verbose=0)
        grade_probs = unpack_predictions(preds)
        summary = prediction_summary(grade_probs)
    except Exception as exc:
        st.error(f"Inference failed: {exc}")
        st.stop()

    st.markdown("---")
    st.markdown('<div class="dr-section-label">Analysis result</div>', unsafe_allow_html=True)
    st.markdown("## Analysis Result")

    result_left, result_right = st.columns([0.95, 1.05], gap="large")
    with result_left:
        st.markdown(
            f"""
            <div class="dr-result">
                <div class="label">Prediction</div>
                <div class="grade">{summary["display_class"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with result_right:
        st.metric("Confidence", f"{summary['confidence'] * 100:.1f}%")
        st.markdown("### What does this result mean?")
        st.write(GRADE_DESCRIPTIONS[summary["display_class"]])

    prob_df = pd.DataFrame(
        {
            "DR grade": CLASS_NAMES,
            "Probability": grade_probs[0],
        }
    )

    st.markdown('<div class="dr-section-label">Probability distribution</div>', unsafe_allow_html=True)
    st.markdown("## Class probabilities")
    st.caption(
        "The chart shows how strongly the model associated the image with each diabetic retinopathy category."
    )
    st.bar_chart(prob_df.set_index("DR grade"), y="Probability", height=280)

    st.markdown('<div class="dr-section-label">Model attention</div>', unsafe_allow_html=True)
    st.markdown("## How the Model Focused on the Image")
    st.caption(
        "The highlighted areas show which parts of the retinal image model focused on when making its prediction."
    )
    try:
        conv_feature_model, grade_from_features_model = build_gradcam_models(
            model, DEFAULT_GRADCAM_TARGET_LAYER
        )
        overlay = compute_gradcam_overlay(
            display_rgb,
            image_batch,
            conv_feature_model,
            grade_from_features_model,
            target_class_idx=summary["display_label"],
        )
        img_col, attention_col = st.columns(2, gap="medium")
        with img_col:
            st.image(display_rgb, caption="Original image", width=300)
        with attention_col:
            st.image(overlay, caption="Model attention", width=300)
    except Exception as exc:
        st.warning(f"Model attention visualization could not be generated: {exc}")


if __name__ == "__main__":
    main()
