
import io
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image, UnidentifiedImageError
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as PDFImage
)


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Image Detection",
    page_icon="🔍",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "ai_image_detector.keras"
CONFIG_PATH = BASE_DIR / "deployment_config.json"


# --------------------------------------------------
# 2. CUSTOM DESIGN
# --------------------------------------------------

st.markdown("""
<style>
.stApp {
    background: #f5f8fc;
    color: #172b4d;
}
.main .block-container {
    max-width: 1150px;
    padding-top: 2rem;
}
.hero {
    background: linear-gradient(120deg, #102a56, #087e8b);
    padding: 38px;
    border-radius: 20px;
    color: white;
    margin-bottom: 25px;
}
.hero h1 {
    color: white;
    font-size: 38px;
}
.hero p {
    color: #e2f5ff;
    font-size: 17px;
}
.result-card {
    padding: 22px;
    border-radius: 15px;
    background: white;
    border: 1px solid #dce5f0;
    margin: 10px 0;
}
.section-title {
    color: #12345b;
    font-size: 25px;
    font-weight: 700;
}
.small-note {
    color: #52657c;
    font-size: 15px;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 3. LOAD CONFIGURATION AND MODEL
# --------------------------------------------------

@st.cache_resource
def load_detector():

    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model file not found: ai_image_detector.keras")

    if not CONFIG_PATH.exists():
        raise FileNotFoundError("deployment_config.json not found.")

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = json.load(file)

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    return model, config


# --------------------------------------------------
# 4. IMAGE PREPROCESSING
# --------------------------------------------------

def preprocess_image(image):

    height = int(config["image_height"])
    width = int(config["image_width"])

    image = image.convert("RGB")
    image = image.resize((width, height))

    image_array = np.asarray(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    # Keep pixel values in the 0–255 range.
    # EfficientNet performs its own preprocessing.

    return image_array


# --------------------------------------------------
# 5. PREDICTION
# --------------------------------------------------

def predict_image(image_array):

    prediction = model.predict(image_array, verbose=0)

    fake_probability = float(np.asarray(prediction).reshape(-1)[0])

    fake_probability = max(0.0, min(1.0, fake_probability))
    real_probability = 1.0 - fake_probability

    threshold = float(config["fake_probability_threshold"])

    if fake_probability >= threshold:
        result = "FAKE"
    else:
        result = "REAL"

    return (
        result,
        real_probability * 100,
        fake_probability * 100,
        fake_probability
    )


# --------------------------------------------------
# 6. GRAD-CAM GENERATION
# --------------------------------------------------

def generate_gradcam(image_array):

    backbone = model.get_layer("efficientnetb0")

    augmentation = model.get_layer("augmentation")

    conv_layer = backbone.get_layer("top_conv")

    conv_model = tf.keras.Model(
        inputs=backbone.input,
        outputs=conv_layer.output
    )

    gap_layer = model.get_layer("global_average_pooling2d_1")
    dropout_layer = model.get_layer("dropout_1")
    classifier_layer = model.get_layer("fake_probability")

    image_tensor = tf.convert_to_tensor(image_array)

    with tf.GradientTape() as tape:

        augmented_image = augmentation(
            image_tensor,
            training=False
        )

        conv_outputs = conv_model(
            augmented_image,
            training=False
        )

        tape.watch(conv_outputs)

        pooled = gap_layer(conv_outputs)

        dropped = dropout_layer(
            pooled,
            training=False
        )

        fake_score = classifier_layer(dropped)

        loss = fake_score[:, 0]

    gradients = tape.gradient(loss, conv_outputs)

    if gradients is None:
        raise ValueError("Could not calculate Grad-CAM gradients.")

    weights = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    conv_outputs = conv_outputs[0]
    weights = weights[0]

    heatmap = tf.reduce_sum(
        conv_outputs * weights,
        axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)

    maximum = tf.reduce_max(heatmap)

    heatmap = heatmap / (maximum + 1e-8)

    heatmap = heatmap.numpy()

    # Resize heatmap to the uploaded image size.
    original_width = int(config["image_width"])
    original_height = int(config["image_height"])

    heatmap_image = Image.fromarray(
        np.uint8(255 * heatmap)
    )

    heatmap_image = heatmap_image.resize(
        (original_width, original_height)
    )

    heatmap_array = np.asarray(heatmap_image)

    # Create a colored heatmap.
    cmap = plt.get_cmap("jet")

    colored_heatmap = cmap(
        heatmap_array / 255.0
    )[:, :, :3]

    colored_heatmap = np.uint8(
        colored_heatmap * 255
    )

    heatmap_rgb = Image.fromarray(colored_heatmap)

    # Overlay the heatmap on the original image.
    original_image = Image.fromarray(
        image_array[0].astype(np.uint8)
    ).convert("RGB")

    overlay = Image.blend(
        original_image,
        heatmap_rgb,
        alpha=0.40
    )

    return overlay


# --------------------------------------------------
# 7. PDF REPORT
# --------------------------------------------------

def create_pdf_report(
    result,
    real_probability,
    fake_probability,
    original_image,
    gradcam_image
):

    pdf_buffer = io.BytesIO()

    document = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4
    )

    styles = getSampleStyleSheet()

    story = []

    story.append(
        Paragraph(
            "AI-GENERATED IMAGE DETECTION REPORT",
            styles["Title"]
        )
    )

    story.append(Spacer(1, 20))

    story.append(
        Paragraph(
            f"<b>Prediction:</b> {result}",
            styles["Heading2"]
        )
    )

    story.append(Spacer(1, 10))

    story.append(
        Paragraph(
            f"REAL Probability: {real_probability:.2f}%",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"FAKE Probability: {fake_probability:.2f}%",
            styles["Normal"]
        )
    )

    story.append(Spacer(1, 20))

    story.append(
        Paragraph(
            "Uploaded Image",
            styles["Heading2"]
        )
    )

    original_buffer = io.BytesIO()
    original_image.save(original_buffer, format="PNG")
    original_buffer.seek(0)

    story.append(
        PDFImage(
            original_buffer,
            width=3.5 * inch,
            height=3.5 * inch
        )
    )

    story.append(Spacer(1, 15))

    story.append(
        Paragraph(
            "Grad-CAM Visualization",
            styles["Heading2"]
        )
    )

    gradcam_buffer = io.BytesIO()
    gradcam_image.save(gradcam_buffer, format="PNG")
    gradcam_buffer.seek(0)

    story.append(
        PDFImage(
            gradcam_buffer,
            width=3.5 * inch,
            height=3.5 * inch
        )
    )

    story.append(Spacer(1, 15))

    story.append(
        Paragraph(
            "Grad-CAM highlights image regions that contributed "
            "to the model's prediction. This visualization is an "
            "explanation aid and does not independently prove "
            "whether an image is real or AI-generated.",
            styles["Normal"]
        )
    )

    document.build(story)

    pdf_buffer.seek(0)

    return pdf_buffer.getvalue()


# --------------------------------------------------
# 8. MAIN WEBSITE
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <h1>🔍 AI-Generated Image Detection</h1>
    <p>
        Detect whether an image is REAL or AI-GENERATED
        using Deep Learning and EfficientNetB0.
    </p>
</div>
""", unsafe_allow_html=True)


st.markdown(
    '<p class="section-title">Upload Your Image</p>',
    unsafe_allow_html=True
)

st.write(
    "Upload an image to analyze its authenticity using our trained deep learning model."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"]
)


# --------------------------------------------------
# 9. PROCESS UPLOADED IMAGE
# --------------------------------------------------

if uploaded_file is not None:

    try:

        original_image = Image.open(
            uploaded_file
        ).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Uploaded Image")
            st.image(
                original_image,
                use_container_width=True
            )

        with st.spinner("Loading model and analyzing image..."):

            model, config = load_detector()

            image_array = preprocess_image(original_image)

            result, real_probability, fake_probability, fake_score = (
                predict_image(image_array)
            )

            gradcam_image = generate_gradcam(image_array)

        st.divider()

        st.markdown(
            '<p class="section-title">Detection Result</p>',
            unsafe_allow_html=True
        )

        if result == "FAKE":
            st.error("FAKE IMAGE — The model predicts that this image is AI-generated.")
        else:
            st.success("REAL IMAGE — The model predicts that this image is real.")

        col_real, col_fake = st.columns(2)

        with col_real:
            st.metric(
                "REAL Probability",
                f"{real_probability:.2f}%"
            )
            st.progress(
                min(100, max(0, int(real_probability)))
            )

        with col_fake:
            st.metric(
                "FAKE Probability",
                f"{fake_probability:.2f}%"
            )
            st.progress(
                min(100, max(0, int(fake_probability)))
            )

        st.divider()

        st.markdown(
            '<p class="section-title">Grad-CAM Explainability</p>',
            unsafe_allow_html=True
        )

        st.write(
            "Grad-CAM (Gradient-weighted Class Activation Mapping) "
            "highlights the image regions that influenced the model's prediction."
        )

        with col2:
            st.subheader("Grad-CAM Visualization")
            st.image(
                gradcam_image,
                use_container_width=True
            )

        st.info(
            "Brighter highlighted regions indicate areas that contributed "
            "more strongly to the model's prediction. Grad-CAM is an "
            "interpretability aid, not proof of image authenticity."
        )

        st.divider()

        st.markdown(
            '<p class="section-title">Download Report</p>',
            unsafe_allow_html=True
        )

        pdf_data = create_pdf_report(
            result,
            real_probability,
            fake_probability,
            original_image,
            gradcam_image
        )

        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_data,
            file_name="AI_Image_Detection_Report.pdf",
            mime="application/pdf"
        )

    except (UnidentifiedImageError, OSError):
        st.error("Please upload a valid image file.")

    except Exception as error:
        st.error(f"An error occurred: {error}")
        st.info(
            "Please check the model file, configuration and Grad-CAM layer names."
        )


# --------------------------------------------------
# 10. ABOUT SECTION
# --------------------------------------------------

st.divider()

st.markdown(
    '<p class="section-title">How It Works</p>',
    unsafe_allow_html=True
)

st.write("""
1. Upload an image in JPG, JPEG, PNG or WEBP format.
2. The image is resized to 224 × 224 pixels.
3. EfficientNetB0 analyzes the image.
4. The classifier calculates REAL and FAKE probabilities.
5. Grad-CAM highlights important image regions.
6. Download the detection report as a PDF.
""")

st.caption(
    "Note: This model provides a prediction, not a guarantee of authenticity. "
    "Its performance depends on the images and conditions it was trained on."
)
