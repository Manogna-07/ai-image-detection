
import os
import json
import uuid
import io

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import tensorflow as tf

from PIL import Image, UnidentifiedImageError
from flask import (
    Flask, render_template, request, send_from_directory,
    send_file, url_for
)
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as PDFImage,
    Table, TableStyle
)


# ==========================================
# 1. APPLICATION PATHS AND CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "ai_image_detector.keras")
CONFIG_PATH = os.path.join(BASE_DIR, "deployment_config.json")

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
GENERATED_FOLDER = os.path.join(BASE_DIR, "static", "generated")
RESULTS_FOLDER = os.path.join(BASE_DIR, "results")
REPORTS_FOLDER = os.path.join(BASE_DIR, "reports")

for folder in [
    UPLOAD_FOLDER, GENERATED_FOLDER, RESULTS_FOLDER, REPORTS_FOLDER
]:
    os.makedirs(folder, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}


# ==========================================
# 2. LOAD CONFIGURATION AND TRAINED MODEL
# ==========================================

with open(CONFIG_PATH, "r") as file:
    config = json.load(file)

IMAGE_HEIGHT = config["image_height"]
IMAGE_WIDTH = config["image_width"]
LABELS = config["labels"]
FAKE_THRESHOLD = float(config["fake_probability_threshold"])

print("Loading trained EfficientNetB0 model...")

model = load_model(MODEL_PATH, compile=False)

# The nested EfficientNetB0 backbone
backbone = model.get_layer("efficientnetb0")

# Grad-CAM target layer identified from your model
last_conv_layer = backbone.get_layer("top_conv")

# Build a feature extractor for the nested backbone
backbone_grad_model = tf.keras.Model(
    inputs=backbone.input,
    outputs=[
        last_conv_layer.output,
        backbone.output
    ]
)

# Retrieve the original classification head
global_pool = model.get_layer("global_average_pooling2d_1")
dropout_layer = model.get_layer("dropout_1")
output_layer = model.get_layer("fake_probability")

print("Model loaded successfully!")
print("Grad-CAM target layer:", last_conv_layer.name)


# ==========================================
# 3. MODEL PERFORMANCE METRICS
# ==========================================

# Actual test metrics supplied from your evaluation
MODEL_METRICS = {
    "accuracy": 0.856389,
    "precision": 0.824810,
    "recall": 0.905000,
    "f1_score": 0.863046,
    "roc_auc": 0.934122
}

CONFUSION_MATRIX_PATH = os.path.join(
    RESULTS_FOLDER, "confusion_matrix.png"
)


# ==========================================
# 4. FILE VALIDATION
# ==========================================

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ==========================================
# 5. IMAGE PREPROCESSING
# ==========================================

def preprocess_image(image_path):

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))

        # Keep 0-255 float32 values.
        # Do not divide by 255: EfficientNet handles its own rescaling.
        image_array = np.asarray(image, dtype=np.float32)

    return np.expand_dims(image_array, axis=0)


# ==========================================
# 6. PREDICTION AND GRAD-CAM
# ==========================================

def predict_and_gradcam(image_path):

    image_array = preprocess_image(image_path)

    input_tensor = tf.convert_to_tensor(image_array)

    # Calculate gradients through the nested EfficientNetB0
    with tf.GradientTape() as tape:

        augmented = model.get_layer("augmentation")(
            input_tensor,
            training=False
        )

        conv_outputs, backbone_outputs = backbone_grad_model(
            augmented,
            training=False
        )

        tape.watch(conv_outputs)

        pooled = global_pool(backbone_outputs)

        dropped = dropout_layer(
            pooled,
            training=False
        )

        prediction_output = output_layer(dropped)

        fake_probability_tensor = tf.reshape(
            prediction_output, [-1]
        )[0]

        fake_probability = float(
            fake_probability_tensor.numpy()
        )

        # Explain the predicted class
        if fake_probability >= FAKE_THRESHOLD:
            class_score = fake_probability_tensor
        else:
            class_score = 1.0 - fake_probability_tensor

    gradients = tape.gradient(class_score, conv_outputs)

    if gradients is None:
        raise RuntimeError(
            "Could not calculate Grad-CAM gradients."
        )

    # Average gradients over spatial dimensions
    weights = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    # Combine feature maps using gradient weights
    cam = tf.reduce_sum(
        conv_outputs[0] * weights[0],
        axis=-1
    )

    cam = tf.maximum(cam, 0)

    maximum = tf.reduce_max(cam)

    cam = tf.math.divide_no_nan(cam, maximum)

    heatmap = cam.numpy()

    # Convert heatmap into a displayable image
    heatmap_image = Image.fromarray(
        np.uint8(255 * heatmap)
    )

    heatmap_image = heatmap_image.resize(
        (IMAGE_WIDTH, IMAGE_HEIGHT)
    )

    heatmap_array = np.asarray(heatmap_image)

    # Read original image for overlay
    with Image.open(image_path) as original:
        original = original.convert("RGB")
        original = original.resize(
            (IMAGE_WIDTH, IMAGE_HEIGHT)
        )
        original_array = np.asarray(original)

    # Create heatmap and overlay
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(original_array)
    axes[0].set_title("Original Image")

    axes[1].imshow(
        heatmap_array,
        cmap="jet",
        vmin=0,
        vmax=255
    )
    axes[1].set_title("Grad-CAM Heatmap")

    axes[2].imshow(original_array)
    axes[2].imshow(
        heatmap_array,
        cmap="jet",
        alpha=0.40,
        vmin=0,
        vmax=255
    )
    axes[2].set_title("Grad-CAM Overlay")

    for axis in axes:
        axis.axis("off")

    plt.tight_layout()

    heatmap_filename = uuid.uuid4().hex + "_gradcam.png"

    heatmap_path = os.path.join(
        GENERATED_FOLDER,
        heatmap_filename
    )

    plt.savefig(
        heatmap_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    # Final class probabilities
    fake_probability = max(
        0.0, min(1.0, fake_probability)
    )

    real_probability = 1.0 - fake_probability

    if fake_probability >= FAKE_THRESHOLD:
        predicted_label = LABELS["1"]
        confidence = fake_probability
    else:
        predicted_label = LABELS["0"]
        confidence = real_probability

    result = {
        "prediction": predicted_label,
        "confidence": round(confidence * 100, 2),
        "real_probability": round(real_probability * 100, 2),
        "fake_probability": round(fake_probability * 100, 2),
        "threshold": FAKE_THRESHOLD
    }

    return result, heatmap_filename


# ==========================================
# 7. HOME PAGE AND IMAGE ANALYSIS
# ==========================================

@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    image_url = None
    heatmap_url = None
    report_url = None
    error = None

    if request.method == "POST":

        file = request.files.get("image")

        if file is None or file.filename == "":
            error = "Please select an image."

        elif not allowed_file(file.filename):
            error = "Upload JPG, JPEG, PNG or WEBP images only."

        else:

            safe_name = secure_filename(file.filename)

            unique_name = (
                uuid.uuid4().hex + "_" + safe_name
            )

            image_path = os.path.join(
                UPLOAD_FOLDER, unique_name
            )

            try:

                file.save(image_path)

                # Validate image contents
                with Image.open(image_path) as image:
                    image.verify()

                result, heatmap_filename = predict_and_gradcam(
                    image_path
                )

                image_url = url_for(
                    "uploaded_file",
                    filename=unique_name
                )

                heatmap_url = url_for(
                    "generated_file",
                    filename=heatmap_filename
                )

                report_url = url_for(
                    "download_report",
                    filename=unique_name
                )

            except (
                UnidentifiedImageError,
                OSError,
                ValueError
            ):

                if os.path.exists(image_path):
                    os.remove(image_path)

                error = "Invalid image. Please upload a valid image."

            except Exception:

                app.logger.exception("Image analysis failed")

                error = (
                    "Image analysis failed. "
                    "Please check the terminal for details."
                )

    return render_template(
        "index.html",
        result=result,
        image_url=image_url,
        heatmap_url=heatmap_url,
        report_url=report_url,
        error=error,
        metrics=MODEL_METRICS,
        confusion_matrix_exists=os.path.exists(
            CONFUSION_MATRIX_PATH
        )
    )


# ==========================================
# 8. SERVE UPLOADED AND GENERATED IMAGES
# ==========================================

@app.route("/uploads/<filename>")
def uploaded_file(filename):

    return send_from_directory(
        UPLOAD_FOLDER,
        filename
    )


@app.route("/generated/<filename>")
def generated_file(filename):

    return send_from_directory(
        GENERATED_FOLDER,
        filename
    )


@app.route("/confusion-matrix")
def confusion_matrix():

    if not os.path.exists(CONFUSION_MATRIX_PATH):
        return "Confusion matrix not found.", 404

    return send_from_directory(
        RESULTS_FOLDER,
        "confusion_matrix.png"
    )


# ==========================================
# 9. PDF REPORT GENERATION
# ==========================================

@app.route("/download-report/<filename>")
def download_report(filename):

    image_path = os.path.join(
        UPLOAD_FOLDER, filename
    )

    if not os.path.isfile(image_path):
        return "Image not found.", 404

    # Recalculate prediction and Grad-CAM for the report
    try:
        result, heatmap_filename = predict_and_gradcam(
            image_path
        )

    except Exception:
        app.logger.exception("PDF report generation failed")
        return "Unable to generate report.", 500

    heatmap_path = os.path.join(
        GENERATED_FOLDER,
        heatmap_filename
    )

    pdf_path = os.path.join(
        REPORTS_FOLDER,
        uuid.uuid4().hex + "_report.pdf"
    )

    document = SimpleDocTemplate(
        pdf_path,
        pagesize=A4
    )

    styles = getSampleStyleSheet()

    story = []

    story.append(
        Paragraph(
            "AI Image Detection Report",
            styles["Title"]
        )
    )

    story.append(Spacer(1, 12))

    story.append(
        Paragraph(
            "EfficientNetB0 AI Image Detector",
            styles["Heading2"]
        )
    )

    story.append(Spacer(1, 15))

    data = [
        ["Analysis Detail", "Result"],
        ["Prediction", result["prediction"]],
        ["Confidence", str(result["confidence"]) + "%"],
        ["REAL probability", str(result["real_probability"]) + "%"],
        ["FAKE probability", str(result["fake_probability"]) + "%"],
        ["FAKE threshold", str(FAKE_THRESHOLD)],
        ["Image size", "224 x 224"],
        ["Model", "EfficientNetB0"]
    ]

    table = Table(data, colWidths=[190, 230])

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#173B65")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F3F8FC"))
    ]))

    story.append(table)
    story.append(Spacer(1, 20))

    story.append(
        Paragraph("Uploaded Image", styles["Heading2"])
    )

    story.append(
        PDFImage(image_path, width=230, height=170)
    )

    story.append(Spacer(1, 15))

    story.append(
        Paragraph("Grad-CAM Explanation", styles["Heading2"])
    )

    story.append(
        PDFImage(heatmap_path, width=430, height=145)
    )

    story.append(Spacer(1, 15))

    story.append(
        Paragraph(
            "Disclaimer: This report contains a prediction from a "
            "trained deep learning model. It does not prove the "
            "original source or authenticity of the image. "
            "Grad-CAM is an approximate visual explanation.",
            styles["Normal"]
        )
    )

    document.build(story)

    return send_file(
        pdf_path,
        as_attachment=True,
        download_name="AI_Image_Detection_Report.pdf"
    )


# ==========================================
# 10. ERROR HANDLING
# ==========================================

@app.errorhandler(413)
def file_too_large(error):

    return render_template(
        "index.html",
        result=None,
        image_url=None,
        heatmap_url=None,
        report_url=None,
        metrics=MODEL_METRICS,
        error="Image must be smaller than 10 MB."
    ), 413


# ==========================================
# 11. RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
