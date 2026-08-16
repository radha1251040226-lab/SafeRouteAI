import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="SafeRouteAI",
    page_icon="🚗",
    layout="wide"
)

# -----------------------------
# LOAD MODEL
# -----------------------------
MODEL_PATH = r"runs\detect\pothole_detection-4\weights\best.pt"

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# -----------------------------
# TITLE
# -----------------------------
st.title("🚗 SafeRouteAI")
st.subheader("AI-Powered Pothole Detection & Road Safety Analysis")

st.write(
    "Upload a road image and SafeRouteAI will detect potholes, "
    "calculate confidence, estimate severity, and provide a road-safety recommendation."
)

st.divider()

# -----------------------------
# IMAGE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "📤 Upload a road image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Display uploaded image
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Input Image")
        st.image(image, use_container_width=True)

    # Save temporary image
    suffix = os.path.splitext(uploaded_file.name)[1]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp_file:

        image.save(temp_file.name)
        temp_path = temp_file.name

    # -----------------------------
    # RUN YOLO
    # -----------------------------
    with st.spinner("🔍 Analyzing road image..."):

        results = model.predict(
            source=temp_path,
            conf=0.25,
            verbose=False
        )

    result = results[0]

    # -----------------------------
    # DETECTION RESULTS
    # -----------------------------
    pothole_count = len(result.boxes)

    with col2:
        st.subheader("🤖 Detection Result")

        annotated_image = result.plot()

        st.image(
            annotated_image,
            channels="BGR",
            use_container_width=True
        )

    st.divider()

    # -----------------------------
    # BASIC RESULT
    # -----------------------------
    st.subheader("📊 Road Safety Analysis")

    st.metric(
        "Potholes Detected",
        pothole_count
    )

    if pothole_count == 0:

        st.success("✅ ROAD STATUS: SAFE")

        st.write(
            "No potholes were detected in the uploaded image."
        )

    else:

        confidences = [
            float(box.conf[0])
            for box in result.boxes
        ]

        average_confidence = (
            sum(confidences) / len(confidences)
        )

        high_confidence = sum(
            1 for c in confidences if c >= 0.70
        )

        # -----------------------------
        # RISK CALCULATION
        # -----------------------------
        if pothole_count >= 3 or high_confidence >= 2:
            risk_level = "HIGH"
            status = "⚠️ HIGH RISK"
            recommendation = (
                "SLOW DOWN. Multiple or significant potholes "
                "were detected. Drive carefully."
            )

        elif pothole_count >= 1:
            risk_level = "MEDIUM"
            status = "⚠️ CAUTION"
            recommendation = (
                "Reduce speed and watch carefully for potholes."
            )

        else:
            risk_level = "LOW"
            status = "✅ SAFE"
            recommendation = (
                "Road appears relatively safe based on this image."
            )

        # -----------------------------
        # DISPLAY METRICS
        # -----------------------------
        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Potholes",
                pothole_count
            )

        with c2:
            st.metric(
                "Average Confidence",
                f"{average_confidence * 100:.2f}%"
            )

        with c3:
            st.metric(
                "Risk Level",
                risk_level
            )

        st.subheader(status)

        st.info(recommendation)

        # -----------------------------
        # INDIVIDUAL POTHOLES
        # -----------------------------
        st.subheader("🔎 Individual Detections")

        for i, confidence in enumerate(confidences):

            if confidence >= 0.70:
                severity = "High"
            elif confidence >= 0.40:
                severity = "Medium"
            else:
                severity = "Low"

            st.write(
                f"**Pothole {i + 1}** — "
                f"Confidence: **{confidence * 100:.2f}%** — "
                f"Severity: **{severity}**"
            )

    # Delete temporary file
    try:
        os.remove(temp_path)
    except:
        pass

# -----------------------------
# FOOTER
# -----------------------------
st.divider()

st.caption(
    "SafeRouteAI | YOLOv8-based Intelligent Road Safety System"
)