import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import json
from PIL import Image
from pathlib import Path

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="AI Smart Checkout",
    page_icon="🛒",
    layout="wide"
)

ROOT_DIR = Path(__file__).resolve().parent

MODEL_PATH = ROOT_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"
PRICES_PATH = ROOT_DIR / "prices.json"

# -------------------------------
# Load Model & Prices
# -------------------------------
@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))

@st.cache_data
def load_prices():
    with open(PRICES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

model = load_model()
price_map = load_prices()
CLASS_NAMES = model.names  # id → class name

# -------------------------------
# UI Header
# -------------------------------
st.markdown(
    "<h1 style='text-align:center;'>🛒 AI Smart Checkout System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;'>Upload a checkout image and get an instant bill.</p>",
    unsafe_allow_html=True
)

st.divider()

# -------------------------------
# Image Upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload a checkout image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # -------------------------------
    # Run YOLO Inference
    # -------------------------------
    with st.spinner("Detecting products..."):
        results = model.predict(
            image_np,
            conf=0.6,
            device="cpu"
        )[0]

    detections = {}
    annotated = image_np.copy()

    # -------------------------------
    # Draw Boxes & Count Products
    # -------------------------------
    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = CLASS_NAMES[cls_id]
        conf = float(box.conf[0])

        detections[cls_name] = detections.get(cls_name, 0) + 1

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"{cls_name} ({conf:.2f})",
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # -------------------------------
    # Layout
    # -------------------------------
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🖼️ Detected Products")
        st.image(annotated, use_column_width=True)

    with col2:
        st.subheader("🧾 Checkout Bill")

        if detections:
            total = 0
            rows = []

            for product, qty in detections.items():
                price = price_map.get(product, 0)
                subtotal = price * qty
                total += subtotal
                rows.append((product, qty, price, subtotal))

            st.table({
                "Product": [r[0] for r in rows],
                "Qty": [r[1] for r in rows],
                "Unit Price (₹)": [r[2] for r in rows],
                "Subtotal (₹)": [r[3] for r in rows]
            })

            st.markdown(f"### 💰 Total: ₹ {total}")
        else:
            st.warning("No products detected.")

    st.success("Checkout completed successfully!")

