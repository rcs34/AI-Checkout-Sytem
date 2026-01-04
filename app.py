import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import json
from PIL import Image
from pathlib import Path
from collections import Counter

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="AI Smart Checkout",
    page_icon="🛒",
    layout="wide"
)

# -------------------------------
# Paths & Constants
# -------------------------------
ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"
PRICES_PATH = ROOT_DIR / "prices.json"

CONF_THRESHOLD = 0.5
TOP_K = 3

# -------------------------------
# Session State Initialization
# -------------------------------
if "cart_items" not in st.session_state:
    st.session_state.cart_items = []

if "cart_images" not in st.session_state:
    st.session_state.cart_images = 0

if "stage" not in st.session_state:
    st.session_state.stage = "cart"   # cart | payment

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
CLASS_NAMES = model.names
ALL_PRODUCTS = sorted(price_map.keys())

# -------------------------------
# Similar Products (Heuristic)
# -------------------------------
SIMILAR_PRODUCTS = {
    "India Gate - Pure Basmati Rice Super": [
        "Betty Crocker - Complete Pancake Mix Classic",
        "Patanjali - Atta Noodles"
    ],
    "Betty Crocker - Complete Pancake Mix Classic": [
        "India Gate - Pure Basmati Rice Super",
        "Blue Bird - Baking Powder"
    ],
}

# -------------------------------
# UI Header
# -------------------------------
st.markdown("<h1 style='text-align:center;'>🛒 AI Smart Checkout System</h1>", unsafe_allow_html=True)

if st.session_state.stage == "cart":
    st.markdown(
        "<p style='text-align:center;'>Place <b>one product</b> in the frame and upload the image.</p>",
        unsafe_allow_html=True
    )
else:
    st.markdown(
        "<p style='text-align:center;'>Confirm payment to complete checkout.</p>",
        unsafe_allow_html=True
    )

st.divider()

# =====================================================
# ===================== CART STAGE =====================
# =====================================================
if st.session_state.stage == "cart":

    # -------------------------------
    # Image Upload
    # -------------------------------
    uploaded_file = st.file_uploader(
        "Upload a product image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        annotated = image_np.copy()

        with st.spinner("Detecting product..."):
            results = model.predict(
                image_np,
                conf=CONF_THRESHOLD,
                device="cpu"
            )[0]

        col1, col2 = st.columns([2, 1])

        # -------------------------------
        # Detection Rules (Single Product)
        # -------------------------------
        if len(results.boxes) == 0:
            with col1:
                st.image(image_np, use_container_width=True)
            with col2:
                st.error("❌ No product detected. Please try again.")

        elif len(results.boxes) > 1:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)

            with col1:
                st.image(annotated, use_container_width=True)
            with col2:
                st.error("⚠️ Multiple products detected.")
                st.info("Please place only ONE product in the frame.")

        else:
            box = results.boxes[0]
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            predicted = CLASS_NAMES[cls_id].strip().strip('"')

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                f"{predicted} ({conf:.2f})",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            with col1:
                st.subheader("Detected Product")
                st.image(annotated, use_container_width=True)

            with col2:
                st.subheader("Confirm Product")

                suggestions = SIMILAR_PRODUCTS.get(predicted, [])
                options = list(dict.fromkeys(
                    [predicted] + suggestions + ["Other (manual)"]
                ))

                choice = st.radio(
                    "Select the correct product:",
                    options,
                    index=0
                )

                if choice == "Other (manual)":
                    final_product = st.selectbox(
                        "Search product:",
                        ALL_PRODUCTS
                    )
                else:
                    final_product = choice

                st.write(f"**Price:** ₹ {price_map.get(final_product, 0)}")

                if st.button("➕ Add to Cart"):
                    st.session_state.cart_items.append(final_product)
                    st.session_state.cart_images += 1
                    st.success("Added to cart!")

    # -------------------------------
    # CART SUMMARY
    # -------------------------------
    st.divider()
    st.subheader(f"🛒 Cart Summary ({len(st.session_state.cart_items)} items)")

    if st.session_state.cart_items:
        counts = Counter(st.session_state.cart_items)
        total = 0

        rows = []
        for product, qty in counts.items():
            price = price_map.get(product, 0)
            subtotal = price * qty
            total += subtotal
            rows.append((product, qty, price, subtotal))

        st.table({
            "Product": [r[0] for r in rows],
            "Qty": [r[1] for r in rows],
            "Unit Price (₹)": [r[2] for r in rows],
            "Subtotal (₹)": [r[3] for r in rows],
        })

        st.markdown(f"### 💰 Total: ₹ {total}")

        if st.button("✅ Verify & Checkout"):
            st.session_state.stage = "payment"
    else:
        st.info("Cart is empty. Upload products to add items.")

# =====================================================
# =================== PAYMENT STAGE ====================
# =====================================================
if st.session_state.stage == "payment":

    st.divider()
    st.subheader("💳 Payment")

    counts = Counter(st.session_state.cart_items)
    total = sum(price_map[p] * q for p, q in counts.items())

    st.markdown("### 🧾 Order Summary")
    for product, qty in counts.items():
        st.write(f"{product} × {qty} = ₹ {price_map.get(product, 0) * qty}")

    st.markdown(f"### 💰 Total Payable: ₹ {total}")

    payment_method = st.radio(
        "Select payment method:",
        ["Cash", "UPI", "Card"]
    )

    if payment_method == "UPI":
        st.info("Scan UPI QR on POS device (simulated)")
    elif payment_method == "Card":
        st.info("Insert / Tap card on POS terminal (simulated)")
    else:
        st.info("Pay at counter (cash)")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("⬅️ Back to Cart"):
            st.session_state.stage = "cart"

    with col2:
        if st.button("💰 Pay Now"):
            st.success("✅ Payment Successful!")
            st.balloons()

            st.markdown("### 🧾 Receipt")
            for product, qty in counts.items():
                st.write(f"{product} × {qty} = ₹ {price_map.get(product, 0) * qty}")

            st.markdown(f"### 💵 Total Paid: ₹ {total}")
            st.markdown(f"**Payment Method:** {payment_method}")

            # RESET SYSTEM
            st.session_state.cart_items.clear()
            st.session_state.cart_images = 0
            st.session_state.stage = "cart"

            st.success("Checkout complete. Ready for next customer 🛒")
