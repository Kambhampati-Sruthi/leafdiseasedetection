import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import joblib
import pandas as pd

# --- Configuration ---
MODEL_PATH = "leaf_disease_model.keras"
LABELS_PATH = "class_labels1.joblib"
IMG_SIZE = 128

# --- Load model and labels ---
@st.cache_resource
def load_model_and_labels():
    model = tf.keras.models.load_model(MODEL_PATH)
    try:
        class_names = joblib.load(LABELS_PATH)
        if not isinstance(class_names, dict):
            raise ValueError("Label file is not a dictionary.")
    except Exception as e:
        st.error(f"❌ Failed to load label file: {e}")
        class_names = {}
    return model, class_names

model, CLASS_NAMES = load_model_and_labels()

# --- Streamlit UI ---
st.title("🌿 Rice Leaf Disease Detection")
st.write("Upload a rice leaf image to detect possible diseases.")

uploaded_file = st.file_uploader("📷 Upload a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing image..."):
        try:
            img_array = np.array(image)
            img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array, verbose=0)
            predicted_index = int(np.argmax(prediction))
            confidence = round(float(np.max(prediction)) * 100, 2)

            if predicted_index in CLASS_NAMES:
                predicted_label = CLASS_NAMES[predicted_index]
                st.subheader("🔍 Prediction Result")
                st.success(f"**Detected Disease:** {predicted_label}")
                st.info(f"**Confidence:** {confidence} %")
            else:
                st.error(f"⚠️ Predicted index `{predicted_index}` not found in label map.")
                st.stop()

            st.subheader("📊 Confidence for All Classes")
            pred_df = pd.DataFrame({
                "Class": [CLASS_NAMES.get(i, f"Class {i}") for i in range(len(prediction[0]))],
                "Confidence (%)": np.round(prediction[0] * 100, 2)
            }).sort_values(by="Confidence (%)", ascending=False)

            st.dataframe(pred_df.reset_index(drop=True), use_container_width=True)

        except Exception as e:
            st.error(f"🚨 Prediction error:\n`{e}`")
else:
    st.info("👈 Upload a rice leaf image to get started.")
