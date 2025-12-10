import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import tempfile
import os

st.set_page_config(page_title="Brain Tumor Detection App", layout="centered")


# -------------------------
# Load Model
# -------------------------
@st.cache_resource
def load_brain_model():
    model = load_model("brain_tumor_cnn_full.h5")
    return model


model = load_brain_model()

CLASS_NAMES = ["No Tumor", "Tumor"]


st.title("🧠 Brain Tumor Classification App")
st.write("Upload an MRI image to detect tumor and visualize the Grad-CAM heatmap.")


# -------------------------
# Grad-CAM Function
# -------------------------
def get_grad_cam(model, img_array, layer_name="last_conv_layer"):

    # 1. Fetch last conv layer
    last_conv_layer = model.get_layer(layer_name)

    # 2. Create a Grad-CAM model (Sequential-safe)
    grad_model = tf.keras.models.Model(
        inputs=model.input, outputs=[last_conv_layer.output, model.output]
    )

    # 3. Compute gradient of the predicted class
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]  # Tumor score (binary)

    grads = tape.gradient(loss, conv_outputs)

    # 4. Mean of gradients (importance weights)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 5. Multiply conv maps by weights
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    # 6. Normalize heatmap
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (np.max(heatmap) + 1e-9)

    return heatmap


# -------------------------
# File Upload
# -------------------------
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(uploaded_file.read())

    img = cv2.imread(temp.name)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(img_rgb, caption="Uploaded MRI Image", use_container_width=True)

    # preprocess
    img_resized = cv2.resize(img_rgb, (128, 128))
    img_normalized = img_resized / 255.0
    img_array = np.expand_dims(img_normalized, axis=0)

    # prediction
    pred = model.predict(img_array)[0][0]
    predicted_class = "Tumor" if pred > 0.5 else "No Tumor"

    st.subheader("Prediction:")
    st.success(f"🧠 Tumor Type: **{predicted_class}**")

    st.write("Class Probabilities:")
    st.write({"No Tumor": 1 - float(pred), "Tumor": float(pred)})

    # -------------------------
    # Grad-CAM Heatmap
    # -------------------------
    st.subheader("🔥 Grad-CAM Heatmap")

    cam = get_grad_cam(model, img_array, layer_name="last_conv_layer")

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    superimposed = cv2.addWeighted(img_resized, 0.6, heatmap, 0.4, 0)
    st.image(superimposed, caption="Grad-CAM Heatmap", use_container_width=True)

    os.remove(temp.name)
