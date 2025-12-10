import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # optional: reduces TF warnings

import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import tempfile

st.set_page_config(page_title="Brain Tumor Detection App", layout="centered")


# -------------------------
# Load CNN model
# -------------------------
@st.cache_resource
def load_brain_model():
    return load_model("brain_tumor_cnn_full.h5")


model = load_brain_model()


# -------------------------
# Grad-CAM FUNCTION (CORRECTED)
# -------------------------
def get_grad_cam(model, img_array, layer_name="last_conv_layer"):

    last_conv_layer = model.get_layer(layer_name)

    grad_model = tf.keras.models.Model(
        inputs=model.input, outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)[0]

    conv_outputs = conv_outputs[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))

    cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)

    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)
    cam = cv2.resize(cam.numpy(), (128, 128))

    return cam


# -------------------------
# UI
# -------------------------
st.title("🧠 Brain Tumor Classification App")
st.write("Upload an MRI image to predict tumor type and view Grad-CAM heatmap.")


uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())

    img = cv2.imread(temp_file.name)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(img_rgb, caption="Uploaded MRI", use_container_width=True)

    # Preprocess
    img_resized = cv2.resize(img_rgb, (128, 128))
    img_normalized = img_resized / 255.0
    img_array = np.expand_dims(img_normalized, axis=0)

    # Prediction
    pred = model.predict(img_array)[0][0]
    predicted_class = "Tumor" if pred > 0.5 else "No Tumor"

    st.subheader("Prediction:")
    st.success(f"🧠 Tumor Type: **{predicted_class}**")

    st.write("Class Probabilities:")
    st.write({"No Tumor": float(1 - pred), "Tumor": float(pred)})

    # Grad-CAM
    st.subheader("🔥 Grad-CAM Heatmap")
    cam = get_grad_cam(model, img_array, layer_name="last_conv_layer")

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    superimposed = cv2.addWeighted(img_resized, 0.6, heatmap, 0.4, 0)
    st.image(superimposed, caption="Grad-CAM Heatmap", use_container_width=True)

    os.remove(temp_file.name)
