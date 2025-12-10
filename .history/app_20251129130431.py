import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
import tempfile
import os

st.set_page_config(page_title="Brain Tumor App", layout="centered")


# ---------------------------
# Load model safely
# ---------------------------
@st.cache_resource
def load_brain_model():
    model = load_model("brain_tumor_cnn_full.h5")

    # ⭐ FORCE TRACE THE MODEL (Fix for undefined input error)
    dummy = np.zeros((1, 128, 128, 3), dtype=np.float32)
    model.predict(dummy)

    return model


model = load_brain_model()
CLASS_NAMES = ["No Tumor", "Tumor"]

st.title("🧠 Brain Tumor Detection – Stable Version")
st.write("Upload an MRI image and visualize Tumor activation using Grad-CAM.")


# ---------------------------
# Grad-CAM (Final Working)
# ---------------------------
def grad_cam(model, img_array, layer_name="last_conv_layer"):

    try:
        last_conv = model.get_layer(layer_name)
    except:
        st.error(f"Layer '{layer_name}' not found in model.")
        return None

    grad_model = Model(inputs=model.input, outputs=[last_conv.output, model.output])

    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(img_array)
        class_idx = tf.argmax(preds[0])
        class_output = preds[:, class_idx]

    grads = tape.gradient(class_output, conv_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]

    heatmap = tf.zeros(conv_output.shape[0:2], dtype=tf.float32)
    for i in range(pooled_grads.shape[-1]):
        heatmap += pooled_grads[i] * conv_output[:, :, i]

    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / np.max(heatmap)

    heatmap = cv2.resize(heatmap.numpy(), (128, 128))

    return heatmap


# ---------------------------
# File Upload
# ---------------------------
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(uploaded_file.read())

    img = cv2.imread(temp.name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(img, caption="Uploaded MRI", use_container_width=True)

    # Preprocess
    img_resized = cv2.resize(img, (128, 128))
    img_norm = img_resized / 255.0
    img_batch = np.expand_dims(img_norm, axis=0)

    # Prediction
    pred = float(model.predict(img_batch)[0][0])
    predicted = "Tumor" if pred > 0.5 else "No Tumor"

    st.subheader("Prediction")
    st.success(f"🧠 **{predicted}**")
    st.write({"No Tumor": 1 - pred, "Tumor": pred})

    # Grad-CAM
    st.subheader("🔥 Grad-CAM Activation Heatmap")

    cam = grad_cam(model, img_batch, "last_conv_layer")

    if cam is None:
        st.error("Grad-CAM failed.")
    else:
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        superimposed = cv2.addWeighted(img_resized, 0.6, heatmap, 0.4, 0)

        st.image(superimposed, caption="Grad-CAM", use_container_width=True)

    os.remove(temp.name)
