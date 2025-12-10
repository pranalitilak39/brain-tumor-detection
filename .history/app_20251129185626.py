import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
import tempfile
import os

st.set_page_config(page_title="Brain Tumor Detection", layout="centered")


# -------------------------------------------------------
# 1) Load model + force build
# -------------------------------------------------------
@st.cache_resource
def load_brain_model(path="brain_tumor_cnn_full.h5"):
    model = load_model(path)

    # Try to get input shape, fallback to 128x128x3
    try:
        ish = model.input_shape
        H = ish[1] or 128
        W = ish[2] or 128
        C = ish[3] or 3
    except:
        H, W, C = 128, 128, 3

    dummy = np.zeros((1, H, W, C), dtype=np.float32)
    model.predict(dummy, verbose=0)  # important — now model.input exists

    return model


# 👉 MODEL MUST BE HERE
model = load_brain_model()
# -------------------------------------------------------


# -------------------------------------------------------
# 2) Grad-CAM
# -------------------------------------------------------
def compute_gradcam_for_model(m, img_batch, last_conv_name):

    _ = m.predict(img_batch, verbose=0)

    try:
        last_conv_layer = m.get_layer(last_conv_name)
    except Exception as e:
        raise RuntimeError(f"Cannot find layer '{last_conv_name}': {e}")

    conv_model = Model(inputs=m.input, outputs=last_conv_layer.output)

    img_tf = tf.convert_to_tensor(img_batch, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs = conv_model(img_tf)
        preds = m(img_tf)

        if preds.shape[-1] == 1:
            loss = preds[:, 0]
        else:
            loss = preds[:, tf.argmax(preds[0])]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise RuntimeError("Gradients are None — cannot compute Grad-CAM.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
    conv_out = conv_outputs[0].numpy()

    heatmap = np.zeros(conv_out.shape[:2], dtype=np.float32)
    for i in range(conv_out.shape[-1]):
        heatmap += pooled_grads[i] * conv_out[:, :, i]

    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max() + 1e-9
    return heatmap


# -------------------------------------------------------
# 3) UI + File Upload
# -------------------------------------------------------
st.title("🧠 Brain Tumor Detection")

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

    # ---------------------------------------------------
    # Prediction  (MODEL IS DEFINED ABOVE → no NameError)
    # ---------------------------------------------------
    pred = float(model.predict(img_batch)[0][0])
    predicted = "Tumor" if pred > 0.5 else "No Tumor"

    st.subheader("Prediction")
    st.success(f"{predicted} (score = {pred:.3f})")

    st.write({"No Tumor": round(1 - pred, 6), "Tumor": round(pred, 6)})

    # ---------------------------------------------------
    # Grad-CAM
    # ---------------------------------------------------
    heatmap = compute_gradcam_for_model(model, img_batch, "last_conv_layer")
    heatmap = cv2.resize(heatmap, (128, 128))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img_resized, 0.6, heatmap, 0.4, 0)

    st.subheader("Grad-CAM Heatmap")
    st.image(overlay, use_container_width=True)

    os.remove(temp.name)
