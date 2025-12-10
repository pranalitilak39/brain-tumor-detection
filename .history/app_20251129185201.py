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
def load_brain_model(path="brain_tumor_cnn_full.h5"):
    # Load the Keras model and force a dummy forward pass so model.input is defined
    model = load_model(path)

    # Force-build the model (important). Use the model.input_shape if available,
    # otherwise fall back to 128x128x3 which your model summary shows was used.
    try:
        ish = model.input_shape  # (None, H, W, C) if available
        H = int(ish[1]) if ish[1] is not None else 128
        W = int(ish[2]) if ish[2] is not None else 128
        C = int(ish[3]) if ish[3] is not None else 3
    except Exception:
        H, W, C = 128, 128, 3

    dummy = np.zeros((1, H, W, C), dtype=np.float32)
    # run one forward pass to initialize internal tensors (this fixes the "never been called" issue)
    model.predict(dummy, verbose=0)

    return model


model = load_brain_model()


# ---------------------------
# Grad-CAM (Final Working)
# ---------------------------
def compute_gradcam_for_model(m, img_batch, last_conv_name):
    """
    Robust Grad-CAM for a Sequential model:
    - builds a conv-only submodel (conv_model)
    - inside the same GradientTape calls conv_model(img) and m(img)
    - returns a normalized 2D heatmap (values 0..1)
    """
    # Ensure model is built (should be from load_brain_model) but call predict again defensively
    _ = m.predict(img_batch, verbose=0)

    # Get last conv layer object
    try:
        last_conv_layer = m.get_layer(last_conv_name)
    except Exception as e:
        raise RuntimeError(f"Cannot find layer '{last_conv_name}': {e}")

    # conv-only model that outputs feature maps of last conv layer
    conv_model = Model(inputs=m.input, outputs=last_conv_layer.output)

    img_tf = tf.convert_to_tensor(img_batch, dtype=tf.float32)

    # compute conv feature maps and predictions inside same tape
    with tf.GradientTape() as tape:
        conv_outputs = conv_model(img_tf)  # shape (1, Hc, Wc, C)
        preds = m(img_tf)  # shape (1, num_out)

        # choose loss for binary or multiclass
        if preds.shape[-1] == 1:
            loss = preds[:, 0]  # binary: use positive score
        else:
            loss = preds[:, tf.argmax(preds[0])]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise RuntimeError("Gradients are None — cannot compute Grad-CAM.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()  # shape (C,)
    conv_out_np = conv_outputs[0].numpy()  # Hc x Wc x C

    # weighted sum of feature maps
    heatmap = np.zeros(conv_out_np.shape[:2], dtype=np.float32)
    for i in range(conv_out_np.shape[-1]):
        heatmap += pooled_grads[i] * conv_out_np[:, :, i]

    # relu + normalize
    heatmap = np.maximum(heatmap, 0)
    maxv = np.max(heatmap) if np.max(heatmap) != 0 else 1e-9
    heatmap /= maxv

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
