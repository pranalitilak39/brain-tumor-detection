import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input
import tempfile, traceback

st.set_page_config(page_title="Brain Tumor Detection", layout="centered")
st.title("🧠 Brain Tumor Detection (Robust Version)")


# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_brain_model(path="brain_tumor_cnn_full.h5"):
    try:
        m = load_model(path)
        return m, None
    except Exception as e:
        return None, f"Failed to load model '{path}': {e}"


model, load_err = load_brain_model("brain_tumor_cnn_full.h5")
if load_err:
    st.error(load_err)
    st.stop()
st.info("Model loaded successfully.")


# -------------------- FIND LAST CONV LAYER --------------------
def find_last_conv_layer(m):
    for layer in reversed(m.layers):
        if "conv" in layer.__class__.__name__.lower():
            return layer.name
    return None


LAST_CONV = find_last_conv_layer(model)
if LAST_CONV is None:
    st.error("No convolutional layer found. Cannot compute Grad-CAM.")
    st.stop()

st.write(f"Using last conv layer for Grad-CAM: **{LAST_CONV}**")


# -------------------- CHECK OUTPUT SHAPE --------------------
try:
    out_shape = model.output_shape
except:
    dummy = np.zeros((1, 128, 128, 3), dtype=np.float32)
    _ = model.predict(dummy, verbose=0)
    out_shape = model.output_shape

num_outputs = int(out_shape[-1])
is_binary = num_outputs == 1
st.write(
    f"Model output units: {num_outputs} → **{'Binary' if is_binary else 'Multiclass'}**"
)


# -------------------- GRADCAM FUNCTION (FINAL FIXED VERSION) --------------------
def compute_gradcam_for_model(m, img_batch, last_conv_name):
    """
    Fully corrected Grad-CAM function.
    """

    # Determine image shape
    _, H, W, C = img_batch.shape

    # Build functional wrapper (important!)
    try:
        input_tensor = Input(shape=(H, W, C))
        x = input_tensor
        for layer in m.layers:
            x = layer(x)
        functional = Model(inputs=input_tensor, outputs=x)
    except Exception as e:
        raise RuntimeError(f"Cannot reconstruct model: {e}")

    # Build grad model
    try:
        grad_model = Model(
            inputs=functional.inputs,
            outputs=[functional.get_layer(last_conv_name).output, functional.output],
        )
    except:
        raise RuntimeError(f"Layer '{last_conv_name}' not found in model.")

    img_tf = tf.convert_to_tensor(img_batch, dtype=tf.float32)

    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tf)

        if predictions.shape[-1] == 1:
            loss = predictions[:, 0]
        else:
            loss = predictions[:, tf.argmax(predictions[0])]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise RuntimeError("Gradients returned None.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
    conv = conv_outputs[0].numpy()

    # Weighted sum
    heatmap = np.zeros(conv.shape[:2], dtype=np.float32)
    for i in range(conv.shape[-1]):
        heatmap += pooled_grads[i] * conv[:, :, i]

    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    return heatmap


# -------------------- FILE UPLOADER --------------------
uploaded = st.file_uploader("Upload MRI image", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.stop()

# Save temporary file
tmp = tempfile.NamedTemporaryFile(delete=False)
tmp.write(uploaded.read())
tmp.flush()
tmp_name = tmp.name
tmp.close()

try:
    # Read image
    img_bgr = cv2.imread(tmp_name)
    if img_bgr is None:
        raise Exception("Invalid image file.")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="Uploaded MRI", use_container_width=True)

    # Preprocess
    H = model.input_shape[1] or 128
    W = model.input_shape[2] or 128

    img_resized = cv2.resize(img_rgb, (W, H))
    img_norm = img_resized.astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_norm, axis=0)

    # Predict
    preds = model.predict(img_batch, verbose=0)

    if is_binary:
        score = float(preds[0][0])
        label = "Tumor" if score > 0.5 else "No Tumor"
        st.success(f"Prediction: **{label}** (score {score:.3f})")

    else:
        probs = preds[0]
        idx = int(np.argmax(probs))
        st.success(f"Predicted Class: {idx} (prob {probs[idx]:.3f})")

    # -------------------- GRAD-CAM --------------------
    heatmap = compute_gradcam_for_model(model, img_batch, LAST_CONV)
    heatmap_resized = cv2.resize(heatmap, (W, H))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img_resized, 0.6, heatmap_color, 0.4, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    st.subheader("🔥 Grad-CAM Heatmap Overlay")
    st.image(overlay_rgb, use_container_width=True)

except Exception as e:
    st.error("Unexpected error occurred:")
    st.text(str(e))
    st.text(traceback.format_exc())

finally:
    try:
        os.remove(tmp_name)
    except:
        pass
