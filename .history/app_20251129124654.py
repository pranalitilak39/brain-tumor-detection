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
st.title("🧠 Brain Tumor Detection — Final Stable")


# ---- Load model ----
@st.cache_resource
def load_brain_model(path="brain_tumor_cnn_full.h5"):
    try:
        m = load_model(path)
        return m, None
    except Exception as e:
        return None, str(e)


model, load_err = load_brain_model("brain_tumor_cnn_full.h5")
if load_err:
    st.error(f"Failed to load model: {load_err}")
    st.stop()

st.success("Model loaded.")


# ---- Build a Functional wrapper that guarantees tensors exist ----
def build_functional_wrapper(m, input_shape=(128, 128, 3)):
    """
    Create a Functional model that re-uses the same layer objects from m,
    by creating a new Input and calling each layer on it.
    Returns the functional model or raises an exception.
    """
    # determine H,W,C from m.input_shape if available
    try:
        ish = m.input_shape
        H = int(ish[1]) if ish[1] is not None else input_shape[0]
        W = int(ish[2]) if ish[2] is not None else input_shape[1]
        C = int(ish[3]) if ish[3] is not None else input_shape[2]
    except Exception:
        H, W, C = input_shape

    inp = Input(shape=(H, W, C))
    x = inp
    for layer in m.layers:
        x = layer(x)
    functional = Model(inputs=inp, outputs=x)
    return functional, (H, W, C)


# Build wrapper (catch errors and show to UI)
try:
    functional_model, (MODEL_H, MODEL_W, MODEL_C) = build_functional_wrapper(model)
except Exception as e:
    st.error("Failed to create functional wrapper for Grad-CAM. Paste this error here:")
    st.text(str(e))
    st.stop()

st.write(f"Functional wrapper built. Model input: ({MODEL_H},{MODEL_W},{MODEL_C})")


# ---- find last conv layer name ----
def find_last_conv(m):
    for layer in reversed(m.layers):
        if "conv" in layer.__class__.__name__.lower():
            return layer.name
    return None


LAST_CONV = find_last_conv(functional_model)
if LAST_CONV is None:
    st.error("No Conv layer found in model (cannot compute Grad-CAM).")
    st.stop()
st.write(f"Using conv layer: **{LAST_CONV}**")

# ---- helper: is binary? ----
try:
    out_shape = functional_model.output_shape
except Exception:
    out_shape = (None, 1)
num_out = int(out_shape[-1])
is_binary = num_out == 1
st.write(f"Model output units: {num_out} → {'binary' if is_binary else 'multiclass'}")


# ---- Grad-CAM function using the functional_model ----
def gradcam_from_functional(func_model, img_batch, last_conv_name):
    grad_model = Model(
        inputs=func_model.inputs,
        outputs=[func_model.get_layer(last_conv_name).output, func_model.output],
    )
    img_tf = tf.convert_to_tensor(img_batch, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_outs, preds = grad_model(img_tf)
        if is_binary:
            loss = preds[:, 0]
        else:
            loss = preds[:, tf.argmax(preds[0])]
    grads = tape.gradient(loss, conv_outs)
    if grads is None:
        raise RuntimeError("Gradients are None")
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
    conv = conv_outs[0].numpy()
    heatmap = np.zeros(conv.shape[:2], dtype=np.float32)
    for i in range(conv.shape[-1]):
        heatmap += pooled[i] * conv[:, :, i]
    heatmap = np.maximum(heatmap, 0)
    maxv = np.max(heatmap) if np.max(heatmap) != 0 else 1e-9
    heatmap /= maxv
    return heatmap


# ---- UI: uploader ----
uploaded = st.file_uploader("Upload MRI image (jpg/png)", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.info("Upload an MRI image to run prediction + Grad-CAM.")
    st.stop()

tmp = tempfile.NamedTemporaryFile(delete=False)
tmp.write(uploaded.read())
tmp.flush()
tmp_name = tmp.name
tmp.close()

try:
    img_bgr = cv2.imread(tmp_name)
    if img_bgr is None:
        raise RuntimeError("cv2.imread returned None — invalid image.")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="Uploaded image", use_container_width=True)

    # preprocess to model size
    img_resized = cv2.resize(img_rgb, (MODEL_W, MODEL_H))
    img_norm = img_resized.astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_norm, axis=0)

    # prediction (use functional_model to ensure consistency)
    preds = functional_model.predict(img_batch, verbose=0)
    if is_binary:
        score = float(preds[0][0])
        label = "Tumor" if score > 0.5 else "No Tumor"
        st.success(f"Prediction: **{label}** (score={score:.3f})")
        st.write({"No Tumor": float(1 - score), "Tumor": float(score)})
    else:
        probs = preds[0]
        idx = int(np.argmax(probs))
        st.success(f"Predicted class {idx} (prob={probs[idx]:.3f})")
        st.write({f"class_{i}": float(probs[i]) for i in range(len(probs))})

    # Grad-CAM
    heatmap = gradcam_from_functional(functional_model, img_batch, LAST_CONV)
    heatmap_resized = cv2.resize(heatmap, (MODEL_W, MODEL_H))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)  # BGR
    overlay_bgr = cv2.addWeighted(
        cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR), 0.6, heatmap_color, 0.4, 0
    )
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    st.subheader("Grad-CAM overlay")
    st.image(overlay_rgb, use_container_width=True)

except Exception as e:
    # show a single clear error line and full trace below for debugging
    st.error(f"Error: {str(e).splitlines()[0]}")
    st.text(traceback.format_exc())
finally:
    try:
        os.remove(tmp_name)
    except Exception:
        pass
