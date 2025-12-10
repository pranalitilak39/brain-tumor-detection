# app.py — Final corrected version using conv_model + model inside GradientTape
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
st.title("🧠 Brain Tumor Detection — Fixed Grad-CAM")


# ---- Load model safely ----
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


# ---- find last conv layer name ----
def find_last_conv_layer(m):
    for layer in reversed(m.layers):
        if "conv" in layer.__class__.__name__.lower():
            return layer.name
    return None


LAST_CONV = find_last_conv_layer(model)
if LAST_CONV is None:
    st.error("No convolutional layer found. Cannot compute Grad-CAM.")
    st.stop()
st.write(f"Using last conv layer: **{LAST_CONV}**")

# ---- ensure model is built and determine input size ----
try:
    in_shape = model.input_shape  # (None, H, W, C)
    MODEL_H = int(in_shape[1]) if in_shape[1] is not None else 128
    MODEL_W = int(in_shape[2]) if in_shape[2] is not None else 128
    MODEL_C = int(in_shape[3]) if in_shape[3] is not None else 3
except Exception:
    MODEL_H, MODEL_W, MODEL_C = 128, 128, 3


# Build functional wrapper for consistent predict (optional)
def build_functional_wrapper(m, H=128, W=128, C=3):
    inp = Input(shape=(H, W, C))
    x = inp
    for layer in m.layers:
        x = layer(x)
    return Model(inputs=inp, outputs=x)


# Try to build wrapper — if it fails we will still use model directly
try:
    functional_model = build_functional_wrapper(model, MODEL_H, MODEL_W, MODEL_C)
except Exception:
    functional_model = model  # fallback

# Determine binary vs multiclass
try:
    out_shape = model.output_shape
    num_out = int(out_shape[-1])
except Exception:
    num_out = 1
is_binary = num_out == 1
st.write(f"Model outputs: {num_out} → {'binary' if is_binary else 'multiclass'}")


# ---- Robust Grad-CAM using conv_model + model inside same tape ----
def compute_gradcam_via_conv_and_model(m, img_batch, last_conv_name):
    """
    m: original model (Sequential or Functional)
    img_batch: numpy (1,H,W,3)
    last_conv_name: name of conv layer
    returns heatmap (2D ndarray normalized 0..1)
    """
    # make sure model layers are built
    _ = m.predict(img_batch, verbose=0)

    # build a conv-only submodel that outputs the last conv feature maps
    try:
        last_conv = m.get_layer(last_conv_name)
    except Exception as e:
        raise RuntimeError(f"Layer not found: {e}")

    conv_model = Model(inputs=m.input, outputs=last_conv.output)

    # convert to tensor
    img_tf = tf.convert_to_tensor(img_batch, dtype=tf.float32)

    # compute conv features and predictions inside the same tape
    with tf.GradientTape() as tape:
        # forward through conv_model
        conv_outputs = conv_model(img_tf)  # shape (1, Hc, Wc, C)
        # forward through full model
        preds = m(img_tf)  # shape (1, num_out)
        # choose loss
        if preds.shape[-1] == 1:
            loss = preds[:, 0]
        else:
            loss = preds[:, tf.argmax(preds[0])]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise RuntimeError("Gradients are None — cannot compute Grad-CAM.")

    # pooled grads: mean over spatial dims
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()  # shape (C,)
    conv_out_np = conv_outputs[0].numpy()  # Hc x Wc x C

    # weighted sum of conv maps
    heatmap = np.zeros(conv_out_np.shape[:2], dtype=np.float32)
    for i in range(conv_out_np.shape[-1]):
        heatmap += pooled_grads[i] * conv_out_np[:, :, i]

    # ReLU + normalize
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

    # ensure model built (call once)
    try:
        _ = model.predict(img_batch, verbose=0)
    except Exception as e:
        st.error(f"model.predict failed: {e}")
        st.stop()

    # prediction
    preds = model.predict(img_batch, verbose=0)
    if is_binary:
        score = float(preds[0][0])
        label = "Tumor" if score > 0.5 else "No Tumor"
        st.success(f"Prediction: **{label}** (score={score:.3f})")
        st.write({"No Tumor": float(1 - score), "Tumor": float(score)})
    else:
        probs = preds[0]
        idx = int(np.argmax(probs))
        st.success(f"Predicted class {idx} (prob {probs[idx]:.3f})")
        st.write({f"class_{i}": float(probs[i]) for i in range(len(probs))})

    # compute Grad-CAM using robust method
    try:
        heatmap = compute_gradcam_via_conv_and_model(model, img_batch, LAST_CONV)
    except Exception as e:
        st.error(f"Grad-CAM failed: {e}")
        st.text(traceback.format_exc())
        st.stop()

    # overlay heatmap
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
    st.error(f"Error: {str(e).splitlines()[0]}")
    st.text(traceback.format_exc())

finally:
    try:
        os.remove(tmp_name)
    except Exception:
        pass
