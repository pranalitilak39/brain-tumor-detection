# app.py — Final corrected, robust Streamlit app for Brain Tumor Detection + Grad-CAM
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # optional: reduce TF oneDNN messages

import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import tempfile, traceback

st.set_page_config(page_title="Brain Tumor Detection", layout="centered")
st.title("🧠 Brain Tumor Detection (Robust)")


# ---- Load model safely ----
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


# ---- helper: find last conv layer name ----
def find_last_conv_layer(m):
    # Prefer explicit name
    for layer in reversed(m.layers):
        if layer.name == "last_conv_layer":
            return layer.name
    # Otherwise pick last layer whose class name includes 'conv'
    for layer in reversed(m.layers):
        if "conv" in layer.__class__.__name__.lower():
            return layer.name
    return None


LAST_CONV = find_last_conv_layer(model)
if LAST_CONV is None:
    st.error(
        "No convolutional layer found in the model. Grad-CAM requires at least one Conv layer."
    )
    st.stop()
st.write(f"Using convolutional layer for Grad-CAM: **{LAST_CONV}**")

# ---- helper: check output shape and type ----
try:
    out_shape = model.output_shape
except Exception:
    # if model not built yet, try to build with dummy input
    try:
        dummy = np.zeros((1, 128, 128, 3), dtype=np.float32)
        _ = model.predict(dummy, verbose=0)
        out_shape = model.output_shape
    except Exception as e:
        st.error(f"Model output_shape unavailable and dummy predict failed: {e}")
        st.stop()

num_outputs = int(out_shape[-1]) if isinstance(out_shape, (tuple, list)) else 1
is_binary = num_outputs == 1
st.write(
    f"Model output dimension: **{num_outputs}** → interpreted as **{'binary' if is_binary else 'multiclass'}**"
)


# ---- Grad-CAM function (uses model after a forward pass) ----
def compute_gradcam_for_model(m, img_batch, last_conv_name):
    """
    m: loaded keras model (Sequential or Functional)
    img_batch: numpy array shape (1,H,W,3), float32
    last_conv_name: name of conv layer to inspect
    returns: heatmap resized to model conv feature map (Hc, Wc) normalized 0..1
    """
    # Ensure model has been called at least once (build input/output tensors)
    try:
        _ = m.predict(img_batch, verbose=0)
    except Exception as e:
        # try to call via functional wrapper if direct predict fails
        raise RuntimeError(f"Failed to run model.predict before Grad-CAM: {e}")

    # get the last conv layer object
    try:
        last_conv_layer = m.get_layer(last_conv_name)
    except Exception as e:
        raise RuntimeError(f"Cannot find layer '{last_conv_name}' in model: {e}")

    # Build a small model that outputs conv features and final predictions
    grad_model = Model(inputs=m.input, outputs=[last_conv_layer.output, m.output])

    img_tf = tf.convert_to_tensor(img_batch, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tf)
        if is_binary:
            # binary: predictions shape (1,1) -> take positive score
            loss = predictions[:, 0]
        else:
            # multiclass: take score for predicted class
            pred_index = tf.argmax(predictions[0])
            loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise RuntimeError("Gradients returned None — cannot compute Grad-CAM.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # shape (C,)
    conv_out = conv_outputs[0].numpy()  # Hc x Wc x C
    pooled_np = pooled_grads.numpy()

    # weighted sum of feature maps
    heatmap = np.zeros(conv_out.shape[:2], dtype=np.float32)
    for i in range(conv_out.shape[-1]):
        heatmap += pooled_np[i] * conv_out[:, :, i]

    # ReLU + normalize
    heatmap = np.maximum(heatmap, 0)
    maxv = np.max(heatmap) if np.max(heatmap) != 0 else 1e-9
    heatmap /= maxv
    return heatmap  # values in [0,1]


# ---- UI: file uploader ----
uploaded = st.file_uploader("Upload MRI image (jpg/png)", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.info("Upload an MRI image to run prediction and Grad-CAM.")
    st.stop()

# Save to temporary file
tmp = tempfile.NamedTemporaryFile(delete=False)
tmp.write(uploaded.read())
tmp.flush()
tmp_name = tmp.name
tmp.close()

try:
    # Read image with OpenCV
    img_bgr = cv2.imread(tmp_name, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(
            "cv2.imread returned None. The uploaded file may not be a valid image."
        )
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="Uploaded image (RGB)", use_container_width=True)

    # Determine model input size
    try:
        in_shape = model.input_shape  # (None, H, W, C)
        H = int(in_shape[1]) if in_shape[1] is not None else 128
        W = int(in_shape[2]) if in_shape[2] is not None else 128
    except Exception:
        H, W = 128, 128

    # Preprocess
    img_resized = cv2.resize(img_rgb, (W, H))
    img_norm = img_resized.astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_norm, axis=0)

    # Ensure model is callable (builds internal tensors)
    try:
        preds = model.predict(img_batch, verbose=0)
    except Exception as e:
        st.error(f"model.predict failed on the input. Details: {e}")
        st.stop()

    # Prediction display
    if is_binary:
        score = float(preds[0][0])
        label = "Tumor" if score > 0.5 else "No Tumor"
        st.success(f"Prediction: **{label}**  (score = {score:.3f})")
        st.write({"No Tumor": float(1 - score), "Tumor": float(score)})
    else:
        probs = preds[0]
        pred_idx = int(np.argmax(probs))
        st.success(f"Predicted class index: {pred_idx} (prob={probs[pred_idx]:.3f})")
        st.write({f"class_{i}": float(probs[i]) for i in range(len(probs))})
        from tensorflow.keras.layers import Input

def compute_gradcam_for_model(m, img_batch, last_conv_name):
    """
    Robust Grad-CAM builder:
    - Builds a Functional wrapper by creating a new Input and passing it through
      the model.layers (works for Sequential and many saved Functional models).
    - Computes Grad-CAM heatmap (normalized) and returns a 2D array in [0,1].
    """

    # 1) Determine input shape from the img_batch
    if isinstance(img_batch, np.ndarray):
        _, H, W, C = img_batch.shape
    else:
        # fallback - try model input shape
        in_shape = getattr(m, "input_shape", (None, 128, 128, 3))
        H = int(in_shape[1] or 128); W = int(in_shape[2] or 128); C = int(in_shape[3] or 3)

    # 2) Build a functional wrapper safely
    try:
        # Create a fresh Input tensor matching the runtime input size
        input_tensor = Input(shape=(H, W, C))
        x = input_tensor
        for layer in m.layers:
            # call layer on x to re-create computation graph using same layer objects (weights reused)
            x = layer(x)
        functional = Model(inputs=input_tensor, outputs=x)
    except Exception as e:
        raise RuntimeError(f"Failed to build functional wrapper for Grad-CAM: {e}")

    # 3) Ensure model (original) can run on the provided input (already done upstream),
    #    but we will run the functional wrapper for gradients.
    grad_model = Model(inputs=functional.inputs, outputs=[functional.get_layer(last_conv_name).output, functional.output])

    img_tf = tf.convert_to_tensor(img_batch, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tf)
        if conv_outputs is None:
            raise RuntimeError("conv_outputs is None for the chosen layer.")
        # select loss: binary or multiclass
        if predictions.shape[-1] == 1:
            loss = predictions[:, 0]
        else:
            loss = predictions[:, tf.argmax(predictions[0])]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise RuntimeError("Gradients are None — cannot compute Grad-CAM.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
    conv_out = conv_outputs[0].numpy()  # Hc x Wc x C

    # weighted sum
    heatmap = np.zeros(conv_out.shape[:2], dtype=np.float32)
    for i in range(conv_out.shape[-1]):
        heatmap += pooled_grads[i] * conv_out[:, :, i]

    # relu + normalize
    heatmap = np.maximum(heatmap, 0)
    maxv = np.max(heatmap) if np.max(heatmap) != 0 else 1e-9
    heatmap /= maxv

    return heatmap


    # Resize heatmap to model input size & overlay
    heatmap_resized = cv2.resize(heatmap, (W, H))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)  # BGR
    overlay_bgr = cv2.addWeighted(
        cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR), 0.6, heatmap_color, 0.4, 0
    )
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    st.subheader("Grad-CAM Overlay")
    st.image(overlay_rgb, use_container_width=True)

except Exception as e:
    st.error("An unexpected error occurred. See details below.")
    st.text(str(e))
    st.text(traceback.format_exc())

finally:
    try:
        os.remove(tmp_name)
    except Exception:
        pass
