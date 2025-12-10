# final_app.py - robust Streamlit app for your brain tumor model + Grad-CAM
import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input
import tempfile, os, sys, traceback

st.set_page_config(page_title="Brain Tumor Detection (Robust)", layout="centered")

st.title("🧠 Brain Tumor Detection")


# ---- load model safely ----
@st.cache_resource
def safe_load_model(path="brain_tumor_cnn_full.h5"):
    try:
        m = load_model(path)
        return m, None
    except Exception as e:
        return None, f"Failed loading model '{path}': {e}"


model, load_err = safe_load_model("brain_tumor_cnn_full.h5")
if load_err:
    st.error(load_err)
    st.stop()

st.info("Model loaded successfully.")


# ---- helper: find last conv layer name ----
def find_last_conv_layer(m):
    # prefer a layer explicitly named 'last_conv_layer' if exists
    for layer in reversed(m.layers):
        if layer.name == "last_conv_layer":
            return layer.name
    # otherwise choose last Conv2D-like layer
    for layer in reversed(m.layers):
        # string checks to avoid importing layer classes
        lname = layer.__class__.__name__.lower()
        if "conv" in lname:
            return layer.name
    return None


LAST_CONV = find_last_conv_layer(model)
if LAST_CONV is None:
    st.error(
        "Could not find any convolutional layer in the model. Make sure your model has Conv2D layers."
    )
    st.stop()
st.write(f"Using last convolutional layer: **{LAST_CONV}**")

# ---- helper: check output shape (binary vs multiclass) ----
try:
    output_shape = model.output_shape  # may be e.g. (None,1) or (None,4)
except Exception:
    # If model not built yet, build by calling on a dummy input
    try:
        # try to build with an input of shape (1,128,128,3)
        dummy = np.zeros((1, 128, 128, 3), dtype=np.float32)
        _ = model.predict(dummy)
        output_shape = model.output_shape
    except Exception as e:
        st.error(f"Model not built and dummy call failed: {e}")
        st.stop()

num_outputs = int(output_shape[-1]) if isinstance(output_shape, (tuple, list)) else 1
is_binary = num_outputs == 1

st.write(
    f"Model output dimension: **{num_outputs}** — interpreted as **{'binary' if is_binary else 'multiclass'}** model"
)


# ---- build a functional wrapper for grad-cam safely ----
def build_functional_for_gradcam(m):
    try:
        # If model.input and model.output exist and are tensors, try direct use
        if hasattr(m, "inputs") and hasattr(m, "outputs") and m.inputs is not None:
            return m  # already usable
    except Exception:
        pass

    # Otherwise create a new Input and pass through the layers (Sequential-safe)
    try:
        in_shape = (
            m.layers[0].input_shape
            if hasattr(m.layers[0], "input_shape")
            else (None, 128, 128, 3)
        )
        # remove batch dim if present
        if isinstance(in_shape, tuple) and len(in_shape) >= 3:
            h, w, c = in_shape[-3], in_shape[-2], in_shape[-1]
            input_tensor = Input(shape=(h or 128, w or 128, c or 3))
        else:
            input_tensor = Input(shape=(128, 128, 3))
        x = input_tensor
        for layer in m.layers:
            x = layer(x)
        functional = Model(inputs=input_tensor, outputs=x)
        return functional
    except Exception as e:
        return None


functional_model = build_functional_for_gradcam(model)
if functional_model is None:
    st.error(
        "Could not create a Functional model for Grad-CAM. Please ensure the model is standard Sequential/Functional and was saved with architecture."
    )
    st.stop()


# ---- Grad-CAM implementation (robust) ----
def compute_gradcam(func_model, img_tensor, last_conv_name):
    """
    func_model: Functional model (inputs, outputs)
    img_tensor: np array shape (1,H,W,3), float32
    last_conv_name: name of conv layer to inspect
    returns: heatmap (Hconv,Wconv) normalized 0..1
    """
    try:
        last_conv = func_model.get_layer(last_conv_name)
    except Exception as e:
        raise RuntimeError(f"Layer '{last_conv_name}' not found in model: {e}")

    # Build grad model mapping input -> (last_conv_output, prediction)
    grad_model = Model(
        inputs=func_model.inputs, outputs=[last_conv.output, func_model.output]
    )

    # Ensure tensors are float32
    img_tf = tf.convert_to_tensor(img_tensor, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_outs, preds = grad_model(img_tf)
        if is_binary:
            # binary: preds shape (1,1)
            loss = preds[:, 0]
        else:
            # multiclass: use predicted class index
            pred_index = tf.argmax(preds[0])
            loss = preds[:, pred_index]

    grads = tape.gradient(loss, conv_outs)
    if grads is None:
        raise RuntimeError(
            "Gradients are None — check that the loss depends on the conv outputs"
        )

    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))  # channel importance
    conv_out = conv_outs[0].numpy()  # Hc x Wc x C
    pooled_np = pooled.numpy()

    # compute weighted sum of channels
    heatmap = np.zeros(conv_out.shape[:2], dtype=np.float32)
    for i in range(conv_out.shape[-1]):
        heatmap += pooled_np[i] * conv_out[:, :, i]

    # relu and normalize
    heatmap = np.maximum(heatmap, 0)
    maxh = np.max(heatmap) if np.max(heatmap) != 0 else 1e-9
    heatmap /= maxh
    return heatmap


# ---- UI: file upload and flow ----
uploaded = st.file_uploader("Upload MRI (jpg/png)", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.info("Upload an MRI image to run prediction and Grad-CAM.")
    st.stop()

# Save temporary file
tmp = tempfile.NamedTemporaryFile(delete=False)
tmp.write(uploaded.read())
tmp.flush()
tmp_name = tmp.name
tmp.close()

try:
    # Read with CV2
    img_bgr = cv2.imread(tmp_name, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(
            "cv2.imread returned None. The file may be corrupted or not an image."
        )
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Show original
    st.image(img_rgb, caption="Uploaded Image (RGB)", use_column_width=True)

    # preprocess to model input size (use model.input shape if available)
    try:
        in_shape = model.input_shape  # e.g. (None,128,128,3)
        h = in_shape[1] or 128
        w = in_shape[2] or 128
    except Exception:
        h, w = 128, 128

    img_resized = cv2.resize(img_rgb, (w, h))
    img_norm = img_resized.astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_norm, axis=0)

    # ensure model is built: call predict once
    try:
        _ = model.predict(img_batch, verbose=0)
    except Exception:
        # if direct predict fails on saved model, try functional wrapper
        _ = functional_model.predict(img_batch, verbose=0)

    # get prediction
    preds = model.predict(img_batch, verbose=0)
    if is_binary:
        score = float(preds[0][0])
        label = "Tumor" if score > 0.5 else "No Tumor"
        st.success(f"Prediction: **{label}** (Score = {score:.3f})")
        st.write({"No Tumor": float(1 - score), "Tumor": float(score)})
    else:
        probs = preds[0]
        pred_index = int(np.argmax(probs))
        st.success(
            f"Prediction class index: {pred_index} (prob {probs[pred_index]:.3f})"
        )
        st.write({f"class_{i}": float(probs[i]) for i in range(len(probs))})

    # compute grad-cam (using the functional wrapper to be safe)
    try:
        heatmap = compute_gradcam(functional_model, img_batch, LAST_CONV)
    except Exception as e:
        st.error(f"Grad-CAM failed: {e}")
        st.stop()

    # resize heatmap to original resized image and overlay
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(
        img_resized, 0.6, heatmap_color[..., ::-1], 0.4, 0
    )  # heatmap_color is BGR

    st.subheader("Grad-CAM overlay")
    st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_column_width=True)

except Exception as e:
    st.error("An unexpected error occurred during processing. See details below.")
    st.text(str(e))
    st.text(traceback.format_exc())

finally:
    try:
        os.remove(tmp_name)
    except Exception:
        pass
