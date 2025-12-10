import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
import tempfile
import os
from PIL import Image

st.set_page_config(page_title="Brain Tumor Detection", layout="centered")


# -------------------------------------------------------
# 1) Load model + force build
# -------------------------------------------------------
@st.cache_resource
def load_brain_model(path="brain_tumor_cnn_full.h5"):
    model = load_model(path)

    # Get input shape from model
    try:
        input_shape = model.input_shape
        if input_shape is None:
            # If input_shape is None, try to infer from layers
            for layer in model.layers:
                if hasattr(layer, "input_shape") and layer.input_shape is not None:
                    input_shape = layer.input_shape
                    break
        if input_shape is None:
            raise ValueError("Could not determine input shape from model")

        # input_shape is (None, H, W, C) or (H, W, C)
        if len(input_shape) == 4:
            H, W, C = input_shape[1], input_shape[2], input_shape[3]
        elif len(input_shape) == 3:
            H, W, C = input_shape[0], input_shape[1], input_shape[2]
        else:
            raise ValueError(f"Unexpected input shape format: {input_shape}")

        # Validate dimensions
        if H is None or W is None or C is None:
            H, W, C = 128, 128, 3  # fallback

    except Exception as e:
        st.error(f"Error determining model input shape: {e}")
        H, W, C = 128, 128, 3

    dummy = np.zeros((1, H, W, C), dtype=np.float32)
    model.predict(dummy, verbose=0)  # important — now model.input exists

    return model, (H, W, C)


# 👉 MODEL MUST BE HERE
model, input_shape = load_brain_model()
H, W, C = input_shape
# -------------------------------------------------------


# -------------------------------------------------------
# 2) Grad-CAM (Robust version)
# -------------------------------------------------------
def compute_gradcam_for_model(m, img_batch, last_conv_name):
    """
    Robust Grad-CAM that does NOT rely on model.input being defined.
    It creates a new Input tensor and passes it through the SAME layer objects
    (re-using their weights) until the last_conv layer and then until the final output.
    Returns heatmap (2D numpy array, values 0..1) sized to conv feature map.
    """
    import tensorflow as tf
    from tensorflow.keras.layers import Input

    # 1) Get index of last_conv layer
    layer_names = [layer.name for layer in m.layers]
    if last_conv_name not in layer_names:
        raise RuntimeError(
            f"Layer '{last_conv_name}' not found in model. Available: {layer_names}"
        )

    last_idx = layer_names.index(last_conv_name)

    # 2) Determine run shape from img_batch
    if isinstance(img_batch, np.ndarray):
        _, H, W, C = img_batch.shape
    else:
        raise RuntimeError("img_batch must be numpy array with shape (1,H,W,3)")

    # 3) Build new functional graph by calling layers on a fresh Input
    new_input = Input(shape=(H, W, C))
    x = new_input
    last_conv_tensor = None

    # call each layer in order on the tensor `x`
    for i, layer in enumerate(m.layers):
        x = layer(x)  # reuse same layer object (weights preserved)
        if i == last_idx:
            last_conv_tensor = x  # capture the conv feature tensor
            # NOTE: do not break; continue to get final output tensor too

    final_output_tensor = x  # after loop this is the model's final output

    # 4) Create a functional model that yields [last_conv_tensor, final_output_tensor]
    from tensorflow.keras.models import Model as KModel

    functional_for_grad = KModel(
        inputs=new_input, outputs=[last_conv_tensor, final_output_tensor]
    )

    # 5) Run gradient tape on the functional_for_grad with the actual image
    img_tf = tf.convert_to_tensor(img_batch, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, preds = functional_for_grad(img_tf)
        # choose loss: binary sigmoid -> preds shape (1,1)
        if preds.shape[-1] == 1:
            loss = preds[:, 0]
        else:
            loss = preds[:, tf.argmax(preds[0])]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise RuntimeError("Gradients are None — cannot compute Grad-CAM.")

    # 6) Pool the gradients and compute weighted sum of conv feature maps
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()  # shape (C,)
    conv_out = conv_outputs[0].numpy()  # Hc x Wc x C

    heatmap = np.zeros(conv_out.shape[:2], dtype=np.float32)
    for i in range(conv_out.shape[-1]):
        heatmap += pooled_grads[i] * conv_out[:, :, i]

    heatmap = np.maximum(heatmap, 0)
    maxv = np.max(heatmap) if np.max(heatmap) != 0 else 1e-9
    heatmap /= maxv

    # return heatmap (values between 0 and 1) sized to conv feature map
    return heatmap


# -------------------------------------------------------
# 3) UI + File Upload
# -------------------------------------------------------
st.title("🧠 Brain Tumor Detection")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        # Use context manager for temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            temp.write(uploaded_file.read())
            temp.flush()  # Ensure data is written to disk
            temp_path = temp.name

        # Load and process image
        img = cv2.imread(temp_path)
        if img is None:
            st.error("Failed to load image. Please upload a valid image file.")
            os.remove(temp_path)
            st.stop()

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img, caption="Uploaded MRI", use_container_width=True)

        # Preprocess using model's input shape
        img_resized = cv2.resize(img, (W, H))
        img_norm = img_resized / 255.0
        img_batch = np.expand_dims(img_norm, axis=0)

        # ---------------------------------------------------
        # Prediction
        # ---------------------------------------------------
        pred = float(model.predict(img_batch, verbose=0)[0][0])
        predicted = "Tumor" if pred > 0.5 else "No Tumor"

        st.subheader("Prediction")
        st.success(f"{predicted} (score = {pred:.3f})")
        st.write({"No Tumor": round(1 - pred, 6), "Tumor": round(pred, 6)})

        # ---------------------------------------------------
        # Grad-CAM
        # ---------------------------------------------------
        try:
            # Find the last convolutional layer automatically
            conv_layers = [
                layer for layer in model.layers if "conv" in layer.name.lower()
            ]
            if not conv_layers:
                st.warning(
                    "No convolutional layers found for Grad-CAM. Skipping heatmap."
                )
            else:
                last_conv_name = conv_layers[-1].name
                heatmap = compute_gradcam_for_model(model, img_batch, last_conv_name)
                heatmap = cv2.resize(heatmap, (W, H))
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

                overlay = cv2.addWeighted(img_resized, 0.6, heatmap, 0.4, 0)
                st.subheader("Grad-CAM Heatmap")
                st.image(overlay, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate Grad-CAM heatmap: {str(e)}")

    except Exception as e:
        st.error(f"An error occurred during processing: {str(e)}")
    finally:
        # Clean up temp file
        if "temp_path" in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
