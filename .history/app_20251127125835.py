import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import tempfile
import os

st.set_page_config(page_title="Brain Tumor Detection App", layout="centered")


# -------------------------
# Load the saved CNN model
# -------------------------
@st.cache_resource
def load_brain_model():
    model = load_model("brain_tumor_cnn_full.h5")
    return model


model = load_brain_model()

CLASS_NAMES = ["No Tumor", "Tumor"]


st.title("🧠 Brain Tumor Classification App")
st.write("Upload an MRI image to predict tumor type and visualize Grad-CAM heatmap.")


# -------------------------
# Function: Grad-CAM
# -------------------------
def get_grad_cam(model, img_array, layer_name="conv2d_2"):

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, np.argmax(predictions[0])]

    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]

    guided_grads = grads
    weights = np.mean(guided_grads, axis=(0, 1))

    cam = np.dot(output, weights)
    cam = cv2.resize(cam, (128, 128))
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()

    return cam


# -------------------------
# File uploader
# -------------------------
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    # save temp file
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())

    # read image
    img = cv2.imread(temp_file.name)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(img_rgb, caption="Uploaded MRI", use_container_width=True)

    # preprocess
    img_resized = cv2.resize(img_rgb, (128, 128))
    img_normalized = img_resized / 255.0
    img_array = np.expand_dims(img_normalized, axis=0)

    # prediction
    pred = model.predict(img_array)[0]
    predicted_class = CLASS_NAMES[np.argmax(pred)]

    st.subheader("Prediction:")
    st.success(f"🧠 Tumor Type: **{predicted_class}**")
    st.write("Class Probabilities:")
    tumor_prob = float(pred[0])

st.write({
    "No Tumor": 1 - tumor_prob,
    "Tumor": tumor_prob
})


    # Grad-CAM
    st.subheader("🔥 Grad-CAM Heatmap")

    cam = get_grad_cam(model, img_array)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    superimposed = cv2.addWeighted(img_resized, 0.6, heatmap, 0.4, 0)

    st.image(superimposed, caption="Grad-CAM Heatmap", use_container_width=True)

    # cleanup
    os.remove(temp_file.name)
