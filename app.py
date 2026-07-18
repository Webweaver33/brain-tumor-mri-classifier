"""
app.py — Brain Tumor MRI Classifier Dashboard
Run with:  streamlit run app.py
"""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from utils import (
    get_class_names,
    load_vit_model,
    load_resnet_model,
    HybridModel,
    predict_image,
    ViTGradCAM,
    overlay_gradcam,
    DEVICE,
)

st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="wide"
)


# ---------------------------------------------------------------------------
# Cached model loading — only runs once per session, not on every rerun
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained models...")
def load_models():
    class_names = get_class_names()
    vit_model = load_vit_model(len(class_names))
    resnet_model = load_resnet_model(len(class_names))
    hybrid_model = HybridModel(vit_model, resnet_model)
    gradcam = ViTGradCAM(vit_model)
    return class_names, vit_model, resnet_model, hybrid_model, gradcam


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🧠 Brain Tumor MRI Classifier")
st.sidebar.markdown(
    "Hybrid **ViT + ResNet50** model for classifying brain MRI scans, "
    "with Grad-CAM to visualize what the model is focusing on."
)
st.sidebar.markdown(f"**Device:** `{DEVICE}`")

model_choice = st.sidebar.radio(
    "Prediction model",
    ["Hybrid (ViT + ResNet50)", "ViT only", "ResNet50 only"],
    index=0
)

top_k = st.sidebar.slider("Top-K predictions to show", min_value=1, max_value=4, value=4)

st.sidebar.divider()
st.sidebar.caption(
    "Models are trained in Google Colab (GPU) and loaded here for fast inference. "
    "See models/ folder for setup instructions."
)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.title("Brain Tumor MRI Classification with Grad-CAM")

try:
    class_names, vit_model, resnet_model, hybrid_model, gradcam = load_models()
except FileNotFoundError as e:
    st.error(str(e))
    st.info(
        "Place your trained model files in the `models/` folder:\n\n"
        "- `models/vit-base-brain-tumor/` (folder, from `vit_model.save_pretrained()`)\n"
        "- `models/vit_base_best.pth`\n"
        "- `models/resnet50_best.pth`"
    )
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a brain MRI scan (JPG/PNG)", type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    st.info("Upload an MRI image above to get a prediction.")
    st.stop()

image = Image.open(uploaded_file).convert("RGB")

col1, col2 = st.columns(2)
with col1:
    st.image(image, caption="Uploaded MRI Scan", use_container_width=True)

# ---------------------------------------------------------------------------
# Run prediction
# ---------------------------------------------------------------------------
with st.spinner("Analyzing image..."):
    if model_choice == "Hybrid (ViT + ResNet50)":
        predictions, input_tensor = predict_image(
            hybrid_model, image, class_names, top_k=top_k, use_logits_attr=False
        )
    elif model_choice == "ViT only":
        predictions, input_tensor = predict_image(
            vit_model, image, class_names, top_k=top_k, use_logits_attr=True
        )
    else:
        predictions, input_tensor = predict_image(
            resnet_model, image, class_names, top_k=top_k, use_logits_attr=False
        )

    # Grad-CAM always runs through the ViT branch (needs a fresh forward pass with grad)
    input_tensor.requires_grad_(False)
    cam_2d, predicted_class_idx = gradcam.generate_cam(input_tensor)
    overlay, heatmap = overlay_gradcam(image, cam_2d, alpha=0.4)

top1_class, top1_prob = predictions[0]

with col2:
    st.image(overlay, caption="Grad-CAM Overlay (model attention)", use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
res_col1, res_col2 = st.columns([1, 1])

with res_col1:
    st.subheader("Prediction")
    st.metric("Top prediction", top1_class, f"{top1_prob*100:.1f}% confidence")

    st.write("**Top-K probabilities:**")
    for cls, prob in predictions:
        st.write(f"{cls}")
        st.progress(min(prob, 1.0), text=f"{prob*100:.1f}%")

with res_col2:
    st.subheader("Model attention (Grad-CAM heatmap)")
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cam_2d, cmap="jet")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig)

st.divider()
st.caption(
    "⚠️ This tool is for educational/portfolio demonstration purposes only "
    "and is not a substitute for professional medical diagnosis."
)
