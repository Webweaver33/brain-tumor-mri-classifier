"""
utils.py
Shared model definitions, loading, prediction, and Grad-CAM logic
for the Brain Tumor MRI Classifier Streamlit dashboard.

This is adapted directly from the original Colab training/inference code,
just reorganized into reusable functions/classes for a local app.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from PIL import Image
from torchvision import transforms, models
from transformers import ViTForImageClassification

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
VIT_SAVE_DIR = os.path.join(MODELS_DIR, "vit-base-brain-tumor")
VIT_WEIGHTS_PATH = os.path.join(MODELS_DIR, "vit_base_best.pth")
RESNET_WEIGHTS_PATH = os.path.join(MODELS_DIR, "resnet50_best.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Same normalization used during training
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def get_class_names():
    """
    Class names must match the ImageFolder ordering used during training
    (alphabetical order of subfolder names in your Training/ directory).
    Edit this list to match your actual dataset classes.
    """
    return ["glioma", "meningioma", "notumor", "pituitary"]


def load_vit_model(num_classes):
    """Load the trained ViT model from the saved pretrained directory."""
    if not os.path.exists(VIT_SAVE_DIR):
        raise FileNotFoundError(
            f"ViT model folder not found at {VIT_SAVE_DIR}. "
            "Copy the 'vit-base-brain-tumor' folder from Colab into models/."
        )
    model = ViTForImageClassification.from_pretrained(
        VIT_SAVE_DIR,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    model.to(DEVICE)
    model.eval()
    return model


def load_resnet_model(num_classes):
    """Load the trained ResNet50 model from its .pth weights file."""
    if not os.path.exists(RESNET_WEIGHTS_PATH):
        raise FileNotFoundError(
            f"ResNet weights not found at {RESNET_WEIGHTS_PATH}. "
            "Copy resnet50_best.pth from Colab into models/."
        )
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model.load_state_dict(torch.load(RESNET_WEIGHTS_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


class HybridModel(nn.Module):
    """Averages ViT and ResNet50 logits, same as the Colab hybrid model."""

    def __init__(self, vit_model, resnet_model):
        super().__init__()
        self.vit = vit_model
        self.resnet = resnet_model

    def forward(self, x):
        with torch.no_grad():
            vit_logits = self.vit(x).logits
            resnet_logits = self.resnet(x)
        return (vit_logits + resnet_logits) / 2


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict_image(model, image, class_names, top_k=5, use_logits_attr=True):
    """
    Predict top-k classes for a PIL image.
    model: either the raw ViT model (has .logits) or the HybridModel (returns tensor directly)
    """
    input_tensor = test_transforms(image).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        logits = outputs.logits if (use_logits_attr and hasattr(outputs, "logits")) else outputs
        probabilities = F.softmax(logits, dim=1)

    top_k = min(top_k, len(class_names))
    top_probs, top_indices = torch.topk(probabilities, top_k)
    top_probs = top_probs.cpu().numpy()[0]
    top_indices = top_indices.cpu().numpy()[0]

    predictions = [(class_names[idx], float(prob)) for idx, prob in zip(top_indices, top_probs)]
    return predictions, input_tensor


# ---------------------------------------------------------------------------
# Grad-CAM for ViT
# ---------------------------------------------------------------------------
def _find_last_vit_layer(model):
    """
    Find the last transformer encoder block (ViTLayer) inside the model,
    regardless of the exact attribute path (this varies between
    transformers versions, e.g. model.vit.encoder.layer vs other layouts).
    """
    candidates = [m for m in model.modules() if type(m).__name__ == "ViTLayer"]
    if not candidates:
        raise RuntimeError(
            "Could not automatically locate a ViTLayer inside the model for Grad-CAM. "
            "Your installed `transformers` version may use a different internal "
            "structure than expected."
        )
    return candidates[-1]


class ViTGradCAM:
    """Grad-CAM implementation for the ViT branch."""

    def __init__(self, vit_model, target_layer=None):
        self.model = vit_model
        self.gradients = None
        self.activations = None

        if target_layer is None:
            target_layer = _find_last_vit_layer(self.model)
        self.target_layer = target_layer

        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output[0] if isinstance(output, tuple) else output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0] if isinstance(grad_output, tuple) else grad_output

    def generate_cam(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)
        logits = output.logits if hasattr(output, "logits") else output

        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        self.model.zero_grad()
        class_loss = logits[0, target_class]
        class_loss.backward()

        gradients = self.gradients.detach().cpu()
        activations = self.activations.detach().cpu()

        weights = gradients.mean(dim=1, keepdim=True)
        cam = (weights * activations).sum(dim=2)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        cam_np = cam.squeeze().numpy()
        num_patches = cam_np.shape[0]
        grid_size = int(np.sqrt(num_patches))

        if num_patches == grid_size * grid_size:
            cam_2d = cam_np.reshape(grid_size, grid_size)
        else:
            grid_size = int(np.ceil(np.sqrt(num_patches)))
            padded = np.zeros(grid_size * grid_size)
            padded[:num_patches] = cam_np
            cam_2d = padded.reshape(grid_size, grid_size)

        return cam_2d, target_class


def overlay_gradcam(image, cam, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """Overlay Grad-CAM heatmap on the original PIL image. Returns (overlay_rgb, heatmap_rgb)."""
    img_np = np.array(image.convert("RGB"))
    h, w = img_np.shape[:2]

    cam_resized = cv2.resize(cam, (w, h))
    cam_resized = np.uint8(255 * cam_resized)

    heatmap = cv2.applyColorMap(cam_resized, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img_np, 1 - alpha, heatmap, alpha, 0)
    return overlay, heatmap