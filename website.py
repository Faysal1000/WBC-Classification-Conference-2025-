"""
website.py

Dependencies:
  pip install streamlit tensorflow staintools opencv-python pillow matplotlib numpy

Run:
  streamlit run website.py
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt

# ---------------------------
# User settings / labels
# ---------------------------
DEFAULT_MODEL_PATH = r"WBC CLassifier.keras"
DEFAULT_STAIN_MATRIX_PATH = r"custom_stain_matrix_vahadane.npy"
IMG_SIZE = (224, 224)
labels = [
    "basophil",
    "eosinophil",
    "erythroblast",
    "ig",
    "lymphocyte",
    "monocyte",
    "neutrophil",
    "platelet"
]

# ---------------------------
# dependencies functions
# ---------------------------
@tf.keras.utils.register_keras_serializable(package="Custom", name="F1Score")
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * (p * r) / (p + r + tf.keras.backend.epsilon())

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()

# ---------------------------
# Utility: caching model & normalizer
# ---------------------------
@st.cache_resource
def load_keras_model(uploaded_file=None):
    if uploaded_file is not None:
        tmp_path = "tmp_uploaded_model"
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        try:
            model = tf.keras.models.load_model(tmp_path, compile=False)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
    else:
        if not os.path.exists(DEFAULT_MODEL_PATH):
            return None
        model = tf.keras.models.load_model(DEFAULT_MODEL_PATH, compile=False)
    return model

@st.cache_resource
def load_normalizer(uploaded_file=None):
    if uploaded_file is not None:
        tmp = "tmp_stain_matrix.npy"
        with open(tmp, "wb") as f:
            f.write(uploaded_file.getvalue())
        try:
            data = np.load(tmp, allow_pickle=True)
            normalizer = data.item() if hasattr(data, "item") else data
        finally:
            try:
                os.remove(tmp)
            except Exception:
                pass
        return normalizer
    if not os.path.exists(DEFAULT_STAIN_MATRIX_PATH):
        return None
    data = np.load(DEFAULT_STAIN_MATRIX_PATH, allow_pickle=True)
    normalizer = data.item() if hasattr(data, "item") else data
    return normalizer

# ---------------------------
# Preprocessing 
# ---------------------------
def preprocess_image2(image):
    img = image.astype(np.float32)
    img = (img - np.min(img)) / (np.ptp(img) + 1e-8)
    return img

# ---------------------------
# Recursive layer extraction and shape inference
# ---------------------------
def get_all_layers(model):
    layers = []
    for layer in model.layers:
        layers.append(layer)
        if hasattr(layer, 'layers') and layer.layers:
            layers += get_all_layers(layer)
    return layers

def infer_layer_output_shape(layer, input_shape=(1, 224, 224, 3)):
    try:
        dummy_input = tf.zeros(input_shape)
        if isinstance(layer, tf.keras.layers.InputLayer):
            return input_shape
        model_temp = tf.keras.Sequential([tf.keras.Input(shape=input_shape[1:]), layer])
        output = model_temp(dummy_input)
        return output.shape.as_list()
    except Exception:
        return None

# ---------------------------
# Grad-CAM utilities
# ---------------------------
def find_last_4d_layer(all_layers):
    for layer in reversed(all_layers):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
            out_shape = getattr(layer, 'output_shape', None)
            if out_shape and isinstance(out_shape, (list, tuple)) and len(out_shape) == 4:
                H, W = out_shape[1:3]
                if (H is None or H > 1) and (W is None or W > 1):
                    return layer.name
            # Fallback: infer shape with dummy input
            out_shape = infer_layer_output_shape(layer)
            if out_shape and len(out_shape) == 4 and out_shape[1] > 1 and out_shape[2] > 1:
                return layer.name
    return None

def build_gradcam_submodel(model, last_conv_name, all_layers):
    last_conv_layer = next(l for l in all_layers if l.name == last_conv_name)
    last_layer = model.layers[-1]
    if hasattr(last_layer, "activation") and last_layer.activation is not None and 'softmax' in str(last_layer.activation).lower():
        logits_output = last_layer.input
    else:
        logits_output = last_layer.output
    return tf.keras.Model(inputs=model.inputs,
                          outputs=[last_conv_layer.output, logits_output])

def gradcam(model, img_array, class_index=None, preferred_last_conv="top_conv", eps=1e-8):
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    all_layers = get_all_layers(model)

    # pick conv layer
    last_conv_name = None
    if preferred_last_conv and any(l.name == preferred_last_conv for l in all_layers):
        last_conv_name = preferred_last_conv
    if last_conv_name is None:
        last_conv_name = find_last_4d_layer(all_layers)

    if last_conv_name is None:
        layer_info = [(l.name, str(l.__class__.__name__), infer_layer_output_shape(l) or getattr(l, 'output_shape', None)) for l in all_layers if isinstance(l, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D))]
        st.error(f"No valid conv layer found. Conv/DepthwiseConv2D layers (name, type, output_shape):\n{layer_info}\nTry selecting layers like 'top_conv', 'block6i_project_conv', or 'block6i_dwconv2'.")
        raise RuntimeError("No valid conv layer found for Grad-CAM")

    conv_model = build_gradcam_submodel(model, last_conv_name, all_layers)

    with tf.GradientTape() as tape:
        conv_outputs, logits = conv_model(img_tensor, training=False)
        tape.watch(conv_outputs)
        if class_index is None:
            class_index = int(tf.argmax(logits[0]).numpy())
        loss = logits[0, class_index]  

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise RuntimeError(f"Gradients are None for layer '{last_conv_name}'")

    max_grad_abs = tf.reduce_max(tf.abs(grads)).numpy()
    if max_grad_abs < 1e-10:
        st.warning(f"Gradients are effectively zero (max abs: {max_grad_abs:.2e}) in layer '{last_conv_name}'. "
                   f"Try a shallower layer (e.g., 'block5a_project_conv').")

    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.tensordot(conv_outputs[0], weights, axes=[[2], [0]])
    cam = tf.nn.relu(cam).numpy().astype(np.float32)

    if np.max(cam) > 1e-8:
        cam /= np.max(cam)

    st.write(f"[Grad-CAM] layer={last_conv_name}, "
             f"min={cam.min():.4f}, max={cam.max():.4f}, shape={cam.shape}")

    return cam

def overlay_heatmap_on_image(img_bgr, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    overlay = cv2.addWeighted(img_bgr, 1-alpha, heatmap_color, alpha, 0)
    return overlay

# ---------------------------
# Helpers to read image and apply stain normalization
# ---------------------------
def pil_to_rgb_array(pil_img):
    pil_img = pil_img.convert("RGB")
    arr = np.array(pil_img)
    return arr

def apply_vahadane(normalizer, rgb_image):
    if rgb_image.dtype != np.uint8:
        mn = float(rgb_image.min())
        mx = float(rgb_image.max())
        rgb_image = ((rgb_image - mn) / (mx - mn + 1e-8) * 255.0).astype(np.uint8)
    try:
        transformed = normalizer.transform(rgb_image)
    except Exception as e:
        st.warning(f"Vahadane transform failed: {e}")
        transformed = rgb_image
    if transformed.dtype != np.uint8:
        transformed = np.clip(transformed, 0, 255).astype(np.uint8)
    return transformed

def prepare_for_model(img_rgb, target_size=IMG_SIZE):
    img_resized = cv2.resize(img_rgb, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
    img_pre = preprocess_image2(img_resized.astype(np.float32))
    if img_pre.ndim == 2:
        img_pre = np.stack([img_pre]*3, axis=-1)
    if img_pre.shape[-1] != 3:
        img_pre = img_pre[..., :3]
    return np.expand_dims(img_pre, axis=0).astype(np.float32)

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="WBC Classifier + Vahadane + Grad-CAM", layout="centered")
st.title("WBC Classifier — Vahadane → EfficientV2B1 → Grad-CAM")

st.sidebar.header("Model & Stain Matrix")
uploaded_model_file = st.sidebar.file_uploader("Upload Keras model (.h5/.keras)", type=["h5", "keras"], key="model_upload")
uploaded_stain_file = st.sidebar.file_uploader("Upload stain matrix (.npy)", type=["npy"], key="stain_upload")

with st.sidebar.expander("Load model / normalizer"):
    model = None
    normalizer = None
    try:
        model = load_keras_model(uploaded_file=uploaded_model_file)
        if model is not None:
            # Build model to populate output shapes
            input_shape = (None, IMG_SIZE[0], IMG_SIZE[1], 3)
            model.build(input_shape)
    except Exception as e:
        st.sidebar.error(f"Could not load or build model: {e}")
    try:
        normalizer = load_normalizer(uploaded_file=uploaded_stain_file)
    except Exception as e:
        st.sidebar.error(f"Could not load stain matrix: {e}")

    if model is None:
        st.sidebar.warning("Model not loaded. Please upload a model file or ensure 'WBC CLassifier.keras' is in the correct directory.")
    else:
        st.sidebar.success("Model loaded.")
    if normalizer is None:
        st.sidebar.warning("Stain normalizer not loaded. Please upload .npy file or ensure 'custom_stain_matrix_vahadane.npy' is in the correct directory.")
    else:
        st.sidebar.success("Stain normalizer loaded.")

# Grad-CAM layer selection
conv_candidates = []
if model is not None:
    all_layers = get_all_layers(model)
    for l in all_layers:
        if isinstance(l, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
            conv_candidates.append(l.name)
layer_options = conv_candidates if conv_candidates else ['top_conv', 'block6i_project_conv', 'block6i_dwconv2']
preferred_conv_layer = st.sidebar.selectbox("Grad-CAM conv layer (deeper layers first)", layer_options[::-1], index=0, help="Select a shallower layer (e.g., block5a_project_conv) if gradients vanish.")

uploaded_file = st.file_uploader("Upload an image to classify and inspect.", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        pil_img = Image.open(uploaded_file)
        rgb = pil_to_rgb_array(pil_img)
    except Exception as e:
        st.error(f"Could not read uploaded image: {e}")
        st.stop()

    if model is None:
        st.error("Model not loaded. Please upload a model file or ensure 'WBC CLassifier.keras' is in the correct directory.")
        st.stop()

    if normalizer is not None:
        try:
            vahadane_rgb = apply_vahadane(normalizer, rgb)
        except Exception as e:
            st.warning(f"Vahadane apply failed: {e}")
            vahadane_rgb = rgb.copy()
    else:
        st.info("No normalizer loaded — using original as normalized output.")
        vahadane_rgb = rgb.copy()

    model_input = prepare_for_model(vahadane_rgb, target_size=IMG_SIZE)

    # Predict without any scaling (use raw model outputs)
    logits = model(model_input, training=False).numpy()  # raw output
    if logits.ndim == 1:
        logits = np.expand_dims(logits, axis=0)

    pred_class = int(np.argmax(logits[0]))
    pred_conf = float(logits[0][pred_class])

    # Grad-CAM
    try:
        cam = gradcam(model, model_input, class_index=pred_class, preferred_last_conv=preferred_conv_layer)
    except Exception as e:
        st.warning(f"Grad-CAM failed: {e}")
        cam = None

    # Convert image to displayable uint8
    def to_display_uint8(arr):
        if arr.dtype == np.uint8:
            return arr
        mn = float(arr.min())
        mx = float(arr.max())
        if mn >= 0.0 and mx <= 1.0:
            return (arr * 255).astype(np.uint8)
        return np.clip(((arr - mn) / (mx - mn + 1e-8) * 255.0), 0, 255).astype(np.uint8)

    disp_orig = to_display_uint8(rgb)
    disp_vah = to_display_uint8(vahadane_rgb)

    if cam is not None:
        cam_resized = cv2.resize(cam, (disp_orig.shape[1], disp_orig.shape[0]))
        disp_bgr = cv2.cvtColor(disp_vah, cv2.COLOR_RGB2BGR)
        overlay_bgr = overlay_heatmap_on_image(disp_bgr, cam_resized, alpha=0.5)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    else:
        overlay_rgb = disp_vah.copy()

    # Display results
    st.markdown("### Results")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(disp_orig, caption="Original", use_container_width=True)
    with col2:
        st.image(disp_vah, caption="Vahadane normalized", use_container_width=True)
    with col3:
        st.image(overlay_rgb, caption=f"Grad-CAM — Pred: {labels[pred_class]} ({pred_conf:.2f} raw)", use_container_width=True)

    # Confidence bar plot (raw logits)
    st.markdown("#### Raw confidence scores (all classes)")
    fig, ax = plt.subplots(figsize=(8, 3))
    x = np.arange(len(labels))
    scores = logits[0]
    bars = ax.bar(x, scores, tick_label=labels, color=['orange' if i==pred_class else 'skyblue' for i in range(len(labels))])
    ax.set_ylabel("Raw confidence")
    ax.set_ylim([min(scores)*0.9, max(scores)*1.1])  # dynamic scaling
    for rect, val in zip(bars, scores):
        ax.text(rect.get_x() + rect.get_width()/2.0, val + (max(scores)*0.02), f"{val:.2f}", ha='center', va='bottom', fontsize=8)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

else:
    st.info("Upload an image to start. You can upload a deep learning model and stain normalizer (.npy) to override defaults.")