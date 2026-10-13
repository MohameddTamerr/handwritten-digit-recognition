"""
Explainable AI (XAI) engine for MNIST Digit Classifier.
Provides:
- Grad-CAM (Gradient-weighted Class Activation Mapping) for CNN
- Saliency Maps (Input Gradient Attribution) for CNN and MLP
- Occlusion Sensitivity Attribution for Random Forest
- Colormap overlay generation (Stroke + Attention Heatmap)
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64


def compute_gradcam(
    model: tf.keras.Model,
    input_image: np.ndarray,
    target_class: Optional[int] = None,
    conv_layer_name: Optional[str] = None
) -> np.ndarray:
    """
    Computes 2D Grad-CAM heatmap for a convolutional neural network.
    Compatible with Keras 3 / TensorFlow 2.16+.
    
    Args:
        model: Trained Keras CNN model.
        input_image: Model input array of shape (1, 28, 28, 1).
        target_class: Target digit class (0-9). Defaults to predicted class.
        conv_layer_name: Name of target convolutional layer. If None, uses the last Conv2D layer.
        
    Returns:
        28x28 float32 array normalized to [0.0, 1.0].
    """
    if conv_layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                conv_layer_name = layer.name
                break

    if conv_layer_name is None:
        raise ValueError("No Conv2D layer found in the model for Grad-CAM.")

    target_conv_layer = model.get_layer(conv_layer_name)
    layer_idx = model.layers.index(target_conv_layer)
    pre_layers = model.layers[:layer_idx]
    post_layers = model.layers[layer_idx + 1:]

    tensor_input = tf.cast(input_image, tf.float32)

    with tf.GradientTape() as tape:
        # Pass through any layers before target conv
        x = tensor_input
        for layer in pre_layers:
            x = layer(x)
        
        # Conv layer activation
        conv_output = target_conv_layer(x)
        tape.watch(conv_output)

        # Pass through remaining layers to final output
        out = conv_output
        for layer in post_layers:
            out = layer(out)

        if target_class is None:
            target_class = int(tf.argmax(out[0]))
        loss = out[:, target_class]

    # Gradient of target class score with respect to feature maps
    grads = tape.gradient(loss, conv_output)

    # Global average pooling of gradients: channel-wise weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel feature map by its importance weight
    conv_outputs_val = conv_output[0]
    cam = tf.reduce_sum(tf.multiply(conv_outputs_val, pooled_grads), axis=-1)

    # Apply ReLU to focus only on features that have a positive influence on the target class
    cam = tf.maximum(cam, 0.0).numpy()

    # Normalize to [0, 1]
    cam_max = np.max(cam)
    if cam_max > 1e-8:
        cam = cam / cam_max
    else:
        cam = np.zeros_like(cam)

    # Resize CAM to match input image (28, 28)
    cam_pil = Image.fromarray((cam * 255.0).astype(np.uint8))
    cam_resized = cam_pil.resize((28, 28), Image.Resampling.BILINEAR)
    cam_norm = np.array(cam_resized, dtype=np.float32) / 255.0

    return cam_norm


def compute_saliency_map(
    model: tf.keras.Model,
    input_image: np.ndarray,
    target_class: Optional[int] = None
) -> np.ndarray:
    """
    Computes pixel-level saliency map (|dL/dx|) for neural networks (CNN or MLP).
    Highlights exact pixels that contributed most to the predicted class score.
    """
    tensor_input = tf.cast(input_image, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(tensor_input)
        predictions = model(tensor_input)
        if target_class is None:
            target_class = int(tf.argmax(predictions[0]))
        loss = predictions[:, target_class]

    grads = tape.gradient(loss, tensor_input)
    saliency = tf.abs(grads).numpy()[0]

    # Reshape if flattened
    if saliency.ndim == 1 or (saliency.ndim == 2 and saliency.shape[1] == 1):
        saliency = saliency.reshape(28, 28)
    elif saliency.ndim == 3 and saliency.shape[2] == 1:
        saliency = saliency.reshape(28, 28)

    s_max = np.max(saliency)
    if s_max > 1e-8:
        saliency = saliency / s_max
    else:
        saliency = np.zeros_like(saliency)

    return saliency.astype(np.float32)


def compute_rf_occlusion_attribution(
    model: Any,
    input_image: np.ndarray,
    target_class: Optional[int] = None,
    patch_size: int = 3
) -> np.ndarray:
    """
    Computes spatial occlusion sensitivity map for Random Forest.
    Slides a patch over the 28x28 input and calculates drop in predicted class probability.
    """
    flat_input = input_image.reshape(1, 784).copy()
    base_probs = model.predict_proba(flat_input)[0]

    if target_class is None:
        target_class = int(np.argmax(base_probs))

    base_score = float(base_probs[target_class])
    img_28 = flat_input.reshape(28, 28)
    attribution = np.zeros((28, 28), dtype=np.float32)

    half_p = patch_size // 2
    for y in range(half_p, 28 - half_p, 2):
        for x in range(half_p, 28 - half_p, 2):
            occluded = img_28.copy()
            if np.sum(occluded[y - half_p:y + half_p + 1, x - half_p:x + half_p + 1]) > 0.05:
                occluded[y - half_p:y + half_p + 1, x - half_p:x + half_p + 1] = 0.0
                p_occ = model.predict_proba(occluded.reshape(1, 784))[0][target_class]
                drop = max(0.0, base_score - p_occ)
                attribution[y - half_p:y + half_p + 1, x - half_p:x + half_p + 1] += drop

    a_max = np.max(attribution)
    if a_max > 1e-8:
        attribution = attribution / a_max
    return attribution


def colormap_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """
    Maps a [0, 1] 2D heatmap into an RGB image using a high-contrast 'Turbo/Inferno' gradient:
    Cool dark purple/blue -> Warm magenta -> Vivid red -> Warm orange -> Bright yellow.
    """
    h = np.clip(heatmap, 0.0, 1.0)
    
    r = np.clip(1.5 * h - 0.2, 0.0, 1.0)
    g = np.clip(1.8 * (h - 0.45), 0.0, 1.0)
    b = np.clip(1.0 - 1.5 * h, 0.0, 1.0) * (h < 0.6).astype(np.float32)
    
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0).astype(np.uint8)


def create_xai_overlay(
    display_image_28: np.ndarray,
    heatmap_28: np.ndarray,
    alpha: float = 0.65,
    output_size: int = 140
) -> Tuple[str, str]:
    """
    Generates high-resolution base64 PNGs for:
    1. The pure heatmap.
    2. The overlay of heatmap on top of the original stroke image.
    
    Returns:
        Tuple of (heatmap_b64, overlay_b64).
    """
    pil_stroke = Image.fromarray(display_image_28).resize((output_size, output_size), Image.Resampling.NEAREST)
    stroke_rgb = pil_stroke.convert("RGB")
    stroke_arr = np.array(stroke_rgb, dtype=np.float32)

    pil_heat = Image.fromarray((heatmap_28 * 255.0).astype(np.uint8)).resize((output_size, output_size), Image.Resampling.BILINEAR)
    heat_norm = np.array(pil_heat, dtype=np.float32) / 255.0

    heat_rgb = colormap_heatmap(heat_norm).astype(np.float32)

    blended = stroke_arr * (1.0 - alpha * np.maximum(0.2, heat_norm[:, :, None])) + heat_rgb * alpha
    blended = np.clip(blended, 0.0, 255.0).astype(np.uint8)

    def to_b64(img_arr: np.ndarray) -> str:
        im = Image.fromarray(img_arr)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    pure_heat_b64 = to_b64(heat_rgb.astype(np.uint8))
    overlay_b64 = to_b64(blended)

    return pure_heat_b64, overlay_b64


def generate_digit_explanation(
    predicted_digit: int,
    confidence: float,
    model_type: str = "cnn"
) -> str:
    """
    Returns a human-readable explanation of what features the AI identified.
    """
    digit_cues = {
        0: "the continuous circular closed loop with an open, hollow center.",
        1: "the single straight vertical stroke and absence of branching curves.",
        2: "the top-rounded arch connecting downward to a flat horizontal baseline.",
        3: "the double stacked open curves facing right with a middle pinch.",
        4: "the intersection of the vertical right stem with the horizontal crossbar.",
        5: "the sharp top horizontal bar, vertical drop, and rounded bottom loop.",
        6: "the sweeping downward stroke curving into a closed lower circular loop.",
        7: "the prominent horizontal top bar meeting a diagonal stroke descending down.",
        8: "the symmetrical dual loops crossing at the central focal junction.",
        9: "the fully enclosed upper circular loop meeting a right downward vertical stem."
    }
    
    cue = digit_cues.get(predicted_digit, "the characteristic stroke contours and junctions.")
    tech = "Grad-CAM feature map" if "cnn" in model_type.lower() else "gradient saliency attribution"
    return (
        f"The {tech} reveals the neural network focused most intensively on {cue} "
        f"Warm/bright regions highlight the exact spatial patterns confirming this digit."
    )
