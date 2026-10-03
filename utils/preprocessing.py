"""
Preprocessing pipeline for converting canvas drawings into MNIST-compatible inputs.
Supports both single-digit and multi-digit recognition (segmenting multiple digits left-to-right).
Matches the exact preprocessing and normalization from 1_keras_sequential_exercise.ipynb.
"""

from typing import Tuple, Dict, Any, List
import numpy as np
import scipy.ndimage as ndi
from PIL import Image


class EmptyCanvasError(ValueError):
    """Raised when no digit has been drawn on the canvas."""
    pass


def extract_stroke_intensity(image_data: np.ndarray) -> np.ndarray:
    """
    Extracts stroke intensity from canvas RGBA image.
    Supports black and colored strokes drawn on a white canvas.
    Returns a 2D float32 array where background is 0.0 and strokes are up to 255.0.
    """
    if image_data is None or image_data.size == 0:
        raise EmptyCanvasError("Please draw a digit before making a prediction.")

    if len(image_data.shape) == 3 and image_data.shape[2] == 4:
        r = image_data[:, :, 0].astype(np.float32)
        g = image_data[:, :, 1].astype(np.float32)
        b = image_data[:, :, 2].astype(np.float32)
        a = image_data[:, :, 3].astype(np.float32) / 255.0

        diff_r = np.maximum(0.0, 255.0 - r)
        diff_g = np.maximum(0.0, 255.0 - g)
        diff_b = np.maximum(0.0, 255.0 - b)
        stroke = np.maximum(np.maximum(diff_r, diff_g), diff_b) * a
    elif len(image_data.shape) == 3 and image_data.shape[2] == 3:
        r = image_data[:, :, 0].astype(np.float32)
        g = image_data[:, :, 1].astype(np.float32)
        b = image_data[:, :, 2].astype(np.float32)
        diff_r = np.maximum(0.0, 255.0 - r)
        diff_g = np.maximum(0.0, 255.0 - g)
        diff_b = np.maximum(0.0, 255.0 - b)
        stroke = np.maximum(np.maximum(diff_r, diff_g), diff_b)
    else:
        stroke = 255.0 - image_data.astype(np.float32)

    return stroke


def center_image_by_mass(img_28: np.ndarray) -> np.ndarray:
    """
    Centers the 28x28 digit using its center of mass,
    aligning with the standard MNIST dataset preprocessing methodology.
    """
    total_mass = np.sum(img_28)
    if total_mass <= 1e-4:
        return img_28

    y_indices, x_indices = np.indices(img_28.shape)
    cy = np.sum(y_indices * img_28) / total_mass
    cx = np.sum(x_indices * img_28) / total_mass

    # Target center is 13.5 (the middle of 0..27)
    shift_y = int(np.round(13.5 - cy))
    shift_x = int(np.round(13.5 - cx))

    # Keep shift bounded so we do not shift the digit out of frame
    shift_y = int(np.clip(shift_y, -5, 5))
    shift_x = int(np.clip(shift_x, -5, 5))

    if shift_y == 0 and shift_x == 0:
        return img_28

    shifted = np.roll(img_28, (shift_y, shift_x), axis=(0, 1))

    if shift_y > 0:
        shifted[:shift_y, :] = 0.0
    elif shift_y < 0:
        shifted[shift_y:, :] = 0.0

    if shift_x > 0:
        shifted[:, :shift_x] = 0.0
    elif shift_x < 0:
        shifted[:, shift_x:] = 0.0

    return shifted


def crop_and_normalize_digit(cropped_stroke: np.ndarray, model_type: str = "cnn") -> Tuple[np.ndarray, np.ndarray]:
    """
    Takes a cropped 2D stroke image of a single digit:
    1. Pads slightly.
    2. Rescales to fit inside a 20x20 box preserving aspect ratio.
    3. Centers into 28x28 using center of mass.
    4. Normalizes to [0.0, 1.0].
    5. Formats input shape for the target model.
    """
    # Add a small padding around the cropped digit
    padded = np.pad(cropped_stroke, pad_width=4, mode="constant", constant_values=0.0)

    # Scale cropped digit into at most 20x20 while preserving aspect ratio
    h, w = padded.shape
    scale = 20.0 / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    pil_crop = Image.fromarray(padded.astype(np.uint8))
    pil_resized = pil_crop.resize((new_w, new_h), Image.Resampling.LANCZOS)
    resized_arr = np.array(pil_resized, dtype=np.float32)

    # Embed within a 28x28 canvas
    canvas_28 = np.zeros((28, 28), dtype=np.float32)
    top = (28 - new_h) // 2
    left = (28 - new_w) // 2
    canvas_28[top:top + new_h, left:left + new_w] = resized_arr

    # Center by center of mass (standard MNIST centering)
    canvas_28 = center_image_by_mass(canvas_28)

    # Normalize to [0.0, 1.0] float32 as in training notebook
    normalized_28 = np.clip(canvas_28 / 255.0, 0.0, 1.0).astype(np.float32)

    # 28x28 display image (uint8, 0=black, 255=white)
    display_image = (normalized_28 * 255.0).astype(np.uint8)

    # Format model input
    model_type_clean = model_type.lower()
    if "cnn" in model_type_clean:
        model_input = normalized_28.reshape(1, 28, 28, 1)
    else:
        model_input = normalized_28.reshape(1, 784)

    return model_input, display_image


def preprocess_canvas_image(
    image_data: np.ndarray,
    model_type: str = "cnn"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Single-digit preprocessing pipeline (backward compatible).
    """
    stroke = extract_stroke_intensity(image_data)

    active_pixels = stroke > 25.0
    if np.sum(active_pixels) < 35 or np.max(stroke) < 30.0:
        raise EmptyCanvasError("Please draw a digit before making a prediction.")

    rows, cols = np.where(active_pixels)
    rmin, rmax = np.min(rows), np.max(rows)
    cmin, cmax = np.min(cols), np.max(cols)

    cropped = stroke[rmin:rmax + 1, cmin:cmax + 1]
    return crop_and_normalize_digit(cropped, model_type)


def preprocess_multi_digit_canvas(
    image_data: np.ndarray,
    model_type: str = "cnn"
) -> List[Dict[str, Any]]:
    """
    Multi-digit segmentation & preprocessing:
    1. Extracts stroke intensity.
    2. Detects whether multiple separate digits were drawn horizontally.
    3. Segments digits from left to right.
    4. Preprocesses each digit individually into standard 28x28 MNIST format.

    Returns:
        List of dicts: [
            {"model_input": np.ndarray, "display_image": np.ndarray, "bbox": (ymin, ymax, xmin, xmax)},
            ...
        ]
    """
    stroke = extract_stroke_intensity(image_data)

    active_pixels = stroke > 25.0
    if np.sum(active_pixels) < 35 or np.max(stroke) < 30.0:
        raise EmptyCanvasError("Please draw a digit before making a prediction.")

    # Morphological dilation to connect strokes within the same digit (e.g. crossbars in '7' or '4')
    # but preserve separation between distinct side-by-side digits
    struct = np.ones((9, 14), dtype=bool)
    dilated = ndi.binary_dilation(active_pixels, structure=struct)
    labeled, num_features = ndi.label(dilated)

    slices = ndi.find_objects(labeled)

    # Filter out tiny noise specks (< 35 active pixels)
    valid_components = []
    for s in slices:
        sy, sx = s
        comp_mask = active_pixels[sy, sx]
        if np.sum(comp_mask) >= 35:
            valid_components.append(s)

    if len(valid_components) == 0:
        # Fallback to single bounding box
        rows, cols = np.where(active_pixels)
        cropped = stroke[np.min(rows):np.max(rows) + 1, np.min(cols):np.max(cols) + 1]
        m_input, disp = crop_and_normalize_digit(cropped, model_type)
        return [{"model_input": m_input, "display_image": disp}]

    # Sort digits from left to right by their horizontal start coordinate
    valid_components = sorted(valid_components, key=lambda s: s[1].start)

    digit_items = []
    for s in valid_components:
        sy, sx = s
        # Crop the actual stroke within this component's bounding box
        digit_crop = stroke[sy, sx]
        m_input, disp = crop_and_normalize_digit(digit_crop, model_type)
        digit_items.append({
            "model_input": m_input,
            "display_image": disp,
            "bbox": (sy.start, sy.stop, sx.start, sx.stop)
        })

    return digit_items
