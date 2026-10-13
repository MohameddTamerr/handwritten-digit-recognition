"""
Out-of-Distribution (OOD) & Non-Digit Validator for MNIST Classifier.
Detects whether an input drawing is a genuine digit (0-9) or a letter, doodle,
arbitrary drawing, or noise. Prevents false digit predictions on non-digit inputs.
Powered by a deep CNN trained on 80,000 real MNIST digits, EMNIST handwritten letters,
and doodle shapes.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import joblib
import numpy as np
import scipy.ndimage as ndi
import streamlit as st
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent.parent
VALIDATOR_KERAS_PATH = BASE_DIR / "models" / "digit_validator.keras"
VALIDATOR_JOBLIB_PATH = BASE_DIR / "models" / "digit_validator.joblib"


@st.cache_resource(show_spinner=False)
def load_validator_cnn() -> Optional[Any]:
    """Loads and caches the trained CNN digit-vs-non-digit validator model."""
    if not VALIDATOR_KERAS_PATH.exists():
        return None
    try:
        model = tf.keras.models.load_model(str(VALIDATOR_KERAS_PATH))
        # Warmup
        model.predict(np.zeros((1, 28, 28, 1), dtype=np.float32), verbose=0)
        return model
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_validator_joblib() -> Optional[Dict[str, Any]]:
    """Loads and caches the joblib PCA manifold bundle if available."""
    if not VALIDATOR_JOBLIB_PATH.exists():
        return None
    try:
        return joblib.load(str(VALIDATOR_JOBLIB_PATH))
    except Exception:
        return None


def count_holes_and_components(image_28: np.ndarray) -> Tuple[int, int]:
    """
    Computes topological properties of the stroke:
    - holes_count: number of enclosed topological holes (Euler characteristic check)
    - comp_count: number of disconnected stroke components
    """
    binary = image_28 > 0.15
    if not np.any(binary):
        return 0, 0

    filled = ndi.binary_fill_holes(binary)
    holes_mask = filled & (~binary)

    labeled_holes, num_holes = ndi.label(holes_mask)
    valid_holes = 0
    for h_id in range(1, num_holes + 1):
        if np.sum(labeled_holes == h_id) >= 4:
            valid_holes += 1

    labeled_comp, num_comp = ndi.label(binary)
    valid_comp = 0
    for c_id in range(1, num_comp + 1):
        if np.sum(labeled_comp == c_id) >= 6:
            valid_comp += 1

    return valid_holes, valid_comp


def validate_digit(
    model_input: np.ndarray,
    display_image: Optional[np.ndarray] = None,
    model_probabilities: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Validates whether the drawn image is a legitimate digit (0-9) or an out-of-distribution
    input (letters, doodles, scribbles, arbitrary drawings).
    
    Returns:
        dict: {
            "is_digit": bool,
            "digit_score": float (0-100%),
            "reason": str,
            "reconstruction_error": float,
            "holes": int,
            "components": int
        }
    """
    flat = model_input.reshape(1, 784).astype(np.float32)
    img_28 = flat.reshape(28, 28)

    active_pixel_count = int(np.sum(img_28 > 0.12))
    
    # Check 1: Empty or extremely faint stroke
    if active_pixel_count < 20:
        return {
            "is_digit": False,
            "digit_score": 0.0,
            "reason": "Drawing is too faint or empty to be a digit.",
            "reconstruction_error": 0.0,
            "holes": 0,
            "components": 0
        }

    # Check 2: Dense solid scribbles (massive fill)
    if active_pixel_count > 460:
        return {
            "is_digit": False,
            "digit_score": 0.0,
            "reason": "Drawing is heavily filled or scribbled, not a standard digit.",
            "reconstruction_error": 0.08,
            "holes": 0,
            "components": 1
        }

    # Check 3: Topological hole & component count
    holes, components = count_holes_and_components(img_28)

    # Standard digits (0-9) never have more than 2 holes ('8' has 2, '0','4','6','9' have 1, others have 0)
    if holes >= 3:
        return {
            "is_digit": False,
            "digit_score": 0.0,
            "reason": "Drawing contains too many closed loops or complex doodle patterns.",
            "reconstruction_error": 0.06,
            "holes": holes,
            "components": components
        }

    # Digits rarely have 4+ disconnected fragments
    if components >= 4:
        return {
            "is_digit": False,
            "digit_score": 0.0,
            "reason": "Input contains scattered fragments or stray marks.",
            "reconstruction_error": 0.05,
            "holes": holes,
            "components": components
        }

    # Check 4: Deep CNN Validator (trained on 80,000 real MNIST digits & EMNIST letters)
    cnn_model = load_validator_cnn()
    prob_digit = 1.0

    if cnn_model is not None:
        tensor_in = img_28.reshape(1, 28, 28, 1)
        raw_pred = cnn_model.predict(tensor_in, verbose=0)
        prob_digit = float(raw_pred[0, 0])
    else:
        # Fallback to joblib bundle if keras model is missing
        bundle = load_validator_joblib()
        if bundle and "classifier" in bundle:
            clf = bundle["classifier"]
            prob_digit = float(clf.predict_proba(flat)[0][1])

    # Check 5: PCA Reconstruction Manifold Error
    recon_error = 0.02
    bundle = load_validator_joblib()
    if bundle and "pca" in bundle:
        pca = bundle["pca"]
        recon = pca.inverse_transform(pca.transform(flat))
        recon_error = float(np.mean((flat - recon) ** 2))

    # Check 6: Target Classifier Uncertainty
    classifier_uncertain = False
    if model_probabilities is not None and len(model_probabilities) == 10:
        sorted_p = np.sort(model_probabilities)
        max_p = float(sorted_p[-1])
        runner_up = float(sorted_p[-2])
        margin = max_p - runner_up
        if max_p < 0.55 or margin < 0.18:
            classifier_uncertain = True

    digit_score_pct = float(prob_digit * 100.0)

    # Decision Logic:
    # 1. Letters and drawings score < 0.70 on the EMNIST/MNIST CNN validator
    # 2. Extreme manifold error (> 0.068) rejects complex doodles / stars
    # 3. High classifier uncertainty or high reconstruction error with borderline score
    if prob_digit < 0.70:
        is_digit = False
        reason = "Drawing recognized as a letter, symbol, or non-digit sketch."
    elif recon_error > 0.068:
        is_digit = False
        reason = "Drawing shape does not match the standard morphology of digits (0–9)."
    elif prob_digit < 0.85 and (recon_error > 0.048 or classifier_uncertain):
        is_digit = False
        reason = "Ambiguous strokes detected; does not clearly represent a single digit."
    else:
        is_digit = True
        reason = "Valid handwritten digit."

    return {
        "is_digit": is_digit,
        "digit_score": digit_score_pct,
        "reason": reason,
        "reconstruction_error": recon_error,
        "holes": holes,
        "components": components
    }
