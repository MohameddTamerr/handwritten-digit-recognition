"""
Model loading and prediction utilities for MNIST Digit Classifier.
Uses @st.cache_resource for caching and handles CNN, MLP, and Random Forest models.
Integrates Out-of-Distribution (non-digit) validation and Explainable AI (XAI: Grad-CAM / Saliency).
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import joblib
import numpy as np
import streamlit as st
import tensorflow as tf

from utils.ood_detector import validate_digit
from utils.xai import (
    compute_gradcam,
    compute_saliency_map,
    compute_rf_occlusion_attribution,
    create_xai_overlay,
    generate_digit_explanation
)

# Suppress TF logging
tf.get_logger().setLevel("ERROR")

# Base directory for relative paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# Available models defined from 1_keras_sequential_exercise.ipynb
AVAILABLE_MODELS: Dict[str, Dict[str, Any]] = {
    "Convolutional Neural Network (CNN)": {
        "file": "mnist_cnn.keras",
        "type": "cnn",
        "description": "Convolutional Neural Network with 32 filters, 3x3 Conv2D, Max Pooling, and Dense layer.",
        "input_shape": "(1, 28, 28, 1)",
        "test_accuracy": "98.38%"
    },
    "Neural Network (MLP)": {
        "file": "mnist_mlp.keras",
        "type": "mlp",
        "description": "Fully connected Multi-Layer Perceptron with 256 hidden units and Dropout (0.2).",
        "input_shape": "(1, 784)",
        "test_accuracy": "98.00%"
    },
    "Random Forest Classifier": {
        "file": "mnist_random_forest.joblib",
        "type": "rf",
        "description": "Ensemble of 150 decision trees with sqrt max features and min_samples_leaf=1.",
        "input_shape": "(1, 784)",
        "test_accuracy": "96.98%"
    }
}


@st.cache_resource(show_spinner=False)
def load_trained_model(model_name: str) -> Any:
    """
    Loads and caches the trained model from disk.
    Supports Keras (.keras) and Scikit-Learn (.joblib) models.
    Pre-warms the model to ensure instant subsequent predictions.
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model name: {model_name}")

    model_info = AVAILABLE_MODELS[model_name]
    model_path = MODELS_DIR / model_info["file"]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at: {model_path}. "
            f"Please ensure all models are present in the 'models' directory."
        )

    model_type = model_info["type"]
    if model_type in ("cnn", "mlp"):
        model = tf.keras.models.load_model(str(model_path))
        dummy_shape = (1, 28, 28, 1) if model_type == "cnn" else (1, 784)
        dummy_input = np.zeros(dummy_shape, dtype=np.float32)
        model.predict(dummy_input, verbose=0)
    elif model_type == "rf":
        model = joblib.load(str(model_path))
        dummy_input = np.zeros((1, 784), dtype=np.float32)
        model.predict_proba(dummy_input)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model


def predict_digit(
    model_name: str,
    model_input: np.ndarray,
    display_image: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Runs prediction using the chosen model and returns:
    - is_digit (bool): Whether input is a valid handwritten digit or a letter/drawing
    - predicted_digit (Optional[int])
    - confidence (float percentage)
    - probabilities (np.ndarray of length 10)
    - top_5 (list of (digit, probability_percent))
    - has_probabilities (bool)
    - xai (dict): Explainable AI artifacts (heatmap, overlay b64, text explanation)
    - validation_reason (str)
    """
    model = load_trained_model(model_name)
    model_info = AVAILABLE_MODELS[model_name]
    model_type = model_info["type"]

    has_probabilities = True
    probabilities = np.zeros(10, dtype=np.float32)

    if model_type in ("cnn", "mlp"):
        raw_preds = model.predict(model_input, verbose=0)
        probabilities = raw_preds[0].astype(np.float32)
    elif model_type == "rf":
        if hasattr(model, "predict_proba"):
            raw_preds = model.predict_proba(model_input)
            probabilities = raw_preds[0].astype(np.float32)
        else:
            has_probabilities = False
            pred_class = model.predict(model_input)[0]
            probabilities[pred_class] = 1.0

    predicted_digit = int(np.argmax(probabilities))
    max_prob = float(probabilities[predicted_digit])
    confidence = max_prob * 100.0

    # Top 5 sorted predictions
    sorted_indices = np.argsort(probabilities)[::-1]
    top_5 = [
        (int(idx), float(probabilities[idx] * 100.0))
        for idx in sorted_indices[:5]
    ]

    # Run Out-of-Distribution & Non-Digit Validation
    validation = validate_digit(model_input, display_image, probabilities)
    is_digit = validation["is_digit"]

    # Generate Explainable AI (XAI) visualizations
    xai_data = None
    if display_image is not None:
        try:
            if model_type == "cnn":
                heatmap = compute_gradcam(model, model_input, target_class=predicted_digit)
                xai_method = "Grad-CAM"
            elif model_type == "mlp":
                heatmap = compute_saliency_map(model, model_input, target_class=predicted_digit)
                xai_method = "Saliency Map"
            else:
                heatmap = compute_rf_occlusion_attribution(model, model_input, target_class=predicted_digit)
                xai_method = "Occlusion Sensitivity"

            pure_heat_b64, overlay_b64 = create_xai_overlay(display_image, heatmap)
            explanation_text = generate_digit_explanation(predicted_digit, confidence, model_type)

            xai_data = {
                "method": xai_method,
                "heatmap_b64": pure_heat_b64,
                "overlay_b64": overlay_b64,
                "explanation": explanation_text
            }
        except Exception as e:
            xai_data = None

    return {
        "is_digit": is_digit,
        "validation": validation,
        "validation_reason": validation["reason"],
        "predicted_digit": predicted_digit if is_digit else None,
        "candidate_digit": predicted_digit,
        "confidence": confidence,
        "probabilities": probabilities,
        "top_5": top_5,
        "has_probabilities": has_probabilities,
        "model_type": model_type,
        "model_info": model_info,
        "xai": xai_data
    }
