import streamlit as st
import numpy as np
import io
import base64
from pathlib import Path
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from utils.preprocessing import preprocess_multi_digit_canvas, EmptyCanvasError
from utils.prediction import AVAILABLE_MODELS, load_trained_model, predict_digit

# --- Page Configuration ---
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="assets/app_icon.png" if Path("assets/app_icon.png").exists() else None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Cache optimized background and logo base64 once to eliminate flashing and DOM bloat
@st.cache_data
def get_static_assets():
    bg_jpg = Path("assets/bg_optimized.jpg")
    bg_png = Path("Pastel Numbers Learning Background.png")
    bg_b64 = ""
    if bg_jpg.exists():
        with open(bg_jpg, "rb") as f:
            bg_b64 = base64.b64encode(f.read()).decode("utf-8")
    elif bg_png.exists():
        with open(bg_png, "rb") as f:
            bg_b64 = base64.b64encode(f.read()).decode("utf-8")
            
    logo_path = Path("assets/app_icon.png")
    logo_b64 = ""
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            logo_b64 = base64.b64encode(f.read()).decode("utf-8")
            
    return bg_b64, logo_b64

BG_B64, LOGO_B64 = get_static_assets()

# Dynamic background CSS
if BG_B64:
    bg_css = f"""
    .stApp {{
        background-image: url('data:image/jpeg;base64,{BG_B64}') !important;
        background-size: cover !important;
        background-position: center center !important;
        background-attachment: fixed !important;
        background-repeat: no-repeat !important;
    }}
    """
else:
    bg_css = ".stApp { background: #FCFBF9 !important; }"

# --- Inject Clean Styling, Transitions & Professional SVG Icon Masks ---
st.html(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

/* Fast Non-Flashing Background */
{bg_css}

:root, html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stMain"], section[data-testid="stMain"], .main {{
    background-color: transparent !important;
    color: #312C51 !important;
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}}

/* Main Block Container */
.block-container {{
    padding-top: 1.4rem !important;
    padding-bottom: 2rem !important;
    max-width: 1140px !important;
    margin: 0 auto !important;
}}

@media (max-width: 768px) {{
    .block-container {{
        padding-left: 14px !important;
        padding-right: 14px !important;
        padding-top: 0.9rem !important;
    }}
}}

/* Hide Streamlit default headers & menu */
#MainMenu, footer, header, [data-testid="stToolbar"], [data-testid="stDecoration"] {{
    display: none !important;
}}

/* Keyframe Animations */
@keyframes fadeInUp {{
    from {{
        opacity: 0;
        transform: translateY(10px);
    }}
    to {{
        opacity: 1;
        transform: translateY(0);
    }}
}}

@keyframes predPop {{
    0% {{
        transform: scale(0.92);
        opacity: 0.5;
    }}
    60% {{
        transform: scale(1.03);
    }}
    100% {{
        transform: scale(1);
        opacity: 1;
    }}
}}

@keyframes subtlePulse {{
    0%, 100% {{
        box-shadow: 0 0 0 0 rgba(64, 57, 116, 0.4);
    }}
    50% {{
        box-shadow: 0 0 0 6px rgba(64, 57, 116, 0);
    }}
}}

/* Header Animation & Layout */
.header-wrapper {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
    flex-wrap: wrap;
    gap: 14px;
    animation: fadeInUp 0.35s cubic-bezier(0.16, 1, 0.3, 1) both;
}}

.header-left {{
    display: flex;
    align-items: center;
    gap: 14px;
}}

.header-logo-badge {{
    width: 48px;
    height: 48px;
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 2px 8px rgba(50, 46, 83, 0.05);
    flex-shrink: 0;
    transition: transform 0.2s ease;
}}

.header-logo-badge:hover {{
    transform: scale(1.05);
}}

.header-logo-badge img {{
    width: 36px;
    height: 36px;
    object-fit: contain;
    border-radius: 6px;
}}

.header-title-box h1 {{
    font-size: 24px !important;
    font-weight: 800 !important;
    color: #1E1B4B !important;
    margin: 0 !important;
    letter-spacing: -0.02em !important;
    line-height: 1.2 !important;
}}

.header-title-box p {{
    font-size: 13.5px !important;
    color: #64748B !important;
    margin: 3px 0 0 0 !important;
    font-weight: 500 !important;
}}

.header-tip-card {{
    background: #FFFFFF;
    border: 1px solid #FEF08A;
    border-radius: 12px;
    padding: 10px 18px;
    display: flex;
    align-items: center;
    gap: 10px;
    box-shadow: 0 2px 8px rgba(50, 46, 83, 0.03);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}}

.header-tip-card:hover {{
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(245, 158, 11, 0.1);
}}

.tip-text {{
    font-size: 12px;
    color: #475569;
    font-weight: 500;
    line-height: 1.35;
}}

/* Card Containers */
div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: #FFFFFF !important;
    border-radius: 18px !important;
    border: 1px solid #EDEEF2 !important;
    box-shadow: 0 4px 20px rgba(50, 46, 83, 0.05) !important;
    padding: 20px 22px !important;
    margin-bottom: 14px !important;
    animation: fadeInUp 0.4s cubic-bezier(0.16, 1, 0.3, 1) both;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}}

div[data-testid="stVerticalBlockBorderWrapper"]:hover {{
    box-shadow: 0 6px 24px rgba(50, 46, 83, 0.08) !important;
}}

@media (max-width: 768px) {{
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        padding: 14px 14px !important;
        border-radius: 16px !important;
    }}
}}

/* Card Headers */
.card-header {{
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 16px;
    font-weight: 700;
    color: #1E1B4B;
    margin-bottom: 10px;
}}

.card-divider {{
    border-bottom: 1px dashed #E2E8F0;
    margin-top: 4px;
    margin-bottom: 14px;
}}

/* Canvas Outer Frame */
.canvas-outer-frame {{
    border: 1.5px solid #E2E8F0;
    border-radius: 12px;
    padding: 0;
    background: #FFFFFF;
    display: flex;
    justify-content: center;
    align-items: center;
    margin-bottom: 14px;
    overflow: hidden;
    width: 100%;
}}

.canvas-outer-frame iframe {{
    border-radius: 10px !important;
    display: block !important;
    margin: 0 auto !important;
}}

/* Toolbar Labels */
.toolbar-label {{
    font-size: 12px;
    font-weight: 600;
    color: #64748B;
    margin-bottom: 6px;
    display: block;
}}

/* Keep toolbar sub-columns horizontal */
div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-brush_btn_"]),
div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-color_btn_"]),
div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-act_"]) {{
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    align-items: center !important;
    gap: 6px !important;
}}

div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-brush_btn_"]) > div[data-testid="column"],
div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-color_btn_"]) > div[data-testid="column"],
div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-act_"]) > div[data-testid="column"] {{
    width: auto !important;
    flex: 1 1 0px !important;
    min-width: 0 !important;
}}

/* Action Buttons (Draw, Rubber, Undo, Redo, Clear) */
div[class*="st-key-act_"] button {{
    height: 40px !important;
    font-size: 13px !important;
    border-radius: 10px !important;
    width: 100% !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 6px !important;
    transition: all 0.15s cubic-bezier(0.2, 0.8, 0.2, 1) !important;
    cursor: pointer !important;
}}

div[class*="st-key-act_"] button:hover {{
    transform: translateY(-1px) !important;
}}

div[class*="st-key-act_"] button:active {{
    transform: scale(0.97) !important;
}}

div[class*="st-key-act_draw_active"] button,
div[class*="st-key-act_rubber_active"] button {{
    background: #403974 !important;
    color: #FFFFFF !important;
    border: none !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 8px rgba(64, 57, 116, 0.25) !important;
}}

div[class*="st-key-act_rubber_active"] button {{
    animation: subtlePulse 2s infinite !important;
}}

div[class*="st-key-act_draw_inactive"] button,
div[class*="st-key-act_rubber_inactive"] button,
div[class*="st-key-act_undo"] button,
div[class*="st-key-act_redo"] button,
div[class*="st-key-act_clear"] button {{
    background: #FFFFFF !important;
    color: #334155 !important;
    border: 1px solid #E2E8F0 !important;
    font-weight: 500 !important;
    box-shadow: none !important;
}}

div[class*="st-key-act_draw_inactive"] button:hover,
div[class*="st-key-act_rubber_inactive"] button:hover,
div[class*="st-key-act_undo"] button:hover,
div[class*="st-key-act_redo"] button:hover,
div[class*="st-key-act_clear"] button:hover {{
    background: #F8FAFC !important;
    border-color: #CBD5E1 !important;
}}

/* Professional SVG Icons for Action Buttons via CSS Masks */
div[class*="st-key-act_draw"] button::before {{
    content: "";
    display: inline-block;
    width: 14px;
    height: 14px;
    background-color: currentColor;
    -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2.3' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M17 3a2.828 2.828 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5L17 3z'/%3E%3C/svg%3E") no-repeat center;
    mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2.3' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M17 3a2.828 2.828 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5L17 3z'/%3E%3C/svg%3E") no-repeat center;
    -webkit-mask-size: contain;
    mask-size: contain;
}}

div[class*="st-key-act_rubber"] button::before {{
    content: "";
    display: inline-block;
    width: 14px;
    height: 14px;
    background-color: currentColor;
    -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2.3' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m7 21-4.3-4.3c-1-1-1-2.5 0-3.4l9.6-9.6c1-1 2.5-1 3.4 0l5.6 5.6c1 1 1 2.5 0 3.4L13 21'/%3E%3Cpath d='M22 21H7'/%3E%3Cpath d='m5 11 9 9'/%3E%3C/svg%3E") no-repeat center;
    mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2.3' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m7 21-4.3-4.3c-1-1-1-2.5 0-3.4l9.6-9.6c1-1 2.5-1 3.4 0l5.6 5.6c1 1 1 2.5 0 3.4L13 21'/%3E%3Cpath d='M22 21H7'/%3E%3Cpath d='m5 11 9 9'/%3E%3C/svg%3E") no-repeat center;
    -webkit-mask-size: contain;
    mask-size: contain;
}}

div[class*="st-key-act_undo"] button::before {{
    content: "";
    display: inline-block;
    width: 14px;
    height: 14px;
    background-color: currentColor;
    -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2.3' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M3 7v6h6'/%3E%3Cpath d='M21 17a9 9 0 0 0-9-9 9 9 0 0 0-6 2.3L3 13'/%3E%3C/svg%3E") no-repeat center;
    mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2.3' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M3 7v6h6'/%3E%3Cpath d='M21 17a9 9 0 0 0-9-9 9 9 0 0 0-6 2.3L3 13'/%3E%3C/svg%3E") no-repeat center;
    -webkit-mask-size: contain;
    mask-size: contain;
}}

div[class*="st-key-act_redo"] button::before {{
    content: "";
    display: inline-block;
    width: 14px;
    height: 14px;
    background-color: currentColor;
    -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2.3' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M21 7v6h-6'/%3E%3Cpath d='M3 17a9 9 0 0 1 9-9 9 9 0 0 1 6 2.3L21 13'/%3E%3C/svg%3E") no-repeat center;
    mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2.3' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M21 7v6h-6'/%3E%3Cpath d='M3 17a9 9 0 0 1 9-9 9 9 0 0 1 6 2.3L21 13'/%3E%3C/svg%3E") no-repeat center;
    -webkit-mask-size: contain;
    mask-size: contain;
}}

div[class*="st-key-act_clear"] button::before {{
    content: "";
    display: inline-block;
    width: 14px;
    height: 14px;
    background-color: currentColor;
    -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2.3' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='3 6 5 6 21 6'/%3E%3Cpath d='M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2'/%3E%3C/svg%3E") no-repeat center;
    mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2.3' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='3 6 5 6 21 6'/%3E%3Cpath d='M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2'/%3E%3C/svg%3E") no-repeat center;
    -webkit-mask-size: contain;
    mask-size: contain;
}}

/* Circular Brush Buttons */
div[class*="st-key-brush_btn_"] button {{
    border-radius: 50% !important;
    width: 30px !important;
    height: 30px !important;
    min-width: 30px !important;
    min-height: 30px !important;
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    box-shadow: none !important;
    color: #475569 !important;
    transition: all 0.15s ease !important;
}}

div[class*="st-key-brush_btn_"] button:hover {{
    transform: scale(1.1) !important;
}}

div[class*="st-key-brush_btn_active"] button {{
    border: 1.5px solid #403974 !important;
    box-shadow: 0 0 0 2px #FFFFFF, 0 0 0 4px #403974 !important;
    color: #403974 !important;
}}

/* Circular Color Buttons */
div[class*="st-key-color_btn_"] button {{
    border-radius: 50% !important;
    width: 24px !important;
    height: 24px !important;
    min-width: 24px !important;
    min-height: 24px !important;
    padding: 0 !important;
    margin: 0 auto !important;
    border: none !important;
    box-shadow: none !important;
    transition: all 0.15s ease !important;
}}

div[class*="st-key-color_btn_"] button:hover {{
    transform: scale(1.18) !important;
}}

div[class*="st-key-color_btn_active"] button {{
    box-shadow: 0 0 0 2px #FFFFFF, 0 0 0 4px #403974 !important;
    transform: scale(1.05) !important;
}}

/* Predict Button */
div[class*="st-key-predict_btn"] button {{
    background: #574EA6 !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 13.5px !important;
    border-radius: 10px !important;
    padding: 10px !important;
    width: 100% !important;
    border: none !important;
    box-shadow: 0 4px 12px rgba(87, 78, 166, 0.25) !important;
    margin-top: 6px !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 7px !important;
    transition: all 0.2s cubic-bezier(0.2, 0.8, 0.2, 1) !important;
}}

div[class*="st-key-predict_btn"] button::before {{
    content: "";
    display: inline-block;
    width: 14px;
    height: 14px;
    background-color: #FFFFFF;
    -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='currentColor'%3E%3Cpolygon points='13 2 3 14 12 14 11 22 21 10 12 10 13 2'/%3E%3C/svg%3E") no-repeat center;
    mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='currentColor'%3E%3Cpolygon points='13 2 3 14 12 14 11 22 21 10 12 10 13 2'/%3E%3C/svg%3E") no-repeat center;
    -webkit-mask-size: contain;
    mask-size: contain;
}}

div[class*="st-key-predict_btn"] button:hover {{
    background: #4B4292 !important;
    box-shadow: 0 6px 16px rgba(87, 78, 166, 0.35) !important;
    transform: translateY(-1px) !important;
}}

div[class*="st-key-predict_btn"] button:active {{
    transform: scale(0.98) !important;
}}

/* Prediction Result Card */
.prediction-card {{
    background: #FEF8F2;
    border: 1.5px solid #FBEADB;
    border-radius: 18px;
    padding: 16px 20px;
    text-align: center;
    margin-bottom: 14px;
    box-shadow: 0 4px 18px rgba(241, 194, 142, 0.1);
    animation: fadeInUp 0.4s ease both;
}}

.pred-header {{
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 15px;
    font-weight: 700;
    color: #1E1B4B;
    margin-bottom: 4px;
}}

.pred-digit {{
    font-size: 56px;
    font-weight: 800;
    color: #1E1B4B;
    line-height: 1;
    margin: 6px 0;
    letter-spacing: 0.02em;
    animation: predPop 0.35s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}}

@media (max-width: 768px) {{
    .pred-digit {{
        font-size: 46px;
    }}
}}

.pred-confidence {{
    font-size: 13.5px;
    font-weight: 500;
    color: #475569;
    margin: 0;
}}

.pred-confidence span {{
    font-weight: 700;
    color: #1E1B4B;
}}

/* Model Details Section */
.details-subhead {{
    font-size: 12px;
    font-weight: 700;
    color: #1E1B4B;
    margin-bottom: 8px;
}}

.processed-container {{
    display: flex;
    gap: 8px;
    align-items: center;
    flex-wrap: wrap;
}}

.processed-img-box {{
    background: #000000;
    border-radius: 8px;
    padding: 4px;
    width: 68px;
    height: 68px;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: transform 0.2s ease;
}}

.processed-img-box:hover {{
    transform: scale(1.05);
}}

.processed-img-box img {{
    image-rendering: pixelated;
    border-radius: 4px;
}}

.prob-table {{
    display: flex;
    flex-direction: column;
    gap: 4px;
    margin-top: 2px;
    margin-bottom: 8px;
}}

.prob-row {{
    display: flex;
    align-items: center;
    font-size: 12.5px;
}}

.prob-digit {{
    width: 14px;
    color: #1E1B4B;
    font-weight: 700;
}}

.prob-bar-track {{
    flex-grow: 1;
    height: 7px;
    background-color: #F1F5F9;
    border-radius: 4px;
    margin: 0 8px;
    overflow: hidden;
}}

.prob-bar-fill-top {{
    height: 100%;
    background: linear-gradient(90deg, #F1C28E 0%, #F3AB9D 100%);
    border-radius: 4px;
    transition: width 0.4s cubic-bezier(0.2, 0.8, 0.2, 1);
}}

.prob-bar-fill-sub {{
    height: 100%;
    background-color: #E2E8F0;
    border-radius: 4px;
    transition: width 0.4s cubic-bezier(0.2, 0.8, 0.2, 1);
}}

.prob-pct {{
    width: 44px;
    text-align: right;
    color: #64748B;
    font-size: 11.5px;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
}}

.digit-label-badge {{
    font-size: 11px;
    font-weight: 700;
    color: #C2410C;
    background: #FFF7ED;
    border: 1px solid #FFEDD5;
    padding: 1px 7px;
    border-radius: 5px;
    display: inline-block;
    margin-bottom: 4px;
}}
</style>
""")

# --- State Management ---
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

if "brush_size" not in st.session_state:
    st.session_state.brush_size = "Medium"

if "selected_color" not in st.session_state:
    st.session_state.selected_color = "#312C51"

if "is_eraser" not in st.session_state:
    st.session_state.is_eraser = False

if "canvas_history" not in st.session_state:
    st.session_state.canvas_history = []

if "redo_stack" not in st.session_state:
    st.session_state.redo_stack = []

if "initial_drawing" not in st.session_state:
    st.session_state.initial_drawing = None

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if "last_canvas_sig" not in st.session_state:
    st.session_state.last_canvas_sig = None

# --- Top Header with Professional Icons ---
logo_html = f'<img src="data:image/png;base64,{LOGO_B64}" />' if LOGO_B64 else '<span style="font-weight:800; color:#312C51;">?</span>'

st.html(f"""
<div class="header-wrapper">
    <div class="header-left">
        <div class="header-logo-badge">
            {logo_html}
        </div>
        <div class="header-title-box">
            <h1>MNIST Digit Classifier</h1>
            <p>Draw digits (0–9) and see what the model predicts</p>
        </div>
    </div>
    <div class="header-tip-card">
        <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#D97706" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0;"><path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/><path d="M9 18h6"/><path d="M10 22h4"/></svg>
        <div class="tip-text">
            <div>Try single or multi-digit numbers</div>
            <div>to test model segmentation & recognition!</div>
        </div>
    </div>
</div>
""")

# --- Main Columns Layout: Left ~65%, Right ~35% on Desktop; 100% stacked on Mobile ---
col_left, col_right = st.columns([65, 35], gap="medium")

# ==============================================================
# LEFT SECTION: DRAW YOUR DIGIT
# ==============================================================
with col_left:
    with st.container(border=True):
        st.html("""
        <div class="card-header">
            <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#48426D" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0;"><path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/></svg>
            <span>Draw Your Digit</span>
        </div>
        <div class="card-divider"></div>
        """)

        # Brush size mapping: Drawing strokes vs Rubber eraser stroke widths
        BRUSH_MAP_DRAW = {
            "Small": 12,
            "Medium": 22,
            "Large": 32,
        }
        BRUSH_MAP_RUBBER = {
            "Small": 24,
            "Medium": 36,
            "Large": 48,
        }

        # Color Palette matching target mockup
        PALETTE = [
            ("#312C51", "aubergine"),
            ("#48426D", "violet"),
            ("#F1C28E", "peach"),
            ("#F3AB9D", "coral"),
            ("#1E1E1E", "black"),
            ("#2563EB", "blue"),
            ("#10B981", "teal"),
        ]

        # Determine stroke color and width for canvas
        if st.session_state.is_eraser:
            active_stroke = "#FFFFFF"
            active_width = BRUSH_MAP_RUBBER[st.session_state.brush_size]
        else:
            active_stroke = st.session_state.selected_color
            active_width = BRUSH_MAP_DRAW[st.session_state.brush_size]

        # Canvas Frame - Kept at a stable position with NO preceding conditional elements
        st.html('<div class="canvas-outer-frame">')
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=active_width,
            stroke_color=active_stroke,
            background_color="#FFFFFF",
            height=360,
            width=560,
            drawing_mode="freedraw",
            initial_drawing=st.session_state.initial_drawing,
            display_toolbar=False,
            key=f"canvas_mnist_{st.session_state.canvas_key}",
            update_streamlit=True,
        )
        st.html('</div>')

        # Synchronize stroke history without altering initial_drawing
        # Keeping initial_drawing constant guarantees fabric.js NEVER wipes new strokes on rerun!
        if canvas_result and canvas_result.json_data and "objects" in canvas_result.json_data:
            current_objects = canvas_result.json_data["objects"]
            if len(current_objects) > len(st.session_state.canvas_history):
                st.session_state.canvas_history = list(current_objects)
                st.session_state.redo_stack.clear()
            elif len(current_objects) > 0:
                st.session_state.canvas_history = list(current_objects)

        # --- Controls: Row 1 - Action Buttons (Draw, Rubber, Undo, Redo, Clear) ---
        act_c1, act_c2, act_c3, act_c4, act_c5 = st.columns([1, 1, 1, 1, 1])

        with act_c1:
            draw_state = "inactive" if st.session_state.is_eraser else "active"
            if st.button("Draw", key=f"act_draw_{draw_state}", help="Draw mode"):
                if st.session_state.is_eraser:
                    st.session_state.is_eraser = False
                    st.rerun()

        with act_c2:
            rubber_state = "active" if st.session_state.is_eraser else "inactive"
            if st.button("Rubber", key=f"act_rubber_{rubber_state}", help="Activate rubber eraser"):
                if not st.session_state.is_eraser:
                    st.session_state.is_eraser = True
                    st.rerun()

        with act_c3:
            if st.button("Undo", key="act_undo", help="Undo last stroke"):
                if st.session_state.canvas_history and len(st.session_state.canvas_history) > 0:
                    popped = st.session_state.canvas_history.pop()
                    st.session_state.redo_stack.append(popped)
                    st.session_state.initial_drawing = {
                        "version": "4.4.0",
                        "objects": list(st.session_state.canvas_history),
                    }
                    st.session_state.canvas_key += 1
                    st.session_state.last_canvas_sig = None
                    st.rerun()

        with act_c4:
            if st.button("Redo", key="act_redo", help="Redo undone stroke"):
                if st.session_state.redo_stack and len(st.session_state.redo_stack) > 0:
                    restored = st.session_state.redo_stack.pop()
                    st.session_state.canvas_history.append(restored)
                    st.session_state.initial_drawing = {
                        "version": "4.4.0",
                        "objects": list(st.session_state.canvas_history),
                    }
                    st.session_state.canvas_key += 1
                    st.session_state.last_canvas_sig = None
                    st.rerun()

        with act_c5:
            if st.button("Clear", key="act_clear", help="Clear canvas"):
                st.session_state.canvas_history = []
                st.session_state.redo_stack = []
                st.session_state.initial_drawing = None
                st.session_state.canvas_key += 1
                st.session_state.last_prediction = None
                st.session_state.last_canvas_sig = None
                st.rerun()

        # Rubber active subtle notification placed AFTER canvas so it never shifts canvas delta path
        if st.session_state.is_eraser:
            st.html("""
            <div style="font-size: 12px; color: #6B21A8; background: #FDF4FF; border: 1px solid #D8B4FE; border-radius: 8px; padding: 6px 12px; margin-top: 8px; display: flex; align-items: center; gap: 6px; animation: fadeInUp 0.2s ease both;">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#6B21A8" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0;"><path d="m7 21-4.3-4.3c-1-1-1-2.5 0-3.4l9.6-9.6c1-1 2.5-1 3.4 0l5.6 5.6c1 1 1 2.5 0 3.4L13 21"/><path d="M22 21H7"/><path d="m5 11 9 9"/></svg>
                <span><b>Rubber Active:</b> Drag across strokes to erase. Click <b>Draw</b> when finished.</span>
            </div>
            """)

        st.html("<div style='height: 12px;'></div>")

        # --- Controls: Row 2 - Brush Size and Color Palette ---
        r2_c1, r2_c2 = st.columns([32, 68])

        with r2_c1:
            st.html('<span class="toolbar-label">Brush Size</span>')
            b_c1, b_c2, b_c3 = st.columns([1, 1, 1])

            with b_c1:
                is_s = "active" if st.session_state.brush_size == "Small" else "inactive"
                if st.button("•", key=f"brush_btn_{is_s}_s", help="Small stroke / eraser"):
                    st.session_state.brush_size = "Small"
                    st.rerun()

            with b_c2:
                is_m = "active" if st.session_state.brush_size == "Medium" else "inactive"
                if st.button("●", key=f"brush_btn_{is_m}_m", help="Medium stroke / eraser"):
                    st.session_state.brush_size = "Medium"
                    st.rerun()

            with b_c3:
                is_l = "active" if st.session_state.brush_size == "Large" else "inactive"
                if st.button("⬤", key=f"brush_btn_{is_l}_l", help="Large stroke / eraser"):
                    st.session_state.brush_size = "Large"
                    st.rerun()

        with r2_c2:
            st.html('<span class="toolbar-label">Color</span>')
            c_cols = st.columns(len(PALETTE))

            for idx, (hex_code, name) in enumerate(PALETTE):
                with c_cols[idx]:
                    is_col = "active" if (st.session_state.selected_color == hex_code and not st.session_state.is_eraser) else "inactive"
                    st.html(f"""
                    <style>
                    div[class*="st-key-color_btn_{is_col}_{name}"] button {{
                        background-color: {hex_code} !important;
                    }}
                    </style>
                    """)
                    if st.button(" ", key=f"color_btn_{is_col}_{name}", help=f"Draw with {name.capitalize()}"):
                        st.session_state.selected_color = hex_code
                        st.session_state.is_eraser = False
                        st.rerun()


# ==============================================================
# RIGHT SECTION: MODEL SELECTION, PREDICTION, MODEL DETAILS
# ==============================================================
with col_right:
    # ----------------------------------------------------------
    # CARD 1: SELECT MODEL
    # ----------------------------------------------------------
    with st.container(border=True):
        st.html("""
        <div class="card-header" style="margin-bottom: 8px;">
            <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#48426D" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0;"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
            <span>Select Model</span>
        </div>
        """)

        model_options = list(AVAILABLE_MODELS.keys())
        selected_model_name = st.selectbox(
            "Model Selection",
            options=model_options,
            index=0,
            label_visibility="collapsed",
        )
        selected_meta = AVAILABLE_MODELS[selected_model_name]

        predict_btn_clicked = st.button("Predict Digit", key="predict_btn")

    # ----------------------------------------------------------
    # Smart Multi-Digit Prediction Engine (Uses latest visible canvas)
    # ----------------------------------------------------------
    canvas_objects_count = len(canvas_result.json_data.get("objects", [])) if (canvas_result and canvas_result.json_data) else 0
    current_canvas_sig = (canvas_objects_count, selected_model_name)

    needs_prediction = (
        predict_btn_clicked or 
        (canvas_result is not None and canvas_result.image_data is not None and canvas_objects_count > 0 and current_canvas_sig != st.session_state.last_canvas_sig)
    )

    if needs_prediction and canvas_result is not None and canvas_result.image_data is not None:
        try:
            digit_segments = preprocess_multi_digit_canvas(
                canvas_result.image_data,
                model_type=selected_meta["type"],
            )

            digit_preds = []
            for seg in digit_segments:
                res = predict_digit(selected_model_name, seg["model_input"])
                digit_preds.append({
                    "digit": res["predicted_digit"],
                    "confidence": res["confidence"],
                    "top_5": res["top_5"],
                    "display_image": seg["display_image"]
                })

            combined_str = "".join(str(item["digit"]) for item in digit_preds)
            avg_conf = float(np.mean([item["confidence"] for item in digit_preds]))

            st.session_state.last_prediction = {
                "combined_digits": combined_str,
                "confidence": avg_conf,
                "digit_details": digit_preds,
            }
            st.session_state.last_canvas_sig = current_canvas_sig
        except EmptyCanvasError:
            if predict_btn_clicked:
                st.warning("Please draw a digit before making a prediction.")
        except Exception as e:
            if predict_btn_clicked:
                st.error(f"Error: {e}")

    active_pred = st.session_state.last_prediction

    # ----------------------------------------------------------
    # CARD 2: PREDICTION RESULT
    # ----------------------------------------------------------
    if active_pred is not None:
        comb_digits = active_pred["combined_digits"]
        p_conf = active_pred["confidence"]
        num_digits = len(active_pred["digit_details"])

        conf_label = f"Confidence: <span>{p_conf:.1f}%</span>" if num_digits == 1 else f"Avg Confidence: <span>{p_conf:.1f}%</span> ({num_digits} digits)"

        st.html(f"""
        <div class="prediction-card">
            <div class="pred-header">
                <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#E06D53" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0;"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>
                <span>Prediction Result</span>
            </div>
            <div class="pred-digit">{comb_digits}</div>
            <p class="pred-confidence">{conf_label}</p>
        </div>
        """)
    else:
        st.html("""
        <div class="prediction-card" style="border: 1.5px dashed #FCDCC3; background: #FFFCF9;">
            <div class="pred-header">
                <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#E06D53" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0;"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>
                <span>Prediction Result</span>
            </div>
            <div class="pred-digit" style="color: #CBD5E1; font-weight: 300;">—</div>
            <p class="pred-confidence" style="color: #94A3B8;">Confidence: <span style="color: #94A3B8;">--%</span></p>
        </div>
        """)

    # ----------------------------------------------------------
    # CARD 3: MODEL DETAILS
    # ----------------------------------------------------------
    with st.container(border=True):
        st.html("""
        <div class="card-header" style="margin-bottom: 8px;">
            <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#48426D" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0;"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>
            <span>Model Details</span>
        </div>
        """)

        det_col1, det_col2 = st.columns([38, 62], gap="medium")

        with det_col1:
            st.html('<div class="details-subhead">Processed (28 × 28)</div>')
            if active_pred is not None and "digit_details" in active_pred:
                imgs_html = '<div class="processed-container">'
                for idx, d_info in enumerate(active_pred["digit_details"]):
                    pil_img = Image.fromarray(d_info["display_image"])
                    buffered = io.BytesIO()
                    pil_img.save(buffered, format="PNG")
                    img_b64 = base64.b64encode(buffered.getvalue()).decode()

                    label_str = f"Digit {idx+1}" if len(active_pred["digit_details"]) > 1 else "Input"
                    imgs_html += f"""
                    <div style="text-align: center;">
                        <span style="font-size:10.5px; color:#64748B; font-weight:600; display:block; margin-bottom:2px;">{label_str}</span>
                        <div class="processed-img-box">
                            <img src="data:image/png;base64,{img_b64}" width="60" height="60" />
                        </div>
                    </div>
                    """
                imgs_html += '</div>'
                st.html(imgs_html)
            else:
                st.html("""
                <div class="processed-img-box" style="color: #64748B; font-size: 10.5px; text-align: center; width: 100%;">
                    Waiting for drawing...
                </div>
                """)

        with det_col2:
            st.html('<div class="details-subhead">Top Predictions</div>')

            if active_pred is not None and len(active_pred.get("digit_details", [])) > 0:
                digits_list = active_pred["digit_details"]
                num_digits = len(digits_list)

                bars_html = ""
                for d_idx, d_info in enumerate(digits_list):
                    if num_digits > 1:
                        bars_html += f"""
                        <div class="digit-label-badge">Digit {d_idx + 1} ({d_info['digit']})</div>
                        """

                    top_records = d_info["top_5"]
                    display_records = top_records[:3] if num_digits > 1 else top_records

                    bars_html += '<div class="prob-table">'
                    for rank_idx, (r_digit, r_prob) in enumerate(display_records):
                        fill_class = "prob-bar-fill-top" if rank_idx == 0 else "prob-bar-fill-sub"
                        digit_color = "#1E1B4B" if rank_idx == 0 else "#64748B"
                        bars_html += f"""
                        <div class="prob-row">
                            <span class="prob-digit" style="color: {digit_color};">{r_digit}</span>
                            <div class="prob-bar-track">
                                <div class="{fill_class}" style="width: {min(100.0, max(2.0, r_prob)):.1f}%;"></div>
                            </div>
                            <span class="prob-pct">{r_prob:.1f}%</span>
                        </div>
                        """
                    bars_html += '</div>'

                st.html(bars_html)
            else:
                bars_html = '<div class="prob-table">'
                for placeholder_d in [0, 1, 2, 3, 4]:
                    bars_html += f"""
                    <div class="prob-row">
                        <span class="prob-digit" style="color: #94A3B8;">{placeholder_d}</span>
                        <div class="prob-bar-track">
                            <div class="prob-bar-fill-sub" style="width: 0%;"></div>
                        </div>
                        <span class="prob-pct" style="color: #94A3B8;">0.0%</span>
                    </div>
                    """
                bars_html += '</div>'
                st.html(bars_html)
