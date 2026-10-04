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

/* ===== TOUCH-FRIENDLY GLOBAL ===== */
*, *::before, *::after {{
    -webkit-tap-highlight-color: transparent;
}}

button, a, [role="button"] {{
    touch-action: manipulation;
}}

/* ===== FLUID RESPONSIVE CONTAINER & SPACING ===== */
.block-container {{
    padding-top: clamp(0.5rem, 1.5vw, 1.2rem) !important;
    padding-bottom: clamp(1rem, 2vw, 2rem) !important;
    padding-left: clamp(8px, 2.5vw, 24px) !important;
    padding-right: clamp(8px, 2.5vw, 24px) !important;
    max-width: min(1320px, 95vw) !important;
    margin: 0 auto !important;
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
    margin-bottom: clamp(12px, 2vw, 20px);
    flex-wrap: wrap;
    gap: clamp(8px, 1.8vw, 14px);
    animation: fadeInUp 0.35s cubic-bezier(0.16, 1, 0.3, 1) both;
}}

.header-left {{
    display: flex;
    align-items: center;
    gap: clamp(8px, 1.8vw, 14px);
}}

.header-logo-badge {{
    width: clamp(38px, 6.5vw, 48px);
    height: clamp(38px, 6.5vw, 48px);
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: clamp(10px, 1.5vw, 12px);
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
    width: clamp(28px, 5vw, 36px);
    height: clamp(28px, 5vw, 36px);
    object-fit: contain;
    border-radius: 6px;
}}

.header-title-box h1 {{
    font-size: clamp(18px, 3.8vw, 24px) !important;
    font-weight: 800 !important;
    color: #1E1B4B !important;
    margin: 0 !important;
    letter-spacing: -0.02em !important;
    line-height: 1.2 !important;
}}

.header-title-box p {{
    font-size: clamp(11.5px, 2.2vw, 13.5px) !important;
    color: #64748B !important;
    margin: 3px 0 0 0 !important;
    font-weight: 500 !important;
}}

.header-tip-card {{
    background: #FFFFFF;
    border: 1px solid #FEF08A;
    border-radius: clamp(9px, 1.5vw, 12px);
    padding: clamp(7px, 1.5vw, 10px) clamp(10px, 2vw, 18px);
    display: flex;
    align-items: center;
    gap: clamp(6px, 1.2vw, 10px);
    box-shadow: 0 2px 8px rgba(50, 46, 83, 0.03);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}}

.header-tip-card:hover {{
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(245, 158, 11, 0.1);
}}

.tip-text {{
    font-size: clamp(11px, 2.1vw, 12px);
    color: #475569;
    font-weight: 500;
    line-height: 1.35;
}}

@media (max-width: 768px) {{
    .header-wrapper {{
        flex-direction: column;
        align-items: flex-start;
        gap: 10px;
    }}
    .header-tip-card {{
        width: 100%;
        box-sizing: border-box;
    }}
}}

/* Card Containers */
div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: #FFFFFF !important;
    border-radius: clamp(12px, 1.8vw, 18px) !important;
    border: 1px solid #EDEEF2 !important;
    box-shadow: 0 4px 20px rgba(50, 46, 83, 0.05) !important;
    padding: clamp(12px, 2vw, 20px) clamp(10px, 2vw, 22px) !important;
    margin-bottom: clamp(10px, 1.5vw, 14px) !important;
    animation: fadeInUp 0.4s cubic-bezier(0.16, 1, 0.3, 1) both;
    transition: box-shadow 0.2s ease !important;
    width: 100% !important;
    box-sizing: border-box !important;
}}

div[data-testid="stVerticalBlockBorderWrapper"]:hover {{
    box-shadow: 0 6px 24px rgba(50, 46, 83, 0.08) !important;
}}

/* Card Headers */
.card-header {{
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: clamp(14px, 2.5vw, 16px);
    font-weight: 700;
    color: #1E1B4B;
    margin-bottom: 8px;
}}

.card-divider {{
    border-bottom: 1px dashed #E2E8F0;
    margin-top: 4px;
    margin-bottom: clamp(8px, 1.5vw, 14px);
}}

/* ===== CANVAS FRAME & IFRAME STYLING ===== */
div[data-testid="stCustomComponentV1"]:has(iframe[title*="st_canvas"]) {{
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    width: 100% !important;
    border: 1.5px solid #E2E8F0 !important;
    border-radius: clamp(10px, 1.6vw, 14px) !important;
    background: #FFFFFF !important;
    box-sizing: border-box !important;
    margin-bottom: clamp(10px, 1.5vw, 14px) !important;
    box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.02) !important;
    overflow: hidden !important;
}}

iframe[title*="st_canvas"] {{
    display: block !important;
    margin: 0 auto !important;
    max-width: 100% !important;
    touch-action: none !important;
    border: none !important;
}}

/* Toolbar Labels */
.toolbar-label {{
    font-size: clamp(11px, 2.2vw, 12px);
    font-weight: 600;
    color: #64748B;
    margin-bottom: 5px;
    display: block;
}}

/* ===== FLUID ACTION BUTTONS (Draw, Rubber, Undo, Redo, Clear) ===== */
div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-act_"]) {{
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    align-items: center !important;
    gap: clamp(3px, 1vw, 8px) !important;
    width: 100% !important;
}}

div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-act_"]) > div[data-testid="column"] {{
    width: auto !important;
    flex: 1 1 0px !important;
    min-width: 0 !important;
}}

div[class*="st-key-act_"] button {{
    height: clamp(38px, 5.5vw, 42px) !important;
    min-height: clamp(38px, 5.5vw, 42px) !important;
    font-size: clamp(10.5px, 2.5vw, 13px) !important;
    border-radius: clamp(7px, 1.4vw, 10px) !important;
    width: 100% !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: clamp(2px, 0.8vw, 6px) !important;
    padding: clamp(2px, 0.8vw, 6px) clamp(2px, 0.8vw, 8px) !important;
    white-space: nowrap !important;
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
    width: clamp(12px, 2.5vw, 14px);
    height: clamp(12px, 2.5vw, 14px);
    background-color: currentColor;
    -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2.3' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M17 3a2.828 2.828 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5L17 3z'/%3E%3C/svg%3E") no-repeat center;
    mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2.3' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M17 3a2.828 2.828 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5L17 3z'/%3E%3C/svg%3E") no-repeat center;
    -webkit-mask-size: contain;
    mask-size: contain;
    flex-shrink: 0;
}}

div[class*="st-key-act_rubber"] button::before {{
    content: "";
    display: inline-block;
    width: clamp(12px, 2.5vw, 14px);
    height: clamp(12px, 2.5vw, 14px);
    background-color: currentColor;
    -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2.3' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m7 21-4.3-4.3c-1-1-1-2.5 0-3.4l9.6-9.6c1-1 2.5-1 3.4 0l5.6 5.6c1 1 1 2.5 0 3.4L13 21'/%3E%3Cpath d='M22 21H7'/%3E%3Cpath d='m5 11 9 9'/%3E%3C/svg%3E") no-repeat center;
    mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2.3' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m7 21-4.3-4.3c-1-1-1-2.5 0-3.4l9.6-9.6c1-1 2.5-1 3.4 0l5.6 5.6c1 1 1 2.5 0 3.4L13 21'/%3E%3Cpath d='M22 21H7'/%3E%3Cpath d='m5 11 9 9'/%3E%3C/svg%3E") no-repeat center;
    -webkit-mask-size: contain;
    mask-size: contain;
    flex-shrink: 0;
}}

div[class*="st-key-act_undo"] button::before {{
    content: "";
    display: inline-block;
    width: clamp(12px, 2.5vw, 14px);
    height: clamp(12px, 2.5vw, 14px);
    background-color: currentColor;
    -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2.3' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M3 7v6h6'/%3E%3Cpath d='M21 17a9 9 0 0 0-9-9 9 9 0 0 0-6 2.3L3 13'/%3E%3C/svg%3E") no-repeat center;
    mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2.3' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M3 7v6h6'/%3E%3Cpath d='M21 17a9 9 0 0 0-9-9 9 9 0 0 0-6 2.3L3 13'/%3E%3C/svg%3E") no-repeat center;
    -webkit-mask-size: contain;
    mask-size: contain;
    flex-shrink: 0;
}}

div[class*="st-key-act_redo"] button::before {{
    content: "";
    display: inline-block;
    width: clamp(12px, 2.5vw, 14px);
    height: clamp(12px, 2.5vw, 14px);
    background-color: currentColor;
    -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2.3' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M21 7v6h-6'/%3E%3Cpath d='M3 17a9 9 0 0 1 9-9 9 9 0 0 1 6 2.3L21 13'/%3E%3C/svg%3E") no-repeat center;
    mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2.3' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M21 7v6h-6'/%3E%3Cpath d='M3 17a9 9 0 0 1 9-9 9 9 0 0 1 6 2.3L21 13'/%3E%3C/svg%3E") no-repeat center;
    -webkit-mask-size: contain;
    mask-size: contain;
    flex-shrink: 0;
}}

div[class*="st-key-act_clear"] button::before {{
    content: "";
    display: inline-block;
    width: clamp(12px, 2.5vw, 14px);
    height: clamp(12px, 2.5vw, 14px);
    background-color: currentColor;
    -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2.3' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='3 6 5 6 21 6'/%3E%3Cpath d='M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2'/%3E%3C/svg%3E") no-repeat center;
    mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2.3' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='3 6 5 6 21 6'/%3E%3Cpath d='M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2'/%3E%3C/svg%3E") no-repeat center;
    -webkit-mask-size: contain;
    mask-size: contain;
    flex-shrink: 0;
}}

/* ===== FLUID BRUSH & COLOR BUTTONS ===== */
div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-brush_btn_"]),
div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-color_btn_"]) {{
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    align-items: center !important;
    gap: clamp(3px, 1vw, 8px) !important;
    width: 100% !important;
}}

div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-brush_btn_"]) > div[data-testid="column"],
div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-color_btn_"]) > div[data-testid="column"] {{
    width: auto !important;
    flex: 1 1 0px !important;
    min-width: 0 !important;
}}

div[class*="st-key-brush_btn_"] button {{
    border-radius: 50% !important;
    width: clamp(28px, 5.5vw, 34px) !important;
    height: clamp(28px, 5.5vw, 34px) !important;
    min-width: clamp(28px, 5.5vw, 34px) !important;
    min-height: clamp(28px, 5.5vw, 34px) !important;
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    padding: 0 !important;
    margin: 0 auto !important;
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

div[class*="st-key-color_btn_"] button {{
    border-radius: 50% !important;
    width: clamp(24px, 4.8vw, 30px) !important;
    height: clamp(24px, 4.8vw, 30px) !important;
    min-width: clamp(24px, 4.8vw, 30px) !important;
    min-height: clamp(24px, 4.8vw, 30px) !important;
    padding: 0 !important;
    margin: 0 auto !important;
    border: none !important;
    box-shadow: none !important;
    transition: all 0.15s ease !important;
}}

div[class*="st-key-color_btn_"] button:hover {{
    transform: scale(1.15) !important;
}}

div[class*="st-key-color_btn_active"] button {{
    box-shadow: 0 0 0 2px #FFFFFF, 0 0 0 4px #403974 !important;
    transform: scale(1.05) !important;
}}

/* Stack Brush Size and Color row vertically on screens <= 640px */
@media (max-width: 640px) {{
    div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-brush_btn_"]):has(div[class*="st-key-color_btn_"]) {{
        flex-direction: column !important;
        align-items: stretch !important;
        gap: 10px !important;
    }}
    div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-brush_btn_"]):has(div[class*="st-key-color_btn_"]) > div[data-testid="column"] {{
        width: 100% !important;
        flex: 1 1 100% !important;
    }}
}}

/* ===== FLUID PREDICT BUTTON ===== */
div[class*="st-key-predict_btn"] button {{
    background: #574EA6 !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: clamp(13px, 2.8vw, 14.5px) !important;
    border-radius: clamp(8px, 1.8vw, 11px) !important;
    height: clamp(42px, 6vw, 48px) !important;
    min-height: clamp(42px, 6vw, 48px) !important;
    padding: clamp(8px, 1.5vw, 12px) !important;
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
    flex-shrink: 0;
}}

div[class*="st-key-predict_btn"] button:hover {{
    background: #4B4292 !important;
    box-shadow: 0 6px 16px rgba(87, 78, 166, 0.35) !important;
    transform: translateY(-1px) !important;
}}

div[class*="st-key-predict_btn"] button:active {{
    transform: scale(0.98) !important;
}}

/* ===== FLUID PREDICTION RESULT CARD ===== */
.prediction-card {{
    background: #FEF8F2;
    border: 1.5px solid #FBEADB;
    border-radius: clamp(12px, 2vw, 18px);
    padding: clamp(12px, 2vw, 18px) clamp(12px, 2vw, 20px);
    text-align: center;
    margin-bottom: clamp(10px, 1.5vw, 14px);
    box-shadow: 0 4px 18px rgba(241, 194, 142, 0.1);
    animation: fadeInUp 0.4s ease both;
}}

.pred-header {{
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    font-size: clamp(13px, 2.5vw, 15px);
    font-weight: 700;
    color: #1E1B4B;
    margin-bottom: 4px;
}}

.pred-digit {{
    font-size: clamp(38px, 8.5vw, 58px);
    font-weight: 800;
    color: #1E1B4B;
    line-height: 1.1;
    margin: 4px 0;
    letter-spacing: 0.02em;
    animation: predPop 0.35s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    word-break: break-all;
    overflow-wrap: break-word;
}}

.pred-confidence {{
    font-size: clamp(12px, 2.4vw, 13.5px);
    font-weight: 500;
    color: #475569;
    margin: 0;
}}

.pred-confidence span {{
    font-weight: 700;
    color: #1E1B4B;
}}

/* ===== FLUID MODEL DETAILS SECTION ===== */
.details-subhead {{
    font-size: clamp(11.5px, 2.3vw, 12px);
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
    width: clamp(54px, 11vw, 68px);
    height: clamp(54px, 11vw, 68px);
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
    width: 100%;
    height: 100%;
    object-fit: contain;
}}

.prob-table {{
    display: flex;
    flex-direction: column;
    gap: 4px;
    margin-top: 2px;
    margin-bottom: 8px;
    width: 100%;
}}

.prob-row {{
    display: flex;
    align-items: center;
    font-size: clamp(11.5px, 2.3vw, 12.5px);
    width: 100%;
}}

.prob-digit {{
    width: 14px;
    color: #1E1B4B;
    font-weight: 700;
    flex-shrink: 0;
}}

.prob-bar-track {{
    flex-grow: 1;
    height: 7px;
    background-color: #F1F5F9;
    border-radius: 4px;
    margin: 0 clamp(4px, 1.2vw, 8px);
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
    width: clamp(38px, 8vw, 44px);
    text-align: right;
    color: #64748B;
    font-size: clamp(10.5px, 2.2vw, 11.5px);
    font-weight: 600;
    font-variant-numeric: tabular-nums;
    flex-shrink: 0;
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

/* Stack Model Details inner columns on mobile/small tablets */
@media (max-width: 640px) {{
    div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stHorizontalBlock"]:has(.details-subhead) {{
        flex-direction: column !important;
        gap: 12px !important;
    }}
    div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stHorizontalBlock"]:has(.details-subhead) > div[data-testid="column"] {{
        width: 100% !important;
        flex: 1 1 100% !important;
    }}
    .processed-container {{
        justify-content: center !important;
    }}
}}

/* Clean Column Stacking on screens <= 860px */
@media (max-width: 860px) {{
    div[data-testid="stHorizontalBlock"]:has(> div[data-testid="column"]:first-child:has(iframe[title*="st_canvas"])) {{
        flex-direction: column !important;
        gap: 12px !important;
    }}
    div[data-testid="stHorizontalBlock"]:has(> div[data-testid="column"]:first-child:has(iframe[title*="st_canvas"])) > div[data-testid="column"] {{
        width: 100% !important;
        flex: 1 1 100% !important;
        max-width: 100% !important;
    }}
}}

/* iOS Safari fixed background fix */
@media (max-width: 768px) {{
    .stApp {{
        background-attachment: scroll !important;
    }}
}}

/* Touch-Friendly Selectbox */
@media (max-width: 768px) {{
    div[data-testid="stSelectbox"] > div {{
        min-height: 44px !important;
        font-size: 14px !important;
    }}
}}
</style>
""")

# --- State Management ---
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

if "canvas_ratio" not in st.session_state:
    st.session_state.canvas_ratio = "Standard"

if "history_updated" not in st.session_state:
    st.session_state.history_updated = False

# --- Dynamic Fluid Canvas Dimension Engine ---
CANVAS_RATIOS = {
    "Standard": 0.625,  # 16:10 balanced ratio
    "Wide": 0.55,      # 16:9 panoramic ratio
    "Compact": 0.72,   # 4:3 taller ratio
}

def get_device_canvas_config():
    """
    Dynamically computes canvas dimensions and stroke widths based on device type
    (phones, tablets, and desktops) and selected aspect ratio.
    """
    ratio_mode = st.session_state.get("canvas_ratio", "Standard")
    aspect_ratio = CANVAS_RATIOS.get(ratio_mode, 0.625)

    ua = ""
    sec_mobile = ""
    try:
        headers = getattr(st.context, "headers", {})
        ua = headers.get("user-agent", "").lower()
        sec_mobile = headers.get("sec-ch-ua-mobile", "")
    except Exception:
        pass

    is_tablet = "ipad" in ua or "tablet" in ua
    is_mobile = sec_mobile == "?1" or any(p in ua for p in ["mobile", "iphone", "android", "ipod", "blackberry", "windows phone"])

    if is_mobile and not is_tablet:
        canvas_w = 340  # Ergonomic fluid phone width fitting all standard phones
    elif is_tablet:
        canvas_w = 520  # Ergonomic fluid tablet width
    else:
        canvas_w = 560  # Ergonomic fluid desktop width

    canvas_h = max(200, int(round(canvas_w * aspect_ratio)))

    # Compute proportionally scaled brush strokes
    scale = max(0.65, min(1.35, canvas_w / 560.0))
    brush_draw = {
        "Small": max(8, int(round(12 * scale))),
        "Medium": max(14, int(round(20 * scale))),
        "Large": max(22, int(round(30 * scale)))
    }
    brush_rubber = {
        "Small": max(16, int(round(22 * scale))),
        "Medium": max(26, int(round(34 * scale))),
        "Large": max(36, int(round(46 * scale)))
    }

    device_label = "Phone" if (is_mobile and not is_tablet) else ("Tablet" if is_tablet else "Desktop")
    return canvas_w, canvas_h, brush_draw, brush_rubber, device_label

# Compute initial dimensions and preserve in session state across all reruns
INIT_CANVAS_W, INIT_CANVAS_H, BRUSH_MAP_DRAW, BRUSH_MAP_RUBBER, DETECTED_DEVICE = get_device_canvas_config()

if "detected_device" not in st.session_state or st.session_state.detected_device != DETECTED_DEVICE:
    st.session_state.detected_device = DETECTED_DEVICE
    st.session_state.canvas_w = INIT_CANVAS_W
    st.session_state.canvas_h = INIT_CANVAS_H

CANVAS_W = st.session_state.canvas_w
CANVAS_H = st.session_state.canvas_h

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

        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=active_width,
            stroke_color=active_stroke,
            background_color="#FFFFFF",
            height=st.session_state.canvas_h,
            width=st.session_state.canvas_w,
            drawing_mode="freedraw",
            initial_drawing=st.session_state.initial_drawing,
            display_toolbar=False,
            key="mnist_drawing_canvas",
            update_streamlit=True,
        )

        # Synchronize stroke history without altering initial_drawing during active drawing
        if canvas_result and canvas_result.json_data and "objects" in canvas_result.json_data:
            current_objects = canvas_result.json_data["objects"]
            if st.session_state.get("history_updated", False):
                if len(current_objects) == len(st.session_state.canvas_history):
                    st.session_state.history_updated = False
            else:
                if len(current_objects) > len(st.session_state.canvas_history):
                    st.session_state.canvas_history = list(current_objects)
                    st.session_state.redo_stack.clear()
                elif len(current_objects) < len(st.session_state.canvas_history) and len(current_objects) > 0:
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
                    st.session_state.history_updated = True
                    st.session_state.last_canvas_sig = None
                    if len(st.session_state.canvas_history) == 0:
                        st.session_state.last_prediction = None
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
                    st.session_state.history_updated = True
                    st.session_state.last_canvas_sig = None
                    st.rerun()

        with act_c5:
            if st.button("Clear", key="act_clear", help="Clear canvas"):
                st.session_state.canvas_history = []
                st.session_state.redo_stack = []
                st.session_state.initial_drawing = {
                    "version": "4.4.0",
                    "objects": [],
                }
                st.session_state.history_updated = True
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

        # Dynamic Fluid Layout Indicator & Aspect Ratio Switcher
        st.html(f"""
        <div style="margin-top: 14px; margin-bottom: 6px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 6px;">
            <span style="font-size: 11.5px; font-weight: 600; color: #64748B;">Drawing Aspect</span>
            <span style="font-size: 10.5px; color: #4338CA; font-weight: 700; background: #EEF2FF; padding: 2px 8px; border-radius: 6px; display: inline-flex; align-items: center; gap: 4px;">
                <span style="display:inline-block; width:6px; height:6px; border-radius:50%; background:#10B981;"></span>
                Dynamic {CANVAS_W} × {CANVAS_H}px
            </span>
        </div>
        """)

        ratio_val = st.pills(
            "Drawing Aspect",
            options=["Standard", "Wide", "Compact"],
            default=st.session_state.canvas_ratio,
            label_visibility="collapsed",
            key="canvas_ratio_pills",
        )
        if ratio_val and ratio_val != st.session_state.canvas_ratio:
            st.session_state.canvas_ratio = ratio_val
            aspect_ratio = CANVAS_RATIOS.get(ratio_val, 0.625)
            st.session_state.canvas_h = max(200, int(round(st.session_state.canvas_w * aspect_ratio)))
            st.session_state.last_prediction = None
            st.session_state.last_canvas_sig = None
            st.rerun()

    # ----------------------------------------------------------
    # Smart Multi-Digit Prediction Engine (Uses latest visible canvas)
    # ----------------------------------------------------------
    canvas_objects_count = len(st.session_state.canvas_history)
    current_canvas_sig = (canvas_objects_count, selected_model_name)

    needs_prediction = (
        predict_btn_clicked or 
        (not st.session_state.get("history_updated", False) and canvas_result is not None and canvas_result.image_data is not None and canvas_objects_count > 0 and current_canvas_sig != st.session_state.last_canvas_sig)
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
