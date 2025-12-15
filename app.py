import streamlit as st
from PIL import Image
import numpy as np
from collections import Counter
from io import BytesIO

# -----------------------------
# Pillow-safe resampling constant
# -----------------------------
try:
    NEAREST = Image.Resampling.NEAREST  # Pillow >= 9.1
except AttributeError:
    NEAREST = Image.NEAREST  # Older Pillow fallback

# -----------------------------
# App config
# -----------------------------
st.set_page_config(layout="wide")
st.title("🎨 Pixel Toolkit")

tool = st.sidebar.radio(
    "Select tool:",
    ["Color Pixel Counter", "Pixel Deleter", "Rubik Mosaic Checker"]
)
display_zoom = st.sidebar.slider("Display Zoom Multiplier", 1, 50, 20)

# -----------------------------
# Shared palettes / helpers
# -----------------------------
color_families = {
    "White":  {(242, 246, 239)},
    "Yellow": {(239, 223, 42)},
    "Orange": {(239, 125, 35)},
    "Red":    {(188, 39, 55)},
    "Blue":   {(19, 80, 152)},
    "Green":  {(10, 140, 0)}
}
color_values = {k: i + 1 for i, k in enumerate(color_families)}
TOLERANCE = 10
MAX_PIXELS = 500 * 500

def is_close(c1, c2):
    return all(abs(a - b) <= TOLERANCE for a, b in zip(c1, c2))

rubik_colors = {
    "White":  (0xF2, 0xF6, 0xEF),
    "Yellow": (0xEF, 0xDF, 0x2A),
    "Red":    (0xBC, 0x27, 0x37),
    "Orange": (0xEF, 0x7D, 0x23),
    "Blue":   (0x13, 0x50, 0x98),
    "Green":  (0x0A, 0x8C, 0x00)
}
opposites = {
    "White": "Yellow", "Yellow": "Whi
