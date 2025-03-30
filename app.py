import streamlit as st
from PIL import Image
import numpy as np
from collections import Counter

st.title("🎨 Color Pixel Counter (Dalí Test Ready)")

uploaded_file = st.file_uploader("Upload a pixelated image with 5 known colors", type=["png", "jpg", "jpeg"])

# Corrected colors extracted from your test image, with assigned values
color_value_map = {
    (252, 255, 251): 1,  # White
    (242, 230, 0): 2,    # Yellow
    (237, 100, 3): 3,    # Orange
    (192, 0, 37): 4,     # Red
    (0, 61, 167): 5      # Blue
}

TOLERANCE = 10  # adjust if needed

def is_close(color1, color2, tolerance=TOLERANCE):
    return all(abs(a - b) <= tolerance for a, b in zip(color1, color2))

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(img)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    flat_pixels = img_array.reshape(-1, img_array.shape[-1])
    pixel_tuples = [tuple(pixel) for pixel in flat_pixels]

    st.subheader("🎯 Mapped Color Counts and Values:")
    total_value = 0

    for i, (target_color, value) in enumerate(color_value_map.items(), start=1):
        count = sum(1 for pixel in pixel_tuples if is_close(pixel, target_color))
        total = count * value
        total_value += total

        hex_color = '#%02x%02x%02x' % target_color
        st.markdown(f"**{i}. Color (RGB): {target_color} → Pixels: {count} × Value: {value} = {total}**")
        st.color_picker(f"Preview Color {i}", value=hex_color, key=i, label_visibility="collapsed", disabled=True)

    st.subheader(f"🧮 Total Image Value: {total_value}")
