import streamlit as st
from PIL import Image
import numpy as np
from collections import Counter

st.title("🎨 Color Pixel Counter (with Value Mapping)")

uploaded_file = st.file_uploader("Upload a pixelated image with 5 known colors", type=["png", "jpg", "jpeg"])

# Define your 5 known colors and their values
color_value_map = {
    (255, 255, 255): 1,  # White
    (255, 255, 0): 2,    # Yellow
    (255, 165, 0): 3,    # Orange
    (255, 0, 0): 4,      # Red
    (0, 0, 255): 5       # Blue
}

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(img)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Flatten the image array to a list of RGB tuples
    flat_pixels = img_array.reshape(-1, img_array.shape[-1])
    pixel_tuples = [tuple(pixel) for pixel in flat_pixels]

    # Count occurrences of each color
    color_counts = Counter(pixel_tuples)

    st.subheader("🎯 Mapped Color Counts and Values:")
    total_value = 0

    for i, (color, value) in enumerate(color_value_map.items(), start=1):
        count = color_counts.get(color, 0)
        total = count * value
        total_value += total

        hex_color = '#%02x%02x%02x' % color
        st.markdown(f"**{i}. Color (RGB): {color} → Pixels: {count} × Value: {value} = {total}**")
        st.color_picker(f"Preview Color {i}", value=hex_color, key=i, label_visibility="collapsed", disabled=True)

    st.subheader(f"🧮 Total Image Value: {total_value}")
