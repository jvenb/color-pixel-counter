import streamlit as st
from PIL import Image
import numpy as np
from collections import Counter

st.title("🎨 Color Pixel Counter (with Value Mapping)")

uploaded_file = st.file_uploader("Upload a pixelated image with 5 known colors", type=["png", "jpg", "jpeg"])

# Define your actual 5 colors and their assigned values
color_value_map = {
    (252, 255, 251): 1,  # White
    (242, 230, 0): 2,    # Yellow
    (238, 102, 7): 3,    # Orange
    (190, 0, 35): 4,     # Red
    (0, 61, 174): 5      # Blue
}

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(img)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Flatten image to list of RGB tuples
    flat_pixels = img_array.reshape(-1, img_array.shape[-1])
    pixel_tuples = [tuple(pixel) for pixel in flat_pixels]

    # Count how many pixels per color
    color_counts = Counter(pixel_tuples)

    st.subheader("🎯 Mapped Color Counts and Values:")
    total_value = 0

    for i, (color, value) in enumerate(color_value_map.items(), start=1):
        count = color_counts.get(color, 0)
        total = count * value
        total_value += total

        hex_color = '#%02x%02x%02x' % color
        st.markdown(f"**{i}. Color (RGB): {color} → Pixels: {count} × Value: {value} = {total}**")
        st.color_picker(f"Preview Color {i}", value=hex_color, key=i, label_visibility=\"collapsed\", disabled=True)

    st.subheader(f\"🧮 Total Image Value: {total_value}\")
