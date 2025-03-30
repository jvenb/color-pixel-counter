import streamlit as st
from PIL import Image
import numpy as np
from collections import Counter

st.title("🎨 Color Pixel Counter (Auto 5-Color Mode)")

uploaded_file = st.file_uploader("Upload a pixelated image with 5 colors", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(img)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Flatten the image array to a list of RGB tuples
    flat_pixels = img_array.reshape(-1, img_array.shape[-1])
    pixel_tuples = [tuple(pixel) for pixel in flat_pixels]

    # Count occurrences of each color
    color_counts = Counter(pixel_tuples)

    # Get the 5 most common colors
    most_common_colors = color_counts.most_common(5)

    st.subheader("🖐️ Top 5 Colors and Their Counts:")
    for i, (color, count) in enumerate(most_common_colors, start=1):
        st.markdown(f"**{i}. Color (RGB): {color} → {count} pixels**")
        st.color_picker(f"Preview Color {i}", value='#%02x%02x%02x' % color, key=i)
