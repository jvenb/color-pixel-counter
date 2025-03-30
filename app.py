import streamlit as st
from PIL import Image
import numpy as np

st.title("🎨 Color Pixel Counter")

uploaded_file = st.file_uploader("Upload a pixelated image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(img)
    st.image(img, caption="Click on the image to select a color", use_column_width=True)

    x = st.number_input("X Coordinate (pixel)", min_value=0, max_value=img.width - 1, step=1)
    y = st.number_input("Y Coordinate (pixel)", min_value=0, max_value=img.height - 1, step=1)

    if st.button("Count Pixels of This Color"):
        selected_color = tuple(img_array[int(y), int(x)])
        match_count = np.sum(np.all(img_array == selected_color, axis=-1))

        st.markdown(f"**Selected Color (RGB):** {selected_color}")
        st.markdown(f"**Matching Pixels:** {match_count}")
