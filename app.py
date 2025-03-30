import streamlit as st
from PIL import Image
import numpy as np
from collections import Counter

st.title("🎨 Color Pixel Counter (Dalí Test Ready + Pixel Audit)")

uploaded_file = st.file_uploader("Upload a pixelated image with 5 known colors", type=["png", "jpg", "jpeg"])

# Corrected colors extracted from your test image, with assigned values
color_value_map = {
    (252, 255, 251): 1,  # White
    (242, 230, 0): 2,    # Yellow
    (237, 100, 3): 3,    # Orange
    (192, 0, 37): 4,     # Red
    (0, 61, 167): 5      # Blue
}

TOLERANCE = 10

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
    total_counted_pixels = 0
    matched_indices = set()

    for i, (target_color, value) in enumerate(color_value_map.items(), start=1):
        count = 0
        for idx, pixel in enumerate(pixel_tuples):
            if is_close(pixel, target_color):
                count += 1
                matched_indices.add(idx)
        total = count * value
        total_value += total
        total_counted_pixels += count

        hex_color = '#%02x%02x%02x' % target_color
        st.markdown(f"**{i}. Color (RGB): {target_color} → Pixels: {count} × Value: {value} = {total}**")
        st.color_picker(f"Preview Color {i}", value=hex_color, key=i, label_visibility="collapsed", disabled=True)

    st.subheader(f"🧮 Total Image Value: {total_value}")

    # Check and list unmatched pixels
    total_pixels = len(pixel_tuples)
    unmatched_indices = [i for i in range(total_pixels) if i not in matched_indices]
    unmatched_pixels = [pixel_tuples[i] for i in unmatched_indices]

    if not unmatched_pixels:
        st.success("✅ All pixels in the image were matched to known colors. No pixels were left out.")
    else:
        st.warning(f"⚠️ {len(unmatched_pixels)} pixel(s) were not matched to any of the defined colors.")
        unmatched_summary = Counter(unmatched_pixels).most_common(10)
        st.markdown("### ❌ Unmatched Colors (Top 10):")
        for color, count in unmatched_summary:
            hex_color = '#%02x%02x%02x' % color
            st.markdown(f"- {color} → {count} pixel(s)")
            st.color_picker("", value=hex_color, key=f"unmatched-{hex_color}", disabled=True, label_visibility="collapsed")
