import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
from collections import Counter

st.title("🎨 Color Pixel Counter")

uploaded_file = st.file_uploader("Upload a pixelated image with 5 known colors", type=["png", "jpg", "jpeg"])

# Define color families for each group and their values
color_families = {
    "White": {(252, 255, 251), (255, 255, 255)},
    "Yellow": {(242, 230, 0)},
    "Orange": {(238, 102, 7), (237, 100, 3)},
    "Red": {(192, 0, 37), (190, 0, 35)},
    "Blue": {(0, 61, 174), (0, 61, 167)}
}

color_values = {
    "White": 1,
    "Yellow": 2,
    "Orange": 3,
    "Red": 4,
    "Blue": 5
}

TOLERANCE = 10
MAX_PIXELS = 500 * 500

def is_close(color1, color2, tolerance=TOLERANCE):
    return all(abs(a - b) <= tolerance for a, b in zip(color1, color2))

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")

    if img.width * img.height > MAX_PIXELS:
        st.error("🚫 Image too large. Please upload an image smaller than 500×500 pixels.")
    else:
        enlargement_factor = st.slider("🔍 Image Zoom (Pixel Size Multiplier)", min_value=1, max_value=50, value=20)

        img_array = np.array(img)
        flat_pixels = img_array.reshape(-1, img_array.shape[-1])
        pixel_tuples = [tuple(pixel) for pixel in flat_pixels]

        st.subheader("🎯 Mapped Color Counts and Values:")
        total_value = 0
        total_counted_pixels = 0
        matched_indices = set()

        for i, (label, family) in enumerate(color_families.items(), start=1):
            value = color_values[label]
            count = 0
            for idx, pixel in enumerate(pixel_tuples):
                if any(is_close(pixel, variant) for variant in family):
                    count += 1
                    matched_indices.add(idx)
            total = count * value
            total_value += total
            total_counted_pixels += count

            sample_color = next(iter(family))
            hex_color = '#%02x%02x%02x' % sample_color
            st.markdown(f"**{i}. {label} → Pixels: {count} × Value: {value} = {total}**")
            st.color_picker(f"Preview {label}", value=hex_color, key=i, label_visibility="collapsed", disabled=True)

        st.subheader(f"🧮 Total Image Value: {total_value}")

        total_pixels = len(pixel_tuples)
        unmatched_indices = [i for i in range(total_pixels) if i not in matched_indices]
        unmatched_pixels = [pixel_tuples[i] for i in unmatched_indices]

        if not unmatched_pixels:
            st.success("✅ All pixels in the image were matched to known colors. No pixels were left out.")
            enlarged = img.resize((img.width * enlargement_factor, img.height * enlargement_factor), Image.NEAREST)
            st.image(enlarged, caption="Crisp Enlarged Image", use_container_width=False)
        else:
            st.warning(f"⚠️ {len(unmatched_pixels)} pixel(s) were not matched to any of the defined color families.")
            unmatched_summary = Counter(unmatched_pixels).most_common(10)
            st.markdown("### ❌ Unmatched Colors (Top 10):")
            for color, count in unmatched_summary:
                hex_color = '#%02x%02x%02x' % color
                st.markdown(f"- {color} → {count} pixel(s)")
                st.color_picker("", value=hex_color, key=f"unmatched-{hex_color}", disabled=True, label_visibility="collapsed")

            grayscale = img.convert("L").convert("RGB")
            draw = ImageDraw.Draw(grayscale)
            width, height = img.size
            for idx in unmatched_indices:
                x = idx % width
                y = idx // width
                draw.point((x, y), fill=img.getpixel((x, y)))

            enlarged_highlight = grayscale.resize((width * enlargement_factor, height * enlargement_factor), Image.NEAREST)
            st.image(enlarged_highlight, caption="Unmatched Pixels Highlighted (color on grayscale)", use_container_width=False)
