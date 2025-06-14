import streamlit as st
from PIL import Image
import numpy as np
from collections import Counter
from io import BytesIO

# Configure wide layout
st.set_page_config(layout="wide")
st.title("🎨 Pixel Toolkit")

# Sidebar controls
tool = st.sidebar.radio(
    "Select tool:",
    ["Color Pixel Counter", "Pixel Deleter", "Rubik Mosaic Checker"]
)
display_zoom = st.sidebar.slider("Display Zoom Multiplier", 1, 50, 20)

# Standard Rubik's colors (RGB)
rubik_colors = {
    "White": (255, 255, 255),
    "Yellow": (255, 213, 0),
    "Red": (196, 30, 58),
    "Orange": (255, 88, 0),
    "Blue": (0, 70, 173),
    "Green": (0, 155, 72)
}
opposites = {
    "White": "Yellow",
    "Yellow": "White",
    "Red": "Orange",
    "Orange": "Red",
    "Blue": "Green",
    "Green": "Blue"
}

def nearest_rubik_color(pixel):
    # Find the cube color with minimum Euclidean distance
    dists = {name: sum((c - p) ** 2 for c, p in zip(rgb, pixel)) for name, rgb in rubik_colors.items()}
    return min(dists, key=dists.get)

if tool == "Color Pixel Counter":
    # ... existing Color Pixel Counter code (unchanged) ...
    pass

elif tool == "Pixel Deleter":
    # ... existing Pixel Deleter code (unchanged) ...
    pass

else:  # Rubik Mosaic Checker
    st.header("🔍 Rubik Mosaic Checker")
    st.sidebar.header("Rubik Mosaic Checker Settings")
    invariant_file = st.sidebar.file_uploader("Upload invariant design", type=["png","jpg","jpeg"] )
    target_file = st.sidebar.file_uploader("Upload target design", type=["png","jpg","jpeg"] )
    if invariant_file and target_file:
        inv_img = Image.open(invariant_file).convert("RGB")
        tgt_img = Image.open(target_file).convert("RGB")
        # Resize target to match invariant if different
        if inv_img.size != tgt_img.size:
            tgt_img = tgt_img.resize(inv_img.size, Image.NEAREST)

        inv_arr = np.array(inv_img)
        tgt_arr = np.array(tgt_img)
        h, w = inv_img.size[1], inv_img.size[0]

        # Map both images to nearest Rubik colors
        inv_mapped = np.zeros((h, w), dtype=object)
        tgt_mapped = np.zeros((h, w), dtype=object)
        for y in range(h):
            for x in range(w):
                inv_mapped[y, x] = nearest_rubik_color(tuple(inv_arr[y, x]))
                tgt_mapped[y, x] = nearest_rubik_color(tuple(tgt_arr[y, x]))

        # Check center invariance
        cy, cx = h // 2, w // 2
        if tgt_mapped[cy, cx] != inv_mapped[cy, cx]:
            st.error(f"Invariant violated: center color must remain {inv_mapped[cy, cx]}.")
        else:
            st.success("Center invariant holds.")

        # Check opposite-color adjacency rule (optional)
        violations = []
        for name, opp in opposites.items():
            # Example rule: no two adjacent pixels with opposite colors
            mask1 = (tgt_mapped == name)
            mask2 = (tgt_mapped == opp)
            # Check horizontal neighbors
            for y in range(h):
                for x in range(w - 1):
                    if mask1[y, x] and mask2[y, x + 1]:
                        violations.append(((x, y), (x + 1, y), name, opp))
        if violations:
            st.warning(f"Found {len(violations)} adjacent opposite-color violations.")
        else:
            st.success("No immediate opposite-color adjacency violations.")

        # Display mapped target image
        display_arr = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                display_arr[y, x] = rubik_colors[tgt_mapped[y, x]]
        display_img = Image.fromarray(display_arr)
        disp = display_img.resize((w * display_zoom, h * display_zoom), Image.NEAREST)
        st.image(disp, caption="Target mapped to Rubik colors", use_container_width=False)

        # Optionally show summary
        counts = Counter(tgt_mapped.flatten())
        st.markdown("### Sticker counts on target:")
        for color, cnt in counts.items():
            st.write(f"- {color}: {cnt}")
        # (Further cube-invariant checks could be added later)

        # Download mapped design
        buf = BytesIO()
        display_img.save(buf, format="PNG")
        buf.seek(0)
        st.sidebar.download_button("Download Mapped PNG", data=buf, file_name="mapped_rubik.png", mime="image/png")

# Note: Fill in existing modes or extract them to helper functions for brevity
