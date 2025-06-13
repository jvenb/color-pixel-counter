import streamlit as st
from PIL import Image
import numpy as np
from collections import Counter
from io import BytesIO
import traceback

# Configure wide layout
st.set_page_config(layout="wide")

# Main app with error handling
def main():
    st.title("🎨 Pixel Toolkit")

    # Sidebar controls
    tool = st.sidebar.radio("Select tool:", ["Color Pixel Counter", "Pixel Deleter"])
    display_zoom = st.sidebar.slider("Display Zoom Multiplier", 1, 50, 20)

    if tool == "Color Pixel Counter":
        st.sidebar.header("🔢 Color Pixel Counter Settings")
        uploaded = st.sidebar.file_uploader("Upload a pixelated image (≤500×500)", type=["png","jpg","jpeg"])

        # Define color families and values
        color_families = {
            "White": {(252,255,251),(255,255,255)},
            "Yellow": {(242,230,0)},
            "Orange": {(238,102,7),(237,100,3)},
            "Red": {(192,0,37),(190,0,35)},
            "Blue": {(0,61,174),(0,61,167)}
        }
        color_values = {k:i+1 for i,k in enumerate(color_families)}
        TOLERANCE = 10
        MAX_PIXELS = 500 * 500
        def is_close(a, b): return all(abs(x - y) <= TOLERANCE for x, y in zip(a, b))

        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            w, h = img.size
            if w * h > MAX_PIXELS:
                st.error("🚫 Image too large.")
            else:
                arr = np.array(img)
                flat = arr.reshape(-1, 3)

                st.header("🎯 Color Counts & Values")
                total_val, matched = 0, set()
                for label, fam in color_families.items():
                    val = color_values[label]
                    cnt = 0
                    for i, px in enumerate(flat):
                        if any(is_close(px, v) for v in fam):
                            cnt += 1
                            matched.add(i)
                    total_val += cnt * val
                    sample = next(iter(fam))
                    hexc = f"#{sample[0]:02x}{sample[1]:02x}{sample[2]:02x}"
                    st.markdown(f"**{label}: {cnt} px × {val} = {cnt * val}**")
                    st.color_picker(label, value=hexc, disabled=True, label_visibility="collapsed")

                st.subheader(f"🧮 Total Value: {total_val}")
                unmatched = [tuple(flat[i]) for i in range(flat.shape[0]) if i not in matched]
                if unmatched:
                    st.warning(f"⚠️ {len(unmatched)} unmatched pixels.")
                    top = Counter(unmatched).most_common(10)
                    st.markdown("### ❌ Top 10 Unmatched Colors:")
                    for col, cnt in top:
                        hexc = f"#{col[0]:02x}{col[1]:02x}{col[2]:02x}"
                        st.markdown(f"- {col}: {cnt} px")
                        st.color_picker("", value=hexc, disabled=True, label_visibility="collapsed")

                disp = img.resize((w * display_zoom, h * display_zoom), Image.NEAREST)
                st.image(disp, caption="Crisp Enlarged Image", use_container_width=False)

    elif tool == "Pixel Deleter":
        st.sidebar.header("🗑️ Pixel Deleter Settings")
        uploaded = st.sidebar.file_uploader("Upload a pixelated image", type=["png","jpg","jpeg"])
        if uploaded:
            img = Image.open(uploaded).convert("RGBA")
            # Initialize session state
            if 'upload_name' not in st.session_state or st.session_state.upload_name != uploaded.name:
                st.session_state.upload_name = uploaded.name
                st.session_state.orig_arr = np.array(img)
                st.session_state.work_arr = st.session_state.orig_arr.copy()
                st.session_state.undo_stack = []

            # Reset & Undo
            if st.sidebar.button("Reset Effects"):
                st.session_state.work_arr = st.session_state.orig_arr.copy()
                st.session_state.undo_stack = []
            if st.sidebar.button("Undo Last Effect") and st.session_state.undo_stack:
                st.session_state.work_arr = st.session_state.undo_stack.pop()

            # Chaining toggle and base array
            chain = st.sidebar.checkbox("Chain effects", False)
            base_arr = st.session_state.work_arr if chain else st.session_state.orig_arr.copy()
            h, w = base_arr.shape[:2]

            # Pattern selection
            pattern = st.sidebar.selectbox("Select pattern:", [
                "Original", "Checkerboard", "Alternate Rows", "Alternate Columns",
                "Diagonal Stripes", "Horizontal Stripes", "Vertical Stripes",
                "Random Mask", "Concentric Rings", "Border Only", "Custom Grid"
            ])

            # Generate mask
            if pattern == "Original":
                mask = np.ones((h, w), dtype=bool)
            elif pattern == "Checkerboard":
                inv = st.sidebar.checkbox("Invert checkerboard", False)
                mask = np.fromfunction(lambda y, x: ((x + y) % 2 == (1 if inv else 0)), (h, w))
            elif pattern == "Alternate Rows":
                inv = st.sidebar.checkbox("Invert rows", False)
                mask = np.fromfunction(lambda y, x: (y % 2 == (1 if inv else 0)), (h, w))
            elif pattern == "Alternate Columns":
                inv = st.sidebar.checkbox("Invert cols", False)
                mask = np.fromfunction(lambda y, x: (x % 2 == (1 if inv else 0)), (h, w))
            elif pattern == "Diagonal Stripes":
                N = st.sidebar.slider("Stripe width N", 1, min(h, w) // 2, 10)
                inv = st.sidebar.checkbox("Invert diagonal", False)
                mask = np.fromfunction(lambda y, x: (((abs(x - y) % (2 * N)) < N) ^ inv), (h, w))
            elif pattern == "Horizontal Stripes":
                M = st.sidebar.slider("Stripe height M", 1, h // 2, 10)
                inv = st.sidebar.checkbox("Invert horiz", False)
                mask = np.fromfunction(lambda y, x: (((y // M) % 2) == 0) ^ inv, (h, w))
            elif pattern == "Vertical Stripes":
                M = st.sidebar.slider("Stripe width M", 1, w // 2, 10)
                inv = st.sidebar.checkbox("Invert vert", False)
                mask = np.fromfunction(lambda y, x: (((x // M) % 2) == 0) ^ inv, (h, w))
            elif pattern == "Random Mask":
                pct = st.sidebar.slider("Delete %", 0, 100, 50)
                seed = st.sidebar.number_input("Seed", value=0)
                rng = np.random.default_rng(seed)
                mask = rng.random((h, w)) >= pct / 100
            elif pattern == "Concentric Rings":
                R = st.sidebar.slider("Ring thickness", 1, min(h, w) // 4, 10)
                inv = st.sidebar.checkbox("Invert rings", False)
                cy, cx = h / 2, w / 2
                mask = np.fromfunction(
                    lambda y, x: ((np.floor(np.hypot(x - cx, y - cy) / R) % 2) == 0) ^ inv,
                    (h, w)
                )
            elif pattern == "Border Only":
                K = st.sidebar.slider("Border width K", 0, min(h, w) // 2, 10)
                inv = st.sidebar.checkbox("Invert border", False)
                mask = np.fromfunction(
                    lambda y, x: (((x < K) | (x >= w - K) | (y < K) | (y >= h - K))) ^ inv,
                    (h, w)
                )
            else:  # Custom Grid
                A = st.sidebar.slider("Block width A", 1, w, 10)
                B = st.sidebar.slider("Block height B", 1, h, 10)
                inv = st.sidebar.checkbox("Invert grid", False)
                mask = np.fromfunction(
                    lambda y, x: (((x // A + y // B) % 2) == 0) ^ inv,
                    (h, w)
                )

            # Compute preview
            preview = base_arr.copy()
            preview[..., 3] *= mask.astype(np.uint8)

            # Apply effect
            if st.sidebar.button("Apply Effect"):
                st.session_state.undo_stack.append(st.session_state.work_arr.copy())
                st.session_state.work_arr = preview.copy()

            # Display preview
            disp_img = Image.fromarray(preview).resize((w * display_zoom, h * display_zoom), Image.NEAREST)
            st.image(disp_img, caption="Image Preview", use_container_width=False)

            # Download
            buf = BytesIO()
            Image.fromarray(preview).save(buf, format="PNG")
            buf.seek(0)
            st.sidebar.download_button("Download PNG", data=buf, file_name="output.png", mime="image/png")

# Wrap execution to catch errors
try:
    main()
except Exception:
    st.error("😢 The app encountered an error:")
    st.code(traceback.format_exc())

