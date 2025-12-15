import streamlit as st
from PIL import Image
import numpy as np
from collections import Counter
from io import BytesIO

# -----------------------------
# App config
# -----------------------------
st.set_page_config(layout="wide")
st.title("🎨 Pixel Toolkit")

# Sidebar controls
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

# Rubik palette
rubik_colors = {
    "White":  (0xF2, 0xF6, 0xEF),
    "Yellow": (0xEF, 0xDF, 0x2A),
    "Red":    (0xBC, 0x27, 0x37),
    "Orange": (0xEF, 0x7D, 0x23),
    "Blue":   (0x13, 0x50, 0x98),
    "Green":  (0x0A, 0x8C, 0x00)
}
opposites = {
    "White": "Yellow", "Yellow": "White",
    "Red": "Orange", "Orange": "Red",
    "Blue": "Green", "Green": "Blue"
}

def nearest_rubik_color(px):
    d = {name: sum((c - p) ** 2 for c, p in zip(rgb, px))
         for name, rgb in rubik_colors.items()}
    return min(d, key=d.get)

# =========================================================
# Tool 1: Color Pixel Counter
# =========================================================
if tool == "Color Pixel Counter":
    st.header("🔢 Color Pixel Counter")
    uploaded = st.sidebar.file_uploader(
        "Upload a pixelated image (≤500×500)",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        w, h = img.size

        if w * h > MAX_PIXELS:
            st.error("🚫 Image too large.")
        else:
            arr = np.array(img)
            flat = arr.reshape(-1, 3)

            st.subheader("🎯 Counts & Values")
            total = 0
            matched = set()

            # Count pixels per family
            for label, fam in color_families.items():
                val = color_values[label]
                cnt = sum(any(is_close(px, v) for v in fam) for px in flat)
                total += cnt * val

                # Track matched pixels
                matched.update(
                    i for i, px in enumerate(flat)
                    if any(is_close(px, v) for v in fam)
                )

                sample = next(iter(fam))
                hexc = f"#{sample[0]:02x}{sample[1]:02x}{sample[2]:02x}"

                st.markdown(f"**{label}: {cnt} px × {val} = {cnt * val}**")

                # IMPORTANT FIX: unique key per widget
                st.color_picker(
                    " ",
                    value=hexc,
                    disabled=True,
                    label_visibility="collapsed",
                    key=f"fam_picker_{label}"
                )

            st.subheader(f"🧮 Total Value: {total}")

            if len(matched) == len(flat):
                st.success("✅ All pixels matched!")
            else:
                unmatched = [tuple(flat[i]) for i in range(len(flat)) if i not in matched]
                st.warning(f"⚠️ {len(unmatched)} unmatched pixels.")
                top = Counter(unmatched).most_common(10)

                st.markdown("### ❌ Top 10 Unmatched Colors")
                for idx, (col, cnt) in enumerate(top):
                    hexc = f"#{col[0]:02x}{col[1]:02x}{col[2]:02x}"
                    st.markdown(f"- {col}: {cnt} px")

                    # IMPORTANT FIX: unique key per widget
                    st.color_picker(
                        " ",
                        value=hexc,
                        disabled=True,
                        label_visibility="collapsed",
                        key=f"unmatched_picker_{idx}_{col}_{cnt}"
                    )

            st.image(
                img.resize((w * display_zoom, h * display_zoom), Image.NEAREST),
                caption="Crisp Enlarged Image",
                use_container_width=False
            )

# =========================================================
# Tool 2: Pixel Deleter / Adder
# =========================================================
elif tool == "Pixel Deleter":
    st.header("🗑️ Pixel Deleter / ➕ Pixel Adder")
    uploaded = st.sidebar.file_uploader(
        "Upload a pixelated image", type=["png", "jpg", "jpeg"]
    )

    if uploaded:
        img = Image.open(uploaded).convert("RGBA")

        if "name" not in st.session_state or st.session_state.name != uploaded.name:
            st.session_state.name = uploaded.name
            st.session_state.orig = np.array(img)
            st.session_state.current = st.session_state.orig.copy()
            st.session_state.undo = []

        if st.sidebar.button("Reset"):
            st.session_state.current = st.session_state.orig.copy()
            st.session_state.undo = []

        if st.sidebar.button("Undo") and st.session_state.undo:
            st.session_state.current = st.session_state.undo.pop()

        chain = st.sidebar.checkbox("Chain effects")
        base = st.session_state.current if chain else st.session_state.orig.copy()
        h, w = base.shape[:2]

        mode = st.sidebar.selectbox(
            "Pattern/Mode:",
            ["Original", "Checkerboard", "Alt Rows", "Alt Cols",
             "Diagonal", "H Stripes", "V Stripes", "Random Mask",
             "Rings", "Border", "Grid", "Pixel Adder"]
        )

        # Pixel Adder
        if mode == "Pixel Adder":
            dup = st.sidebar.slider("Duplicates per side", 1, 10, 1)
            maintain = st.sidebar.checkbox("Maintain original width")
            exp = np.repeat(base, 2 * dup + 1, axis=1)

            if not maintain:
                preview = exp
            else:
                seed = int(st.sidebar.number_input("Random seed", 0))
                rng = np.random.default_rng(seed)
                dist = st.sidebar.selectbox("Distribution:", ["Uniform", "Column", "Sparse"])

                if dist == "Uniform":
                    pct = st.sidebar.slider("% pixels offset", 0, 100, 50) / 100.0
                    preview = np.zeros_like(base)
                    for i in range(h):
                        for j in range(w):
                            offset = rng.integers(-dup, dup + 1) if rng.random() < pct else 0
                            newcol = int(np.clip(j + offset, 0, w - 1))
                            preview[i, j] = base[i, newcol]

                elif dist == "Column":
                    w_e = exp.shape[1]
                    idx = rng.choice(w_e, size=w, replace=False)
                    idx.sort()
                    preview = exp[:, idx, :]

                else:  # Sparse
                    pct = st.sidebar.slider("% pixels sparsed", 0, 100, 20) / 100.0
                    preview = base.copy()
                    for i in range(h):
                        available = set(range(w))
                        picks = []
                        for _ in range(int(w * pct)):
                            if not available:
                                break
                            j = int(rng.choice(list(available)))
                            picks.append(j)
                            rem = set(range(max(0, j - dup), min(w, j + dup + 1)))
                            available -= rem
                        for j in picks:
                            preview[i, j] = base[i, j]
                            if j
