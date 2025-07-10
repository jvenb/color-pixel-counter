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

# --- Color Pixel Counter setup ---
color_families = {
    "White":  {(242,246,239)},
    "Yellow": {(239,223,42)},
    "Orange": {(239,125,35)},
    "Red":    {(188,39,55)},
    "Blue":   {(19,80,152)},
    "Green":  {(10,140,0)}
}
color_values = {k: i+1 for i, k in enumerate(color_families)}
TOLERANCE  = 10
MAX_PIXELS = 500 * 500

def is_close(c1, c2):
    return all(abs(a-b) <= TOLERANCE for a,b in zip(c1,c2))

# --- Rubik-style palette setup ---
rubik_colors = {
    "White":  (0xf2,0xf6,0xef),
    "Yellow": (0xef,0xdf,0x2a),
    "Red":    (0xbc,0x27,0x37),
    "Orange": (0xef,0x7d,0x23),
    "Blue":   (0x13,0x50,0x98),
    "Green":  (0x0a,0x8c,0x00)
}
opposites = {"White":"Yellow","Yellow":"White",
             "Red":"Orange","Orange":"Red",
             "Blue":"Green","Green":"Blue"}

def nearest_rubik_color(px):
    d = {name: sum((c-p)**2 for c,p in zip(rgb,px))
         for name,rgb in rubik_colors.items()}
    return min(d, key=d.get)

# --- Tool 1: Color Pixel Counter ---
if tool == "Color Pixel Counter":
    st.header("🔢 Color Pixel Counter")
    uploaded = st.sidebar.file_uploader(
        "Upload a pixelated image (≤500×500)",
        type=["png","jpg","jpeg"]
    )
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        w,h = img.size
        if w*h > MAX_PIXELS:
            st.error("🚫 Image too large.")
        else:
            arr = np.array(img)
            flat = arr.reshape(-1,3)
            st.subheader("🎯 Counts & Values")
            total = 0; matched = set()
            for label,fam in color_families.items():
                val = color_values[label]
                cnt = sum(any(is_close(px,v) for v in fam) for px in flat)
                total += cnt*val
                matched.update(i for i,px in enumerate(flat)
                               if any(is_close(px,v) for v in fam))
                sample = next(iter(fam))
                hexc = f"#{sample[0]:02x}{sample[1]:02x}{sample[2]:02x}"
                st.markdown(f"**{label}: {cnt} px × {val} = {cnt*val}**")
                st.color_picker("", value=hexc,
                                disabled=True,
                                label_visibility="collapsed")
            st.subheader(f"🧮 Total Value: {total}")
            if len(matched)==len(flat):
                st.success("✅ All pixels matched!")
            else:
                unmatched = [tuple(flat[i]) for i in range(len(flat))
                             if i not in matched]
                st.warning(f"⚠️ {len(unmatched)} unmatched pixels.")
                top = Counter(unmatched).most_common(10)
                st.markdown("### ❌ Top 10 Unmatched Colors")
                for col,cnt in top:
                    hexc = f"#{col[0]:02x}{col[1]:02x}{col[2]:02x}"
                    st.markdown(f"- {col}: {cnt} px")
                    st.color_picker("", value=hexc,
                                    disabled=True,
                                    label_visibility="collapsed")
            st.image(img.resize((w*display_zoom,h*display_zoom),
                                Image.NEAREST),
                     caption="Crisp Enlarged Image",
                     use_container_width=False)

# --- Tool 2: Pixel Deleter / Adder ---
elif tool == "Pixel Deleter":
    st.header("🗑️ Pixel Deleter / ➕ Pixel Adder")
    uploaded = st.sidebar.file_uploader(
        "Upload a pixelated image",
        type=["png","jpg","jpeg"]
    )
    if uploaded:
        img = Image.open(uploaded).convert("RGBA")
        if ('name' not in st.session_state
            or st.session_state.name != uploaded.name):
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
        base = (st.session_state.current
                if chain
                else st.session_state.orig.copy())
        h,w = base.shape[:2]

        mode = st.sidebar.selectbox(
            "Pattern/Mode:",
            ["Original","Checkerboard","Alt Rows","Alt Cols",
             "Diagonal","H Stripes","V Stripes","Random Mask",
             "Rings","Border","Grid","Pixel Adder"]
        )

        if mode == "Pixel Adder":
            dup = st.sidebar.slider(
                "Duplicates per side", 1, 10, 1
            )
            maintain = st.sidebar.checkbox(
                "Maintain original width"
            )
            exp = np.repeat(base, 2*dup+1, axis=1)
            if not maintain:
                preview = exp
            else:
                seed = st.sidebar.number_input("Random seed", 0)
                rng = np.random.default_rng(seed)
                dist = st.sidebar.selectbox(
                    "Distribution:", ["Uniform","Column"]
                )
                if dist == "Uniform":
                    pct = st.sidebar.slider(
                        "% pixels expanded", 0, 100, 50
                    )
                    mask = rng.random((h, w)) < pct/100.0
                    rand_idx = rng.integers(
                        0, 2*dup+1, size=(h, w)
                    )
                    default_idx = np.full((h, w), dup, dtype=int)
                    idx = np.where(mask, rand_idx, default_idx)
                else:  # Column mode
                    col_idx = rng.integers(0, 2*dup+1)
                    idx = np.full((h, w), col_idx, dtype=int)
                rows = np.arange(h)[:, None]
                preview = exp[rows, idx]
        else:
            mask = np.ones((h, w), bool)
            if mode == "Checkerboard":
                inv = st.sidebar.checkbox("Invert")
                mask = np.fromfunction(
                    lambda y,x: ((x+y)%2 == int(inv)),
                    (h,w)
                )
            elif mode == "Alt Rows":
                inv = st.sidebar.checkbox("Invert")
                mask = np.fromfunction(
                    lambda y,x: (y%2 == int(inv)),
                    (h,w)
                )
            elif mode == "Alt Cols":
                inv = st.sidebar.checkbox("Invert")
                mask = np.fromfunction(
                    lambda y,x: (x%2 == int(inv)),
                    (h,w)
                )
            elif mode == "Diagonal":
                N = st.sidebar.slider(
                    "Width N", 1, min(h,w)//2, 10
                )
                inv = st.sidebar.checkbox("Invert")
                mask = np.fromfunction(
                    lambda y,x: (((abs(x-y)%(2*N))<N) ^ inv),
                    (h,w)
                )
            elif mode == "H Stripes":
                M = st.sidebar.slider(
                    "Height M", 1, h//2, 10
                )
                inv = st.sidebar.checkbox("Invert")
                mask = np.fromfunction(
                    lambda y,x: (((y//M)%2)==0) ^ inv,
                    (h,w)
                )
            elif mode == "V Stripes":
                M = st.sidebar.slider(
                    "Width M", 1, w//2, 10
                )
                inv = st.sidebar.checkbox("Invert")
                mask = np.fromfunction(
                    lambda y,x: (((x//M)%


