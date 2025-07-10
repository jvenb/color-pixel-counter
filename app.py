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
        "Upload a pixelated image (≤500×500)", type=["png","jpg","jpeg"]
    )
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        w,h = img.size
        if w*h > MAX_PIXELS:
            st.error("🚫 Image too large (over 500×500).")
        else:
            arr = np.array(img)
            flat = arr.reshape(-1,3)
            st.subheader("🎯 Counts & Values")
            total = 0
            matched = set()
            for label,fam in color_families.items():
                val = color_values[label]
                cnt = sum(any(is_close(px,v) for v in fam) for px in flat)
                total += cnt * val
                matched.update(i for i,px in enumerate(flat)
                               if any(is_close(px,v) for v in fam))
                sample = next(iter(fam))
                hexc = f"#{sample[0]:02x}{sample[1]:02x}{sample[2]:02x}"
                st.markdown(f"**{label}: {cnt} px × {val} = {cnt*val}**")
                st.color_picker("", value=hexc, disabled=True,
                                label_visibility="collapsed")
            st.subheader(f"🧮 Total Value: {total}")
            if len(matched) == len(flat):
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
                    st.color_picker("", value=hexc, disabled=True,
                                    label_visibility="collapsed")
            disp = img.resize((w*display_zoom, h*display_zoom), Image.NEAREST)
            st.image(disp, caption="Crisp Enlarged Image", use_container_width=False)

# --- Tool 2: Pixel Deleter / Adder ---
elif tool == "Pixel Deleter":
    st.header("🗑️ Pixel Deleter / ➕ Pixel Adder")
    uploaded = st.sidebar.file_uploader(
        "Upload a pixelated image", type=["png","jpg","jpeg"]
    )
    if uploaded:
        img = Image.open(uploaded).convert("RGBA")
        if ('name' not in st.session_state or
            st.session_state.name != uploaded.name):
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
        base = (st.session_state.current if chain
                else st.session_state.orig.copy())
        h,w = base.shape[:2]
        mode = st.sidebar.selectbox(
            "Pattern/Mode:",
            ["Original","Checkerboard","Alternate Rows","Alternate Columns",
             "Diagonal Stripes","Horizontal Stripes","Vertical Stripes",
             "Random Mask","Concentric Rings","Border Only","Custom Grid",
             "Pixel Adder"]
        )
        # --- Pixel Adder ---
        if mode == "Pixel Adder":
            dup = st.sidebar.slider("Duplicates per side", 1, 10, 1)
            maintain = st.sidebar.checkbox("Maintain original width")
            exp = np.repeat(base, 2*dup+1, axis=1)
            if not maintain:
                preview = exp
            else:
                seed = st.sidebar.number_input("Random seed", 0)
                rng = np.random.default_rng(seed)
                dist = st.sidebar.selectbox("Distribution:", ["Uniform", "Column"])
                if dist == "Uniform":
                    pct = st.sidebar.slider("% pixels expanded", 0, 100, 50)
                else:
                    col_choice = rng.integers(0, 2*dup+1)
                # build preview manually to maintain width
                preview = np.zeros_like(base)
                for i in range(h):
                    for j in range(w):
                        block = exp[i, j*(2*dup+1):(j+1)*(2*dup+1)]
                        if dist == "Uniform":
                            if rng.random() < pct/100:
                                pick = rng.integers(0, 2*dup+1)
                            else:
                                pick = dup
                        else:
                            pick = col_choice
                        preview[i,j] = block[pick]
        else:
                seed = st.sidebar.number_input("Random seed", 0)
                rng = np.random.default_rng(seed)
                distribution = st.sidebar.selectbox(
                    "Distribution:", ["Uniform", "Column"]
                )
                # Get indices
                if distribution == "Uniform":
                    pct = st.sidebar.slider("% pixels expanded", 0, 100, 50)
                    mask = rng.random((h, w)) < pct/100.0
                    random_idx = rng.integers(0, 2*dup+1, size=(h, w))
                    default_idx = np.full((h, w), dup, dtype=int)
                    idx = np.where(mask, random_idx, default_idx)
                else:  # Column
                    col_idx = rng.integers(0, 2*dup+1)
                    idx = np.full((h, w), col_idx, dtype=int)
                rows = np.arange(h)[:, None]
                preview = exp[rows, idx]
        else:
            # --- Pixel Deleter Patterns ---
            mask = np.ones((h, w), dtype=bool)
            if mode == "Checkerboard":
                inv = st.sidebar.checkbox("Invert checkerboard", False)
                mask = np.fromfunction(
                    lambda y,x: ((x+y)%2 == (1 if inv else 0)),
                    (h,w)
                )
            elif mode == "Alternate Rows":
                inv = st.sidebar.checkbox("Invert rows", False)
                mask = np.fromfunction(
                    lambda y,x: (y%2 == (1 if inv else 0)),
                    (h,w)
                )
            elif mode == "Alternate Columns":
                inv = st.sidebar.checkbox("Invert columns", False)
                mask = np.fromfunction(
                    lambda y,x: (x%2 == (1 if inv else 0)),
                    (h,w)
                )
            elif mode == "Diagonal Stripes":
                N = st.sidebar.slider("Stripe width N", 1, min(h,w)//2, 10)
                inv = st.sidebar.checkbox("Invert diagonal", False)
                mask = np.fromfunction(
                    lambda y,x: (((abs(x-y) % (2*N)) < N) ^ inv),
                    (h,w)
                )
            elif mode == "Horizontal Stripes":
                M = st.sidebar.slider("Stripe height M", 1, h//2, 10)
                inv = st.sidebar.checkbox("Invert horiz", False)
                mask = np.fromfunction(
                    lambda y,x: (((y//M) % 2) == 0) ^ inv,
                    (h,w)
                )
            elif mode == "Vertical Stripes":
                M = st.sidebar.slider("Stripe width M", 1, w//2, 10)
                inv = st.sidebar.checkbox("Invert vert", False)
                mask = np.fromfunction(
                    lambda y, x: ((((x//M) % 2) == 0) ^ inv),
                    (h, w)
                )
                )
            elif mode == "Random Mask":
                pct = st.sidebar.slider("Delete %", 0, 100, 50)
                seed = st.sidebar.number_input("Seed", 0)
                rng = np.random.default_rng(seed)
                mask = rng.random((h,w)) >= pct/100
            elif mode == "Concentric Rings":
                R = st.sidebar.slider("Ring thickness", 1, min(h,w)//4, 10)
                inv = st.sidebar.checkbox("Invert rings", False)
                cy, cx = h/2, w/2
                mask = np.fromfunction(
                    lambda y,x: (((
                        np.floor(np.hypot(x-cx, y-cy)/R) % 2) == 0) ^ inv),
                    (h,w)
                )
            elif mode == "Border Only":
                K = st.sidebar.slider("Border width K", 0, min(h,w)//2, 10)
                inv = st.sidebar.checkbox("Invert border", False)
                mask = np.fromfunction(
                    lambda y,x: (((x < K) | (x >= w-K) |
                                  (y < K) | (y >= h-K)) ^ inv),
                    (h,w)
                )
            else:  # Custom Grid
                A = st.sidebar.slider("Block width A", 1, w, 10)
                B = st.sidebar.slider("Block height B", 1, h, 10)
                inv = st.sidebar.checkbox("Invert grid", False)
                mask = np.fromfunction(
                    lambda y,x: (((x//A + y//B) % 2) == 0) ^ inv,
                    (h,w)
                )
            preview = base.copy()
            preview[...,3] = preview[...,3] * mask.astype(np.uint8)

        # Apply and display
        if st.sidebar.button("Apply"):
            st.session_state.undo.append(
                st.session_state.current.copy()
            )
            st.session_state.current = preview.copy()
        disp_img = Image.fromarray(preview)
        disp = disp_img.resize(
            (disp_img.width * display_zoom,
             disp_img.height * display_zoom),
            Image.NEAREST
        )
        st.image(disp, caption="Image Preview", use_container_width=False)
        buf = BytesIO()
        Image.fromarray(preview).save(buf, format="PNG")
        buf.seek(0)
        st.sidebar.download_button(
            "Download PNG", data=buf,
            file_name="output.png", mime="image/png"
        )

# --- Tool 3: Rubik Mosaic Checker ---
else:
    st.header("🔍 Rubik Mosaic Checker")
    inv_file = st.sidebar.file_uploader("Invariant (A)", type=["png","jpg","jpeg"])
    tgt_file = st.sidebar.file_uploader("Target (B)", type=["png","jpg","jpeg"])
    if inv_file and tgt_file:
        inv = Image.open(inv_file).convert("RGB")
        tgt = Image.open(tgt_file).convert("RGB")
        if inv.size != tgt.size:
            tgt = tgt.resize(inv.size, Image.NEAREST)
        w,h = inv.size
        inv_arr = np.array(inv)
        tgt_arr = np.array(tgt)
        inv_map = np.empty((h,w), dtype=object)
        tgt_map = np.empty((h,w), dtype=object)
        for y in range(h):
            for x in range(w):
                inv_map[y,x] = nearest_rubik_color(tuple(inv_arr[y,x]))
                tgt_map[y,x] = nearest_rubik_color(tuple(tgt_arr[y,x]))
        tgt_map = np.fliplr(tgt_map)
        cy, cx = h//2, w//2
        if tgt_map[cy,cx] != opposites[inv_map[cy,cx]]:
            st.error(
                f"Invariant violated: center must be {opposites[inv_map[cy,cx]]}"
            )
        else:
            st.success("✅ Opposite-face mosaic feasible!")
        dispA = np.uint8([
            [rubik_colors[inv_map[y,x]] for x in range(w)]
            for y in range(h)
        ])
        dispB = np.uint8([
            [rubik_colors[tgt_map[y,x]] for x in range(w)]
            for y in range(h)
        ])
        sideA = Image.fromarray(dispA).resize(
            (w*display_zoom, h*display_zoom), Image.NEAREST
        )
        sideB = Image.fromarray(dispB).resize(
            (w*display_zoom, h*display_zoom), Image.NEAREST
        )
        st.image(np.hstack([np.array(sideA), np.array(sideB)]),
                 use_container_width=False)
        st.markdown("### Sticker counts on B")
        for c, cnt in Counter(tgt_map.flatten()).items():
            st.write(f"- {c}: {cnt}")
        buf = BytesIO()
        Image.fromarray(dispB).save(buf, format="PNG")
        buf.seek(0)
        st.sidebar.download_button(
            "Download mapped PNG", data=buf,
            file_name="mapped_rubik.png", mime="image/png"
        )
