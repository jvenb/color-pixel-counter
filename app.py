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
color_values = {
    "White":  1,
    "Yellow": 2,
    "Orange": 3,
    "Red":    4,
    "Blue":   5,
    "Green":  6
}
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
opposites = {
    "White":"Yellow","Yellow":"White",
    "Red":"Orange","Orange":"Red",
    "Blue":"Green","Green":"Blue"
}

def nearest_rubik_color(px):
    d = {name: sum((c-p)**2 for c,p in zip(rgb,px)) for name,rgb in rubik_colors.items()}
    return min(d, key=d.get)

# --- Mode 1: Color Pixel Counter ---
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
                cnt = 0
                for i,px in enumerate(flat):
                    if any(is_close(px,v) for v in fam):
                        cnt += 1; matched.add(i)
                subtotal = cnt * val
                total += subtotal
                sample = next(iter(fam))
                hexc = f"#{sample[0]:02x}{sample[1]:02x}{sample[2]:02x}"
                st.markdown(f"**{label}: {cnt} px × {val} = {subtotal}**")
                st.color_picker("", value=hexc, disabled=True, label_visibility="collapsed")
            st.subheader(f"🧮 Total Value: {total}")
            if len(matched) == len(flat):
                st.success("✅ All pixels matched to known colors!")
            else:
                unmatched = [tuple(flat[i]) for i in range(len(flat)) if i not in matched]
                st.warning(f"⚠️ {len(unmatched)} unmatched pixels.")
                top = Counter(unmatched).most_common(10)
                st.markdown("### ❌ Top 10 Unmatched Colors:")
                for col,cnt in top:
                    hexc = f"#{col[0]:02x}{col[1]:02x}{col[2]:02x}"
                    st.markdown(f"- {col}: {cnt} px")
                    st.color_picker("", value=hexc, disabled=True, label_visibility="collapsed")
            disp = img.resize((w*display_zoom, h*display_zoom), Image.NEAREST)
            st.image(disp, caption="Crisp Enlarged Image", use_container_width=False)

# --- Mode 2: Pixel Deleter / Adder ---
elif tool == "Pixel Deleter":
    st.header("🗑️ Pixel Deleter / ➕ Pixel Adder")
    uploaded = st.sidebar.file_uploader("Upload a pixelated image", type=["png","jpg","jpeg"])
    if uploaded:
        img = Image.open(uploaded).convert("RGBA")
        if 'upload_name' not in st.session_state or st.session_state.upload_name != uploaded.name:
            st.session_state.upload_name = uploaded.name
            st.session_state.orig_arr = np.array(img)
            st.session_state.work_arr = st.session_state.orig_arr.copy()
            st.session_state.undo_stack = []
        if st.sidebar.button("Reset Effects"):
            st.session_state.work_arr = st.session_state.orig_arr.copy(); st.session_state.undo_stack = []
        if st.sidebar.button("Undo Last Effect") and st.session_state.undo_stack:
            st.session_state.work_arr = st.session_state.undo_stack.pop()

        chain = st.sidebar.checkbox("Chain effects", False)
        base = st.session_state.work_arr if chain else st.session_state.orig_arr.copy()
        h, w = base.shape[:2]

        pattern = st.sidebar.selectbox(
            "Select pattern:",
            [
                "Original","Checkerboard","Alternate Rows","Alternate Columns",
                "Diagonal Stripes","Horizontal Stripes","Vertical Stripes",
                "Random Mask","Concentric Rings","Border Only","Custom Grid",
                "Pixel Adder"
            ]
        )

        if pattern == "Pixel Adder":
            arr = base.copy()
            dup = st.sidebar.slider("Duplicates per side", 1, 10, 1)
            maintain = st.sidebar.checkbox("Maintain original width", False)
            if maintain:
                seed = st.sidebar.number_input("Deletion seed", 0)
                expanded = np.repeat(arr, 2*dup+1, axis=1)
                h_exp, w_exp = expanded.shape[:2]
                rng = np.random.default_rng(seed)
                idx = rng.choice(w_exp, size=w, replace=False)
                idx.sort()
                preview = expanded[:, idx, :]
            else:
                preview = np.repeat(arr, 2*dup+1, axis=1)
        else:
            mask = np.ones((h, w), bool)
            if pattern == "Checkerboard":
                inv = st.sidebar.checkbox("Invert checkerboard", False)
                mask = np.fromfunction(lambda y,x: ((x+y)%2==(1 if inv else 0)), (h, w))
            elif pattern == "Alternate Rows":
                inv = st.sidebar.checkbox("Invert rows", False)
                mask = np.fromfunction(lambda y,x: (y%2==(1 if inv else 0)), (h, w))
            elif pattern == "Alternate Columns":
                inv = st.sidebar.checkbox("Invert cols", False)
                mask = np.fromfunction(lambda y,x: (x%2==(1 if inv else 0)), (h, w))
            elif pattern == "Diagonal Stripes":
                N = st.sidebar.slider("Stripe width N", 1, min(h, w)//2, 10)
                inv = st.sidebar.checkbox("Invert diagonal", False)
                mask = np.fromfunction(lambda y,x: (((abs(x-y)%(2*N))<N)^inv), (h, w))
            elif pattern == "Horizontal Stripes":
                M = st.sidebar.slider("Stripe height M", 1, h//2, 10)
                inv = st.sidebar.checkbox("Invert horiz", False)
                mask = np.fromfunction(lambda y,x: (((y//M)%2)==0)^inv, (h, w))
            elif pattern == "Vertical Stripes":
                M = st.sidebar.slider("Stripe width M", 1, w//2, 10)
                inv = st.sidebar.checkbox("Invert vert", False)
                mask = np.fromfunction(lambda y,x: (((x//M)%2)==0)^inv, (h, w))
            elif pattern == "Random Mask":
                pct = st.sidebar.slider("Delete %", 0, 100, 50)
                seed = st.sidebar.number_input("Seed", 0)
                rng = np.random.default_rng(seed)
                mask = rng.random((h, w)) >= pct/100
            elif pattern == "Concentric Rings":
                R = st.sidebar.slider("Ring thickness", 1, min(h, w)//4, 10)
                inv = st.sidebar.checkbox("Invert rings", False)
                cy, cx = h/2, w/2
                mask = np.fromfunction(lambda y,x: ((np.floor(np.hypot(x-cx, y-cy)/R)%2)==0)^inv, (h, w))
            elif pattern == "Border Only":
                K = st.sidebar.slider("Border width K", 0, min(h, w)//2, 10)
                inv = st.sidebar.checkbox("Invert border", False)
                mask = np.fromfunction(lambda y,x: (((x<K)|(x>=w-K)|(y<K)|(y>=h-K)))^inv, (h, w))
            else:
                A = st.sidebar.slider("Block width A", 1, w, 10)
                B = st.sidebar.slider("Block height B", 1, h, 10)
                inv = st.sidebar.checkbox("Invert grid", False)
                mask = np.fromfunction(lambda y,x: (((x//A + y//B)%2)==0)^inv, (h, w))
            preview = base.copy()
            preview[...,3] *= mask.astype(np.uint8)

        if st.sidebar.button("Apply Effect"):
            st.session_state.undo_stack.append(st.session_state.work_arr.copy())
            st.session_state.work_arr = preview.copy()

        disp_img = Image.fromarray(preview)
        disp = disp_img.resize((disp_img.width * display_zoom, disp_img.height * display_zoom), Image.NEAREST)
        st.image(disp, caption="Image Preview", use_container_width=False)

        buf = BytesIO()
        Image.fromarray(preview).save(buf, format="PNG")
        buf.seek(0)
        st.sidebar.download_button("Download PNG", data=buf, file_name="output.png", mime="image/png")

# --- Mode 3: Rubik Mosaic Checker ---
else:
    st.header("🔍 Rubik Mosaic Checker")
    st.sidebar.header("Rubik Mosaic Checker Settings")
    invariant_file = st.sidebar.file_uploader("Upload invariant design", type=["png","jpg","jpeg"])
    target_file    = st.sidebar.file_uploader("Upload target design",    type=["png","jpg","jpeg"])
    if invariant_file and target_file:
        inv_img = Image.open(invariant_file).convert("RGB")
        tgt_img = Image.open(target_file).convert("RGB")
        if inv_img.size != tgt_img.size:
            tgt_img = tgt_img.resize(inv_img.size, Image.NEAREST)
        h, w = inv_img.size[1], inv_img.size[0]
        inv_arr = np.array(inv_img); tgt_arr = np.array(tgt_img)

        inv_map = np.empty((h, w), dtype=object)
        tgt_map = np.empty((h, w), dtype=object)
        for y in range(h):
            for x in range(w):
                inv_map[y,x] = nearest_rubik_color(tuple(inv_arr[y,x]))
                tgt_map[y,x] = nearest_rubik_color(tuple(tgt_arr[y
