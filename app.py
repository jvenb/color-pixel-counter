import streamlit as st
from PIL import Image
import numpy as np
from collections import Counter
from io import BytesIO

# -----------------------------
# Pillow-safe resampling constant
# -----------------------------
try:
    NEAREST = Image.Resampling.NEAREST  # Pillow >= 9.1+
except AttributeError:
    NEAREST = Image.NEAREST

# -----------------------------
# Limits (prevents Streamlit Cloud crashes)
# -----------------------------
MAX_PIXELS = 500 * 500  # adjust if you want, but this is safe for Streamlit Cloud

def guard_size(img: Image.Image, label: str):
    w, h = img.size
    if w * h > MAX_PIXELS:
        st.error(f"🚫 {label} image too large ({w}×{h}). Please upload ≤500×500.")
        st.stop()

# -----------------------------
# App config
# -----------------------------
st.set_page_config(layout="wide")
st.title("🎨 Pixel Toolkit")

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

def is_close(c1, c2):
    return all(abs(a - b) <= TOLERANCE for a, b in zip(c1, c2))

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
# MAIN (wrapped so you see the error instead of “Oh no.”)
# =========================================================
try:
    # =========================================================
    # Tool 1: Color Pixel Counter
    # =========================================================
    if tool == "Color Pixel Counter":
        st.header("🔢 Color Pixel Counter")
        uploaded = st.sidebar.file_uploader(
            "Upload a pixelated image (≤500×500)",
            type=["png", "jpg", "jpeg"],
            key="uploader_counter"
        )

        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            guard_size(img, "Counter")
            w, h = img.size

            arr = np.array(img)
            flat = arr.reshape(-1, 3)

            st.subheader("🎯 Counts & Values")
            total = 0
            matched = set()

            for label, fam in color_families.items():
                val = color_values[label]
                cnt = sum(any(is_close(px, v) for v in fam) for px in flat)
                total += cnt * val

                matched.update(
                    i for i, px in enumerate(flat)
                    if any(is_close(px, v) for v in fam)
                )

                sample = next(iter(fam))
                hexc = f"#{sample[0]:02x}{sample[1]:02x}{sample[2]:02x}"
                st.markdown(f"**{label}: {cnt} px × {val} = {cnt * val}**")

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
                    st.color_picker(
                        " ",
                        value=hexc,
                        disabled=True,
                        label_visibility="collapsed",
                        key=f"unmatched_picker_{idx}_{col}_{cnt}"
                    )

            st.image(
                img.resize((w * display_zoom, h * display_zoom), NEAREST),
                caption="Crisp Enlarged Image",
                use_container_width=False
            )

    # =========================================================
    # Tool 2: Pixel Deleter / Adder
    # =========================================================
    elif tool == "Pixel Deleter":
        st.header("🗑️ Pixel Deleter / ➕ Pixel Adder")

        uploaded = st.sidebar.file_uploader(
            "Upload a pixelated image (≤500×500)",
            type=["png", "jpg", "jpeg"],
            key="uploader_deleter"
        )

        if uploaded:
            img = Image.open(uploaded).convert("RGBA")
            guard_size(img, "Deleter")

            if "name" not in st.session_state or st.session_state.name != uploaded.name:
                st.session_state.name = uploaded.name
                st.session_state.orig = np.array(img)
                st.session_state.current = st.session_state.orig.copy()
                st.session_state.undo = []

            if st.sidebar.button("Reset", key="deleter_reset"):
                st.session_state.current = st.session_state.orig.copy()
                st.session_state.undo = []

            if st.sidebar.button("Undo", key="deleter_undo") and st.session_state.undo:
                st.session_state.current = st.session_state.undo.pop()

            chain = st.sidebar.checkbox("Chain effects", key="deleter_chain")
            base = st.session_state.current if chain else st.session_state.orig.copy()
            h, w = base.shape[:2]

            mode = st.sidebar.selectbox(
                "Pattern/Mode:",
                ["Original", "Checkerboard", "Alt Rows", "Alt Cols",
                 "Diagonal", "H Stripes", "V Stripes", "Random Mask",
                 "Rings", "Border", "Grid", "Pixel Adder"],
                key="deleter_mode"
            )

            if mode == "Pixel Adder":
                dup = st.sidebar.slider("Duplicates per side", 1, 10, 1, key="adder_dup")
                maintain = st.sidebar.checkbox("Maintain original width", key="adder_maintain")
                exp = np.repeat(base, 2 * dup + 1, axis=1)

                if not maintain:
                    preview = exp
                else:
                    seed = int(st.sidebar.number_input("Random seed", 0, key="adder_seed"))
                    rng = np.random.default_rng(seed)
                    dist = st.sidebar.selectbox("Distribution:", ["Uniform", "Column", "Sparse"], key="adder_dist")

                    if dist == "Uniform":
                        pct = st.sidebar.slider("% pixels offset", 0, 100, 50, key="adder_uniform_pct") / 100.0
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

                    else:
                        pct = st.sidebar.slider("% pixels sparsed", 0, 100, 20, key="adder_sparse_pct") / 100.0
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
                                if j > 0:
                                    preview[i, j - 1] = base[i, j]
                                if j < w - 1:
                                    preview[i, j + 1] = base[i, j]

            else:
                mask = np.ones((h, w), bool)

                if mode == "Checkerboard":
                    inv = st.sidebar.checkbox("Invert", key="m_cb_inv")
                    mask = np.fromfunction(lambda y, x: ((x + y) % 2 == int(inv)), (h, w)).astype(bool)
                elif mode == "Alt Rows":
                    inv = st.sidebar.checkbox("Invert", key="m_ar_inv")
                    mask = np.fromfunction(lambda y, x: (y % 2 == int(inv)), (h, w)).astype(bool)
                elif mode == "Alt Cols":
                    inv = st.sidebar.checkbox("Invert", key="m_ac_inv")
                    mask = np.fromfunction(lambda y, x: (x % 2 == int(inv)), (h, w)).astype(bool)
                elif mode == "Diagonal":
                    N = st.sidebar.slider("Width N", 1, max(1, min(h, w) // 2), 10, key="m_diag_n")
                    inv = st.sidebar.checkbox("Invert", key="m_diag_inv")
                    mask = np.fromfunction(lambda y, x: (((abs(x - y) % (2 * N)) < N) ^ inv), (h, w)).astype(bool)
                elif mode == "H Stripes":
                    M = st.sidebar.slider("Stripe height M", 1, max(1, h // 2), 10, key="m_hs_m")
                    inv = st.sidebar.checkbox("Invert horiz", False, key="m_hs_inv")
                    mask = np.fromfunction(lambda y, x: ((((y // M) % 2) == 0) ^ inv), (h, w)).astype(bool)
                elif mode == "V Stripes":
                    M = st.sidebar.slider("Stripe width M", 1, max(1, w // 2), 10, key="m_vs_m")
                    inv = st.sidebar.checkbox("Invert vert", False, key="m_vs_inv")
                    mask = np.fromfunction(lambda y, x: ((((x // M) % 2) == 0) ^ inv), (h, w)).astype(bool)
                elif mode == "Random Mask":
                    pct = st.sidebar.slider("Delete %", 0, 100, 50, key="m_rm_pct")
                    seed = int(st.sidebar.number_input("Seed", 0, key="m_rm_seed"))
                    rng = np.random.default_rng(seed)
                    mask = (rng.random((h, w)) >= pct / 100.0)
                elif mode == "Rings":
                    R = st.sidebar.slider("Ring thickness", 1, max(1, min(h, w) // 4), 10, key="m_ring_r")
                    inv = st.sidebar.checkbox("Invert rings", False, key="m_ring_inv")
                    cy, cx = h / 2.0, w / 2.0
                    mask = np.fromfunction(
                        lambda y, x: (((np.floor(np.hypot(x - cx, y - cy) / R) % 2) == 0) ^ inv),
                        (h, w)
                    ).astype(bool)
                elif mode == "Border":
                    K = st.sidebar.slider("Border width K", 0, min(h, w) // 2, 10, key="m_border_k")
                    inv = st.sidebar.checkbox("Invert border", False, key="m_border_inv")
                    mask = np.fromfunction(
                        lambda y, x: (((x < K) | (x >= w - K) | (y < K) | (y >= h - K)) ^ inv),
                        (h, w)
                    ).astype(bool)
                elif mode == "Grid":
                    A = st.sidebar.slider("Block W", 1, w, 10, key="m_grid_a")
                    B = st.sidebar.slider("Block H", 1, h, 10, key="m_grid_b")
                    inv = st.sidebar.checkbox("Invert grid", False, key="m_grid_inv")
                    mask = np.fromfunction(
                        lambda y, x: ((((x // A + y // B) % 2) == 0) ^ inv),
                        (h, w)
                    ).astype(bool)

                preview = base.copy()
                preview[..., 3] = (preview[..., 3].astype(np.uint16) * mask.astype(np.uint16)).astype(np.uint8)

            if st.sidebar.button("Apply", key="deleter_apply"):
                st.session_state.undo.append(st.session_state.current.copy())
                st.session_state.current = preview.copy()

            disp = Image.fromarray(preview)
            st.image(
                disp.resize((disp.width * display_zoom, disp.height * display_zoom), NEAREST),
                caption="Preview"
            )

            buf = BytesIO()
            Image.fromarray(preview).save(buf, format="PNG")
            buf.seek(0)
            st.sidebar.download_button("Download PNG", data=buf, file_name="output.png", mime="image/png", key="deleter_dl")

    # =========================================================
    # Tool 3: Rubik Mosaic Checker
    # =========================================================
    else:
        st.header("🔍 Rubik Mosaic Checker")

        inv_file = st.sidebar.file_uploader("Invariant (A)", type=["png", "jpg", "jpeg"], key="uploader_inv")
        tgt_file = st.sidebar.file_uploader("Target (B)", type=["png", "jpg", "jpeg"], key="uploader_tgt")

        if inv_file and tgt_file:
            inv = Image.open(inv_file).convert("RGB")
            tgt = Image.open(tgt_file).convert("RGB")
            guard_size(inv, "Invariant")
            guard_size(tgt, "Target")

            if inv.size != tgt.size:
                tgt = tgt.resize(inv.size, NEAREST)

            w, h = inv.size
            inv_arr = np.array(inv)
            tgt_arr = np.array(tgt)

            inv_map = np.empty((h, w), dtype=object)
            tgt_map = np.empty((h, w), dtype=object)

            for y in range(h):
                for x in range(w):
                    inv_map[y, x] = nearest_rubik_color(tuple(inv_arr[y, x]))
                    tgt_map[y, x] = nearest_rubik_color(tuple(tgt_arr[y, x]))

            tgt_map = np.fliplr(tgt_map)

            cy, cx = h // 2, w // 2
            if tgt_map[cy, cx] != opposites[inv_map[cy, cx]]:
                st.error(f"Invariant violated: center must be {opposites[inv_map[cy, cx]]}")
            else:
                st.success("✅ Opposite-face mosaic feasible!")

            dispA = np.uint8([[rubik_colors[inv_map[y, x]] for x in range(w)] for y in range(h)])
            dispB = np.uint8([[rubik_colors[tgt_map[y, x]] for x in range(w)] for y in range(h)])

            sideA = Image.fromarray(dispA).resize((w * display_zoom, h * display_zoom), NEAREST)
            sideB = Image.fromarray(dispB).resize((w * display_zoom, h * display_zoom), NEAREST)

            st.image(np.hstack([np.array(sideA), np.array(sideB)]), use_container_width=False)

            st.markdown("### Sticker counts on B")
            for c, cnt in Counter(tgt_map.flatten()).items():
                st.write(f"- {c}: {cnt}")

            buf = BytesIO()
            Image.fromarray(dispB).save(buf, format="PNG")
            buf.seek(0)
            st.sidebar.download_button("Download mapped PNG", data=buf, file_name="mapped_rubik.png", mime="image/png", key="rubik_dl")

except Exception as e:
    st.error("The app hit an error. The details are below (copy/paste this to me):")
    st.exception(e)
    st.stop()
