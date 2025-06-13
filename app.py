import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
from collections import Counter
from io import BytesIO

st.title("🎨 Pixel Toolkit")

# Choose between two modes
tools = ["Color Pixel Counter", "Pixel Deleter"]
mode = st.sidebar.radio("Select tool:", tools)

if mode == "Color Pixel Counter":
    st.header("🔢 Color Pixel Counter")
    uploaded_file = st.file_uploader(
        "Upload a pixelated image with 5 known colors", type=["png", "jpg", "jpeg"]
    )

    # Define color families and values
    color_families = {
        "White": {(252, 255, 251), (255, 255, 255)},
        "Yellow": {(242, 230, 0)},
        "Orange": {(238, 102, 7), (237, 100, 3)},
        "Red": {(192, 0, 37), (190, 0, 35)},
        "Blue": {(0, 61, 174), (0, 61, 167)}
    }
    color_values = {"White": 1, "Yellow": 2, "Orange": 3, "Red": 4, "Blue": 5}

    TOLERANCE = 10
    MAX_PIXELS = 500 * 500

    def is_close(c1, c2, tol=TOLERANCE):
        return all(abs(a - b) <= tol for a, b in zip(c1, c2))

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        if img.width * img.height > MAX_PIXELS:
            st.error("🚫 Image too large. Please upload an image smaller than 500×500 pixels.")
        else:
            zoom = st.slider("🔍 Image Zoom (Pixel Size Multiplier)", 1, 50, 20)
            arr = np.array(img)
            flat = arr.reshape(-1, 3)

            st.subheader("🎯 Mapped Color Counts and Values:")
            total = 0
            matched = set()
            for i, (label, fam) in enumerate(color_families.items(), 1):
                val = color_values[label]
                cnt = 0
                for idx, px in enumerate(flat):
                    if any(is_close(px, variant) for variant in fam):
                        cnt += 1
                        matched.add(idx)
                subtotal = cnt * val
                total += subtotal
                sample = next(iter(fam))
                hexc = '#%02x%02x%02x' % sample
                st.markdown(f"**{i}. {label} → Pixels: {cnt} × Value: {val} = {subtotal}**")
                st.color_picker(f"Preview {label}", value=hexc, key=i, disabled=True, label_visibility="collapsed")

            st.subheader(f"🧮 Total Image Value: {total}")
            total_px = flat.shape[0]
            unmatched_idxs = [i for i in range(total_px) if i not in matched]
            unmatched = [tuple(px) for i, px in enumerate(flat) if i in unmatched_idxs]

            if not unmatched:
                st.success("✅ All pixels matched.")
                st.image(img.resize((img.width*zoom, img.height*zoom), Image.NEAREST), caption="Crisp Enlarged Image")
            else:
                st.warning(f"⚠️ {len(unmatched)} pixel(s) not matched.")
                top = Counter(unmatched).most_common(10)
                st.markdown("### ❌ Unmatched Colors (Top 10):")
                for col, cnt in top:
                    hexc = '#%02x%02x%02x' % col
                    st.markdown(f"- {col} → {cnt} px")
                    st.color_picker("", value=hexc, disabled=True, key=f"unm-{hexc}", label_visibility="collapsed")
                gray = img.convert("L").convert("RGB")
                draw = ImageDraw.Draw(gray)
                w, h = img.size
                for idx in unmatched_idxs:
                    x, y = idx % w, idx // w
                    draw.point((x, y), fill=img.getpixel((x, y)))
                st.image(gray.resize((w*zoom, h*zoom), Image.NEAREST), caption="Unmatched Highlighted")

elif mode == "Pixel Deleter":
    st.header("🗑️ Pixel Deleter")
    uploaded = st.file_uploader("Upload a pixelated image", type=["png","jpg","jpeg"])
    if uploaded:
        img = Image.open(uploaded).convert("RGBA")
        arr = np.array(img)
        h, w = arr.shape[:2]

        # Deletion patterns
        pattern = st.selectbox(
            "Select deletion pattern:",
            ["Checkerboard", "Alternate Rows", "Alternate Columns",
             "Diagonal Stripes", "Horizontal Stripes", "Vertical Stripes",
             "Random Mask", "Circular Mask", "Border Only", "Custom Grid"]
        )

        if pattern == "Checkerboard":
            inv = st.checkbox("Delete top-left?", value=False)
            mask = np.fromfunction(lambda y, x: ((x+y)%2==1) if inv else ((x+y)%2==0), (h, w))

        elif pattern == "Alternate Rows":
            inv = st.checkbox("Delete first row?", value=False)
            mask = np.fromfunction(lambda y, x: (y%2==1) if inv else (y%2==0), (h, w))

        elif pattern == "Alternate Columns":
            inv = st.checkbox("Delete first column?", value=False)
            mask = np.fromfunction(lambda y, x: (x%2==1) if inv else (x%2==0), (h, w))

        elif pattern == "Diagonal Stripes":
            N = st.slider("Stripe width N:", 1, min(h, w)//2, 10)
            inv = st.checkbox("Invert diagonal?", value=False)
            mask = np.fromfunction(
                lambda y, x: (((x-y) % (2*N) >= N) if inv else ((x-y) % (2*N) < N)),
                (h, w)
            )

        elif pattern == "Horizontal Stripes":
            M = st.slider("Stripe height M:", 1, h//2, 10)
            inv = st.checkbox("Invert horizontal?", value=False)
            mask = np.fromfunction(
                lambda y, x: (((y//M)%2==1) if inv else ((y//M)%2==0)),
                (h, w)
            )

        elif pattern == "Vertical Stripes":
            M = st.slider("Stripe width M:", 1, w//2, 10)
            inv = st.checkbox("Invert vertical?", value=False)
            mask = np.fromfunction(
                lambda y, x: (((x//M)%2==1) if inv else ((x//M)%2==0)),
                (h, w)
            )

        elif pattern == "Random Mask":
            pct = st.slider("% to delete:", 0, 100, 50)
            seed = st.number_input("Random seed:", value=0)
            rng = np.random.default_rng(seed)
            mask = rng.random((h, w)) >= (pct/100)

        elif pattern == "Circular Mask":
            maxr = min(h, w) / 2
            r = st.slider("Radius:", 0.0, maxr, maxr/2)
            inv = st.checkbox("Delete inside?", value=False)
            cy, cx = h/2, w/2
            mask = np.fromfunction(
                lambda y, x: (((x-cx)**2 + (y-cy)**2) > r**2) if not inv else (((x-cx)**2 + (y-cy)**2) <= r**2),
                (h, w)
            )

        elif pattern == "Border Only":
            K = st.slider("Border width K:", 0, min(h, w)//2, 10)
            inv = st.checkbox("Delete border?", value=False)
            mask = np.fromfunction(
                lambda y, x: ((x>=K)&(x< w-K)&(y>=K)&(y< h-K)) if not inv else ~((x>=K)&(x< w-K)&(y>=K)&(y< h-K)),
                (h, w)
            )

        elif pattern == "Custom Grid":
            A = st.slider("Block width A:", 1, w, 10)
            B = st.slider("Block height B:", 1, h, 10)
            inv = st.checkbox("Invert grid?", value=False)
            mask = np.fromfunction(
                lambda y, x: (((x//A + y//B)%2==0) if not inv else ((x//A + y//B)%2==1)),
                (h, w)
            )

        # Apply mask on alpha channel
        m = mask.astype(np.uint8)
        arr[..., 3] = arr[..., 3] * m

        result = Image.fromarray(arr)
        st.image(result, caption="Processed Image", use_container_width=False)
        buf = BytesIO()
        result.save(buf, format="PNG")
        buf.seek(0)
        st.download_button(
            "⬇️ Download PNG with transparency",
            data=buf,
            file_name="pixel_deleted.png",
            mime="image/png"
        )
