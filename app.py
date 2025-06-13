import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
from collections import Counter
from io import BytesIO

st.set_page_config(layout="wide")
st.title("🎨 Pixel Toolkit")

# Sidebar controls
tool = st.sidebar.radio("Select tool:", ["Color Pixel Counter", "Pixel Deleter"])
# Common display zoom
display_zoom = st.sidebar.slider("Display Zoom Multiplier", 1, 50, 20)

if tool == "Color Pixel Counter":
    st.sidebar.header("🔢 Color Pixel Counter Settings")
    uploaded = st.sidebar.file_uploader(
        "Upload a pixelated image (≤500×500)", type=["png", "jpg", "jpeg"]
    )
    # Color families
    color_families = {
        "White": {(252,255,251), (255,255,255)},
        "Yellow": {(242,230,0)},
        "Orange": {(238,102,7), (237,100,3)},
        "Red": {(192,0,37), (190,0,35)},
        "Blue": {(0,61,174), (0,61,167)}
    }
    color_values = {"White":1, "Yellow":2, "Orange":3, "Red":4, "Blue":5}
    TOL, MAX_PX = 10, 500*500
    def is_close(a,b): return all(abs(x-y)<=TOL for x,y in zip(a,b))

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        w,h = img.size
        if w*h>MAX_PX:
            st.error("Image too large.")
        else:
            arr = np.array(img); flat = arr.reshape(-1,3)
            st.header("🎯 Color Counts & Values")
            total, matched = 0, set()
            for i,(label,fam) in enumerate(color_families.items(),1):
                val = color_values[label]; cnt=0
                for idx,px in enumerate(flat):
                    if any(is_close(px,variant) for variant in fam):
                        cnt+=1; matched.add(idx)
                subtotal=cnt*val; total+=subtotal
                sample=next(iter(fam)); hexc=f"#{sample[0]:02x}{sample[1]:02x}{sample[2]:02x}"
                st.markdown(f"**{i}. {label}: {cnt} px × {val} = {subtotal}**")
                st.color_picker(label, value=hexc, key=i, disabled=True, label_visibility="collapsed")
            st.subheader(f"🧮 Total Value: {total}")
            unmatched = [tuple(flat[i]) for i in range(flat.shape[0]) if i not in matched]
            if not unmatched:
                st.success("All pixels matched.")
            else:
                st.warning(f"{len(unmatched)} unmatched pixels.")
                top=Counter(unmatched).most_common(10)
                st.markdown("### Unmatched Top 10 Colors:")
                for col,cnt in top:
                    hexc=f"#{col[0]:02x}{col[1]:02x}{col[2]:02x}"
                    st.markdown(f"- {col}: {cnt} px")
                    st.color_picker("",value=hexc,disabled=True,label_visibility="collapsed")
            # Display image
            disp = img.resize((w*display_zoom, h*display_zoom), Image.NEAREST)
            st.image(disp, caption="Crisp Enlarged Image", use_column_width=False)

elif tool == "Pixel Deleter":
    st.sidebar.header("🗑️ Pixel Deleter Settings")
    uploaded = st.sidebar.file_uploader("Upload a pixelated image", type=["png","jpg","jpeg"])
    chain = st.sidebar.checkbox("Chain effects", False)
    reset = st.sidebar.button("Reset Effects")
    zoom = display_zoom
    if uploaded:
        img = Image.open(uploaded).convert("RGBA")
        if 'orig_arr' not in st.session_state or reset or st.session_state.upload_name!=uploaded.name:
            st.session_state.upload_name = uploaded.name
            st.session_state.orig_arr = np.array(img)
            st.session_state.work_arr = st.session_state.orig_arr.copy()
        arr_src = st.session_state.work_arr if chain else st.session_state.orig_arr.copy()
        h,w = arr_src.shape[:2]

        # Pattern selection
        pattern = st.sidebar.selectbox("Select pattern:", [
            "Checkerboard","Alternate Rows","Alternate Columns",
            "Diagonal Stripes","Horizontal Stripes","Vertical Stripes",
            "Random Mask","Concentric Rings","Border Only","Custom Grid"
        ])
        # Mask generation
        if pattern=="Checkerboard":
            inv=st.sidebar.checkbox("Invert checkerboard", False)
            mask=np.fromfunction(lambda y,x: ((x+y)%2== (1 if inv else 0)),(h,w))
        elif pattern=="Alternate Rows":
            inv=st.sidebar.checkbox("Invert rows", False)
            mask=np.fromfunction(lambda y,x: (y%2== (1 if inv else 0)),(h,w))
        elif pattern=="Alternate Columns":
            inv=st.sidebar.checkbox("Invert cols", False)
            mask=np.fromfunction(lambda y,x: (x%2== (1 if inv else 0)),(h,w))
        elif pattern=="Diagonal Stripes":
            N=st.sidebar.slider("Stripe width N",1,min(h,w)//2,10)
            inv=st.sidebar.checkbox("Invert diagonal", False)
            mask=np.fromfunction(lambda y,x: ((abs(x-y)%(2*N))<N) ^ inv,(h,w))
        elif pattern=="Horizontal Stripes":
            M=st.sidebar.slider("Stripe height M",1,h//2,10)
            inv=st.sidebar.checkbox("Invert horiz", False)
            mask=np.fromfunction(lambda y,x: (((y//M)%2)==0) ^ inv,(h,w))
        elif pattern=="Vertical Stripes":
            M=st.sidebar.slider("Stripe width M",1,w//2,10)
            inv=st.sidebar.checkbox("Invert vert", False)
            mask=np.fromfunction(lambda y,x: (((x//M)%2)==0) ^ inv,(h,w))
        elif pattern=="Random Mask":
            pct=st.sidebar.slider("Delete %",0,100,50)
            seed=st.sidebar.number_input("Seed",0)
            rng=np.random.default_rng(seed)
            mask=rng.random((h,w)) >= pct/100
        elif pattern=="Concentric Rings":
            R=st.sidebar.slider("Ring thickness",1,min(h,w)//4,10)
            inv=st.sidebar.checkbox("Invert rings", False)
            cy, cx = h/2, w/2
            mask = np.fromfunction(
                lambda y,x: ((np.floor(np.hypot(x-cx,y-cy)/R)%2)==0) ^ inv,
                (h,w)
            )
        elif pattern=="Border Only":
            K=st.sidebar.slider("Border width K",0,min(h,w)//2,10)
            inv=st.sidebar.checkbox("Invert border", False)
            mask=np.fromfunction(
                lambda y,x: (((x<K)|(x>=w-K)|(y<K)|(y>=h-K))) ^ inv,(h,w)
            )
        else:  # Custom Grid
            A=st.sidebar.slider("Block width A",1,w,10)
            B=st.sidebar.slider("Block height B",1,h,10)
            inv=st.sidebar.checkbox("Invert grid", False)
            mask=np.fromfunction(
                lambda y,x: (((x//A + y//B)%2)==0) ^ inv,(h,w)
            )
        # Apply mask
        m=mask.astype(np.uint8)
        arr_src[...,3] *= m
        if chain: st.session_state.work_arr = arr_src.copy()
        # Display result
        result = Image.fromarray(arr_src)
        disp = result.resize((w*zoom, h*zoom), Image.NEAREST)
        st.image(disp, caption="Processed Image", use_column_width=False)
        buf=BytesIO(); result.save(buf,format="PNG"); buf.seek(0)
        st.sidebar.download_button("Download PNG", data=buf, file_name="output.png", mime="image/png")
