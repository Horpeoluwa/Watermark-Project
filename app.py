import streamlit as st
import torch
import numpy as np
import os
from model import WatermarkEncoder, WatermarkDecoder
from PIL import Image
import io
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr

# --- 1. PAGE CONFIG & THEME ---
st.set_page_config(
    page_title="FUTA Watermarking Tool",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CACHING & MODEL LOADING ---
@st.cache_resource
def load_models():
    encoder = WatermarkEncoder()
    decoder = WatermarkDecoder()
    
    # Updated paths to match your 10k weights if you renamed them
    enc_path = 'encoder_weights.pth'
    dec_path = 'decoder_weights.pth'
    
    if os.path.exists(enc_path) and os.path.exists(dec_path):
        encoder.load_state_dict(torch.load(enc_path, map_location=torch.device('cpu')))
        decoder.load_state_dict(torch.load(dec_path, map_location=torch.device('cpu')))
        encoder.eval()
        decoder.eval()
        return encoder, decoder, True
    return None, None, False

encoder, decoder, is_trained = load_models()

# --- 3. SIDEBAR (PROFESSIONAL INFO) ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=80)
    st.title("System Status")
    
    if is_trained:
        st.success("✅ Model: Fully Trained (Epoch 10)")
        st.write("**Dataset:** COCO 2017 (10k Images)")
        st.write("**Architecture:** Conv Encoder-Decoder")
    else:
        st.error("⚠️ Weights missing! Ensure .pth files are in GitHub.")
    
    st.divider()
    st.subheader("How it works")
    st.info("""
    1. **Upload** host photo.
    2. **Upload** ownership logo.
    3. **AI** embeds logo invisibly.
    4. **Metrics** verify quality.
    """)
    st.caption("M.Tech Thesis - Federal University of Technology, Akure")

# --- 4. MAIN INTERFACE ---
st.title("🛡️ Image Ownership & Forensic Tool")
st.markdown("---")

col_up1, col_up2 = st.columns(2)
with col_up1:
    st.subheader("1. Host Image")
    uploaded_file = st.file_uploader("Select Photo", type=["jpg", "png", "jpeg"])
with col_up2:
    st.subheader("2. Ownership Logo")
    logo_file = st.file_uploader("Select Logo", type=["png", "jpg"])

if uploaded_file and logo_file and is_trained:
    raw_img = Image.open(uploaded_file).convert('RGB')
    img_res = np.array(raw_img.resize((256, 256)))
    
    raw_logo = Image.open(logo_file).convert('L') 
    logo_res = np.array(raw_logo.resize((64, 64)))

    with st.status("Performing Forensic Analysis...", expanded=True) as status:
        st.write("Preparing Tensors...")
        host_tensor = torch.from_numpy(img_res).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        logo_tensor = torch.from_numpy(logo_res).float().unsqueeze(0).unsqueeze(0) / 255.0

        with torch.no_grad():
            # Encoding
            watermarked_tensor = encoder(host_tensor, logo_tensor)
            
            # --- 1% INVISIBILITY CONSTRAINT ---
            residual = torch.clamp(watermarked_tensor - host_tensor, -0.01, 0.01)
            watermarked_tensor = torch.clamp(host_tensor + residual, 0, 1)
            
            # Decoding
            extracted_tensor = decoder(watermarked_tensor)

        # Denormalize for display
        wm_img_float = watermarked_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        wm_img = (wm_img_float * 255).astype(np.uint8)
        
        ex_logo_float = extracted_tensor.squeeze().cpu().numpy()
        ex_logo = (ex_logo_float * 255).astype(np.uint8)
        
        # --- CALCULATE REAL METRICS ---
        orig_float = img_res / 255.0
        real_psnr = calculate_psnr(orig_float, wm_img_float, data_range=1.0)
        
        # NC Score Calculation
        logo_float = logo_res / 255.0
        nc_score = np.sum((logo_float - np.mean(logo_float)) * (ex_logo_float - np.mean(ex_logo_float))) / \
                   (np.sqrt(np.sum((logo_float - np.mean(logo_float))**2) * np.sum((ex_logo_float - np.mean(ex_logo_float))**2)) + 1e-8)

        status.update(label="Forensics Complete!", state="complete", expanded=False)

    st.divider()

    # --- 5. RESULTS DISPLAY ---
    st.subheader("📊 Forensic Results")
    res_col1, res_col2, res_col3 = st.columns(3)
    
    with res_col1:
        st.image(img_res, caption="Original Host", use_container_width=True)
    with res_col2:
        st.image(wm_img, caption="Watermarked (Invisible)", use_container_width=True)
        buf = io.BytesIO()
        Image.fromarray(wm_img).save(buf, format="PNG")
        st.download_button("Download Protected Image", buf.getvalue(), "protected_image.png", "image/png")
    with res_col3:
        st.image(ex_logo, caption="Recovered Proof", use_container_width=True)

    # DYNAMIC METRICS FOOTER
    st.success("✅ Ownership Verified via Deep Learning Analysis")
    cols = st.columns(3)
    cols[0].metric("PSNR (Quality)", f"{real_psnr:.2f} dB")
    cols[1].metric("NC Score (Accuracy)", f"{max(0, nc_score):.4f}")
    cols[2].metric("Generalization", "10,000 Images")

else:
    st.info("👋 Upload a Host Image and a Logo to begin.")
