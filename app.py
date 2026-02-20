import os
os.environ["KERAS_BACKEND"] = "torch"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import streamlit as st
import keras
import numpy as np
from PIL import Image
import torch


st.set_page_config(
    page_title="EcoVision AI | Natural Scene Classifier",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="collapsed"
)


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Global Background */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    /* Clean Hero Header */
    .hero-container {
        padding: 3rem 1.5rem;
        background-color: #161b22;
        border-radius: 16px;
        text-align: left;
        margin-bottom: 2rem;
        border: 1px solid #30363d;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        color: #f0f6fc;
        margin: 0;
        line-height: 1.1;
    }
    
    .hero-subtitle {
        font-size: 1.25rem;
        font-weight: 300;
        color: #8b949e;
        margin-top: 0.5rem;
    }
    
    /* Clean Content Card */
    .clean-card {
        background-color: #161b22;
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #30363d;
        height: 100%;
    }
    
    /* Prediction Result Card - No Glow */
    .prediction-card {
        background-color: #0d1117;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        border: 2px solid #388bfd;
    }
    
    .result-label {
        color: #8b949e;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.1rem;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    .result-value {
        font-size: 3.5rem;
        font-weight: 700;
        color: #58a6ff;
        margin: 0.5rem 0;
    }
    
    /* Confidence System */
    .confidence-container {
        margin-top: 1.5rem;
    }

    .confidence-text {
        color: #c9d1d9;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }

    .confidence-bar-bg {
        height: 8px;
        background-color: #30363d;
        border-radius: 4px;
        overflow: hidden;
    }
    
    .confidence-bar-fill {
        height: 100%;
        background-color: #388bfd;
        border-radius: 4px;
        transition: width 0.6s ease-in-out;
    }

    /* Minimal Class Badge */
    .class-badge {
        display: inline-block;
        padding: 4px 12px;
        margin: 4px;
        background-color: #21262d;
        color: #8b949e;
        border: 1px solid #30363d;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 500;
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        .hero-title { font-size: 2.5rem; }
        .result-value { font-size: 2.5rem; }
    }

    /* Streamlit UI Tweak */
    .stFileUploader {
        border-radius: 12px;
    }
    
    .stSpinner > div {
        border-top-color: #58a6ff !important;
    }
    
    /* Sidebar removal artifacts */
    [data-testid="stSidebar"] { display: none; }
    [data-testid="stHeader"] { background: rgba(13, 17, 23, 0.8); }
</style>
""", unsafe_allow_html=True)

# --- MODEL CORE ---
@st.cache_resource
def load_trained_model():
    model_path = r"cnn_intel_image_classification_model.keras"
    if os.path.exists(model_path):
        try:
            with torch.no_grad():
                return keras.models.load_model(model_path)
        except Exception as e:
            st.error(f"Intelligence Core Initialization Error: {e}")
            return None
    return None

model = load_trained_model()
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# --- INTERFACE ---
st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">EcoVision AI</h1>
        <p class="hero-subtitle">High-fidelity Scene Classification & Satellite Feed Analysis System • Developed by Abhinav A</p>
    </div>
""", unsafe_allow_html=True)

main_col1, main_col2 = st.columns([1, 1], gap="medium")

with main_col1:
    st.markdown('<div class="clean-card">', unsafe_allow_html=True)
    st.markdown("### 📷 Image Input")
    uploaded_file = st.file_uploader("Upload satellite or natural landscape imagery", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, use_container_width=True)
    else:
        st.write("---")
        st.markdown('<p style="color:#8b949e; text-align:center;">Waiting for image input...</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with main_col2:
    if uploaded_file:
        if model:
            with st.spinner("Analyzing spectral data..."):
                # Inference
                img = img.convert('RGB')
                img_p = img.resize((150, 150))
                img_arr = np.array(img_p).astype('float32') / 255.0
                img_arr = np.expand_dims(img_arr, axis=0)
                
                with torch.no_grad():
                    preds = model.predict(img_arr, verbose=0)
                
                # Probs
                def softmax(x):
                    e = np.exp(x - np.max(x))
                    return e / e.sum()
                
                probs = softmax(preds[0]) if (np.sum(preds[0]) > 1.01 or np.sum(preds[0]) < 0.99) else preds[0]
                idx = np.argmax(probs)
                conf = probs[idx] * 100
                
                # --- CLEAN RESULT ---
                st.markdown(f"""
                    <div class="prediction-card">
                        <div class="result-label">System Classification</div>
                        <div class="result-value">{class_names[idx].upper()}</div>
                        <div class="confidence-container">
                            <div class="confidence-text">Confidence: {conf:.2f}%</div>
                            <div class="confidence-bar-bg">
                                <div class="confidence-bar-fill" style="width: {conf}%"></div>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("Detailed Probability Distribution", expanded=True):
                    for i, name in enumerate(class_names):
                        p = probs[i] * 100
                        col_a, col_b = st.columns([3, 1])
                        col_a.write(f"**{name.capitalize()}**")
                        col_b.write(f"{p:.1f}%")
                        st.progress(int(p))
        else:
            st.error("Model Error: Neural weights file missing.")
    else:
        st.markdown("""
            <div style="height: 100%; min-height: 300px; display: flex; align-items: center; justify-content: center; border: 1px dashed #30363d; border-radius: 12px; background-color: #0d1117;">
                <p style="color: #484f58;">Classification Report Generator</p>
            </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("<br><hr style='border-color: #30363d;'>", unsafe_allow_html=True)
st.markdown("#### Supported Semantic Landscapes")
for cls in class_names:
    st.markdown(f'<span class="class-badge">{cls.upper()}</span>', unsafe_allow_html=True)

st.markdown("""
    <div style="margin-top: 2rem; color: #484f58; font-size: 0.75rem; text-align: center;">
        EcoVision Integrated Environment • Developed by Abhinav A • Powered by Keras 3 (Torch) • v2.1.0
    </div>
""", unsafe_allow_html=True)
