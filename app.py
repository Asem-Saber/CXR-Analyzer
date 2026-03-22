import streamlit as st
import torch
import os
import tempfile
from PIL import Image
from model import UNET
from dataset import get_transform
from inference import inference

# --- Configuration ---
st.set_page_config(page_title="CXR Analyzer App", layout="wide")
BASE_DIR = r'COVID-19_Radiography_Dataset'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ID2LABEL = {0: 'Normal', 1: 'Lung_Opacity', 2: 'Viral Pneumonia', 3: 'COVID'}
IMG_SIZE = 256
NUM_CLASSES = 4
BEST_MODEL = 'best_model.pth'

TRANSFROMS = get_transform(IMG_SIZE)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Model Loading ---
# @st.cache_resource ensures the model only loads once and stays in memory
@st.cache_resource
def load_model():
    model = UNET(num_classes=NUM_CLASSES)
    # Make sure your weights file is named 'best_model.pth' in the same folder
    try:
        model.load_state_dict(torch.load(BEST_MODEL, map_location=DEVICE))
    except FileNotFoundError:
        st.error("Weights file 'best_model.pth' not found. Please add it to the directory.")
    
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# --- Main UI ---
st.title("🫁 Dual-Task Chest X-Ray Analyzer")
st.markdown("Upload a Chest X-Ray image. The model will classify the condition and segment the affected lung regions.")

# File Uploader
uploaded_file = st.file_uploader("Choose an X-Ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Streamlit keeps uploads in memory. We save it to a temporary file 
    # so your run_inference function can read it via file path.
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    with st.spinner('Analyzing X-Ray...'):
        # Pass the temp file path to your inference engine
        pred_class, confidence, pred_mask, overlay_img = inference(
            model=model, 
            img_path=tmp_path, 
            transforms=TRANSFROMS,
            device=DEVICE, 
            id2label=ID2LABEL
        )
        
    # Clean up the temporary file
    os.remove(tmp_path)

    # --- Results Display ---
    st.divider()
    
    # Metrics row
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Predicted Class", value=pred_class)
    with col2:
        st.metric(label="Confidence", value=f"{confidence * 100:.2f}%")
        
    st.divider()

    # Images row
    st.markdown("### Visual Results")
    img_col1, img_col2, img_col3 = st.columns(3)
    
    with img_col1:
        # Display the image the user uploaded
        st.image(uploaded_file, caption="1. Original X-Ray", use_container_width=True)
        
    with img_col2:
        # Display the 0-255 scaled predicted mask
        st.image(pred_mask, caption="2. Predicted Mask", use_container_width=True, clamp=True)
        
    with img_col3:
        # Display the blended red overlay
        st.image(overlay_img, caption="3. Mask Overlay", use_container_width=True)