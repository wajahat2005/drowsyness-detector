import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ==========================================
# 1. PAGE CONFIG
# ==========================================
st.set_page_config(page_title="Driver Drowsiness Detector", page_icon="🚗")
st.title("🚗 AI Driver Drowsiness & Fatigue Detector")
st.write("Upload an image or use your camera to check the driver's alert status using an ensemble of two EfficientNetB3 models.")

# ==========================================
# 2. CACHE & LOAD MODELS
# ==========================================
@st.cache_resource
def load_models():
    # Load both models from the local models/ directory
    drowsy_model = tf.keras.models.load_model("models/drowsiness_detector_final.keras")
    fatigue_model = tf.keras.models.load_model("models/fatigue_detector_final.keras")
    return drowsy_model, fatigue_model

with st.spinner("Loading AI Models into memory..."):
    model_drowsy, model_fatigue = load_models()

# ==========================================
# 3. ENSEMBLE PREDICTION LOGIC
# ==========================================
def predict_driver_state(image, target_size=(300, 300)):
    # Resize the PIL image to match EfficientNetB3 input
    img = image.resize(target_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Add batch dimension

    # Get predictions
    pred_drowsy = model_drowsy.predict(img_array, verbose=0)[0]
    pred_fatigue = model_fatigue.predict(img_array, verbose=0)[0]

    # Extract Danger (Index 0) and Safe (Index 1) probabilities
    prob_drowsy = pred_drowsy[0]
    prob_fatigue = pred_fatigue[0]
    
    prob_awake = pred_drowsy[1]
    prob_non_fatigue = pred_fatigue[1]

    # Calculate combined ensemble averages
    ensemble_danger = (prob_drowsy + prob_fatigue) / 2.0
    ensemble_safe = (prob_awake + prob_non_fatigue) / 2.0

    # Determine final output
    if ensemble_danger > ensemble_safe:
        return "⚠️ DROWSY / FATIGUE DETECTED", ensemble_danger
    else:
        return "✅ AWAKE / NON-FATIGUE", ensemble_safe

# ==========================================
# 4. USER INTERFACE (UPDATED WITH CAMERA)
# ==========================================
# Create tabs for different input methods
tab1, tab2 = st.tabs(["📸 Take a Picture", "📁 Upload an Image"])

image_data = None

with tab1:
    camera_file = st.camera_input("Take a picture from your webcam/phone")
    if camera_file is not None:
        image_data = camera_file

with tab2:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_data = uploaded_file

# If the user has provided an image through EITHER method, process it
if image_data is not None:
    # Display the image
    image = Image.open(image_data).convert('RGB')
    
    # We only show the image again if they uploaded it, 
    # because st.camera_input already shows the captured photo automatically
    if image_data == uploaded_file:
        st.image(image, caption='Uploaded Image', use_column_width=True)

    # Add a button to run the prediction
    if st.button("Analyze Driver State"):
        with st.spinner("Analyzing facial features..."):
            state, confidence = predict_driver_state(image)
            
            st.markdown("---")
            if "DROWSY" in state:
                st.error(f"**Result:** {state}")
            else:
                st.success(f"**Result:** {state}")
                
            st.info(f"**Ensemble Confidence:** {confidence * 100:.2f}%")
