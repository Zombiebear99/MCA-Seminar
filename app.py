import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import time

# 1. PAGE SETUP
st.set_page_config(
    page_title="Deepfake AI Detector", 
    page_icon="🕵️‍♂️", 
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .main-title { font-size: 45px; font-weight: bold; color: #1E3A8A; text-align: center; }
    .sub-text { font-size: 18px; color: #4B5563; text-align: center; margin-bottom: 20px;}
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>Deepfake Detection Engine</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-text'>Convolutional Neural Network (CNN) Visual Analysis</div>", unsafe_allow_html=True)
st.divider()

# 2. LOAD THE AI MODEL
# The @st.cache_resource decorator ensures the heavy AI model only loads once
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('deepfake_detector.h5')
        return model
    except OSError:
        return None

detector_model = load_model()

if detector_model is None:
    st.error("🚨 Error: 'deepfake_detector.h5' not found. Please ensure your trained model is in the same folder as this app.")
    st.stop()

# 3. UI FOR UPLOADING IMAGES
st.markdown("### Step 1: Upload a Suspect Image")
uploaded_file = st.file_uploader("Choose an image file (JPG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    
    st.image(image, caption="Uploaded Image", width=400)
    
    st.markdown("### Step 2: Run AI Analysis")
    
    # The analyze button
    if st.button("🔍 Analyze Pixels", type="primary"):
        
        # Add a dramatic spinner for the audience
        with st.spinner('Scanning facial textures and lighting inconsistencies...'):
            time.sleep(1.5) # Slight artificial pause for dramatic effect in a presentation
            
            # Prepare the image exactly how the CNN expects it (128x128 pixels, scaled)
            img_resized = image.resize((128, 128))
            img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            
            # The AI makes its prediction
            prediction = detector_model.predict(img_array)[0][0]
            
            # 4. DISPLAY THE RESULTS
            st.divider()
            st.subheader("AI Verdict:")
            
            # In our model: closer to 1 is Real, closer to 0 is Fake
            if prediction > 0.5:
                confidence = prediction * 100
                st.success(f"✅ **REAL HUMAN DETECTED**")
                st.info(f"**Confidence Score:** {confidence:.2f}%")
                st.write("The network found natural light dispersion and standard pixel gradients.")
                st.balloons()
            else:
                confidence = (1 - prediction) * 100
                st.error(f"🚨 **DEEPFAKE DETECTED**")
                st.warning(f"**Confidence Score:** {confidence:.2f}%")
                st.write("The network detected anomalies typical of GAN-generated synthetic media (e.g., edge blurring or unnatural artifacts).")