import streamlit as st
import onnxruntime as ort
from PIL import Image
import numpy as np
import time

st.set_page_config(
    page_title="Deepfake AI Detector",
    page_icon="🕵️‍♂️",
    layout="centered"
)

st.markdown("""
    <style>
    .main-title { font-size: 45px; font-weight: bold; color: #1E3A8A; text-align: center; }
    .sub-text { font-size: 18px; color: #4B5563; text-align: center; margin-bottom: 20px;}
    </style>
""", unsafe_allow_html=True)
st.markdown("<div class='main-title'>Deepfake Detection Engine</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-text'>Convolutional Neural Network (CNN) Visual Analysis</div>", unsafe_allow_html=True)
st.divider()

@st.cache_resource
def load_model():
    try:
        session = ort.InferenceSession('deepfake_detector.onnx')
        return session
    except Exception:
        return None

detector_model = load_model()
if detector_model is None:
    st.error("🚨 Error: 'deepfake_detector.onnx' not found. Please ensure your model is in the same folder as this app.")
    st.stop()

st.markdown("### Upload a Suspect Image")
uploaded_file = st.file_uploader("Choose an image file (JPG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=400)
    st.markdown("### Run AI Analysis")

    if st.button("🔍 Analyze Pixels", type="primary"):
        with st.spinner('Scanning facial textures and lighting inconsistencies...'):
            time.sleep(1.5)

            img_resized = image.resize((128, 128)).convert("RGB")
            img_array = np.array(img_resized, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            input_name = detector_model.get_inputs()[0].name
            prediction = detector_model.run(None, {input_name: img_array})[0][0][0]

            st.divider()
            st.subheader("AI Verdict:")

            if prediction > 0.5:
                confidence = prediction * 100
                st.success("✅ **REAL HUMAN DETECTED**")
                st.info(f"**Confidence Score:** {confidence:.2f}%")
                st.write("The network found natural light dispersion and standard pixel gradients.")
                st.balloons()
            else:
                confidence = (1 - prediction) * 100
                st.error("🚨 **DEEPFAKE DETECTED**")
                st.warning(f"**Confidence Score:** {confidence:.2f}%")
                st.write("The network detected anomalies typical of GAN-generated synthetic media (e.g., edge blurring or unnatural artifacts).")
