import streamlit as st
import numpy as np
from joblib import load
from feature_extraction import extract_all_features
import tempfile, os

# --- Page Setup ---
st.set_page_config(
    page_title="AI Speech Depression Test",
    page_icon="🧠",
    layout="wide"
)

# --- CSS Styling (High Contrast, Clean Rainbow) ---
st.markdown("""
<style>
body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(270deg, violet, indigo, blue, green, yellow, orange, red);
    background-size: 1400% 1400%;
    animation: vibgyorGradient 26s ease infinite;
    color: #fff !important;
    margin: 0;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
@keyframes vibgyorGradient {
    0%{background-position:0% 50%}
    50%{background-position:100% 50%}
    100%{background-position:0% 50%}
}
.main .block-container {
    padding: 0 1rem 1rem 1rem;
    max-width: 100% !important;
    width: 100% !important;
}
.header-container {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1.6rem;
    padding-left: 1rem;
}
.logo-img {
    width: 110px;
    height: 110px;
    border-radius: 20px;
    background: #fff9;
    box-shadow: 0 6px 28px rgba(70,0,200,0.2);
    transition: transform 0.25s;
}
.logo-img:hover {
    transform: scale(1.07) rotate(-4deg);
}
.title-card {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(90deg, #fff 30%, #FF3855 60%, #02B2FF 90%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    user-select: none;
    margin: 0;
    line-height: 1.08;
    letter-spacing: 1px;
}
.subtitle {
    font-size: 1.15rem;
    color: #111;
    background: rgba(255, 255, 255, 0.72);
    border-radius: 8px;
    padding: 0.6rem 1.2rem;
    width: fit-content;
    margin-left: 130px;
    font-weight: 530;
    margin-bottom: 1.2rem;
    box-shadow: 0 2px 10px rgba(0,0,0,.10);
}
.fullscreen-container {
    background: rgba(255,255,255,0.16);
    border-radius: 18px;
    padding: 2rem;
    max-width: 650px;
    margin: 2.5rem auto 1rem auto;
    box-shadow: 0 8px 32px rgba(0,0,0,0.17);
}
.upload-card {
    border-radius: 13px;
    background: #fff;
    padding: 18px;
    box-shadow: 0 0 18px #00000019;
    border: none;
    color: #232a2f;
    text-align: center;
    font-size: 1.12rem;
    font-weight: 600;
    cursor: pointer;
    margin-bottom: 24px;
    transition: box-shadow 0.2s, background 0.2s;
}
.upload-card:hover {
    background: #f3f3f3;
    box-shadow: 0 0 32px #ff980080;
}
.prediction-card {
    margin-top: 26px;
    padding: 24px 0;
    border-radius: 13px;
    font-size: 1.25rem;
    font-weight: 800;
    text-align: center;
    letter-spacing: 0.7px;
    background: #fff;
    box-shadow: 0 1.5px 21px #8889;
    color: #0e1a21;
}
.pred-nondepressed {
    border: 3px solid #2196f3;
    color: #2196f3;
    background: #ecfdff;
}
.pred-depressed {
    border: 3px solid #ff3855;
    color: #ff3855;
    background: #fff5f5;
}
.model-confidence {
    font-size: 1.02rem;
    color: #222;
    margin-top: 7px;
    font-weight: 500;
}
footer {
    margin-top: 2.5rem;
    text-align: center;
    color: #eee;
    font-size: 0.95rem;
    letter-spacing: 1px;
}
audio {margin-bottom: 0.7rem;}
::-webkit-scrollbar {width: 10px;}
::-webkit-scrollbar-thumb {background: #333a; border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

# HEADER
st.markdown("""
<div class="header-container">
    <img class="logo-img" src="https://cdn-icons-png.flaticon.com/512/2996/2996797.png" alt="Logo">
    <div>
        <h1 class="title-card">Speech Depression Detection</h1>
        <div class="subtitle">Upload your voice sample as a WAV file. AI will analyze and instantly show if the sample is classified as <strong>depressed</strong> or <strong>non-depressed</strong>.</div>
    </div>
</div>
""", unsafe_allow_html=True)

# MAIN WRAPPER
st.markdown('<div class="fullscreen-container">', unsafe_allow_html=True)

# UPLOAD
st.markdown('<div class="upload-card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Upload your WAV file (at least 3 seconds)", 
    type="wav", label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

# PREDICTION
if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_audio.write(uploaded_file.read())
    temp_audio.close()

    clf = load("models/classifier.joblib")
    scaler = load("models/scaler.joblib")
    top_indices = np.load("models/top_feature_indices.npy")

    try:
        with st.spinner("✨ Analyzing your audio..."):
            features = extract_all_features(temp_audio.name)
            features_sel = features[top_indices]
            features_scaled = scaler.transform([features_sel])

            pred = clf.predict(features_scaled)[0]
            proba = clf.predict_proba(features_scaled)[0] if hasattr(clf, "predict_proba") else None

            klass = "pred-depressed" if pred == 1 else "pred-nondepressed"
            label = "😔 Depressed" if pred == 1 else "😊 Non-Depressed"

            st.markdown(f'<div class="prediction-card {klass}">{label}</div>', unsafe_allow_html=True)

            if proba is not None:
                conf = proba[1]*100 if pred == 1 else proba[0]*100
                st.markdown(f'<div class="model-confidence">Model confidence: {conf:.1f}%</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
    finally:
        os.remove(temp_audio.name)

# FOOTER
st.markdown(
    "<footer>© 2025 Your Lab or Company — Powered by Streamlit</footer>",
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)
