import streamlit as st
import numpy as np
import tensorflow as tf

# ================================
# PAGE SETTINGS
# ================================

st.set_page_config(
    page_title="GWO-DNN MR Predictor",
    layout="centered"
)

# ================================
# CSS STYLING (ALIGNMENT FIX)
# ================================

st.markdown("""
<style>

/* Background */
.main {
    background-color: #f6f9ff;
}

/* Title Banner */
.title-box {
    background: linear-gradient(90deg, #0f2027, #203a43, #2c5364);
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 20px;
}

.title-text {
    font-size: 40px;
    font-weight: 800;
    color: white;
}

.subtitle {
    font-size: 18px;
    color: #e0e0e0;
}

/* Header Cells */
.table-header {
    background-color: #004aad;
    color: white;
    padding: 10px;
    border-radius: 6px;
    font-weight: 600;
    text-align: center;
}

/* Normal Cells */
.table-cell {
    background-color: white;
    padding: 8px;
    border-radius: 6px;
    text-align: center;
    border: 1px solid #e6e6e6;
}

/* Number input height fix */
div[data-baseweb="input"] {
    height: 40px;
}

/* Predict button */
.stButton > button {
    background: linear-gradient(90deg, #ff512f, #dd2476);
    color: white;
    font-size: 18px;
    font-weight: bold;
    border-radius: 10px;
    height: 50px;
}

/* Result box */
.result-box {
    background: #d4edda;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    font-size: 22px;
    font-weight: 700;
    color: #155724;
    margin-top: 15px;
}

</style>
""", unsafe_allow_html=True)

# ================================
# HEADER
# ================================

st.markdown("""
<div class="title-box">
    <div class="title-text">🧠 GWO–DNN Based MR Prediction System</div>
    <div class="subtitle">Hybrid Optimization + Deep Learning Based Resilient Modulus Estimator</div>
</div>
""", unsafe_allow_html=True)

# ================================
# LOAD MODEL
# ================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("GWO_DNN_MR_Model.h5", compile=False)

model = load_model()

# ================================
# NORMALIZATION VALUES
# ================================

X_min = np.array([5.82, 15.50, 0.0, 13.80, 12.30, 0.0])
X_max = np.array([31.08, 20.40, 41.40, 68.90, 41.54, 20.0])

MR_min = 3.0
MR_max = 217.0

# ================================
# TABLE HEADER
# ================================

st.markdown("### 🔢 Input Parameters Table")

h1, h2, h3, h4 = st.columns([3, 2, 3, 2])

with h1: st.markdown('<div class="table-header">Variable</div>', unsafe_allow_html=True)
with h2: st.markdown('<div class="table-header">Symbol</div>', unsafe_allow_html=True)
with h3: st.markdown('<div class="table-header">Input Value</div>', unsafe_allow_html=True)
with h4: st.markdown('<div class="table-header">Unit</div>', unsafe_allow_html=True)

# ================================
# FUNCTION FOR PERFECT ROW ALIGNMENT
# ================================

def table_row(name, symbol, unit, key, step=0.01):
    c1, c2, c3, c4 = st.columns([3,2,3,2])
    with c1:
        st.markdown(f'<div class="table-cell">{name}</div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="table-cell">{symbol}</div>', unsafe_allow_html=True)
    with c3:
        value = st.number_input("", key=key, step=step, label_visibility="collapsed")
    with c4:
        st.markdown(f'<div class="table-cell">{unit}</div>', unsafe_allow_html=True)

    return value

# ================================
# INPUT ROWS (ALIGNED)
# ================================

wPI = table_row("Weighted Plasticity Index", "wPI", "%", "wPI")
gd  = table_row("Dry Unit Weight", "γd", "kN/m³", "gd")
sc  = table_row("Confining Stress", "σ₃", "kPa", "sc")
sd  = table_row("Deviator Stress", "σd", "kPa", "sd")
w   = table_row("Moisture Content", "w", "%", "w")
NFT = table_row("Freeze–Thaw Cycles", "NFT", "Cycles", "NFT", step=1)

# ================================
# PREDICT BUTTON
# ================================

st.markdown(" ")

if st.button("🚀 PREDICT RESILIENT MODULUS (MR)", use_container_width=True):

    X_raw = np.array([[wPI, gd, sc, sd, w, NFT]])

    X_norm = (X_raw - X_min) / (X_max - X_min)

    MR_norm = model.predict(X_norm)

    MR_raw = MR_norm * (MR_max - MR_min) + MR_min

    st.markdown(
        f'<div class="result-box">Predicted MR = {MR_raw[0][0]:.2f} MPa</div>',
        unsafe_allow_html=True
    )

# ================================
# FOOTER
# ================================

st.markdown("---")
st.caption("Hybrid GWO–DNN Model | Smart Geotechnical Prediction Interface")
