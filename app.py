import streamlit as st
import numpy as np
import joblib
import logging
import pandas as pd
import plotly.express as px
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Prediksi Resiko Kesehatan", layout = "wide")

logging.basicConfig(filename = 'app_system.log', level = logging.INFO)

@st.cache_resource
def load_artifact():
    model = load_model("model_resiko_kesehatan.keras")
    scaler = joblib.load("scaler.pkl")
    labeled = joblib.load("label.pkl")

    return model, scaler, labeled

model, scaler, labeled = load_artifact()

st.sidebar.header("Input data Pasien")
with st.sidebar.form("input_form"):
    umur = st.number_input("Umur", min_value = 5, max_value = 90)
    tekanan = st.number_input("Tekanan Darah", min_value = 5, max_value = 150)
    kolesterol = st.number_input("Kolestrol", min_value = 0, max_value = 200)
    bmi = st.number_input("BMI", min_value = 0, max_value = 200)
    gula_darah = st.number_input("Gula Darah", min_value = 0, max_value = 200)
    aktivitas = st.number_input("Aktivitas Fisik(Jam/Minggu)", min_value = 0, max_value = 200)
    perokok = st.selectbox("Perokok(Ya/Tidak)", options = ['Ya', 'Tidak'])
    riwayat = st.selectbox("Riwayat Penyakit Jantung", options = ['Ya', 'Tidak'])

    submit  = st.form_submit_button("Analisis Button")

st.title("Dashboard Prediksi Resiko Kesehatan")
st.markdown("-------")
if submit:
    p = 1 if perokok == 'Ya' else 0
    r = 1 if riwayat == "Ya" else 0

    try:
        feature = np.array([[umur, bmi, tekanan, kolesterol, gula_darah, aktivitas, p, r]])
        feature_scaled = scaler.transform(feature)
        prob = model.predict(feature_scaled)
        idx = np.argmax(prob, axis = 1)
        label = labeled.inverse_transform(idx)
        confidence = np.max(prob)

        col1, col2 = st.columns([1,2])
        with col1:
            st.subheader("Hasil Prediksi")
            st.metric(label = "Tingkat Resiko", value = label[0])
            st.metric(label = "Tingkat Kepercayaan", value = f"{confidence * 100:.2f}")

            if label[0] == "Tinggi":
                st.error("Saran: Segeras konsultasikan ke dokter")
            elif label[0] == "Sedang":
                st.warning("Saran : Perbaiki pola makan dan tingkatkan aktivitas fisik")
            elif label[0] == "Rendah":
                st.success("Saran : Pertahankan gaya hidul sehat anda")
        with col2:
            st.subheader("Distribusi Probabilitas")
            df_prob = pd.DataFrame({'Risiko' : labeled.classes_, 'Prob':prob[0]})
            fig = px.bar(df_prob, x = 'Risiko', y = 'Prob', color = 'Risiko',
                         color_discrete_map ={'Rendah' : '#07e843', 'Sedang' : "#d9e807", 'Tinggi' : "#e80707"})
            st.plotly_chart(fig, use_container_width = True)
    except Exception as e:
        st.error(f"Terjadi Kesalahan : {e}")