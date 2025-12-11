import streamlit as st
import requests
from PIL import Image
import pandas as pd
import numpy as np

# ===========================
# CONFIGURACIÓN GENERAL
# ===========================
st.set_page_config(
    page_title="MRI Tumor Classifier",
    page_icon="🧠",
    layout="wide",
)

API_URL = "https://clasificaci-n-de-tumores-con-cnn-production.up.railway.app/predict"

# ===========================
# ESTILOS PROFESIONALES
# ===========================
st.markdown("""
<style>

    .stApp {
        background-color: #e8f1fb !important;
    }

    .card {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 22px;
        border-radius: 16px;
        box-shadow: 0px 6px 18px rgba(0,0,0,0.10);
        border: 1px solid #d3e2f4;
        backdrop-filter: blur(6px);
        margin-top: 20px;
    }

    .title {
        text-align: center;
        color: #0a4da3;
        font-weight: 900;
        font-size: 42px;
        margin-bottom: -5px;
    }

    .subtitle {
        text-align: center;
        color: #4a6fa1;
        font-size: 18px;
    }

    .result-title {
        color: #1a7c4c;
        font-size: 26px;
        font-weight: bold;
    }

    .badge {
        background-color: #3b82f6;
        padding: 8px 14px;
        border-radius: 10px;
        color: white;
        font-weight: 700;
        display:inline-block;
    }

    .stButton>button {
        background-color: #3b82f6;
        color: white;
        padding: 0.6rem 1rem;
        border-radius: 10px;
        border: none;
        font-size: 1rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #2563eb;
        transform: translateY(-2px);
    }

</style>
""", unsafe_allow_html=True)

# ===========================
# ENCABEZADO
# ===========================
st.markdown("<h1 class='title'>🧠 MRI Tumor Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Clasificación inteligente de tumores en imágenes MRI — Modelo CNN</p>", unsafe_allow_html=True)
st.write("")
st.write("")

# ===========================
# LAYOUT PRINCIPAL
# ===========================
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📤 Selecciona una imagen MRI")
    uploaded_file = st.file_uploader(
        "Formatos: JPG, PNG", type=["jpg", "jpeg", "png"], help="Sube una imagen de resonancia magnética cerebral."
    )

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        # evita el warning usando width en lugar de use_column_width
        st.image(img, caption="Imagen cargada", width=420)

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🔍 Procesamiento y diagnóstico")

    if st.button("Realizar diagnóstico", use_container_width=True):
        if not uploaded_file:
            st.warning("🔽 Por favor sube una imagen primero.")
        else:
            with st.spinner("Analizando imagen con la red neuronal..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "image/jpeg")}
                try:
                    response = requests.post(API_URL, files=files, timeout=30)
                except requests.exceptions.RequestException as e:
                    st.error(f"Error de conexión con la API: {e}")
                    response = None

                if response and response.status_code == 200:
                    data = response.json()

                    prediction = data.get("prediction", "N/A")
                    confidence = data.get("confidence", 0) * 100  # porcentaje
                    probs = data.get("probabilities", None)  # dict esperado

                    # Resultado principal
                    st.markdown(f"<div class='badge'>{prediction}</div>", unsafe_allow_html=True)
                    st.markdown(f"**Confianza:** <b style='color:#0a4da3'>{confidence:.2f}%</b>", unsafe_allow_html=True)
                    st.markdown("---")

                    # TABLA de probabilidades (ordenada desc)
                    if isinstance(probs, dict):
                        # normalizar y ordenar para evitar errores de suma flotante
                        items = [(k, float(v)) for k, v in probs.items()]
                        df = pd.DataFrame(items, columns=["Clase", "Probabilidad"])
                        df["Probabilidad (%)"] = (df["Probabilidad"] * 100).round(2)
                        df = df.sort_values("Probabilidad (%)", ascending=False).reset_index(drop=True)

                        st.write("### 📊 Probabilidades por clase")
                        # mostrar tabla simple
                        st.table(df[["Clase", "Probabilidad (%)"]])

                        # y una gráfica de barras horizontal
                        st.write("### 📈 Distribución")
                        chart_df = df.set_index("Clase")["Probabilidad (%)"]
                        st.bar_chart(chart_df)

                    else:
                        st.info("No se devolvieron probabilidades detalladas desde la API.")

                elif response:
                    st.error(f"Error en la API (status {response.status_code}).")
                # else el error ya fue mostrado

    st.markdown("</div>", unsafe_allow_html=True)

# ===========================
# MENSAJE INICIAL
# ===========================
if not uploaded_file:
    st.info("🔽 Sube una imagen para comenzar el análisis.")




