import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Trader AI Dashboard", layout="centered")

st.title("📈 Trader AI Sinyal Paneli")
st.markdown("AI destekli AL / SAT / BEKLE tahmini")

# Kullanılabilir zaman dilimleri
available_intervals = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"]

# Zaman dilimi seçimi
selected_interval = st.selectbox("Zaman Dilimini Seçin", available_intervals)

model_path = f"btcusdt_{selected_interval}.pkl"

# Modeli yükle
@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    else:
        st.error(f"Model bulunamadı: {path}")
        return None

model = load_model(model_path)

# Girdi CSV dosyası
st.subheader("Girdi Verisi")
uploaded_file = st.file_uploader("Bir CSV dosyası yükleyin (tek satır)", type=["csv"])

if model and uploaded_file:
    input_df = pd.read_csv(uploaded_file)

    st.write("### Model Girdisi")
    st.dataframe(input_df)

    if st.button("Tahmin Et"):
        prediction = model.predict(input_df)[0]
        label_map = {-1: "❌ SAT", 0: "⏳ BEKLE", 1: "✅ AL"}
        st.markdown(f"### Sonuç ({selected_interval}): {label_map[prediction]}")
elif not uploaded_file:
    st.info("Devam etmek için örnek bir CSV dosyası yükleyin.")
