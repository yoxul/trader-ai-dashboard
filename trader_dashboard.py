import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Trader AI Dashboard", layout="centered")

st.title("📈 Trader AI Sinyal Paneli")
st.markdown("AI destekli AL / SAT / BEKLE tahmini")

# Model yükleme
@st.cache_resource
def load_model(path):
    return joblib.load(path)

model = load_model("btcusdt_1m.pkl")

# Örnek veri
st.subheader("Girdi Verisi")
uploaded_file = st.file_uploader("Bir CSV dosyası yükleyin (tek satır)", type=["csv"])

if uploaded_file:
    input_df = pd.read_csv(uploaded_file)

    st.write("### Model Girdisi")
    st.dataframe(input_df)

    # Tahmin
    if st.button("Tahmin Et"):
        prediction = model.predict(input_df)[0]
        label_map = {-1: "❌ SAT", 0: "⏳ BEKLE", 1: "✅ AL"}
        st.markdown(f"### Sonuç: {label_map[prediction]}")
else:
    st.info("Devam etmek için örnek bir CSV dosyası yükleyin.")
