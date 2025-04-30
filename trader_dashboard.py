import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Trader AI Simülasyon", layout="wide")
st.title("🧪 Trader AI Simülasyon Paneli")
st.markdown("Geçmiş verilerle `AL / SAT / BEKLE` sinyalleri")

# Ayarlar
csv_path = "btcusdt_1m.csv"
model_path = "model/btcusdt_1m.pkl"

# Veri kontrolü
if not os.path.exists(csv_path):
    st.error(f"`{csv_path}` bulunamadı.")
    st.stop()

if not os.path.exists(model_path):
    st.error(f"`{model_path}` modeli bulunamadı.")
    st.stop()

# Veriyi oku
df = pd.read_csv(csv_path)

# Girdiler
features = ['open', 'high', 'low', 'close', 'volume',
            'ema_10', 'ema_20', 'ema_50', 'rsi_14',
            'macd', 'macd_signal', 'macd_histogram']

# Modeli yükle
model = joblib.load(model_path)

# Simülasyon: Tahmin sütununu ekle
try:
    df["tahmin"] = model.predict(df[features])
    label_map = {-1: "❌ SAT", 0: "⏳ BEKLE", 1: "✅ AL"}
    df["sinyal"] = df["tahmin"].map(label_map)
except Exception as e:
    st.error(f"Model tahmini yapılamadı: {e}")
    st.stop()

# Tarih/saat sütunu varsa öne al
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

# Sonuçları göster
st.dataframe(df[["close", "rsi_14", "macd", "sinyal"]].tail(50), height=600)

st.success("✅ Simülasyon tamamlandı. Son 50 satır aşağıda gösteriliyor.")
