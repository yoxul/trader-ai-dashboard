import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Trader AI Tahmin Paneli", layout="centered")
st.title("🤖 Trader AI Otomatik Sinyal Paneli")
st.markdown("Tüm modellerle `sample_data.csv` verisi test ediliyor...")

# Dosya yolları
model_dir = "."
sample_csv = "sample_data.csv"

# Girdi verisini yükle
if not os.path.exists(sample_csv):
    st.error(f"`{sample_csv}` dosyası bulunamadı.")
    st.stop()

input_df = pd.read_csv(sample_csv)
if input_df.shape[0] != 1:
    st.warning("Uyarı: `sample_data.csv` dosyası tek bir satır içermelidir.")
    st.stop()

# Tahminleri topla
results = []

for filename in sorted(os.listdir(model_dir)):
    if filename.endswith(".pkl"):
        model_path = os.path.join(model_dir, filename)
        try:
            model = joblib.load(model_path)
            pred = model.predict(input_df)[0]

            label_map = {-1: "❌ SAT", 0: "⏳ BEKLE", 1: "✅ AL"}
            tahmin = label_map.get(pred, "❓")

            zaman_dilimi = filename.replace("btcusdt_", "").replace(".pkl", "")
            results.append({"Zaman Dilimi": zaman_dilimi, "Tahmin": tahmin})
        except Exception as e:
            results.append({"Zaman Dilimi": filename, "Tahmin": f"HATA: {e}"})

# Sonuçları göster
if results:
    st.success("✅ Tahminler tamamlandı.")
    st.table(pd.DataFrame(results))
else:
    st.warning("Hiç model bulunamadı.")
