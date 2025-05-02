import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Trader 33333 AI Simülasyon", layout="wide")
st.title("💸 Trader AI Simülasyon Paneli + Backtest")
st.markdown("Geçmiş verilerle `AL / SAT / BEKLE` sinyallerine göre sermaye simülasyonu")

# Ayarlar
csv_path = "btcusdt_1m.csv"
model_path = "btcusdt_1m.pkl"
initial_cash = 10000.0
fee_rate = 0.001  # %0.1 Binance gibi

# Veri kontrolü
if not os.path.exists(csv_path):
    st.error(f"`{csv_path}` bulunamadı.")
    st.stop()

if not os.path.exists(model_path):
    st.error(f"`{model_path}` modeli bulunamadı.")
    st.stop()

# Veriyi oku
df = pd.read_csv(csv_path)

features = ['open', 'high', 'low', 'close', 'volume',
            'ema_10', 'ema_20', 'ema_50', 'rsi_14',
            'macd', 'macd_signal', 'macd_histogram']

model = joblib.load(model_path)

try:
    df["tahmin"] = model.predict(df[features])
    label_map = {-1: "SAT", 0: "BEKLE", 1: "AL"}
    df["sinyal"] = df["tahmin"].map(label_map)
except Exception as e:
    st.error(f"Model tahmini yapılamadı: {e}")
    st.stop()

# Tarih/saat düzeni
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

# 📊 Backtest: sanal işlem simülasyonu
cash = initial_cash
coin = 0.0
history = []

for i, row in df.iterrows():
    price = row["close"]
    signal = row["sinyal"]

    if signal == "AL" and cash > 0:
        coin = (cash * (1 - fee_rate)) / price
        history.append((i, "AL", price, cash, coin))
        cash = 0
    elif signal == "SAT" and coin > 0:
        cash = (coin * price) * (1 - fee_rate)
        history.append((i, "SAT", price, cash, coin))
        coin = 0
    else:
        history.append((i, "BEKLE", price, cash, coin))

# Son durumu hesapla
net_value = cash + coin * df["close"].iloc[-1]
profit_pct = (net_value - initial_cash) / initial_cash * 100

# ✅ Tanılayıcı yazdırmalar
st.write("🧪 Tanılama: Simülasyon Sonuçları")
st.write("Son fiyat (close):", df['close'].iloc[-1])
st.write("Nakit (cash):", cash)
st.write("Coin miktarı:", coin)
st.write("Net toplam değer (cash + coin):", net_value)
st.write("Kar/Zarar (%) :", profit_pct)

# Eksik sütun kontrolü
missing_cols = [col for col in features if col not in df.columns]
if missing_cols:
    st.warning(f"❗ Eksik sütunlar: {missing_cols}")

# Sonuç tablosu
result_df = pd.DataFrame(history, columns=["timestamp", "işlem", "fiyat", "nakit($)", "coin miktarı"])
result_df.set_index("timestamp", inplace=True)

st.subheader("💼 İşlem Geçmişi ve Portföy")
st.dataframe(result_df.tail(20), use_container_width=True)

st.success(f"✅ Simülasyon tamamlandı. Toplam Değer: ${net_value:,.2f} | Kar/Zarar: {profit_pct:.2f}%")
