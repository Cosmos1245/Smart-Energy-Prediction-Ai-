import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import firebase_admin
from firebase_admin import credentials, db
import matplotlib.pyplot as plt
from datetime import datetime

def init_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate(st.secrets["firebase"])
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://seps-ai-default-rtdb.asia-southeast1.firebasedatabase.app'
        })

def fetch_data():
    readings_ref = db.reference('/readings')
    readings_raw = readings_ref.get() or {}

    def is_valid_unix(key):
        try:
            ts = int(key)
            return 1577836800 <= ts <= 2082758400
        except Exception:
            return False

    readings_filtered = {k: v for k, v in readings_raw.items() if is_valid_unix(k)}
    df = pd.DataFrame.from_dict(readings_filtered, orient='index')
    df.index = pd.to_datetime(df.index.astype(int), unit='s')

    columns = ['light', 'fan', 'iron']
    df = df[columns].apply(pd.to_numeric, errors='coerce')
    df.dropna(inplace=True)
    return df

def normalize_data(df):
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    return df_scaled, scaler

def prepare_sequences(df_scaled, seq_len=10):
    X, y = [], []
    for i in range(len(df_scaled) - seq_len):
        X.append(df_scaled.iloc[i:i+seq_len].values)
        y.append(df_scaled.iloc[i+seq_len].values)
    return np.array(X), np.array(y)

def build_train_model(X, y, seq_len=10):
    model = Sequential([
        tf.keras.Input(shape=(seq_len, 3)),
        LSTM(64, return_sequences=True),
        LSTM(64),
        Dense(3)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=20, batch_size=16, verbose=0)
    return model

def predict_next(model, df_scaled, scaler, seq_len=10):
    next_input = np.array([df_scaled.iloc[-seq_len:].values])
    next_scaled = model.predict(next_input, verbose=0)
    next_pred = scaler.inverse_transform(next_scaled)[0]
    current_actual = scaler.inverse_transform([df_scaled.iloc[-1].values])[0]
    return next_pred, current_actual

def pct_change(now, future):
    return ((future - now) / now * 100) if now != 0 else 0

def forecast_future(model, df_scaled, scaler, seq_len=10, future_steps=30):
    future_preds = []
    input_seq = df_scaled.iloc[-seq_len:].values.copy()
    for _ in range(future_steps):
        input_array = np.array([input_seq])
        next_scaled = model.predict(input_array, verbose=0)[0]
        future_preds.append(next_scaled)
        input_seq = np.vstack([input_seq[1:], next_scaled])
    future_preds_inv = scaler.inverse_transform(future_preds)
    future_df = pd.DataFrame(future_preds_inv, columns=df_scaled.columns)

    COST_PER_AMP_HOUR = {'light': 2, 'fan': 1.5, 'iron': 3}
    future_df['cost'] = (
        future_df['light'] * COST_PER_AMP_HOUR['light'] +
        future_df['fan'] * COST_PER_AMP_HOUR['fan'] +
        future_df['iron'] * COST_PER_AMP_HOUR['iron']
    )
    return future_df

def plot_forecast(future_df):
    future_dates = pd.date_range(start=datetime.now(), periods=len(future_df), freq='D')
    future_df.index = future_dates

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(14, 6))
    future_df[['light', 'fan', 'iron']].plot(ax=ax, marker='o', linewidth=2)
    ax.set_title('Forecasted Appliance Usage (Next 30 Days)', fontsize=16)
    st.pyplot(fig)

    fig2, ax2 = plt.subplots(figsize=(12, 4))
    future_df['cost'].plot(ax=ax2, marker='x', color='crimson', linewidth=2)
    ax2.set_title('Forecasted Cost (Next 30 Days)', fontsize=16)
    st.pyplot(fig2)

def plot_trends(df_original):
    df_original['week_start'] = df_original.index.to_period('W').start_time
    df_original['month_start'] = df_original.index.to_period('M').start_time

    weekly_avg = df_original.groupby('week_start')[['light', 'fan', 'iron']].mean()
    monthly_avg = df_original.groupby('month_start')[['light', 'fan', 'iron']].mean()

    COST = {'light': 2, 'fan': 1.5, 'iron': 3}
    weekly_cost = (weekly_avg * pd.Series(COST)).sum(axis=1)
    monthly_cost = (monthly_avg * pd.Series(COST)).sum(axis=1)

    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    weekly_avg.plot(ax=axs[0,0])
    monthly_avg.plot(ax=axs[0,1])
    weekly_cost.plot(ax=axs[1,0], color='purple')
    monthly_cost.plot(ax=axs[1,1], color='green')

    for ax in axs.flat:
        ax.grid(True)
        ax.tick_params(axis='x', rotation=45)

    st.pyplot(fig)

def detect_anomalies(df_original):
    rolling_mean = df_original[['light', 'fan', 'iron']].rolling(window=12).mean()
    rolling_std = df_original[['light', 'fan', 'iron']].rolling(window=12).std()
    condition = (df_original[['light', 'fan', 'iron']] > rolling_mean + 2 * rolling_std) | \
                (df_original[['light', 'fan', 'iron']] < rolling_mean - 2 * rolling_std)
    anomalies = df_original[condition.any(axis=1)]
    return anomalies

def compute_efficiency(df_original):
    variances = df_original[['light', 'fan', 'iron']].var()
    efficiency = (1 / (1 + variances.replace(0, np.nan))) * 100
    return efficiency.fillna(0).clip(0, 100)

def plot_seasonal_usage(df_original):
    df_original['month'] = df_original.index.month
    df_original['season'] = df_original['month'].apply(
        lambda x: 'Winter' if x in [12, 1, 2] else 'Summer' if x in [5, 6, 7] else 'Other')
    seasonal_avg = df_original.groupby('season')[['light', 'fan', 'iron']].mean()

    fig, ax = plt.subplots()
    seasonal_avg.plot(kind='bar', ax=ax, title='Seasonal Appliance Usage')
    st.pyplot(fig)

def upload_to_firebase(next_pred, light_change, fan_change, iron_change, future_df, anomalies):
    forecast_ref = db.reference('/forecast')
    next_cost = next_pred[0] * 2 + next_pred[1] * 1.5 + next_pred[2] * 3
    forecast_ref.set({
        'next_light': float(next_pred[0]),
        'next_fan': float(next_pred[1]),
        'next_iron': float(next_pred[2]),
        'light_change_pct': float(light_change),
        'fan_change_pct': float(fan_change),
        'iron_change_pct': float(iron_change),
        'forecasted_cost': float(next_cost),
        'anomalies': anomalies.to_dict() if not anomalies.empty else {},
        'future_30days': future_df.to_dict()
    })

def main():
    st.title("⚡ Smart Energy Prediction AI - SEPS")
    init_firebase()
    df_original = fetch_data()

    if df_original.empty:
        st.warning("No valid data fetched from Firebase.")
        return

    st.subheader("Raw Data")
    st.dataframe(df_original.tail(10))

    # Peak usage info
    st.subheader("🔎 Peak Appliance Usage")
    for appliance in ['light', 'fan', 'iron']:
        peak = df_original[appliance].max()
        time = df_original[appliance].idxmax()
        st.write(f"**{appliance.capitalize()}**: {peak:.2f} Amps at {time}")

    # Prediction
    df_scaled, scaler = normalize_data(df_original)
    X, y = prepare_sequences(df_scaled)
    model = build_train_model(X, y)
    next_pred, current_actual = predict_next(model, df_scaled, scaler)

    st.subheader("🔮 Next Prediction")
    st.write(f"Light: {next_pred[0]:.2f}, Fan: {next_pred[1]:.2f}, Iron: {next_pred[2]:.2f}")
    light_change = pct_change(current_actual[0], next_pred[0])
    fan_change = pct_change(current_actual[1], next_pred[1])
    iron_change = pct_change(current_actual[2], next_pred[2])
    st.write(f"Changes — Light: {light_change:.2f}%, Fan: {fan_change:.2f}%, Iron: {iron_change:.2f}%")
    st.write(f"💰 Estimated Cost: ₹{next_pred[0]*2 + next_pred[1]*1.5 + next_pred[2]*3:.2f}")

    # Forecast
    st.subheader("📈 30-Day Forecast")
    future_df = forecast_future(model, df_scaled, scaler)
    st.dataframe(future_df.head())
    plot_forecast(future_df)

    # Trends
    st.subheader("📊 Trends")
    plot_trends(df_original)

    # Anomalies
    st.subheader("⚠️ Anomalies")
    anomalies = detect_anomalies(df_original)
    st.dataframe(anomalies) if not anomalies.empty else st.success("No anomalies detected.")

    # Efficiency
    st.subheader("⚙️ Efficiency")
    efficiency = compute_efficiency(df_original)
    st.dataframe(efficiency.rename_axis("Appliance").to_frame("Efficiency (%)").T)

    # Seasonal usage
    st.subheader("🌦️ Seasonal Usage")
    plot_seasonal_usage(df_original)

    # Upload
    if st.button("Upload Forecast to Firebase"):
        upload_to_firebase(next_pred, light_change, fan_change, iron_change, future_df, anomalies)
        st.success("✅ Data uploaded to Firebase")

if __name__ == "__main__":
    main()
