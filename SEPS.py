import streamlit as st
import json
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

# --- Firebase Initialization ---
def init_firebase():
    firebase_config = dict(st.secrets["firebase"])
    firebase_config["private_key"] = firebase_config["private_key"].replace('\\n', '\n')
    if not firebase_admin._apps:
        cred = credentials.Certificate(firebase_config)
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://seps-ai-default-rtdb.asia-southeast1.firebasedatabase.app/'
        })
    st.success("✅ Firebase initialized")

# --- Fetch data from Firebase ---
def fetch_data():
    readings_ref = db.reference('/readings')
    readings_raw = readings_ref.get()

    def is_valid_unix(key):
        try:
            ts = int(key)
            return 1577836800 <= ts <= 2082758400
        except:
            return False

    readings_filtered = {k: v for k, v in readings_raw.items() if is_valid_unix(k)}
    df = pd.DataFrame.from_dict(readings_filtered, orient='index')
    df.index = pd.to_datetime(df.index.astype(int), unit='s')
    columns = ['light', 'fan', 'iron']
    df = df[columns].apply(pd.to_numeric, errors='coerce')
    df.dropna(inplace=True)
    return df

# --- Normalize data ---
def normalize_data(df):
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    return df_scaled, scaler

# --- Prepare sequences ---
def prepare_sequences(df_scaled, seq_len=10):
    X, y = [], []
    for i in range(len(df_scaled) - seq_len):
        X.append(df_scaled.iloc[i:i+seq_len].values)
        y.append(df_scaled.iloc[i+seq_len].values)
    return np.array(X), np.array(y)

# --- Build and train LSTM model ---
def build_train_model(X, y, seq_len=10):
    model = Sequential([
        tf.keras.Input(shape=(seq_len, 3)),
        LSTM(64, return_sequences=True),
        LSTM(64),
        Dense(3)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=20, batch_size=16, verbose=1)
    return model

# --- Predict next step ---
def predict_next(model, df_scaled, scaler, seq_len=10):
    next_input = np.array([df_scaled.iloc[-seq_len:].values])
    next_scaled = model.predict(next_input)
    next_pred = scaler.inverse_transform(next_scaled)[0]
    current_actual = scaler.inverse_transform([df_scaled.iloc[-1].values])[0]
    return next_pred, current_actual

# --- % Change Calculation ---
def pct_change(now, future):
    return ((future - now) / now * 100) if now else 0

# --- 30-Day Forecast ---
def forecast_future(model, df_scaled, scaler, seq_len=10, future_steps=30):
    future_preds = []
    input_seq = df_scaled.iloc[-seq_len:].values.copy()
    for _ in range(future_steps):
        input_array = np.array([input_seq])
        next_scaled = model.predict(input_array)[0]
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

# --- Plot forecasted usage and cost ---
def plot_forecast(future_df):
    future_dates = pd.date_range(start=datetime.now(), periods=len(future_df), freq='D')
    future_df.index = future_dates

    plt.style.use('seaborn-v0_8-darkgrid')

    fig, ax = plt.subplots(figsize=(14, 6))
    future_df[['light', 'fan', 'iron']].plot(ax=ax, marker='o', linewidth=2)
    ax.set_title('Forecasted Appliance Usage (Next 30 Days)', fontsize=16, weight='bold')
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Usage (Amps)", fontsize=12)
    ax.grid(which='both', linestyle='--', linewidth=0.7, alpha=0.7)
    y_max = future_df[['light', 'fan', 'iron']].values.max()
    ax.set_ylim(0, y_max * 1.1)
    plt.xticks(rotation=45)
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    st.pyplot(fig)

    fig2, ax2 = plt.subplots(figsize=(12, 4))
    future_df['cost'].plot(ax=ax2, marker='x', color='crimson', linewidth=2)
    ax2.set_title('Forecasted Cost (Next 30 Days)', fontsize=16, weight='bold')
    ax2.set_xlabel("Date", fontsize=12)
    ax2.set_ylabel("Cost (₹)", fontsize=12)
    ax2.grid(which='both', linestyle='--', linewidth=0.7, alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig2)

# --- Trends ---
def plot_trends(df_original):
    df_original = df_original.copy()
    df_original.index = pd.to_datetime(df_original.index)
    df_original['week'] = df_original.index.isocalendar().week
    df_original['month'] = df_original.index.month
    weekly_avg = df_original.groupby('week')[['light', 'fan', 'iron']].mean()
    monthly_avg = df_original.groupby('month')[['light', 'fan', 'iron']].mean()

    COST_PER_AMP_HOUR = {'light': 2, 'fan': 1.5, 'iron': 3}
    weekly_cost = (weekly_avg * pd.Series(COST_PER_AMP_HOUR)).sum(axis=1)
    monthly_cost = (monthly_avg * pd.Series(COST_PER_AMP_HOUR)).sum(axis=1)

    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    weekly_avg.plot(ax=axs[0,0], title='Weekly Averages', marker='o')
    monthly_avg.plot(ax=axs[0,1], title='Monthly Averages', marker='o')
    weekly_cost.plot(ax=axs[1,0], title='Weekly Cost (₹)', color='purple', marker='x')
    monthly_cost.plot(ax=axs[1,1], title='Monthly Cost (₹)', color='green', marker='x')
    plt.tight_layout()
    st.pyplot(fig)

# --- Anomaly Detection ---
def detect_anomalies(df_original):
    rolling_mean = df_original[['light', 'fan', 'iron']].rolling(window=12).mean()
    rolling_std = df_original[['light', 'fan', 'iron']].rolling(window=12).std()
    anomalies = df_original[
        (df_original[['light', 'fan', 'iron']] > rolling_mean + 2 * rolling_std) |
        (df_original[['light', 'fan', 'iron']] < rolling_mean - 2 * rolling_std)
    ].dropna()
    return anomalies

# --- Efficiency ---
def compute_efficiency(df_original):
    return (1 / (1 + df_original[['light', 'fan', 'iron']].var())) * 100

# --- Seasonal Usage ---
def plot_seasonal_usage(df_original):
    df_original = df_original.copy()
    df_original.index = pd.to_datetime(df_original.index)
    df_original['month'] = df_original.index.month
    df_original['season'] = df_original['month'].apply(
        lambda x: 'Winter' if x in [12, 1, 2]
        else 'Summer' if x in [5, 6, 7]
        else 'Other'
    )
    seasonal_avg = df_original.groupby('season')[['light', 'fan', 'iron']].mean()
    fig, ax = plt.subplots()
    seasonal_avg.plot(kind='bar', ax=ax, title='Seasonal Appliance Usage')
    ax.set_ylabel("Average Readings")
    st.pyplot(fig)

# --- Upload to Firebase ---
def upload_to_firebase(next_pred, light_change, fan_change, iron_change, future_df, anomalies):
    forecast_ref = db.reference('/forecast')
    next_cost = (
        next_pred[0] * 2 +
        next_pred[1] * 1.5 +
        next_pred[2] * 3
    )
    forecast_ref.set({
        'next_light': float(next_pred[0]),
        'next_fan': float(next_pred[1]),
        'next_iron': float(next_pred[2]),
        'light_change_pct': float(light_change),
        'fan_change_pct': float(fan_change),
        'iron_change_pct': float(iron_change),
        'forecasted_cost': float(next_cost),
        'anomalies': anomalies.to_dict() if not anomalies.empty else {},
        'future_30days': future_df.round(3).to_dict(orient='records'),
        'updated_at': datetime.now().isoformat()
    })

# --- Main ---
def main():
    st.title("Smart Energy Prediction System (SEPS)")
    init_firebase()
    with st.spinner("Fetching data from Firebase..."):
        df = fetch_data()
    st.write("### Recent Energy Usage Data", df.tail(10))
    df_scaled, scaler = normalize_data(df)
    seq_len = 10
    X, y = prepare_sequences(df_scaled, seq_len)
    model = build_train_model(X, y, seq_len)
    next_pred, current_actual = predict_next(model, df_scaled, scaler, seq_len)
    light_change = pct_change(current_actual[0], next_pred[0])
    fan_change = pct_change(current_actual[1], next_pred[1])
    iron_change = pct_change(current_actual[2], next_pred[2])

    st.write("### Next Predicted Usage")
    st.write(f"Light: {next_pred[0]:.3f} (Change: {light_change:.2f}%)")
    st.write(f"Fan: {next_pred[1]:.3f} (Change: {fan_change:.2f}%)")
    st.write(f"Iron: {next_pred[2]:.3f} (Change: {iron_change:.2f}%)")

    future_df = forecast_future(model, df_scaled, scaler, seq_len, future_steps=30)
    st.write("### Forecasted Cost (Next 30 Days)")
    total_cost = future_df['cost'].sum()
    avg_daily_cost = future_df['cost'].mean()
    st.write(f"**Total Cost**: ₹{total_cost:.2f}")
    st.write(f"**Average Daily Cost**: ₹{avg_daily_cost:.2f}")
    st.write("#### Daily Forecasted Cost Table")
    st.dataframe(future_df[['cost']].round(2))
    plot_forecast(future_df)
    plot_trends(df)
    anomalies = detect_anomalies(df)
    st.write("### Detected Anomalies")
    st.write(anomalies if not anomalies.empty else "No anomalies detected")
    efficiency_score = compute_efficiency(df)
    st.write(f"### Appliance Efficiency Score: {efficiency_score.mean():.2f}")
    plot_seasonal_usage(df)
    upload_to_firebase(next_pred, light_change, fan_change, iron_change, future_df, anomalies)
    st.success("✅ Forecast and analytics uploaded to Firebase")

if __name__ == "__main__":
    main()
