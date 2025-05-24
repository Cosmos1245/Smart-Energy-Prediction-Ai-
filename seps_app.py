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
    firebase_config = dict(st.secrets["firebase"])
    if not firebase_admin._apps:
        cred = credentials.Certificate(firebase_config)
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
    model.fit(X, y, epochs=20, batch_size=16, verbose=1)
    return model

def predict_next(model, df_scaled, scaler, seq_len=10):
    next_input = np.array([df_scaled.iloc[-seq_len:].values])
    next_scaled = model.predict(next_input)
    next_pred = scaler.inverse_transform(next_scaled)[0]
    current_actual = scaler.inverse_transform([df_scaled.iloc[-1].values])[0]
    return next_pred, current_actual

def pct_change(now, future):
    return ((future - now) / now * 100) if now else 0

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

def plot_trends(df_original):
    df_original = df_original.copy()
    df_original.index = pd.to_datetime(df_original.index)
    df_original['week_start'] = df_original.index.to_period('W').start_time
    df_original['month_start'] = df_original.index.to_period('M').start_time

    weekly_avg = df_original.groupby('week_start')[['light', 'fan', 'iron']].mean()
    monthly_avg = df_original.groupby('month_start')[['light', 'fan', 'iron']].mean()

    COST_PER_AMP_HOUR = {'light': 2, 'fan': 1.5, 'iron': 3}
    weekly_cost = (weekly_avg * pd.Series(COST_PER_AMP_HOUR)).sum(axis=1)
    monthly_cost = (monthly_avg * pd.Series(COST_PER_AMP_HOUR)).sum(axis=1)

    fig, axs = plt.subplots(2, 2, figsize=(14, 8))

    weekly_avg.plot(ax=axs[0,0], marker='o')
    axs[0,0].set_title('Weekly Averages')
    axs[0,0].set_xlabel("Week Start")
    axs[0,0].set_ylabel("Amps")

    monthly_avg.plot(ax=axs[0,1], marker='o')
    axs[0,1].set_title('Monthly Averages')
    axs[0,1].set_xlabel("Month Start")
    axs[0,1].set_ylabel("Amps")

    weekly_cost.plot(ax=axs[1,0], marker='x', color='purple')
    axs[1,0].set_title('Weekly Cost (₹)')
    axs[1,0].set_xlabel("Week Start")
    axs[1,0].set_ylabel("Cost")

    monthly_cost.plot(ax=axs[1,1], marker='x', color='green')
    axs[1,1].set_title('Monthly Cost (₹)')
    axs[1,1].set_xlabel("Month Start")
    axs[1,1].set_ylabel("Cost")

    for ax in axs.flat:
        ax.grid(True)
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    st.pyplot(fig)

def detect_anomalies(df_original):
    rolling_mean = df_original[['light', 'fan', 'iron']].rolling(window=12).mean()
    rolling_std = df_original[['light', 'fan', 'iron']].rolling(window=12).std()
    anomalies = df_original[
        (df_original[['light', 'fan', 'iron']] > rolling_mean + 2 * rolling_std) |
        (df_original[['light', 'fan', 'iron']] < rolling_mean - 2 * rolling_std)
    ].dropna()
    return anomalies

def compute_efficiency(df_original):
    return (1 / (1 + df_original[['light', 'fan', 'iron']].var())) * 100

def plot_seasonal_usage(df_original):
    df_original = df_original.copy()
    df_original.index = pd.to_datetime(df_original.index)
    df_original['month'] = df_original.index.month
    df_original['season'] = df_original['month'].apply(
        lambda x: 'Winter' if x in [12, 1, 2] else 'Summer' if x in [5, 6, 7] else 'Other')
    seasonal_avg = df_original.groupby('season')[['light', 'fan', 'iron']].mean()
    fig, ax = plt.subplots()
    seasonal_avg.plot(kind='bar', ax=ax, title='Seasonal Appliance Usage')
    ax.set_ylabel("Average Readings")
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

    peak_info = {}
    for appliance in ['light', 'fan', 'iron']:
        peak_usage = df_original[appliance].max()
        peak_time = df_original[appliance].idxmax()
        peak_info[appliance] = (peak_usage, peak_time)

    st.subheader("🔎 Peak Appliance Usage (Historical)")
    for appliance, (usage, time) in peak_info.items():
        st.markdown(f"**{appliance.capitalize()}**: Peak usage was **{usage:.2f} Amps** at **{time.strftime('%Y-%m-%d %H:%M:%S')}**")

    df_scaled, scaler = normalize_data(df_original)
    X, y = prepare_sequences(df_scaled)
    model = build_train_model(X, y)
    next_pred, current_actual = predict_next(model, df_scaled, scaler)

    st.subheader("🔮 Next Prediction")
    st.write(f"Light: {next_pred[0]:.3f} Amps, Fan: {next_pred[1]:.3f} Amps, Iron: {next_pred[2]:.3f} Amps")

    light_change = pct_change(current_actual[0], next_pred[0])
    fan_change = pct_change(current_actual[1], next_pred[1])
    iron_change = pct_change(current_actual[2], next_pred[2])

    st.write(f"Change in Light Usage: {light_change:.2f}%")
    st.write(f"Change in Fan Usage: {fan_change:.2f}%")
    st.write(f"Change in Iron Usage: {iron_change:.2f}%")

    estimated_cost = next_pred[0]*2 + next_pred[1]*1.5 + next_pred[2]*3
    st.markdown(f"💰 **Estimated Cost of Next Usage**: ₹{estimated_cost:.2f}")

    future_df = forecast_future(model, df_scaled, scaler)

    st.subheader("📈 30-Day Forecast")
    st.dataframe(future_df.head())
    plot_forecast(future_df)

    st.subheader("📊 Weekly and Monthly Trends")
    plot_trends(df_original)

    st.subheader("⚠️ Anomalies Detected")
    anomalies = detect_anomalies(df_original)
    if not anomalies.empty:
        st.dataframe(anomalies)
    else:
        st.write("No anomalies detected.")

    st.subheader("⚙️ Appliance Efficiency Scores")
    efficiency = compute_efficiency(df_original)
    st.dataframe(efficiency.rename_axis('Appliance').to_frame('Efficiency Score').T)

    st.subheader("🌦️ Seasonal Usage Patterns")
    plot_seasonal_usage(df_original)

    st.subheader("Upload Forecast & Anomalies to Firebase")
    if st.button("Upload to Firebase"):
        upload_to_firebase(next_pred, light_change, fan_change, iron_change, future_df, anomalies)
        efficiency_score = efficiency.to_dict()
        st.markdown("## 🧾 FINAL REPORT")
        st.markdown(f"""
🔮 **Predicted Next Usage:**
- Light usage {'increased significantly' if light_change > 10 else 'decreased significantly' if light_change < -10 else 'increased moderately' if light_change > 0 else 'decreased moderately'} ({light_change:.2f}%)
- Fan usage {'increased significantly' if fan_change > 10 else 'decreased significantly' if fan_change < -10 else 'increased moderately' if fan_change > 0 else 'decreased moderately'} ({fan_change:.2f}%)
- Iron usage {'increased significantly' if iron_change > 10 else 'decreased significantly' if iron_change < -10 else 'increased moderately' if iron_change > 0 else 'decreased moderately'} ({iron_change:.2f}%)

💰 **Estimated Cost of Next Usage**: ₹{estimated_cost:.2f}

{'✅ No significant anomalies detected.' if anomalies.empty else '⚠️ Anomalies detected!'}

⚙️ **Appliance Efficiency Scores (0–100):**
- Light: {efficiency_score['light']:.1f}
- Fan: {efficiency_score['fan']:.1f}
- Iron: {efficiency_score['iron']:.1f}
""")
        st.success("✅ Forecast and analytics uploaded to Firebase")

if __name__ == "__main__":
    main()
