import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import firebase_admin
from firebase_admin import credentials, db
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Firebase initialization function
def init_firebase():
    firebase_config = dict(st.secrets["firebase"])
    firebase_config["private_key"] = firebase_config["private_key"].replace('\\n', '\n')
    cred = credentials.Certificate(firebase_config)
    try:
        firebase_admin.get_app()
    except ValueError:
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://seps-ai-default-rtdb.asia-southeast1.firebasedatabase.app/'
        })

init_firebase()

def fetch_data():
    readings_ref = db.reference('/readings')
    readings_raw = readings_ref.get()
    if readings_raw is None:
        return pd.DataFrame()
    def is_valid_unix(key):
        try:
            ts = int(key)
            return 1577836800 <= ts <= 2082758400
        except:
            return False
    readings_filtered = {k: v for k, v in readings_raw.items() if is_valid_unix(k)}
    df = pd.DataFrame.from_dict(readings_filtered, orient='index')
    df.index = pd.to_datetime(df.index.astype(int), unit='s')
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
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

def build_train_model(X, y, seq_len=10, lstm_units=64, layers=2, epochs=20, batch_size=16):
    model = Sequential()
    model.add(tf.keras.Input(shape=(seq_len, X.shape[2])))
    for i in range(layers - 1):
        model.add(LSTM(lstm_units, return_sequences=True))
    model.add(LSTM(lstm_units))
    model.add(Dense(X.shape[2]))
    model.compile(optimizer='adam', loss='mse')
    progress_bar = st.progress(0)
    for epoch in range(epochs):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0)
        progress_bar.progress((epoch + 1) / epochs)
    progress_bar.empty()
    return model

def predict_next(model, df_scaled, scaler, seq_len=10):
    next_input = np.array([df_scaled.iloc[-seq_len:].values])
    next_scaled = model.predict(next_input, verbose=0)
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
        next_scaled = model.predict(input_array, verbose=0)[0]
        future_preds.append(next_scaled)
        input_seq = np.vstack([input_seq[1:], next_scaled])
    future_preds_inv = scaler.inverse_transform(future_preds)
    future_df = pd.DataFrame(future_preds_inv, columns=df_scaled.columns)
    return future_df

def plot_forecast_plotly(future_df, cost_per_amp):
    future_dates = pd.date_range(start=datetime.now(), periods=len(future_df), freq='D')
    future_df.index = future_dates
    future_df['cost'] = (
        future_df['light'] * cost_per_amp['light'] +
        future_df['fan'] * cost_per_amp['fan'] +
        future_df['iron'] * cost_per_amp['iron']
    )
    fig = go.Figure()
    for device in ['light', 'fan', 'iron']:
        fig.add_trace(go.Scatter(x=future_df.index, y=future_df[device],
                                 mode='lines+markers', name=device.capitalize()))
    fig.update_layout(title="Forecasted Appliance Usage", xaxis_title="Date", yaxis_title="Usage (Amps)")
    st.plotly_chart(fig, use_container_width=True)

    fig_cost = px.line(future_df, y='cost', title="Forecasted Cost (₹) over Days")
    st.plotly_chart(fig_cost, use_container_width=True)

def detect_anomalies(df_original):
    rolling_mean = df_original[['light', 'fan', 'iron']].rolling(window=12).mean()
    rolling_std = df_original[['light', 'fan', 'iron']].rolling(window=12).std()
    anomalies = df_original[
        (df_original[['light', 'fan', 'iron']] > rolling_mean + 2 * rolling_std) |
        (df_original[['light', 'fan', 'iron']] < rolling_mean - 2 * rolling_std)
    ].dropna()
    return anomalies

def explain_anomalies(df_original, anomalies):
    explanations = []
    for idx, row in anomalies.iterrows():
        seasonal_avg = df_original[df_original.index.month == idx.month][['light', 'fan', 'iron']].mean()
        reasons = []
        for device in ['light', 'fan', 'iron']:
            if abs(row[device] - seasonal_avg[device]) > seasonal_avg[device] * 0.5:
                reasons.append(f"{device.capitalize()} usage unusually high/low compared to seasonal average.")
        explanations.append((idx.strftime("%Y-%m-%d %H:%M:%S"), reasons))
    return explanations

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
        'next_light_change': float(light_change),
        'next_fan_change': float(fan_change),
        'next_iron_change': float(iron_change),
        'next_usage_cost': float(next_cost),
        'timestamp': int(datetime.now().timestamp()),
        'future_forecast': future_df.to_dict(orient='records'),
        'anomalies': anomalies.to_dict(orient='records'),
    })

def main():
    st.set_page_config(page_title="Advanced SEPS", layout="wide", page_icon="⚡")
    
    # Theme toggle
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False

    def toggle_theme():
        st.session_state.dark_mode = not st.session_state.dark_mode

    st.sidebar.title("⚙️ Settings")
    st.sidebar.button("Toggle Light/Dark Theme", on_click=toggle_theme)
    theme = "dark" if st.session_state.dark_mode else "light"
    st.markdown(f"<style>body {{background-color: {'#0E1117' if theme=='dark' else 'white'}; color: {'white' if theme=='dark' else 'black'};}}</style>", unsafe_allow_html=True)

    st.sidebar.markdown("### LSTM Model Parameters")
    lstm_layers = st.sidebar.slider("LSTM Layers", 1, 4, 2)
    lstm_units = st.sidebar.slider("Units per Layer", 16, 256, 64, step=16)
    epochs = st.sidebar.slider("Training Epochs", 5, 50, 20)
    batch_size = st.sidebar.selectbox("Batch Size", [8, 16, 32, 64], index=1)
    
    st.sidebar.markdown("### Forecast Settings")
    future_days = st.sidebar.slider("Days to Forecast", 7, 60, 30)

    st.sidebar.markdown("### Cost Settings (₹ per Amp)")
    cost_light = st.sidebar.number_input("Light cost per Amp", min_value=0.0, value=2.0)
    cost_fan = st.sidebar.number_input("Fan cost per Amp", min_value=0.0, value=1.5)
    cost_iron = st.sidebar.number_input("Iron cost per Amp", min_value=0.0, value=3.0)
    cost_per_amp = {'light': cost_light, 'fan': cost_fan, 'iron': cost_iron}

    st.title("⚡ Advanced Smart Energy Prediction System (SEPS)")

    if st.button("Fetch & Refresh Data"):
        st.session_state.df_original = fetch_data()
        st.success("Data refreshed!")

    if "df_original" not in st.session_state:
        st.session_state.df_original = fetch_data()

    df_original = st.session_state.df_original

    if df_original.empty:
        st.warning("No data found. Please refresh after data is available.")
        return

    st.subheader("Historical Data Preview")
    st.dataframe(df_original.tail(15))

    # Normalize & Prepare
    df_scaled, scaler = normalize_data(df_original)
    seq_len = 10
    X, y = prepare_sequences(df_scaled, seq_len=seq_len)

    if st.button("Train LSTM Model"):
        with st.spinner("Training LSTM..."):
            model = build_train_model(X, y, seq_len=seq_len,
                                    lstm_units=lstm_units, layers=lstm_layers, epochs=epochs, batch_size=batch_size)
        st.session_state.model = model
        st.success("Training completed!")

    if "model" not in st.session_state:
        st.info("Train the LSTM model to proceed.")
        return

    model = st.session_state.model

    st.subheader("Next Day Prediction")
    next_pred, current_actual = predict_next(model, df_scaled, scaler, seq_len=seq_len)

    # Safety check - ensure arrays length match expected appliance count (3)
    if len(current_actual) != 3 or len(next_pred) != 3:
        st.error("Prediction outputs shape mismatch! Expected 3 features (Light, Fan, Iron).")
        st.write(f"current_actual shape: {current_actual.shape}, next_pred shape: {next_pred.shape}")
        return

    light_change = pct_change(current_actual[0], next_pred[0])
    fan_change = pct_change(current_actual[1], next_pred[1])
    iron_change = pct_change(current_actual[2], next_pred[2])

    pred_df = pd.DataFrame({
        'Appliance': ['Light', 'Fan', 'Iron'],
        'Current Usage (A)': np.round(current_actual[:3], 2).tolist(),
        'Next Day Prediction (A)': np.round(next_pred[:3], 2).tolist(),
        'Percentage Change (%)': [light_change, fan_change, iron_change]
    })

    st.table(pred_df)

    # Forecast Future
    future_df = forecast_future(model, df_scaled, scaler, seq_len=seq_len, future_steps=future_days)
    st.subheader(f"{future_days}-Day Forecast")

    plot_forecast_plotly(future_df, cost_per_amp)

    # Anomaly Detection + Explanation
    st.subheader("Anomaly Detection")
    anomalies = detect_anomalies(df_original)
    if anomalies.empty:
        st.info("No anomalies detected.")
    else:
        st.write(f"Detected {len(anomalies)} anomalies:")
        for idx, row in anomalies.iterrows():
            with st.expander(f"Anomaly at {idx.strftime('%Y-%m-%d %H:%M:%S')}"):
                st.write(row)

        explanations = explain_anomalies(df_original, anomalies)
        st.subheader("Anomaly Explanations")
        for date, reasons in explanations:
            with st.expander(date):
                for r in reasons:
                    st.markdown(f"- {r}")

    # Upload to Firebase
    if st.button("Upload Forecast & Anomalies to Firebase"):
        upload_to_firebase(next_pred, light_change, fan_change, iron_change, future_df, anomalies)
        st.success("Data uploaded to Firebase.")

    # Export data options
    st.subheader("Export Data")
    csv_forecast = future_df.to_csv().encode('utf-8')
    csv_anomalies = anomalies.to_csv().encode('utf-8')

    st.download_button("Download Forecast CSV", csv_forecast, "forecast.csv", "text/csv")
    st.download_button("Download Anomalies CSV", csv_anomalies, "anomalies.csv", "text/csv")

if __name__ == "__main__":
    main()
