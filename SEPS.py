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
import io
from fpdf import FPDF

# === Firebase Setup ===
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

# === Fetch Data ===
def fetch_data():
    readings_ref = db.reference('/readings')
    readings_raw = readings_ref.get()
    if readings_raw is None:
        st.error("No data found at /readings")
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

    columns = ['light', 'fan', 'iron']
    df = df[columns].apply(pd.to_numeric, errors='coerce')
    df.dropna(inplace=True)
    return df

# === Normalize ===
def normalize_data(df):
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    return df_scaled, scaler

# === Prepare sequences ===
def prepare_sequences(df_scaled, seq_len=10):
    X, y = [], []
    for i in range(len(df_scaled) - seq_len):
        X.append(df_scaled.iloc[i:i+seq_len].values)
        y.append(df_scaled.iloc[i+seq_len].values)
    return np.array(X), np.array(y)

# === Build and train model ===
def build_train_model(X, y, seq_len=10):
    model = Sequential([
        tf.keras.Input(shape=(seq_len, 3)),
        LSTM(64, return_sequences=True),
        LSTM(64),
        Dense(3)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=20, batch_size=16, verbose=0)  # silent for Streamlit
    return model

# === Predict next usage ===
def predict_next(model, df_scaled, scaler, seq_len=10):
    next_input = np.array([df_scaled.iloc[-seq_len:].values])
    next_scaled = model.predict(next_input, verbose=0)
    next_pred = scaler.inverse_transform(next_scaled)[0]
    current_actual = scaler.inverse_transform([df_scaled.iloc[-1].values])[0]
    return next_pred, current_actual

# === % Change Calculation ===
def pct_change(now, future):
    return ((future - now) / now * 100) if now else 0

# === Forecast future ===
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

# === Plot forecast ===
def plot_forecast(future_df):
    future_dates = pd.date_range(start=datetime.now(), periods=len(future_df), freq='D')
    future_df.index = future_dates
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(14, 6))
    future_df[['light', 'fan', 'iron']].plot(ax=ax, marker='o', linewidth=2)
    ax.set_title('Forecasted Appliance Usage (Next Days)', fontsize=16, weight='bold')
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
    ax2.set_title('Forecasted Cost (Next Days)', fontsize=16, weight='bold')
    ax2.set_xlabel("Date", fontsize=12)
    ax2.set_ylabel("Cost (₹)", fontsize=12)
    ax2.grid(which='both', linestyle='--', linewidth=0.7, alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig2)

# === Weekly & Monthly Trends ===
def plot_trends(df_original):
    df = df_original.copy()
    df.index = pd.to_datetime(df.index)
    df['week'] = df.index.isocalendar().week
    df['month'] = df.index.month
    weekly_avg = df.groupby('week')[['light', 'fan', 'iron']].mean()
    monthly_avg = df.groupby('month')[['light', 'fan', 'iron']].mean()

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

# === Anomaly Detection ===
def detect_anomalies(df_original):
    rolling_mean = df_original[['light', 'fan', 'iron']].rolling(window=12).mean()
    rolling_std = df_original[['light', 'fan', 'iron']].rolling(window=12).std()
    anomalies = df_original[
        (df_original[['light', 'fan', 'iron']] > rolling_mean + 2 * rolling_std) |
        (df_original[['light', 'fan', 'iron']] < rolling_mean - 2 * rolling_std)
    ].dropna()
    return anomalies

# === Explain anomalies ===
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

# === Efficiency Score ===
def compute_efficiency(df_original):
    return (1 / (1 + df_original[['light', 'fan', 'iron']].var())) * 100

# === Seasonal Usage ===
def plot_seasonal_usage(df_original):
    df = df_original.copy()
    df.index = pd.to_datetime(df.index)
    df['month'] = df.index.month
    df['season'] = df['month'].apply(
        lambda x: 'Winter' if x in [12, 1, 2]
        else 'Summer' if x in [5, 6, 7]
        else 'Other'
    )
    seasonal_avg = df.groupby('season')[['light', 'fan', 'iron']].mean()
    fig, ax = plt.subplots()
    seasonal_avg.plot(kind='bar', ax=ax, title='Seasonal Appliance Usage')
    ax.set_ylabel("Average Readings")
    st.pyplot(fig)

# === Upload forecast to Firebase ===
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

# === Advanced Export (Excel + PDF Summary) ===
def export_reports(future_df, anomalies, explanation_text):
    # Excel Export
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        future_df.to_excel(writer, sheet_name='Forecast')
        anomalies.to_excel(writer, sheet_name='Anomalies')
        workbook = writer.book
        worksheet = writer.sheets['Forecast']

        chart = workbook.add_chart({'type': 'line'})
        max_row = len(future_df) + 1
        for i, col in enumerate(['light', 'fan', 'iron']):
            chart.add_series({
                'name':       ['Forecast', 0, i + 1],
                'categories': ['Forecast', 1, 0, max_row, 0],
                'values':     ['Forecast', 1, i + 1, max_row, i + 1],
                'line':       {'width': 1.5},
            })
        chart.set_title({'name': 'Forecasted Appliance Usage'})
        worksheet.insert_chart('H2', chart)

    # PDF summary report
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "Smart Energy Prediction System - Summary Report", ln=True, align='C')
    pdf.ln(5)
    for line in explanation_text.split('\n'):
        pdf.multi_cell(0, 10, line)
    pdf_output = io.BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)

    # Streamlit download buttons
    st.download_button("Download Forecast Excel", excel_buffer.getvalue(), file_name="energy_forecast.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.download_button("Download Summary Report (PDF)", pdf_output.getvalue(), file_name="energy_summary.pdf", mime="application/pdf")

# === Main Streamlit App ===
def main():
    st.title("🚀 Advanced Smart Energy Prediction System")

    df_original = fetch_data()
    if df_original.empty:
        st.warning("Waiting for valid data...")
        return

    st.subheader("Historical Data Preview")
    st.dataframe(df_original.tail(10))

    st.subheader("User Settings")
    future_days = st.slider("Days to Forecast", 7, 60, 30)

    df_scaled, scaler = normalize_data(df_original)
    seq_len = 10
    X, y = prepare_sequences(df_scaled, seq_len=seq_len)

    st.text("Training LSTM model (this may take a moment)...")
    model = build_train_model(X, y, seq_len=seq_len)
    st.success("Model training complete!")

    next_pred, current_actual = predict_next(model, df_scaled, scaler, seq_len=seq_len)

    st.subheader("Next Day Prediction")
    st.write(f"Predicted usage - Light: {next_pred[0]:.2f} A, Fan: {next_pred[1]:.2f} A, Iron: {next_pred[2]:.2f} A")
    st.write(f"Current usage - Light: {current_actual[0]:.2f} A, Fan: {current_actual[1]:.2f} A, Iron: {current_actual[2]:.2f} A")

    light_change = pct_change(current_actual[0], next_pred[0])
    fan_change = pct_change(current_actual[1], next_pred[1])
    iron_change = pct_change(current_actual[2], next_pred[2])

    st.write(f"Percentage change - Light: {light_change:+.2f}%, Fan: {fan_change:+.2f}%, Iron: {iron_change:+.2f}%")

    future_df = forecast_future(model, df_scaled, scaler, seq_len=seq_len, future_steps=future_days)
    st.subheader(f"{future_days}-Day Forecast")
    plot_forecast(future_df)

    st.subheader("Weekly & Monthly Trends")
    plot_trends(df_original)

    anomalies = detect_anomalies(df_original)
    st.subheader("Detected Anomalies")
    if anomalies.empty:
        st.write("No anomalies detected.")
    else:
        st.dataframe(anomalies)

    explanations = explain_anomalies(df_original, anomalies)
    explanation_text = "Anomaly Explanations:\n"
    for date, reasons in explanations:
        explanation_text += f"\n{date}:\n"
        for r in reasons:
            explanation_text += f" - {r}\n"
    st.text_area("Anomaly Explanations", explanation_text, height=200)

    st.subheader("Seasonal Appliance Usage")
    plot_seasonal_usage(df_original)

    efficiency = compute_efficiency(df_original)
    st.subheader("Efficiency Score (Lower variance = higher score)")
    st.write(efficiency.round(2))

    if st.button("Upload Forecast and Anomalies to Firebase"):
        upload_to_firebase(next_pred, light_change, fan_change, iron_change, future_df, anomalies)
        st.success("Uploaded forecast and anomalies to Firebase!")

    st.subheader("Export Reports")
    export_reports(future_df, anomalies, explanation_text)

if __name__ == "__main__":
    main()
