import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import joblib
from datetime import datetime, timedelta

# --- CONFIGURATION ---
st.set_page_config(
    page_title="EcoGuard: AI Real-Time AQI Monitor",
    page_icon="🍃",
    layout="wide"
)

# File Paths
MODEL_PATH = r"C:\Users\chauh\OneDrive\Desktop\REAL TIME AQI\aqi_model.pkl"
DATA_PATH = r"C:\Users\chauh\OneDrive\Desktop\REAL TIME AQI\aqi_data.csv"

# --- DARK MODE DYNAMIC STYLE ---
st.markdown("""
    <style>
    .stApp {
        background: radial-gradient(circle at top left, #1e293b, #0f172a, #020617);
        color: #f8fafc;
    }
    .main-title {
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        color: #4ade80;
        text-align: center;
        font-weight: 800;
        font-size: 3.5rem;
        margin-bottom: 0px;
        text-shadow: 0px 0px 15px rgba(74, 222, 128, 0.5);
    }
    .sub-title {
        text-align: center;
        color: #94a3b8;
        font-size: 1.3rem;
        margin-bottom: 40px;
        font-weight: 300;
    }
    [data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.7);
        padding: 20px;
        border-radius: 20px;
        border: 1px solid rgba(74, 222, 128, 0.3);
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        transition: all 0.3s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: scale(1.05);
        border-color: #4ade80;
        box-shadow: 0 0 20px rgba(74, 222, 128, 0.3);
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 800 !important;
    }
    .precaution-box {
        padding: 30px;
        border-radius: 25px;
        text-align: center;
        font-weight: bold;
        font-size: 24px;
        margin: 20px 0px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        animation: slideUp 1s ease-out;
        border: 2px solid rgba(255,255,255,0.1);
    }
    @keyframes slideUp {
        0% { transform: translateY(20px); opacity: 0; }
        100% { transform: translateY(0); opacity: 1; }
    }
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #1e293b;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        justify-content: center;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e293b;
        color: #94a3b8;
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #4ade80 !important;
        color: #0f172a !important;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- HELPERS ---
def load_model():
    """Loads the trained ML model from disk."""
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.sidebar.error(f"Model Load Error: {e}")
        return None

def get_aqi_precautions(aqi):
    if aqi <= 50:
        return "🟢 Good", "Air quality is satisfying. Enjoy your day! 🌳", "#065f46", "😊"
    elif aqi <= 100:
        return "🟡 Satisfactory", "Air quality is acceptable. Sensitive people should be cautious. 🚶", "#854d0e", "😐"
    elif aqi <= 200:
        return "🟠 Moderate", "Breathe carefully. Avoid prolonged outdoor exercise. 😷", "#9a3412", "😟"
    elif aqi <= 300:
        return "🔴 Poor", "Unhealthy for sensitive groups. Wear a mask! 😷", "#991b1b", "🤒"
    elif aqi <= 400:
        return "🟣 Very Poor", "Health alert! Minimize outdoor activity. 🏠", "#5b21b6", "😫"
    else:
        return "⚫ Severe", "Emergency! Stay indoors. Use air purifiers. 🚨", "#450a0a", "😱"

def ensure_float(val, default):
    if isinstance(val, dict):
        for v in val.values():
            if isinstance(v, (int, float)):
                return float(v)
    try:
        return float(val) if val is not None else default
    except (ValueError, TypeError):
        return default

def calibrate_features(iaqi_data):
    """
    Aligns real-time IAQI indices to the distribution of the training dataset.
    This prevents the 'Huge Gap' by normalizing the input to the model's expected scale.
    """
    try:
        # 1. Load training means to establish a baseline
        df_train = pd.read_csv(DATA_PATH)
        train_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
        train_means = df_train[train_cols].mean()

        # 2. Define the mapping between API keys and Training columns
        mapping = {
            "PM2_5": "PM2.5",
            "PM10": "PM10",
            "NO2": "NO2",
            "SO2": "SO2",
            "CO": "CO",
            "O3": "O3"
        }

        # 3. Dynamic Scaling: (Current Value / 100) * Training Mean
        calibrated = []
        for api_key, train_col in mapping.items():
            current_val = iaqi_data.get(api_key, 100)
            baseline_mean = train_means[train_col]
            scaled_val = (current_val / 100.0) * baseline_mean
            calibrated.append(scaled_val)

        return np.array([calibrated])
    except Exception as e:
        # Fallback to training means if CSV loading fails
        return np.array([[114.5, 224.9, 172.8, 237.4, 2.1, 197.8]])

def get_ai_health_insight(data):
    pollutants = {
        "PM2.5": (data['PM2_5'], "Fine particulate matter (PM2.5) is high. This can penetrate deep into lungs. Recommendation: Use an N95 mask and avoid outdoor jogging."),
        "PM10": (data['PM10'], "Coarse particles (PM10) are elevated. This may irritate the throat and nose. Recommendation: Keep windows closed."),
        "O3": (data['O3'], "Ground-level Ozone (O3) is high. This can trigger asthma. Recommendation: Limit outdoor activity during peak sunlight hours."),
        "NO2": (data['NO2'], "Nitrogen Dioxide (NO2) levels are concerning. This is often due to traffic emissions. Recommendation: Avoid walking near main roads."),
        "SO2": (data['SO2'], "Sulfur Dioxide (SO2) is elevated. This can cause respiratory distress. Recommendation: Stay indoors if you have asthma."),
        "CO": (data['CO'], "Carbon Monoxide (CO) levels are high. This can reduce oxygen delivery to organs. Recommendation: Ensure proper ventilation in your home.")
    }
    dominant_pollutant = max(pollutants, key=lambda k: pollutants[k][0])
    return pollutants[dominant_pollutant][1], dominant_pollutant

def fetch_realtime_data(city, token="demo"):
    """Fetches real-time AQI. Strictly prevents 'Shanghai' drift for Indian cities."""
    # Try multiple query formats to find the best match for India
    queries = [f"{city},India", city]

    for query in queries:
        url = f"https://api.waqi.info/feed/{query}/?token={token}"
        try:
            response = requests.get(url)
            data = response.json()
            if data['status'] == 'ok':
                city_name = data['data']['city']['name']
                # CRITICAL FIX: Prevent Shanghai drift.
                # If the city name contains 'Shanghai' but we didn't ask for it, ignore this result.
                if "Shanghai" in city_name and "shanghai" not in city.lower():
                    continue

                aqi_val = data['data']['aqi']
                aqi = ensure_float(aqi_val, 0)
                iaqi = data['data'].get('iaqi', {})
                return {
                    "aqi": aqi,
                    "PM2_5": ensure_float(iaqi.get('pm25'), 210),
                    "PM10": ensure_float(iaqi.get('pm10'), 420),
                    "NO2": ensure_float(iaqi.get('no2'), 180),
                    "SO2": ensure_float(iaqi.get('so2'), 70),
                    "CO": ensure_float(iaqi.get('co'), 1.5),
                    "O3": ensure_float(iaqi.get('o3'), 360),
                    "city": city_name
                }
        except Exception:
            continue
    return None

def predict_future_aqi(current_aqi, city_name):
    df_hist = pd.read_csv(DATA_PATH)
    df_hist['Date'] = pd.to_datetime(df_hist['Date'], errors='coerce')
    current_month = datetime.now().month
    monthly_avg = df_hist[df_hist['Date'].dt.month == current_month]['AQI'].mean()
    trend = (monthly_avg - current_aqi) / 30
    future_dates = [datetime.now() + timedelta(days=i) for i in range(1, 8)]
    future_values = [current_aqi + (trend * i) for i in range(1, 8)]
    return future_dates, future_values

def main():
    st.markdown('<p class="main-title">🍃 EcoGuard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">AI-Powered Real-Time Air Quality Intelligence for India</p>', unsafe_allow_html=True)

    st.sidebar.header("⚙️ Dashboard Control")
    city_input = st.sidebar.text_input("📍 Enter City Name", "Delhi", help="Enter any city in India")
    api_token = st.sidebar.text_input("🔑 WAQI API Token", "demo", type="password")
    show_heatmap = st.sidebar.checkbox("📊 Show Correlation Heatmap", value=True)

    # Load model without caching to avoid state issues during debugging
    model = load_model()

    if city_input:
        with st.spinner(f"📡 Synchronizing with air quality sensors in {city_input}..."):
            data = fetch_realtime_data(city_input, api_token)

        if data:
            st.markdown("### 🌡️ Current Air Status")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("AQI Index", int(data['aqi']))
            col2.metric("PM 2.5", f"{data['PM2_5']:.1f}")
            col3.metric("PM 10", f"{data['PM10']:.1f}")
            col4.metric("O3 Level", f"{data['O3']:.1f}")

            status, text, color, emoji = get_aqi_precautions(data['aqi'])
            st.markdown(f"""
                <div class="precaution-box" style="background-color: {color}; border: 2px solid rgba(255,255,255,0.2); color: white;">
                    <span style="font-size: 60px;">{emoji}</span><br>
                    <strong style="font-size: 32px;">{status}</strong><br>
                    <span style="font-size: 20px; font-weight: normal; opacity: 0.9;">{text}</span>
                </div>
                """, unsafe_allow_html=True)

            tab1, tab2, tab3 = st.tabs(["🤖 AI Verification", "🔮 Future Forecast", "📊 Detailed Analysis"])

            with tab1:
                st.subheader("🤖 AI-Driven Health Intelligence")
                if model:
                    features = calibrate_features(data)
                    ai_pred = model.predict(features)[0]

                    c1, c2 = st.columns(2)
                    c1.metric("Sensor Reported AQI", int(data['aqi']))
                    c2.metric("AI Model Predicted AQI", int(ai_pred))

                    diff = abs(data['aqi'] - ai_pred)
                    if diff < 50:
                        st.success(f"✅ AI Validation: Consistent with sensor data. (Variance: {diff:.2f})")
                    else:
                        st.info(f"ℹ️ AI Insight: Local conditions may be causing variance. (Variance: {diff:.2f})")

                    st.markdown("---")
                    st.markdown("### 🩺 AI Health Recommendation")
                    insight, pollutant = get_ai_health_insight(data)
                    st.markdown(f"""
                        <div style="background: rgba(74, 222, 128, 0.1); padding: 20px; border-radius: 15px; border-left: 5px solid #4ade80;">
                            <strong style="color: #4ade80; font-size: 18px;">Dominant Pollutant: {pollutant}</strong><br>
                            <p style="font-size: 16px; margin-top: 10px;">{insight}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.error("AI Model file not found. Please run main.py first to train the model.")

            with tab2:
                st.subheader("7-Day AQI Trend Prediction")
                dates, values = predict_future_aqi(data['aqi'], city_input)
                fig_forecast = go.Figure()
                fig_forecast.add_trace(go.Scatter(x=dates, y=values, mode='lines+markers',
                                                 line=dict(color='#4ade80', width=4),
                                                 marker=dict(size=10, color='#ef4444')))
                fig_forecast.update_layout(
                    title=f"Forecast for {data['city']}",
                    xaxis_title="Date",
                    yaxis_title="AQI",
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor='#1e293b')
                )
                st.plotly_chart(fig_forecast, use_container_width=True)

            with tab3:
                st.subheader("Pollutant Signature")
                categories = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
                values = [data['PM2_5'], data['PM10'], data['NO2'], data['SO2'], data['CO'], data['O3']]

                fig_radar = go.Figure(data=go.Scatterpolar(
                    r=values, theta=categories, fill='toself', line_color='#4ade80'
                ))
                fig_radar.update_layout(
                    template="plotly_dark",
                    polar=dict(radialaxis=dict(visible=True, gridcolor='#1e293b')),
                    showlegend=False,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_radar, use_container_width=True)

                if show_heatmap:
                    st.markdown("---")
                    st.subheader("Correlation Analysis")
                    df_hist = pd.read_csv(DATA_PATH)
                    cols = ['PM2.5','PM10','NO2','SO2','CO','O3','AQI']
                    numeric_df = df_hist[cols].fillna(df_hist[cols].mean())
                    plt.style.use('dark_background')
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
                    st.pyplot(fig)
                    plt.close()
        else:
            st.error("Could not find air quality data for this city. Please check the spelling or try another city.")

if __name__ == "__main__":
    main()
