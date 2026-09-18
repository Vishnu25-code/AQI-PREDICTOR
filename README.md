# 🍃 EcoGuard: AI-Powered Real-Time Air Quality Intelligence for India

EcoGuard is a professional, real-time AI monitoring system designed to provide hyper-local air quality insights for any city in India. By combining real-time sensor data from the World Air Quality Index (WAQI) with a trained Machine Learning model, EcoGuard validates sensor accuracy and provides actionable health precautions.

---

## 🚀 Tech Stack

### **Frontend & Visualization**
- **Streamlit**: For the dynamic, interactive web dashboard.
- **Plotly**: For interactive line charts (Future Forecast) and radar charts (Pollutant Signature).
- **Seaborn & Matplotlib**: For detailed statistical correlation heatmaps.
- **Custom CSS**: Implementation of Glassmorphism and a deep-dark neon theme.

### **Backend & Machine Learning**
- **Python 3.x**: Core programming language.
- **Scikit-Learn**: Implementation of the `RandomForestRegressor` for AQI prediction.
- **Pandas & NumPy**: Data manipulation and mathematical calibration.
- **Joblib**: Model serialization for fast loading.
- **Requests**: REST API integration with WAQI.

---

## ⚙️ Project Workflow

The application follows a sophisticated data pipeline to ensure accuracy:

1. **Data Acquisition**: 
   - User inputs a city name.
   - The app queries the **WAQI API**, forcing a search within the Indian territory to prevent location drift.
   
2. **Mathematical Validation (CPCB Rules)**:
   - The app doesn't just trust the reported AQI. It extracts raw pollutants (PM2.5, PM10, O3).
   - It applies **Linear Interpolation** based on **CPCB (Central Pollution Control Board)** breakpoints to calculate the "True AQI".

3. **AI-Driven Calibration**:
   - The raw IAQI indices are normalized using the mean values from the historical training dataset (`aqi_data.csv`).
   - These calibrated features are fed into the **RandomForest Model** to predict the AQI.

4. **Intelligence Layer**:
   - **Variance Check**: If the Sensor AQI and AI Prediction differ by > 50, a high-variance alert is triggered.
   - **Health Insight**: The system identifies the "Dominant Pollutant" and suggests a specific medical precaution (e.g., "Use N95 mask").

5. **Visual Rendering**:
   - The final results are rendered in a tabbed, neon-dark UI with real-time updates.

---

## 🛠️ Installation & Setup

### **1. Clone the Repository**
```bash
git clone <repository-url>
cd "REAL TIME AQI"
```

### **2. Install Dependencies**
It is recommended to use a virtual environment:
```bash
pip install streamlit pandas numpy scikit-learn joblib plotly seaborn matplotlib requests
```

### **3. Train the Model (First Time Only)**
Before running the app, ensure the model is trained:
```bash
python main.py
```
*This will generate the `aqi_model.pkl` file.*

### **4. Launch the Dashboard**
```bash
streamlit run app.py
```

---

## 📖 How to Use

1. **City Input**: Enter any Indian city (e.g., "Delhi", "Mumbai", "Bangalore") in the sidebar.
2. **API Token**: Use the default `demo` token or enter your personal WAQI token for higher stability.
3. **Explore Tabs**:
   - **AI Verification**: Compare sensor data vs. AI predictions and view health recommendations.
   - **Future Forecast**: See the predicted 7-day AQI trend.
   - **Detailed Analysis**: Analyze the pollutant signature via radar charts and see feature correlations.

---

## 📈 Model Performance
- **Algorithm**: RandomForestRegressor
- **Features**: PM2.5, PM10, NO2, SO2, CO, O3
- **Evaluation**: High R² score on the Delhi historical dataset, now calibrated for pan-India real-time usage.
