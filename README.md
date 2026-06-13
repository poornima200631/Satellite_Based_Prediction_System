# 🌍 AeroSatellite Predict
**Air Quality Forecasting Using Spatial Satellite & Weather Intelligence**

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.57-FF4B4B?logo=streamlit)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-success?logo=scikit-learn)
![NASA MODIS](https://img.shields.io/badge/Satellite-NASA%20MODIS-black)
![ESA Sentinel-5P](https://img.shields.io/badge/Satellite-ESA%20Sentinel--5P-blue)

AeroSatellite Predict is a revolutionary **"Zero-Hardware"** air quality intelligence platform. By fusing real-time spatial data from NASA and ESA satellites with meteorological trends, we have transformed satellite imagery into a highly accurate, software-defined PM2.5 sensor. 

Powered by a hyper-parameter tuned **XGBoost Regressor** delivering an unprecedented **97.3% R² accuracy**, our interactive dashboard brings hyper-local pollution forecasting and health alerts to millions living in unmonitored regions—completely bypassing the need for expensive physical infrastructure.

---

## 🚀 Key Features & Novelty

* 🛰️ **Space-Data Fusion (The Game Changer):** Instead of relying only on basic weather data, our pipeline extracts **Aerosol Optical Depth (AOD)** from NASA MODIS and **Tropospheric NO2 Trace Gases** from ESA Sentinel-5P via the Google Earth Engine API.
* 🤖 **High-Accuracy AI Engine:** Optimized XGBoost model handles complex non-linear interactions between satellite inputs and weather, outperforming standard models.
* ⏳ **Temporal Intelligence:** Features like 3-day/7-day rolling averages and lag features capture pollution persistence and atmospheric memory.
* 📊 **Interactive Analytics Dashboard:** A beautiful, real-time Streamlit dashboard providing trend analysis, seasonal insights, and instant PM2.5 forecasting.

---

## 📈 Model Performance & Validation

We conducted rigorous training and validation, eliminating spatial-temporal leakage and tuning hyperparameters via GridSearchCV. The **Feature Importance** analysis proved that Satellite AOD was the single most powerful predictor (81% importance).

| Metric | Score | Impact |
| :--- | :---: | :--- |
| **R² Score** | `0.973` | Near-perfect variance capture; extremely high accuracy. |
| **RMSE** | `2.66 µg/m³` | Extremely low average prediction error. |
| **MAE** | `2.03 µg/m³` | High reliability for real-world health advisory warnings. |

---

## 🛠️ Technology Stack

* **Data Engineering & Extraction:** Google Earth Engine (GEE API), Geopy, Pandas, NumPy
* **Machine Learning:** Scikit-Learn, XGBoost Regressor
* **Web Framework & UI:** Streamlit, Plotly (Interactive Visualizations)
* **Data Sources:** 
  * CPCB (Historical Ground Truth)
  * NASA MCD19A2 (MODIS AOD)
  * ESA Copernicus S5P (TROPOMI NO2)
  * ERA5 (Meteorology)

---

## 🖥️ How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/Priyakatariya/Satellite_Based_Prediction_System.git
cd Satellite_Based_Prediction_System
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the AI Dashboard
```bash
streamlit run main.py
```
> The dashboard will automatically launch in your default web browser at `http://localhost:8501`.

---

## 📂 Project Structure

```text
Satellite_Based_Prediction_System/
│
├── data/
│   └── processed/
│       ├── model_ready_data_satellite.csv   # Final dataset fused with satellite data
│       └── location_coords.json             # Geocoded latitudes & longitudes
│
├── models/
│   ├── xgboost_satellite.pkl                # Trained 97.3% Accuracy Model
│   ├── scaler_satellite.pkl                 # StandardScaler for inference
│   └── encoder.pkl                          # Label encoders for regions
│
├── reports/                                 # Project documentation & presentations
│
├── src/
│   ├── fetch_satellite_data.py              # GEE API extraction logic
│   ├── geocode_locations.py                 # Lat/Lon coordinate mapping
│   └── xgboost_model.py                     # ML Pipeline & Model Training Script
│
├── test_cases.py                            # Testing scripts
├── test_predict.py                          # Prediction test scripts
├── main.py                                  # Live Streamlit Web Application
├── requirements.txt                         # Dependencies
└── README.md                                # Project Documentation
```

---

## 🌟 Future Roadmap
- **Deep Learning Integration:** Implement LSTM & GRU architectures for multi-day advanced forecasting.
- **Automated Pipelines:** Schedule Apache Airflow DAGs for 24/7 autonomous satellite extraction.
- **Healthcare Integration:** Develop open APIs for medical institutions to trigger asthma alerts based on our satellite-predicted PM2.5 levels.

<br>
<p align="center">
  <b>Built with ❤️ for a cleaner, breathable future.</b>
</p>