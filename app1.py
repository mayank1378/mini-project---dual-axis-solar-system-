import pandas as pd
import numpy as np
import requests
import pytz
from datetime import datetime

import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor

import plotly.express as px
import plotly.graph_objects as go


# -------------------------------
# CONFIG
# -------------------------------
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
SOLAR_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_CSV = r"C:\Users\mayan\.cache\kagglehub\datasets\chitwanmanchanda\weather-api-data\versions\1\data\BTECH\Plant_1_Weather_Sensor_Data.csv"
GEN_CSV     = r"C:\Users\mayan\.cache\kagglehub\datasets\chitwanmanchanda\weather-api-data\versions\1\data\BTECH\Plant_1_Generation_Data.csv"


# ============================================================
# UTILS: DATA LOADING
# ============================================================

@st.cache_data
def load_pv_data():
    weather_df = pd.read_csv(WEATHER_CSV)
    generation_df = pd.read_csv(GEN_CSV)

    weather_df["DATE_TIME"] = pd.to_datetime(weather_df["DATE_TIME"], dayfirst=True)
    generation_df["DATE_TIME"] = pd.to_datetime(generation_df["DATE_TIME"], dayfirst=True)

    # Group generation by timestamp and sum across inverters
    generation_grouped = generation_df.groupby("DATE_TIME")[
        ["DC_POWER", "AC_POWER", "DAILY_YIELD", "TOTAL_YIELD"]
    ].sum()

    # Align to weather timestamps
    generation_grouped = generation_grouped.reindex(weather_df["DATE_TIME"]).fillna(0)
    generation_grouped = generation_grouped.reset_index().rename(columns={"index": "DATE_TIME"})

    # Index on DATE_TIME
    weather_df_idx = weather_df.set_index("DATE_TIME")
    generation_idx = generation_grouped.set_index("DATE_TIME")

    # Remove unnecessary columns
    weather_df_idx = weather_df_idx.drop(columns=["PLANT_ID", "SOURCE_KEY"], errors="ignore")
    generation_idx = generation_idx.drop(columns=["DC_POWER", "TOTAL_YIELD"], errors="ignore")

    # Merge
    df = pd.merge(
        generation_idx.reset_index(),
        weather_df_idx.reset_index(),
        on="DATE_TIME",
        how="inner",
        suffixes=("_gen", "_weather"),
    )

    df["DATE"] = df["DATE_TIME"].dt.date
    return df


@st.cache_data
def geocode_city(city_name: str):
    params = {
        "name": city_name,
        "count": 1,
        "language": "en",
        "format": "json",
    }
    resp = requests.get(GEOCODE_URL, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if "results" not in data or len(data["results"]) == 0:
        raise ValueError("City not found.")
    r = data["results"][0]
    return {
        "name": r["name"],
        "country": r.get("country", ""),
        "lat": r["latitude"],
        "lon": r["longitude"],
        "timezone": r["timezone"],
    }


@st.cache_data
def get_today_radiation(lat: float, lon: float, timezone: str):
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "shortwave_radiation",
        "timezone": timezone,
        "forecast_days": 1,
    }
    resp = requests.get(SOLAR_URL, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if "hourly" not in data or "shortwave_radiation" not in data["hourly"]:
        raise ValueError("Solar radiation data not available.")

    times = data["hourly"]["time"]
    radiation = data["hourly"]["shortwave_radiation"]

    df = pd.DataFrame({"time": pd.to_datetime(times), "radiation_wm2": radiation})
    df = df.dropna(subset=["radiation_wm2"]).reset_index(drop=True)

    max_val = df["radiation_wm2"].max()
    if max_val > 0:
        df["efficiency_rel"] = df["radiation_wm2"] / max_val
    else:
        df["efficiency_rel"] = 0.0

    return df


def get_local_now(timezone_name: str):
    tz = pytz.timezone(timezone_name)
    return datetime.now(tz)


def get_current_efficiency(df: pd.DataFrame, timezone_name: str):
    now_local = get_local_now(timezone_name)
    now_naive = now_local.replace(tzinfo=None)

    diffs = (df["time"] - now_naive).abs()
    idx = diffs.idxmin()
    row = df.loc[idx]

    return now_local, row["time"], float(row["efficiency_rel"]), float(row["radiation_wm2"])


def get_best_time_today(df: pd.DataFrame):
    idx = df["radiation_wm2"].idxmax()
    row = df.loc[idx]
    return row["time"], float(row["efficiency_rel"]), float(row["radiation_wm2"])


# ============================================================
# PAGE 1: DUAL-AXIS SOLAR PANEL TRACKING – POWER PREDICTOR
# ============================================================

def page_pv_predictor():
    st.title("⚡ Dual-Axis Solar Panel Tracking – Power Prediction Module")

    st.markdown(
        """
        This module represents the **AI side** of a dual-axis solar panel tracking system.
        
        Using real plant data (ambient temperature, module temperature, and irradiation)
        along with measured AC power output, we:
        
        - Align and merge data from all inverters and weather sensors  
        - Analyse how tracking + weather affect AC power generation  
        - Train a **Feedforward Neural Network (MLPRegressor)**  
          to predict **AC Power** for a dual-axis PV plant under given conditions.
        """
    )

    df = load_pv_data()

    st.subheader("Merged plant + weather data (sample)")
    st.dataframe(df.head(20))

    # ------------ EDA ------------
    st.markdown("---")
    st.subheader("📈 AC Power Over Time (Tracked PV Plant)")
    fig1 = px.line(df, x="DATE_TIME", y="AC_POWER", title="AC Power Over Time – Dual-Axis PV Plant")
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("### 🔍 Explore a Specific Day (Tracking Behaviour vs Power)")
    unique_days = sorted(df["DATE"].unique())
    selected_day = st.selectbox("Select Date", options=unique_days)

    day_data = df[df["DATE"] == selected_day].copy()

    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=day_data["DATE_TIME"],
            y=day_data["IRRADIATION"],
            name="Irradiation",
            yaxis="y1",
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=day_data["DATE_TIME"],
            y=day_data["AC_POWER"],
            name="AC Power",
            yaxis="y2",
        )
    )
    fig2.update_layout(
        title=f"Irradiation vs AC Power on {selected_day} (Dual-Axis Tracking)",
        xaxis_title="Time",
        yaxis=dict(title="Irradiation", side="left"),
        yaxis2=dict(
            title="AC Power (kW)",
            overlaying="y",
            side="right",
        ),
        legend=dict(x=0.01, y=0.99),
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("☀️ AC Power vs Irradiation (Effect of Tracking + Sunlight)")
    fig3 = px.scatter(
        df,
        x="IRRADIATION",
        y="AC_POWER",
        opacity=0.4,
        title="AC Power vs Irradiation – Dual-Axis PV System",
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Correlation of Weather Features with AC Power")
    corr_cols = ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION", "AC_POWER"]
    corr_matrix = df[corr_cols].corr()
    fig4 = px.imshow(corr_matrix, text_auto=True, title="Correlation Matrix – Dual-Axis PV Plant", aspect="auto")
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    st.subheader("📦 Weather Feature Distributions (Dual-Axis System Inputs)")
    fig5 = px.histogram(df, x="AMBIENT_TEMPERATURE", nbins=30, title="Ambient Temperature Distribution")
    fig6 = px.histogram(df, x="MODULE_TEMPERATURE", nbins=30, title="Module Temperature Distribution")
    fig7 = px.histogram(df, x="IRRADIATION", nbins=30, title="Irradiation Distribution")
    st.plotly_chart(fig5, use_container_width=True)
    st.plotly_chart(fig6, use_container_width=True)
    st.plotly_chart(fig7, use_container_width=True)

    # ------------ MODEL PREP ------------
    st.markdown("---")
    st.subheader("🧬 Train/Test Split & Scaling for Dual-Axis Power Prediction")

    df_model = df.copy().reset_index(drop=True)
    unique_dates = df_model["DATE"].drop_duplicates().values

    rng = np.random.default_rng(seed=42)
    rng.shuffle(unique_dates)

    split_ratio = 0.8
    split_index = int(len(unique_dates) * split_ratio)
    train_dates = unique_dates[:split_index]
    test_dates = unique_dates[split_index:]

    train_df = df_model[df_model["DATE"].isin(train_dates)]
    test_df = df_model[df_model["DATE"].isin(test_dates)]

    st.write(f"Train samples: **{len(train_df)}**, Test samples: **{len(test_df)}**")

    features = ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]
    target = "AC_POWER"

    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_train = X_scaler.fit_transform(train_df[features])
    y_train = y_scaler.fit_transform(train_df[[target]])

    X_test = X_scaler.transform(test_df[features])
    y_test = y_scaler.transform(test_df[[target]])

    st.write(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    # ------------ MODEL TRAINING ------------
    st.markdown("---")
    st.subheader("🧠 Train MLPRegressor – Dual-Axis PV Power Model")

    hidden_units = st.slider("Hidden Units", min_value=4, max_value=64, value=16, step=4)
    max_iter = st.slider("Max Iterations", min_value=100, max_value=1000, value=300, step=100)

    if st.button("Train Dual-Axis Power Model"):
        with st.spinner("Training MLPRegressor for dual-axis power prediction..."):
            mlp = MLPRegressor(
                hidden_layer_sizes=(hidden_units,),
                activation="relu",
                solver="adam",
                max_iter=max_iter,
                random_state=42,
            )
            mlp.fit(X_train, y_train.ravel())

        st.success("Dual-axis PV power model trained successfully!")

        # Predictions
        y_pred_scaled = mlp.predict(X_test)
        y_true_scaled = y_test.ravel()

        y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_true = y_scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()

        mae_val = mean_absolute_error(y_true, y_pred)
        rmse_val = mean_squared_error(y_true, y_pred) ** 0.5
        r2_val = r2_score(y_true, y_pred)

        st.markdown("### 📊 Test Performance – Dual-Axis Power Predictor")
        st.write(f"**MAE (kW)**: {mae_val:.4f}")
        st.write(f"**RMSE**: {rmse_val:.4f}")
        st.write(f"**R²**: {r2_val:.4f}")

        # Predictions vs True
        st.markdown("### 🔍 True vs Predicted AC Power (Dual-Axis System, Sample)")
        sample_n = min(500, len(y_true))
        idxs = np.arange(sample_n)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=idxs, y=y_true[:sample_n], name="True"))
        fig_pred.add_trace(go.Scatter(x=idxs, y=y_pred[:sample_n], name="Predicted"))
        fig_pred.update_layout(
            title="True vs Predicted AC Power – Dual-Axis PV Plant (first 500 test samples)",
            xaxis_title="Sample Index",
            yaxis_title="AC Power (kW)",
        )
        st.plotly_chart(fig_pred, use_container_width=True)


# ============================================================
# PAGE 2: LIVE SOLAR CITY ASSISTANT  (UNCHANGED)
# ============================================================

def page_solar_assistant():
    st.title("🌞 Live Solar City Assistant")
    st.write("Enter a city name to check current solar efficiency and best time for generation **today**.")

    city = st.text_input("Enter City Name", value="Pune")

    if st.button("Analyze Solar Output"):
        try:
            info = geocode_city(city)

            rad_df = get_today_radiation(info["lat"], info["lon"], info["timezone"])
            now_local, matched_time, eff_now, rad_now = get_current_efficiency(rad_df, info["timezone"])
            best_time, best_eff, best_rad = get_best_time_today(rad_df)

            st.subheader(f"📍 {info['name']}, {info['country']}")
            st.write(f"🌍 Coordinates: {info['lat']:.4f}, {info['lon']:.4f}")
            st.write(f"🕒 Local Time: {now_local}")

            st.markdown("---")
            st.subheader("☀️ Current Solar Status")
            st.write(f"Nearest Forecast Hour: **{matched_time}**")
            st.write(f"Current Relative Efficiency: **{eff_now * 100:.1f}%**")
            st.write(f"Current Radiation: **{rad_now:.1f} W/m²**")

            st.markdown("---")
            st.subheader("🏆 Best Time Today")
            st.write(f"Best Hour: **{best_time}**")
            st.write(f"Best Efficiency: **{best_eff * 100:.1f}%**")
            st.write(f"Best Radiation: **{best_rad:.1f} W/m²**")

            st.markdown("---")
            st.subheader("📈 Today's Solar Efficiency Curve")
            fig_eff = px.line(rad_df, x="time", y="efficiency_rel", title="Relative Solar Efficiency Today")
            st.plotly_chart(fig_eff, use_container_width=True)

            st.markdown("---")
            st.subheader("📊 Full Hourly Data")
            st.dataframe(rad_df)

        except Exception as e:
            st.error(str(e))


# ============================================================
# MAIN APP
# ============================================================

def main():
    st.set_page_config(page_title="Dual-Axis Solar Tracking – AI App", page_icon="🔆", layout="wide")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "Dual-Axis Tracker – Power Prediction",
            "Live Solar City Assistant",
        ],
    )

    if page == "Dual-Axis Tracker – Power Prediction":
        page_pv_predictor()
    else:
        page_solar_assistant()


if __name__ == "__main__":
    main()
