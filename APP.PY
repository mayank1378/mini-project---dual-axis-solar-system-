#pip install streamlit pandas numpy plotly scikit-learn tensorflow requests pytz
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime
import pytz

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow import keras
from tensorflow.keras import layers


# -----------------------------------------------------------------------------
# APP CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Solar AI Dashboard",
    page_icon="🔆",
    layout="wide"
)

st.title("🔆 Solar AI Dashboard")

# -----------------------------------------------------------------------------
# PAGE SELECTION
# -----------------------------------------------------------------------------
page = st.sidebar.radio(
    "Navigate",
    ["PV Power Predictor (FNN)", "Live Solar City Assistant"]
)

# =============================================================================
# PAGE 1: PV POWER PREDICTOR (FNN)
# =============================================================================
if page == "PV Power Predictor (FNN)":
    st.header("📈 PV Power Prediction using Feedforward Neural Network")

    # -------------------------------------------------------------------------
    # 1. LOAD DATA (CACHED)
    # -------------------------------------------------------------------------
    @st.cache_data
    def load_pv_data():
        weather_df = pd.read_csv("Plant_1_Weather_Sensor_Data.csv")
        generation_df = pd.read_csv("Plant_1_Generation_Data.csv")

        # Convert timestamps
        weather_df["DATE_TIME"] = pd.to_datetime(weather_df["DATE_TIME"], dayfirst=True)
        generation_df["DATE_TIME"] = pd.to_datetime(generation_df["DATE_TIME"], dayfirst=True)

        # Group generation by timestamp (sum across inverters)
        gen_grouped = generation_df.groupby("DATE_TIME")[["DC_POWER", "AC_POWER", "DAILY_YIELD", "TOTAL_YIELD"]].sum()

        # Align to weather timestamps
        gen_grouped = gen_grouped.reindex(weather_df["DATE_TIME"]).fillna(0)
        gen_grouped = gen_grouped.reset_index().rename(columns={"index": "DATE_TIME"})

        # Set index
        weather_df_idx = weather_df.set_index("DATE_TIME")
        gen_idx = gen_grouped.set_index("DATE_TIME")

        # Drop unnecessary columns
        weather_df_idx = weather_df_idx.drop(columns=["PLANT_ID", "SOURCE_KEY"], errors="ignore")
        gen_idx = gen_idx.drop(columns=["DC_POWER", "TOTAL_YIELD"], errors="ignore")

        # Merge
        df = pd.merge(gen_idx, weather_df_idx, left_index=True, right_index=True)
        df = df.reset_index().rename(columns={"index": "DATE_TIME"})
        df["DATE"] = df["DATE_TIME"].dt.date

        return weather_df, generation_df, df

    weather_df, generation_df, df = load_pv_data()

    st.subheader("Raw Data Samples")
    col1, col2 = st.columns(2)
    with col1:
        st.caption("🌦 Weather Data")
        st.dataframe(weather_df.head())
    with col2:
        st.caption("⚡ Generation Data")
        st.dataframe(generation_df.head())

    st.subheader("Merged Dataset Sample")
    st.dataframe(df.head())

    # -------------------------------------------------------------------------
    # 2. INTERACTIVE EDA
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📊 Exploratory Data Analysis (Interactive)")

    # AC Power over time (overall)
    st.markdown("**AC Power Over Time (Entire Period)**")
    fig_ac = px.line(
        df,
        x="DATE_TIME",
        y="AC_POWER",
        title="AC Power Over Time",
        labels={"DATE_TIME": "Time", "AC_POWER": "AC Power (kW)"}
    )
    st.plotly_chart(fig_ac, use_container_width=True)

    # Interactive day selector
    st.markdown("**Irradiation vs AC Power on Selected Day**")
    df_day = df.copy()
    dates_available = sorted(df_day["DATE"].unique())
    selected_date = st.selectbox("Select date for detailed view:", dates_available)

    day_data = df_day[df_day["DATE"] == selected_date]

    if not day_data.empty:
        fig_day = go.Figure()
        fig_day.add_trace(
            go.Scatter(
                x=day_data["DATE_TIME"],
                y=day_data["IRRADIATION"],
                mode="lines",
                name="Irradiation"
            )
        )
        fig_day.add_trace(
            go.Scatter(
                x=day_data["DATE_TIME"],
                y=day_data["AC_POWER"],
                mode="lines",
                name="AC Power",
                yaxis="y2"
            )
        )

        fig_day.update_layout(
            title=f"Irradiation and AC Power on {selected_date}",
            xaxis_title="Time",
            yaxis=dict(title="Irradiation (kW/m²)"),
            yaxis2=dict(
                title="AC Power (kW)",
                overlaying="y",
                side="right"
            ),
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig_day, use_container_width=True)

    # Scatter: AC Power vs Irradiation
    st.markdown("**AC Power vs Irradiation**")
    fig_scatter = px.scatter(
        df,
        x="IRRADIATION",
        y="AC_POWER",
        opacity=0.3,
        title="AC Power vs Irradiation",
        labels={"IRRADIATION": "Irradiation (kW/m²)", "AC_POWER": "AC Power (kW)"}
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Correlation heatmap
    st.markdown("**Feature Correlation Heatmap**")
    corr_cols = ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION", "AC_POWER"]
    corr = df[corr_cols].corr()
    fig_corr = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title="Feature Correlation Matrix"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # Distributions
    st.markdown("**Feature Distributions**")
    for col in ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]:
        fig_hist = px.histogram(
            df,
            x=col,
            nbins=30,
            title=f"Distribution of {col}",
            labels={col: col}
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # -------------------------------------------------------------------------
    # 3. TRAIN / TEST SPLIT BY DATE
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("🧠 Train Feedforward Neural Network")

    features = ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]
    target = "AC_POWER"

    # Prepare split
    df_model = df.copy().reset_index(drop=True)
    unique_dates = df_model["DATE"].drop_duplicates().values
    np.random.seed(42)
    np.random.shuffle(unique_dates)

    split_ratio = st.slider("Train/Test Split (by days)", 0.5, 0.9, 0.8, 0.05)
    split_index = int(len(unique_dates) * split_ratio)
    train_dates = unique_dates[:split_index]
    test_dates = unique_dates[split_index:]

    train_df = df_model[df_model["DATE"].isin(train_dates)].copy()
    test_df = df_model[df_model["DATE"].isin(test_dates)].copy()

    st.write(f"Train samples: **{len(train_df)}** | Test samples: **{len(test_df)}**")

    # Scaling
    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_train = X_scaler.fit_transform(train_df[features])
    y_train = y_scaler.fit_transform(train_df[[target]])

    X_test = X_scaler.transform(test_df[features])
    y_test = y_scaler.transform(test_df[[target]])

    # -------------------------------------------------------------------------
    # 4. MODEL DEFINITION + TRAINING
    # -------------------------------------------------------------------------
    epochs = st.number_input("Epochs", min_value=5, max_value=200, value=20, step=5)
    activation = st.selectbox("Activation function", ["sigmoid", "relu"], index=0)

    if st.button("Train FNN Model"):
        model = keras.Sequential()
        model.add(layers.Input(shape=(X_train.shape[1],)))
        model.add(layers.Dense(1, activation=activation))

        model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        with st.spinner("Training model..."):
            history = model.fit(
                X_train,
                y_train,
                epochs=epochs,
                validation_split=0.2,
                verbose=0,
            )

        # Evaluate
        test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)

        # Predictions (scaled)
        y_pred_scaled = model.predict(X_test).flatten()
        y_true_scaled = y_test.flatten()

        # Inverse scale
        y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_true = y_scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()

        mae_val = mean_absolute_error(y_true, y_pred)
        rmse_val = mean_squared_error(y_true, y_pred) ** 0.5
        r2_val = r2_score(y_true, y_pred)

        st.success("Model trained successfully.")
        st.write(f"**Test MAE (kW):** {mae_val:.4f}")
        st.write(f"**Test RMSE:** {rmse_val:.4f}")
        st.write(f"**Test R²:** {r2_val:.4f}")

        # Save to session state if you want to reuse later
        st.session_state["pv_model"] = model
        st.session_state["X_scaler"] = X_scaler
        st.session_state["y_scaler"] = y_scaler
        st.session_state["test_df"] = test_df
        st.session_state["y_true"] = y_true
        st.session_state["y_pred"] = y_pred

        # Plot predictions vs true
        st.markdown("### 🔍 Predictions vs True AC Power (Test Set)")

        fig_pred = go.Figure()
        fig_pred.add_trace(
            go.Scatter(
                y=y_true,
                mode="lines",
                name="True AC Power"
            )
        )
        fig_pred.add_trace(
            go.Scatter(
                y=y_pred,
                mode="lines",
                name="Predicted AC Power",
                opacity=0.7
            )
        )
        fig_pred.update_layout(
            xaxis_title="Sample Index",
            yaxis_title="AC Power (kW)",
            title="Model Predictions vs True AC Power",
        )
        st.plotly_chart(fig_pred, use_container_width=True)

        # Scatter true vs pred
        fig_scatter_pred = px.scatter(
            x=y_true,
            y=y_pred,
            labels={"x": "True AC Power (kW)", "y": "Predicted AC Power (kW)"},
            title="True vs Predicted AC Power"
        )
        fig_scatter_pred.add_shape(
            type="line",
            x0=min(y_true),
            y0=min(y_true),
            x1=max(y_true),
            y1=max(y_true),
            line=dict(dash="dash")
        )
        st.plotly_chart(fig_scatter_pred, use_container_width=True)

    st.markdown("---")
    st.info(
        "This page uses plant historical data (weather + generation) to train a Feedforward Neural Network "
        "that predicts AC power output from ambient temperature, module temperature, and irradiation."
    )


# =============================================================================
# PAGE 2: LIVE SOLAR CITY ASSISTANT
# =============================================================================
elif page == "Live Solar City Assistant":
    st.header("🌍 Live Solar City Assistant (Open-Meteo API)")

    GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
    SOLAR_URL = "https://api.open-meteo.com/v1/forecast"

    # -------------------------------
    # CORE FUNCTIONS
    # -------------------------------
    def geocode_city(city_name: str):
        params = {
            "name": city_name,
            "count": 1,
            "language": "en",
            "format": "json"
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
            "timezone": r["timezone"]
        }

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

        df = pd.DataFrame({
            "time": pd.to_datetime(times),
            "radiation_wm2": radiation,
        })
        df = df.dropna(subset=["radiation_wm2"]).reset_index(drop=True)
        return df

    def add_efficiency_column(df: pd.DataFrame):
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

    # -------------------------------
    # UI
    # -------------------------------
    st.write("Enter a city name to check current solar efficiency and best time today.")

    default_city = "Pune"
    city = st.text_input("Enter City Name", value=default_city)

    if st.button("Analyze Solar Output"):
        try:
            info = geocode_city(city)
            rad_df = get_today_radiation(info["lat"], info["lon"], info["timezone"])
            rad_df = add_efficiency_column(rad_df)

            now_local, matched_time, eff_now, rad_now = get_current_efficiency(
                rad_df,
                info["timezone"]
            )
            best_time, best_eff, best_rad = get_best_time_today(rad_df)

            st.subheader(f"📍 {info['name']}, {info['country']}")
            st.write(f"🌍 Coordinates: {info['lat']:.4f}, {info['lon']:.4f}")
            st.write(f"🕒 Local Time: {now_local}")

            st.markdown("---")
            st.subheader("☀️ Current Solar Status")
            st.write(f"Nearest Forecast Hour: **{matched_time}**")
            st.write(f"Current Relative Efficiency: **{eff_now*100:.1f}%**")
            st.write(f"Current Radiation: **{rad_now:.1f} W/m²**")

            st.markdown("---")
            st.subheader("🏆 Best Time Today")
            st.write(f"Best Hour: **{best_time}**")
            st.write(f"Best Efficiency: **{best_eff*100:.1f}%**")
            st.write(f"Best Radiation: **{best_rad:.1f} W/m²**")

            st.markdown("---")
            st.subheader("📈 Today's Solar Efficiency Curve")
            fig_eff = px.line(
                rad_df,
                x="time",
                y="efficiency_rel",
                labels={"time": "Time", "efficiency_rel": "Relative Efficiency"},
                title="Relative Solar Efficiency (0–1) Throughout Today"
            )
            st.plotly_chart(fig_eff, use_container_width=True)

            st.markdown("---")
            st.subheader("📊 Full Hourly Data")
            st.dataframe(rad_df)

        except Exception as e:
            st.error(str(e))

    st.info(
        "This page uses Open-Meteo's free API to estimate current and maximum solar radiation for any city, "
        "and shows when solar conditions are best for power generation."
    )
