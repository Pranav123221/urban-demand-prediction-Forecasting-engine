import streamlit as st
import pandas as pd

from src.prediction import predict_bike_demand


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Urban Demand Forecasting",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
<style>

/* ---------- Page ---------- */

.stApp {
    background-color: #f5f7fa;
}

.block-container {
    max-width: 1180px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* ---------- Hide Streamlit UI ---------- */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}


/* ---------- Hero ---------- */

/* ---------- Hero ---------- */

.hero-box {
    background: linear-gradient(135deg, #111827, #273449);
    padding: 18px 42px;
    border-radius: 24px 24px 0 0;
    margin-bottom: 0;
    color: #9ca3af;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2px;
}

h1 {
    color: #111827 !important;
    font-size: 42px !important;
    font-weight: 800 !important;
    margin-top: 10px !important;
    margin-bottom: 8px !important;
}

.hero-box + div {
    color: #6b7280;
}

.hero-small {
    color: #9ca3af;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2px;
}

.hero-title {
    color: white;
    font-size: 44px;
    font-weight: 800;
    margin-top: 8px;
}

.hero-description {
    color: #6b7280 !important;
    font-size: 16px !important;
    line-height: 1.6 !important;
    margin-bottom: 25px !important;
}


/* ---------- Section headers ---------- */

.section-heading {
    font-size: 21px;
    font-weight: 750;
    color: #111827;
    margin-top: 12px;
}

.section-description {
    color: #6b7280;
    font-size: 14px;
    margin-bottom: 18px;
}


/* ---------- Labels ---------- */

.stDateInput label,
.stNumberInput label,
.stSlider label,
.stSelectbox label {
    color: #374151 !important;
    font-weight: 600 !important;
}


/* ---------- Inputs ---------- */

.stDateInput input,
.stNumberInput input {
    border-radius: 10px !important;
}


/* ---------- Button ---------- */

.stButton > button {
    width: 100%;
    height: 54px;
    border-radius: 12px;
    background: #111827;
    color: white;
    border: none;
    font-size: 16px;
    font-weight: 700;
}

.stButton > button:hover {
    background: #374151;
    color: white;
}


/* ---------- Prediction ---------- */

.result-box {
    background: linear-gradient(135deg, #111827, #374151);
    padding: 34px;
    border-radius: 22px;
    text-align: center;
    margin-top: 25px;
    margin-bottom: 25px;
}

.result-label {
    color: #d1d5db;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

.result-number {
    color: white;
    font-size: 52px;
    font-weight: 800;
    margin: 8px 0;
}

.result-unit {
    color: #9ca3af;
    font-size: 14px;
}


/* ---------- Info cards ---------- */

.info-box {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 20px;
    min-height: 115px;
    box-shadow: 0 5px 18px rgba(0,0,0,0.035);
}

.info-title {
    font-weight: 700;
    color: #111827;
    margin-bottom: 7px;
}

.info-text {
    color: #6b7280;
    font-size: 13px;
    line-height: 1.5;
}


/* ---------- Footer ---------- */

.footer-text {
    text-align: center;
    color: #9ca3af;
    font-size: 12px;
    margin-top: 35px;
}

</style>
""",
    unsafe_allow_html=True
)


# =========================================================
# HERO
# =======================================================

st.markdown(
    """
    <div class="hero-box">
        MACHINE LEARNING • URBAN MOBILITY
    </div>
    """,
    unsafe_allow_html=True
)

st.title("Urban Demand Forecasting")

st.markdown(
    '<p style="color:#6b7280; font-size:16px; line-height:1.6;">'
    'Predict hourly bike rental demand using temporal, weather, '
    'seasonal, and operational factors powered by a machine learning model.'
    '</p>',
    unsafe_allow_html=True
)
# =========================================================
# DEMAND CONDITIONS
# =========================================================

st.markdown(
    '<div class="section-heading">📅 Demand Conditions</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">Enter the conditions for the hour you want to forecast.</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    date = st.date_input("Date")

    hour = st.slider(
        "Hour of Day",
        0,
        23,
        18
    )

    season = st.selectbox(
        "Season",
        [
            "Spring",
            "Summer",
            "Autumn",
            "Winter"
        ]
    )


with col2:

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=-30.0,
        max_value=50.0,
        value=25.0,
        step=0.5
    )

    humidity = st.slider(
        "Humidity (%)",
        0,
        100,
        60
    )

    dew_point = st.number_input(
        "Dew Point Temperature (°C)",
        min_value=-30.0,
        max_value=50.0,
        value=17.0,
        step=0.5
    )


with col3:

    wind_speed = st.number_input(
        "Wind Speed (m/s)",
        min_value=0.0,
        max_value=50.0,
        value=2.0,
        step=0.1
    )

    visibility = st.number_input(
        "Visibility (10m)",
        min_value=0,
        max_value=3000,
        value=1500,
        step=50
    )

    solar_radiation = st.number_input(
        "Solar Radiation (MJ/m²)",
        min_value=0.0,
        max_value=10.0,
        value=0.5,
        step=0.1
    )


st.divider()


# =========================================================
# WEATHER & OPERATIONAL CONDITIONS
# =========================================================

st.markdown(
    '<div class="section-heading">🌦️ Weather & Operational Conditions</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">Additional environmental and service conditions.</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    rainfall = st.number_input(
        "Rainfall (mm)",
        min_value=0.0,
        max_value=1000.0,
        value=0.0,
        step=0.1
    )


with col2:

    snowfall = st.number_input(
        "Snowfall (cm)",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=0.1
    )


with col3:

    holiday = st.selectbox(
        "Holiday",
        [
            "No Holiday",
            "Holiday"
        ]
    )


st.divider()


# =========================================================
# SERVICE AVAILABILITY
# =========================================================

st.markdown(
    '<div class="section-heading">🏙️ Service Availability</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">Specify whether the bike-sharing service is operating.</div>',
    unsafe_allow_html=True
)

functioning_day = st.selectbox(
    "Functioning Day",
    [
        "Yes",
        "No"
    ]
)


st.write("")


# =========================================================
# PREDICT
# =========================================================

predict_button = st.button(
    "🚲  Predict Bike Demand",
    type="primary"
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    input_data = pd.DataFrame(
        [
            {
                "Date": date.strftime("%d/%m/%Y"),
                "Hour": hour,
                "Temperature(°C)": temperature,
                "Humidity(%)": humidity,
                "Wind speed (m/s)": wind_speed,
                "Visibility (10m)": visibility,
                "Dew point temperature(°C)": dew_point,
                "Solar Radiation (MJ/m2)": solar_radiation,
                "Rainfall(mm)": rainfall,
                "Snowfall (cm)": snowfall,
                "Seasons": season,
                "Holiday": holiday,
                "Functioning Day": functioning_day
            }
        ]
    )

    try:

        prediction = predict_bike_demand(input_data)

        prediction = max(0, prediction)

        # Result
        st.markdown(
            f"""
            <div class="result-box">

                <div class="result-label">
                    Predicted Hourly Demand
                </div>

                <div class="result-number">
                    {round(prediction):,}
                </div>

                <div class="result-unit">
                    estimated bike rentals
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        # Supporting information
        info1, info2, info3 = st.columns(3)


        with info1:

            st.markdown(
                """
                <div class="info-box">

                    <div class="info-title">
                        🕐 Forecast Time
                    </div>

                    <div class="info-text">
                        The prediction is generated for the
                        selected date and hour.
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        with info2:

            st.markdown(
                """
                <div class="info-box">

                    <div class="info-title">
                        🌡️ Weather Context
                    </div>

                    <div class="info-text">
                        Temperature, humidity, visibility,
                        wind and precipitation are considered.
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        with info3:

            st.markdown(
                """
                <div class="info-box">

                    <div class="info-title">
                        🤖 ML Prediction
                    </div>

                    <div class="info-text">
                        Prediction generated using the trained
                        Random Forest regression pipeline.
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


    except Exception as e:

        st.error(
            "Prediction could not be generated."
        )

        st.exception(e)


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer-text">
        Urban Demand Forecasting • Machine Learning Project
    </div>
    """,
    unsafe_allow_html=True
)