# 🚲 Urban Demand Prediction & Forecasting Engine

### Predicting Hourly Urban Bike Rental Demand Using Machine Learning

<p align="center">
  <b>An end-to-end machine learning system for understanding and predicting urban bike rental demand using temporal, weather, seasonal, and operational data.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--Learn-1.6.1-orange?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Status-Completed-success?style=for-the-badge" />
</p>

<p align="center">
  <a href="https://github.com/Pranav123221/urban-demand-prediction-Forecasting-engine">
    <img src="https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github" />
  </a>
</p>

---

## 🖥️ Application Preview

<p align="center">
  <img src="assets/dashboard.png" width="900">
</p>

---

## 📌 Overview

Urban mobility systems experience continuously changing demand throughout the day.

Bike rental demand can vary significantly depending on:

- 🕐 Time of day
- 🌡️ Temperature
- 💧 Humidity
- 🌧️ Rainfall
- ❄️ Snowfall
- ☀️ Solar radiation
- 🍂 Season
- 📅 Weekday/weekend patterns
- 🎉 Holiday status
- 🚲 System operating status

This project builds a machine learning system capable of predicting **hourly bike rental demand** using these temporal, environmental, seasonal, and operational factors.

The project follows a complete machine learning workflow:

```text
Data
 ↓
Exploratory Data Analysis
 ↓
Feature Engineering
 ↓
Preprocessing
 ↓
Model Training
 ↓
Model Comparison
 ↓
Evaluation
 ↓
Error & Residual Analysis
 ↓
Feature Importance
 ↓
Model Serialization
 ↓
Prediction Pipeline
 ↓
Streamlit Application



The objective is not simply to train a model, but to demonstrate how a machine learning model can be developed, evaluated, packaged, and integrated into a usable application.

----
🎯 Problem Statement

Bike-sharing systems need to understand future demand to improve operational planning.

If demand is underestimated, insufficient bikes may be available during high-demand periods.

If demand is overestimated, resources may remain underutilized.

Therefore, predicting demand from historical and environmental information can support data-driven decisions around:

🚲 Bike availability
📍 Resource allocation
👥 Fleet planning
🕐 Peak-hour preparation
🌦️ Weather-aware operations
📊 Demand monitoring

This project approaches the problem as a supervised regression task, where the target is the number of bikes rented during a particular hour.


---
💡 Project Highlights
🔹 End-to-End ML Workflow

The project covers the complete lifecycle from raw data to application-level inference.

🔹 Time-Aware Data Splitting

Instead of randomly shuffling observations, the dataset is split chronologically to better represent historical training and future evaluation.

🔹 Temporal Feature Engineering

Date information is transformed into useful calendar features such as:

Year
Month
Day
DayOfWeek
IsWeekend
🔹 Multiple Regression Models

Two models are evaluated:

Linear Regression
Random Forest Regressor
🔹 Production-Oriented Pipeline

Preprocessing and model inference are combined into a reusable Scikit-learn pipeline.

🔹 Error Analysis

The project goes beyond accuracy metrics by analyzing:

Prediction errors
Residual distribution
Actual vs predicted values
Feature importance
🔹 Interactive Application

A Streamlit interface allows users to provide input conditions and obtain predicted bike demand.

----

🧠 Machine Learning Architecture
                    ┌─────────────────────┐
                    │   Seoul Bike Data   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Data Validation   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │       EDA           │
                    │ Patterns & Analysis │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Feature Engineering │
                    │ Time & Calendar     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Chronological     │
                    │    Train/Test Split │
                    └──────────┬──────────┘
                               │
                               ▼
                 ┌───────────────────────────┐
                 │      Preprocessing        │
                 │                           │
                 │ Numerical → Passthrough   │
                 │ Categorical → OneHot      │
                 └────────────┬──────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │    Model Training      │
                  │                        │
                  │ Linear Regression      │
                  │ Random Forest          │
                  └────────────┬───────────┘
                               │
                               ▼
                  ┌────────────────────────┐
                  │     Evaluation         │
                  │                        │
                  │ MAE / RMSE / R²        │
                  └────────────┬───────────┘
                               │
                               ▼
                  ┌────────────────────────┐
                  │   Error & Residual     │
                  │      Analysis          │
                  └────────────┬───────────┘
                               │
                               ▼
                  ┌────────────────────────┐
                  │   Feature Importance   │
                  └────────────┬───────────┘
                               │
                               ▼
                  ┌────────────────────────┐
                  │ Final ML Pipeline      │
                  │ + Model Serialization  │
                  └────────────┬───────────┘
                               │
                               ▼
                  ┌────────────────────────┐
                  │ Streamlit Application   │
                  └────────────────────────┘
📊 Dataset
Seoul Bike Sharing Demand Dataset

The project uses the Seoul Bike Sharing Demand Dataset, containing hourly bike rental information along with weather, seasonal, and operational attributes.

The target variable is:

Rented Bike Count

which represents the number of bikes rented during a particular hour.

📋 Dataset Features
Feature	Description
Date	Date of observation
Rented Bike Count	Number of rented bikes — target
Hour	Hour of the day
Temperature	Temperature in °C
Humidity	Relative humidity
Wind speed	Wind speed
Visibility	Visibility
Dew point temperature	Dew point temperature
Solar Radiation	Solar radiation
Rainfall	Rainfall
Snowfall	Snowfall
Seasons	Season
Holiday	Holiday status
Functioning Day	Whether the bike system was functioning
🔎 Exploratory Data Analysis

The project performs extensive exploratory analysis before model training.

🕐 Hourly Demand Analysis

Hourly demand was analyzed to understand recurring usage patterns throughout the day.

The analysis revealed noticeable demand peaks around common commuting periods, particularly:

Morning → around 8 AM
Evening → around 6–7 PM

These patterns demonstrate why the Hour feature is important for demand prediction.

🌤️ Seasonal Analysis

Bike demand was compared across different seasons.

The analysis helps identify differences in demand behavior under different seasonal conditions.

🌡️ Temperature vs Demand

The relationship between temperature and bike demand was visualized using scatter plots.

Demand generally tends to increase with more favorable temperatures, although the relationship is not perfectly linear.

🌧️ Weather Analysis

Weather variables were analyzed using correlation plots and visualizations.

Variables explored include:

Temperature
Humidity
Wind speed
Rainfall
Snowfall
Solar radiation
Visibility
Dew point temperature
📅 Calendar Analysis

Calendar-based patterns were explored using:

Month
Day of week
Weekend status
Holiday status

These features help the model capture recurring temporal demand patterns.

🛠️ Feature Engineering

The original Date feature was converted into multiple useful temporal features.

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["DayOfWeek"] = df["Date"].dt.dayofweek
df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
Generated Features
Feature	Purpose
Year	Captures yearly variation
Month	Captures monthly/seasonal patterns
Day	Captures day-level variation
DayOfWeek	Captures weekday patterns
IsWeekend	Distinguishes weekdays and weekends
🧹 Data Preprocessing

A ColumnTransformer is used to process numerical and categorical features.

Numerical Features

Numerical features are passed through directly.

Categorical Features

Categorical features are transformed using:

OneHotEncoder(handle_unknown="ignore")

Using handle_unknown="ignore" makes the prediction pipeline safer when previously unseen categorical values are encountered during inference.

⏳ Chronological Train/Test Split

Since the dataset represents time-based observations, the project avoids a purely random train/test split.

The data is sorted chronologically before splitting.

Historical Data
──────────────────────────────────────────────► Time

|────────────── Training ──────────────|── Test ──|
                  80%                       20%

This better reflects a real-world scenario:

Train on historical data
          ↓
Predict later observations

This also reduces the risk of temporal leakage caused by randomly mixing observations from different periods.

🤖 Machine Learning Models
1. Linear Regression

Linear Regression is used as the baseline model.

It provides a simple reference point for understanding how much predictive information can be captured through a linear relationship.

2. Random Forest Regressor

The main nonlinear model is a Random Forest Regressor.

Configuration:

RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

Random Forest is useful for this problem because bike demand can have nonlinear relationships and interactions between:

Time
Weather
Season
Calendar patterns
Operational conditions

---
📈 Model Evaluation

The models are evaluated using three standard regression metrics.

MAE — Mean Absolute Error

Measures the average absolute difference between actual and predicted demand.

Lower = Better
RMSE — Root Mean Squared Error

Penalizes larger prediction errors more strongly.

Lower = Better
R² — Coefficient of Determination

Measures how much of the variation in the target is explained by the model.

Higher = Better
--
📊 Model Comparison

The notebook evaluates both models using the same test data.

Model	MAE	RMSE	R²
Linear Regression	Generated in Notebook	Generated in Notebook	Generated in Notebook
Random Forest	Generated in Notebook	Generated in Notebook	Generated in Notebook

The evaluation values are calculated directly from the trained models rather than manually hard-coded into the README.

🎯 Actual vs Predicted Analysis

The Random Forest model's predictions are compared with actual bike rental demand.

Actual Demand
      │
      │       •
      │    •
      │  •
      │ •
      └──────────────── Predicted Demand

The visualization helps determine how closely predictions follow the observed demand values.
--

📉 Residual Analysis

Residuals are calculated as:

Residual = Actual − Predicted

Residual analysis helps investigate:

Large prediction errors
Systematic patterns
Model bias
Regions where predictions are less accurate
Whether errors appear randomly distributed

A residual plot was generated for the final Random Forest model.
--

🔬 Feature Importance

Random Forest feature importance was analyzed to understand which input features contribute most strongly to model predictions.

The analysis helps identify influential:

Temporal features
Weather variables
Seasonal variables
Operational variables

Important:

Feature importance describes how the trained model uses features. It does not establish a causal relationship between a feature and bike demand.
---

🚨 Outlier Analysis

Potential outliers were examined during exploratory analysis.

However, extreme demand and weather observations can represent legitimate real-world conditions.

Therefore:

No blanket outlier removal
No automatic winsorization

was applied.

This preserves potentially meaningful observations rather than removing them purely because they appear statistically extreme.
---

🏗️ Final Production Pipeline

The final model combines preprocessing and Random Forest into one Scikit-learn pipeline.

Raw Input
    │
    ▼
ColumnTransformer
    │
    ├── Numerical Features
    │
    └── Categorical Features
             │
             ▼
       One-Hot Encoding
             │
             ▼
      Random Forest Model
             │
             ▼
       Demand Prediction

This design ensures that the same preprocessing logic used during training is automatically applied during inference.
--

💾 Model Serialization

The final trained pipeline is serialized using Joblib.

joblib.dump(model_pipeline, "bike_demand_model.pkl")

During inference:

model = joblib.load("bike_demand_model.pkl")

The trained model file is approximately 119 MB.

Because GitHub's standard Git file-size limit is 100 MB, the .pkl file is intentionally excluded from the Git repository.

The model can instead be hosted separately and loaded by the application.

--

🧩 Prediction Module

The prediction logic is separated from the Streamlit interface.

src/
└── prediction.py

The module:

Receives input data
Converts the date
Generates temporal features
Removes the original date column
Loads the trained pipeline
Generates the prediction

Example:

prediction = predict_bike_demand(input_data)

This separation keeps the ML inference logic reusable outside the UI.
--

🖥️ Streamlit Application

The project includes an interactive Streamlit interface.

The application allows users to provide the conditions required by the trained model and receive an estimated hourly bike rental demand.

Application Flow
                 User
                  │
                  ▼
          Input Parameters
                  │
                  ▼
        Feature Engineering
                  │
                  ▼
        Prediction Module
                  │
                  ▼
       Trained ML Pipeline
                  │
                  ▼
        Predicted Demand

The interface is designed to present the machine learning system as an actual usable application rather than only a notebook experiment.
--

📁 Project Structure
Urban-Demand-Forecasting/
│
├── app.py
│
├── data/
│   └── SeoulBikeData.csv
│
├── models/
│   └── bike_demand_model.pkl
│
├── notebooks/
│   └── Data_Driven_Energy_Demand_Forecasting_System.ipynb
│
├── src/
│   └── prediction.py
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore

bike_demand_model.pkl is kept outside the Git repository because its size exceeds GitHub's standard file limit.
--

🧰 Tech Stack
Category	Technology
Programming Language	Python
Data Manipulation	Pandas
Numerical Computing	NumPy
Visualization	Matplotlib, Seaborn
Machine Learning	Scikit-learn
Regression	Linear Regression, Random Forest
Model Serialization	Joblib
Application	Streamlit
Development	Jupyter Notebook, VS Code
Version Control	Git, GitHub
--
📦 Installation
1. Clone the Repository
git clone https://github.com/Pranav123221/urban-demand-prediction-Forecasting-engine.git
2. Navigate to the Project
cd urban-demand-prediction-Forecasting-engine
3. Create a Virtual Environment
python -m venv venv
4. Activate the Environment
Windows
venv\Scripts\activate
macOS / Linux
source venv/bin/activate
5. Install Dependencies
pip install -r requirements.txt

▶️ Run the Application

Start the Streamlit application:

streamlit run app.py

The application will open in your browser.

---
🌐 Potential Real-World Applications

A demand prediction system like this can support:

🚲 Bike-sharing companies
🏙️ Smart-city mobility systems
📍 Fleet allocation
📊 Demand planning
🕐 Peak-hour preparation
🌦️ Weather-aware resource planning
🚦 Urban transportation analytics

The model itself does not make operational decisions; it provides a demand estimate that can be incorporated into broader planning systems.


------

👨‍💻 Author

Pranav Sharma

B.Tech Computer Science (AI/ML)

Connect
<p align="left"> <a href="https://github.com/Pranav123221"> <img src="https://img.shields.io/badge/GitHub-Pranav123221-black?style=for-the-badge&logo=github" /> </a> <a href="https://www.linkedin.com/in/pranav-sharma-333b67338/"> <img src="https://img.shields.io/badge/LinkedIn-Pranav%20Sharma-blue?style=for-the-badge&logo=linkedin" /> </a> </p>
⭐ If You Found This Project Useful

Consider giving the repository a ⭐ on GitHub.

<p align="center"> <b>Built with Python • Scikit-learn • Pandas • Streamlit</b> </p> <p align="center"> 🚲 Turning urban mobility data into actionable demand predictions. </p> ```

One important correction: README mein models/bike_demand_model.pkl ko project structure mein dikhaya hai for the intended local project structure, but GitHub par woh file currently nahi hai because of the 100 MB limit. That's okay; the README explicitly explains it.

