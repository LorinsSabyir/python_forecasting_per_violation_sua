import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Panabo City Violation Forecast", layout="wide")

st.title("Traffic Violation Forecast")
st.write("This dashboard shows the SARIMAX model predictions for the top 5 violations in Panabo City.")

# --- Load Data ---
try:
    df = pd.read_csv("totest.csv")
except FileNotFoundError:
    st.error("Error: 'totest.csv' not found. Please ensure the file is in your GitHub repository.")
    st.stop()

# Clean municipal column
df['violator_address_municipal'] = (
    df['violator_address_municipal']
    .astype(str)
    .str.strip()
    .str.lower()
)

# Filter Panabo City 
df_panabo = df[df['violator_address_municipal'] == "panabo city"]

# Convert date column
df_panabo['appre_date'] = pd.to_datetime(df_panabo['appre_date'])

monthly = (
    df_panabo
    .groupby([pd.Grouper(key='appre_date', freq='M'), 'violation'])
    .size()
    .reset_index(name='count')
)

forecast_results = []

# --- SARIMAX Logic ---
for violation in monthly['violation'].unique():
    data = monthly[monthly['violation'] == violation]
    ts = data.set_index('appre_date')['count']

    # Skip if not enough data
    if len(ts) < 12:
        continue

    model = SARIMAX(
        ts,
        order=(1,1,1),
        seasonal_order=(1,1,1,12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    result = model.fit(disp=False)
    forecast = result.forecast(steps=1)

    forecast_results.append({
        "Violation": violation,
        "Forecast Next Month": int(round(forecast.iloc[0]))
    })

forecast_df = pd.DataFrame(forecast_results)

top5 = forecast_df.sort_values(
    by="Forecast Next Month",
    ascending=False
).head(5)

# --- Visualization ---
top5_violations = top5['Violation'].tolist()
actual_top5 = monthly[monthly['violation'].isin(top5_violations)]

plt.figure(figsize=(12, 6))

for violation in top5_violations:
    data = actual_top5[actual_top5['violation'] == violation]

    # ACTUAL values
    plt.plot(
        data['appre_date'],
        data['count'],
        label=f"{violation} (Actual)"
    )

    # PREDICTED value (next month)
    forecast_value = top5[top5['Violation'] == violation]['Forecast Next Month'].iloc[0]
    next_month = data['appre_date'].max() + pd.DateOffset(months=1)

    plt.scatter(
        next_month,
        forecast_value
    )

plt.title("Top 5 Violation Types: Actual vs Forecast (Panabo City)")
plt.xlabel("Month")
plt.ylabel("Violation Count")
plt.legend()

# Display the plot in the Streamlit App
st.pyplot(plt.gcf())

# Display the text results below the graph
st.subheader("Top 5 Forecasted Violations for Next Month")
for i, row in top5.iterrows():
    st.write(f"- **{row['Violation']}**: {row['Forecast Next Month']} cases")
