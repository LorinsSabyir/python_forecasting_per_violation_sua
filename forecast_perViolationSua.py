import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.set_page_config(page_title="Violation Forecast", layout="wide")

# --- DATA LOADING ---
@st.cache_data # This keeps the app fast
def load_data():
    df = pd.read_csv("totest.csv")
    df['violator_address_municipal'] = df['violator_address_municipal'].astype(str).str.strip().str.lower()
    df['appre_date'] = pd.to_datetime(df['appre_date'])
    return df

df = load_data()

# --- STREAMLIT UI ---
st.title("📊 Panabo City Violation Forecast")
st.write("SARIMAX Model: Actual counts vs. Next month prediction")

# Filter for Panabo City
df_panabo = df[df['violator_address_municipal'] == "panabo city"]

monthly = (
    df_panabo
    .groupby([pd.Grouper(key='appre_date', freq='M'), 'violation'])
    .size()
    .reset_index(name='count')
)

forecast_results = []
violations = monthly['violation'].unique()

for violation in violations:
    data = monthly[monthly['violation'] == violation]
    ts = data.set_index('appre_date')['count']
    
    if len(ts) < 12: continue

    model = SARIMAX(ts, order=(1,1,1), seasonal_order=(1,1,1,12),
                    enforce_stationarity=False, enforce_invertibility=False)
    result = model.fit(disp=False)
    forecast = result.forecast(steps=1)
    
    forecast_results.append({
        "Violation": violation,
        "Forecast": int(round(forecast.iloc[0]))
    })

forecast_df = pd.DataFrame(forecast_results)
top5 = forecast_df.sort_values(by="Forecast", ascending=False).head(5)
top5_violations = top5['Violation'].tolist()

# --- PLOTLY CHART (Better for UI/UX) ---
fig = go.Figure()

for violation in top5_violations:
    data = monthly[monthly['violation'] == violation]
    
    # Add Actual Line
    fig.add_trace(go.Scatter(x=data['appre_date'], y=data['count'], name=f"{violation} (Actual)", mode='lines+markers'))
    
    # Add Forecast Point
    forecast_val = top5[top5['Violation'] == violation]['Forecast'].iloc[0]
    next_month = data['appre_date'].max() + pd.DateOffset(months=1)
    
    fig.add_trace(go.Scatter(x=[next_month], y=[forecast_val], 
                             name=f"{violation} (Forecast)",
                             marker=dict(size=12, symbol='star')))

fig.update_layout(template="plotly_white", hovermode="x unified", height=500)
st.plotly_chart(fig, use_container_width=True)

# Show Table
st.subheader("Top 5 Forecasted Violations")
st.table(top5)
