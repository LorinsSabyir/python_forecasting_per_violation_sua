import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX

df = pd.read_csv("DATASET/totest1.csv")

# Clean municipal column
df['violator_address_municipal'] = (
    df['violator_address_municipal']
    .astype(str)
    .str.strip()
    .str.lower()
)

# Filter Panabo City  create df_panabo
df_panabo = df[df['violator_address_municipal'] == "panabo city"]

# Convert date column
df_panabo['appre_date'] = pd.to_datetime(df_panabo['appre_date'])

monthly = (
    df_panabo
    .groupby([pd.Grouper(key='appre_date', freq='M'), 'violation'])
    .size()
    .reset_index(name='count')
) #count

forecast_results = []

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
forecast_df.sort_values(by="Forecast Next Month", ascending=False)

top5 = forecast_df.sort_values(
    by="Forecast Next Month",
    ascending=False
).head(5)
print("Top 5 Violation Types Forecasted for Next Month:\n")

for i, row in top5.iterrows():
    print(f"- {row['Violation']}: {row['Forecast Next Month']} cases")


top5_violations = top5['Violation'].tolist()

actual_top5 = monthly[
    monthly['violation'].isin(top5_violations)
]

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
plt.show()