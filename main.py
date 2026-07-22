import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import matplotlib.pyplot as plt

df = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv')

m = Prophet()
m.fit(df)

future = m.make_future_dataframe(periods=365)
# print(future.tail())

forecast = m.predict(future)
# print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

fig1 = m.plot(forecast)
plt.title("Prophet Time Series Forecast")
plt.xlabel("Date")
plt.ylabel("Value")
plt.show()