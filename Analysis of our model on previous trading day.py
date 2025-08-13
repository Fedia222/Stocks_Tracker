import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
import datetime as dtime
import yfinance as yf
import time

actual_time = dtime.datetime.now().strftime("%Y-%m_%H:%M")

#lets take Louis Vuitton this time, and we weill take the comlete history for yesterday (we want something with volume)

symbol = "MC.PA"  
ticker = yf.Ticker(symbol)

history_2_days = ticker.history(period="2d", interval="1m")

yesterday = pd.Timestamp.today(tz=history_2_days.index.tz) - pd.Timedelta(days=1)

yesterday_data = history_2_days[history_2_days.index.date == yesterday.date()]

yesterday_data.to_csv(f"{symbol}_yesterday_data.csv")

#just for beauty, lets display toghether the price and volume

plt.figure(figsize=(14, 7))
plt.subplot(2, 1, 1)
plt.plot(yesterday_data['Close'], label='Close Price')  #stocks price
plt.title(f"{symbol} - Close Prices")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(yesterday_data['Volume'], label='Volume', color='skyblue')  #stocks volume
plt.title(f"{symbol} - Volume")
plt.xlabel("Time")
plt.ylabel("Volume")
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.legend()
plt.show()
plt.savefig(f"{symbol}_yesterday_plot.png")
plt.close()



#so now, lets take the data , simulate gbm for the next step, write the data to a csv file and plot the resutls
#we will take the previous 30 day data for better estimation, we can change this later

#simulate with GMB fot the next step

#parameters for the simulation

n_paths = 1
risk_neutral = False
r = 0.02

#sigma estimate
hist_30d = yf.Ticker(symbol).history(period="30d", interval="1d")
ret_daily = np.log(hist_30d['Close']).diff().dropna()
mu_daily = ret_daily.mean()
sigma_daily = ret_daily.std()

mu_annual = mu_daily * 252
sigma_annual = sigma_daily * np.sqrt(252)

#1 min setup for GBM
minutes_per_day = len(yesterday_data)               # ~ number of 1-min bars (≈ 510)
dt = 1 / (252 * minutes_per_day)
drift = r if risk_neutral else mu_annual
drift_term = (drift - 0.5 * sigma_annual**2) * dt
vol_term = sigma_annual * np.sqrt(dt)


#how the simulation of 1 step for each minute; one step ahead predictions

y_close = yesterday_data['Close'].dropna()
y_index = yesterday_data.loc[y_close.index].index  # keep matching tz-aware index

predictions = []
pred_times = []

for i in range(len(y_close) - 1):
    S0 = float(y_close.iloc[i])
    Z = np.random.normal()
    incr = drift_term + vol_term * Z
    S_next = S0 * np.exp(incr)            # GBM one-step prediction for next minute
    predictions.append(S_next)
    pred_times.append(y_index[i + 1])     # prediction timestamp = next real timestamp


#build the DataFrame for predictions,real and difference
predicted_series = pd.Series(predictions, index = pd.DatetimeIndex(pred_times, tz=y_index.tz), name = "model")
real_series = y_close.iloc[1:].rename("real")
diff_series_abs = (np.abs(predicted_series - real_series)).rename("abs_diff")
diff_series = (real_series - predicted_series).rename("diff")

gmb_out = pd.concat([predicted_series, real_series, diff_series_abs, diff_series], axis=1)
gmb_out.to_csv(f"{symbol}_GBM_predictions.csv")

#now, let's plot the results in all possible ways

# 1) let's plot the real and predicted prices on the same graph
plt.figure(figsize=(14, 7))
plt.plot(gmb_out['model'], label='Predicted Price', color='orange')
plt.plot(gmb_out['real'], label='Real Price', color='blue')
plt.title(f"{symbol} - GBM Predictions vs Real Prices")
plt.xlabel("Time")
plt.ylabel("Price")
plt.show()
plt.savefig(f"{symbol}_GBM_predictions_vs_real_prices_plot.png")
plt.close()

#not that bad!
# 2) let's plot the difference between the predicted and real prices
plt.figure(figsize=(14, 7))
plt.plot(gmb_out['diff'], label='Price Difference', color='purple')
plt.title(f"{symbol} - Price Difference (Predicted - Real)")
plt.xlabel("Time")
plt.ylabel("Price Difference")
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.legend()
plt.show()
plt.savefig(f"{symbol}_price_difference_plot.png")
plt.close()

# 3) let's plot the absolute difference between the predicted and real prices, it's the same graphique but with absolute values and it will be better fot do fitting
plt.figure(figsize=(14, 7))
plt.plot(gmb_out['abs_diff'], label='Absolute Price Difference', color='red', lw=0.5)
plt.title(f"{symbol} - Absolute Price Difference (Predicted - Real)")
plt.xlabel("Time")
plt.ylabel("Absolute Price Difference")
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.legend()
plt.show()
plt.savefig(f"{symbol}_absolute_price_difference_plot.png")
plt.close()

# Now, let's calculate the statistics of the predictions
mean_diff = gmb_out['abs_diff'].mean()
std_diff = gmb_out['abs_diff'].std()
median_diff = gmb_out['abs_diff'].median()
max_diff = gmb_out['abs_diff'].max()
min_diff = gmb_out['abs_diff'].min()

#print the statistics
print(f"Statistics for {symbol} predictions:")
print(f"Mean Absolute Difference: {mean_diff:.2f}")
print(f"Standard Deviation of Absolute Difference: {std_diff:.2f}")
print(f"Median Absolute Difference: {median_diff:.2f}")
print(f"Maximum Absolute Difference: {max_diff:.2f}")
print(f"Minimum Absolute Difference: {min_diff:.2f}")

