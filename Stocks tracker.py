"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

import yfinance as yf

#object: Ticker for stock data - LVMH

ticker = yf.Ticker("MC.PA")

data_1d = ticker.history(period="1d")
data_1w = ticker.history(period="1wk")

#print(data_1d) with open, high, low, close, volume

#print(data_1d[['Open', 'High', 'Low', 'Close', 'Volume']])

df = ticker.history(period="1d", interval="1m")

df.to_csv("LVMH_1d_15min.csv")

#plotting the close prices

plt.plot(df['Close'])
plt.title("LVMH (MC.PA) - Close Prices")
plt.xlabel("Time")
plt.ylabel("Price")
#plt.show()

#simulate with GMB

# Option parameter:

symbol = "MC.PA"
steps_per_day = 34
n_paths = 1
risk_neutral = False
r = 0.02

#lets take the last 60 days for better estimation

hist_daily = yf.Ticker(symbol).history(period="60d", interval="1d")
if hist_daily.empty:
    raise ValueError("No historical data found for the ticker.")

# log-returns daily
ret = np.log(hist_daily['Close']).diff().dropna()

mu = ret.mean()
mu_annual = mu * 252
sigma = ret.std()
sigma_annual = sigma * np.sqrt(252)

#last price on the starting point S0
S0 = float(df['Close'].iloc[-1])

proj_times = pd.date_range(
    start=df.index[-1],              # start at the last historical timestamp
    periods=steps_per_day + 1,       # +1 to include the starting point
    freq="15T",
    tz=df.index.tz
)
proj_prices = df_paths.iloc[:, 0].values  

#GBM stimulation parameters:
N = steps_per_day
dt = 1 / (steps_per_day * 252)
drift = r if risk_neutral else mu_annual

#simulate the paths
Z = np.random.normal(size=(steps_per_day, n_paths))
increments = (drift - 0.5 * sigma_annual**2) * dt + sigma_annual * np.sqrt(dt) * Z
log_paths = np.vstack([np.zeros((1, n_paths)), np.cumsum(increments, axis=0)])
paths = S0 * np.exp(log_paths) 
paths[0, :] = S0 

#DataFrame for the paths
df_paths = pd.DataFrame(paths, index=pd.RangeIndex(N+1, name='step'))
df_paths.to_csv("LVMH_GBM_paths.csv")

# 5%, median, and 95% quantiles
q05 = df_paths.quantile(0.05, axis=1)
q50 = df_paths.quantile(0.50, axis=1)
q95 = df_paths.quantile(0.95, axis=1)

plt.plot(df_paths.iloc[:, :20], alpha=0.3, color="red")   # quelques chemins pour lisibilité
plt.plot(q50, linewidth=2, label="Médiane")
plt.plot(q05, linestyle="--", label="5%")
plt.plot(q95, linestyle="--", label="95%")
plt.title("LVMH (MC.PA) – 1 day GBM Simulation with 15-minute steps") 
plt.xlabel("Step = 15 min")
plt.ylabel("simuled price")
plt.legend()
plt.tight_layout()
plt.savefig("lvmh_sim_1d_15m.png", dpi=300, bbox_inches="tight")
#plt.show()

#we want now to merge the original data with the simulated paths

#let's contruct the beginning of the time axis 15 min ahed the original data

# Get last historical timestamp and last price
last_time = df.index[-1]
last_price = df['Close'].iloc[-1]

# Get simulated prices (n_paths=1)
proj_prices = df_paths.iloc[:, 0].values  # length N+1
# Drop the first point from the projection because it's just S0 (last historical price)
proj_prices_no_start = proj_prices[1:]

# Create timestamps for projection continuing immediately after history
if isinstance(df.index, pd.DatetimeIndex):
    proj_times = pd.date_range(start=last_time,     #+ pd.Timedelta(minutes=15)
                               periods=steps_per_day, freq="15min", tz=df.index.tz)
else:
    proj_times = range(len(df), len(df) + steps_per_day)

# Combine data for plotting
plt.figure(figsize=(10,5))
plt.plot(df.index, df['Close'], label="Historical Close (1m)", color="blue")
plt.plot(proj_times, proj_prices, label="Projection GBM (15 min)", color="red", linewidth=2)


plt.title("LVMH (MC.PA) — Historical + GBM Projection (merged)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.savefig("lvmh_close_with_projection.png", dpi=300, bbox_inches="tight")

#we want to make the connection beetween the first simulated point and the last historical point



last_time = df.index[-1]
last_price = df['Close'].iloc[-1]
plt.plot([last_time, proj_times[1]], [last_price, proj_prices[1]],
         linestyle="--", color="black", label="Connection")

plt.show()
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

import yfinance as yf

#object: Ticker for stock data - LVMH

ticker = yf.Ticker("MC.PA")

data_1d = ticker.history(period="1d")
data_1w = ticker.history(period="1wk")

#print(data_1d) with open, high, low, close, volume

#print(data_1d[['Open', 'High', 'Low', 'Close', 'Volume']])

df = ticker.history(period="1d", interval="1m")

df.to_csv("LVMH_1d_15min.csv")

#plotting the close prices

plt.plot(df['Close'])
plt.title("LVMH (MC.PA) - Close Prices")
plt.xlabel("Time")
plt.ylabel("Price")
#plt.show()

#simulate with GMB

# Option parameter:

symbol = "MC.PA"
steps_per_day = 510
n_paths = 3
risk_neutral = False
r = 0.02

#lets take the last 60 days for better estimation

hist_daily = yf.Ticker(symbol).history(period="60d", interval="1d")
if hist_daily.empty:
    raise ValueError("No historical data found for the ticker.")

# log-returns daily
ret = np.log(hist_daily['Close']).diff().dropna()

mu = ret.mean()
mu_annual = mu * 252
sigma = ret.std()
sigma_annual = sigma * np.sqrt(252)

#last price on the starting point S0
S0 = float(df['Close'].dropna().iloc[-1])

#GBM stimulation parameters:
N = steps_per_day
dt = 1 / (steps_per_day * 252)
drift = r if risk_neutral else mu_annual

#simulate the paths
Z = np.random.normal(size=(steps_per_day, n_paths))
increments = (drift - 0.5 * sigma_annual**2) * dt + sigma_annual * np.sqrt(dt) * Z
log_paths = np.vstack([np.zeros((1, n_paths)), np.cumsum(increments, axis=0)])
paths = S0 * np.exp(log_paths)
paths[0, :] = S0  # hard guarantee: first simulated price = S0

#DataFrame for the paths
df_paths = pd.DataFrame(paths, index=pd.RangeIndex(N+1, name='step'))
df_paths.to_csv("LVMH_GBM_paths.csv")

# 5%, median, and 95% quantiles
q05 = df_paths.quantile(0.05, axis=1)
q50 = df_paths.quantile(0.50, axis=1)
q95 = df_paths.quantile(0.95, axis=1)

plt.plot(df_paths.iloc[:, :20], alpha=0.3, color="red")   # quelques chemins pour lisibilité
plt.plot(q50, linewidth=2, label="Médiane")
plt.plot(q05, linestyle="--", label="5%")
plt.plot(q95, linestyle="--", label="95%")
plt.title("LVMH (MC.PA) – 1 day GBM Simulation with 15-minute steps") 
plt.xlabel("Step = 15 min")
plt.ylabel("simuled price")
plt.legend()
plt.tight_layout()
plt.savefig("lvmh_sim_1d_15m.png", dpi=300, bbox_inches="tight")
#plt.show()

#we want now to merge the original data with the simulated paths

#let's contruct the beginning of the time axis 15 min ahed the original data
# (build a datetime axis for the projection so it matches df.index, then skip S0 to avoid overlap)
proj_prices = df_paths.iloc[:, 0].values                 # length N+1
proj_times  = pd.date_range(start=df.index[-1],          # same stamp as last historical point
                            periods=steps_per_day + 1,   # N+1 points
                            freq="1min",                                                                                                                   # Here we change the frequency to 15 minutes                                  
                            tz=df.index.tz)

# Combine data for plotting
plt.figure(figsize=(10,5))
plt.plot(df.index, df['Close'], label="Historical Close (1m)", color="blue")
plt.plot(proj_times[1:], proj_prices[1:], label="Projection GBM (15 min)", color="red", linewidth=2)

plt.title("LVMH (MC.PA) — Historical + GBM Projection (merged)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.savefig("lvmh_close_with_projection.png", dpi=300, bbox_inches="tight")

#we want to make the connection beetween the first simulated point and the last historical point
last_time = df.index[-1]
last_price = df['Close'].iloc[-1]
plt.plot([last_time, proj_times[1]], [last_price, proj_prices[1]],
         linestyle="--", color="black", label="Connection")

plt.show()
