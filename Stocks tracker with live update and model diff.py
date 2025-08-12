import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
import time, datetime as dtime
import yfinance as yf

actual_time = dtime.datetime.now().strftime("%Y-%m_%H:%M")



#object: Ticker for stock data - LVMH

ticker = yf.Ticker("MC.PA")

data_1d = ticker.history(period="1d")
data_1w = ticker.history(period="1wk")

#print(data_1d) with open, high, low, close, volume

#print(data_1d[['Open', 'High', 'Low', 'Close', 'Volume']])

df = ticker.history(period="1d", interval="1m")

df.to_csv(f"LVMH_1d_1min_{actual_time}.csv")

today_open = dtime.datetime.now(tz=df.index.tz).replace(hour=9, minute=0, second=0, microsecond=0)
today_close = dtime.datetime.now(tz=df.index.tz).replace(hour=18, minute=0, second=0, microsecond=0)

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
paths[0, :] = S0  # we want 1st simulated price = S0

#DataFrame for the paths
df_paths = pd.DataFrame(paths, index=pd.RangeIndex(N+1, name='step'))
df_paths.to_csv("LVMH_GBM_paths.csv")

# 5%, median, and 95% quantiles
q05 = df_paths.quantile(0.05, axis=1)
q50 = df_paths.quantile(0.50, axis=1)
q95 = df_paths.quantile(0.95, axis=1)

plt.plot(df_paths.iloc[:, :4], alpha=0.3, color="red")   # some paths for visibility
plt.plot(q50, linewidth=2, label="Median")
plt.plot(q05, linestyle="--", label="5%")
plt.plot(q95, linestyle="--", label="95%")
plt.title("LVMH (MC.PA) – 1 day GBM Simulation with 1-minute steps") 
plt.xlabel("Step = 1 min")
plt.ylabel("simuled price")
plt.xlim(today_open, today_close)
plt.legend()
plt.tight_layout()
#plt.savefig("lvmh_sim_1d_15m.png", dpi=300, bbox_inches="tight")
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
plt.savefig(f"lvmh_close_with_projection_{actual_time}.png", dpi=300, bbox_inches="tight")

#we want to make the connection beetween the first simulated point and the last historical point
last_time = df.index[-1]
last_price = df['Close'].iloc[-1]
plt.plot([last_time, proj_times[1]], [last_price, proj_prices[1]],
         linestyle="--", color="black", label="Connection")

#plt.show()


#we will ad now the live update of the stock price with the difference with the model, it has to be interrupted bu the user by ctrl+c

#dataframe to store new data

diff_log = pd.DataFrame({
    "real": pd.Series(index=proj_times, dtype="float64"),
    "model": pd.Series(proj_prices, index=proj_times, dtype="float64"),
    "diff": pd.Series(index=proj_times, dtype="float64")
})


#live update:

print("Start live diff update... (Ctrl+C to stop)")
try:
    while True:
        #get the last price, 2min before the last point and we tale the last point as the last price
        latest = yf.Ticker(symbol).history(period="1d", interval ="1m").tail(1)
        if not latest.empty:
            t_new = latest.index[-1].tz_convert(df.index.tz) if latest.index.tz is not None else latest.index[-1].tz_localize(df.index.tz)
            p_new = float(latest['Close'].iloc[-1])

            #we record the difference
            if t_new in diff_log.index:
                diff_log.loc[t_new, "real"] = p_new
                diff_log.loc[t_new, "diff"] = diff_log.loc[t_new, "real"] - diff_log.loc[t_new, "model"]

                #plot refresh
                valid = diff_log["real"].dropna()
                plt.plot(valid.index, valid.values, label = "Difference (real - model)", color="purple")
                #plt.axhline(0, color="black", linestyle="--", linewidth=1)
                plt.title(f"LVMH (MC.PA) - Live Price Difference (updated at {t_new})")
                plt.xlabel("Time")
                plt.ylabel("Price Difference") 
                #plt.ylim(-500,500) 
                plt.xlim(today_open, today_close)                                                      # Adjust as needed for visibility, may put the today close
                plt.legend()
                plt.tight_layout()
                plt.pause(0.01)

                #we save to a csv and a graph:
                diff_log.to_csv("LVMH_live_diff.csv")
                plt.savefig("LVMH_live_diff.png", dpi=300, bbox_inches="tight")

            
        now = dtime.datetime.now(tz =df.index.tz)
        wait = 60-  (now.second - now.microsecond / 1e6)
        time.sleep(max(1, wait))



except KeyboardInterrupt:
    print("Live update stopped by user.")