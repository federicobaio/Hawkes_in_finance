
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize
from scipy.stats import norm

ticker = "BTC-USD" # (Strictly within the last 730 days to avoid Yahoo limits)
df = yf.download(ticker, start="2024-04-10", end="2026-04-01", interval="1h", progress=False)

df['Log_return'] = np.log(df['Close'] / df['Close'].shift(1))
df = df.dropna() # simply remove n/a

# We define an event as a hour where the absolute log-return exceeds 2%
threshold = 0.015
df['Event'] = np.abs(df['Log_return']) > threshold

# Extract event times (in hours) in a continuous numerical format
start_date = df.index.min()
df['Hours_since_start'] = (df.index - start_date).total_seconds() / 3600.0
event_times = df[df['Event']]['Hours_since_start'].values

print(f"Total trading hours: {len(df)}")
print(f"Number of extreme events detected (> {threshold*100}%): {len(event_times)}")

plt.figure(figsize=(14, 6))
plt.plot(df.index, df['Log_return'], label='Hourly log-returns', color='lightgray', alpha=0.8)
event_dates = df[df['Event']].index
event_returns = df[df['Event']]['Log_return']
plt.scatter(event_dates, event_returns, color='red', s=15, label=f'Extreme events (> {threshold*100}%)', zorder=5)
plt.axhline(threshold, color='blue', linestyle='--', linewidth=1, alpha=0.5)
plt.axhline(-threshold, color='blue', linestyle='--', linewidth=1, alpha=0.5)
plt.title('Bitcoin (BTC-USD) Log-returns and Extreme volatility events', fontsize=14)
plt.xlabel('Hours', fontsize=12)
plt.ylabel('Log-return', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

def nll_hawkes(par, events, T):
    # par contains mu, alpha and beta
    mu = par[0]
    alpha = par[1]
    beta = par[2]
    N = len(events)
    
    # The first part of the integral is simply integral(mu) -> mu*T
    integral_term = mu * T
    
    # Second part of the integral
    integral_term = integral_term + np.sum(alpha * (1 - np.exp(-beta * (T - events))))
    
    # We evaluate now the first part of the likelihood that is sum(log(lambda(t)))
    sum_log_intensity = 0
    

    for i in range(N):
        t_i = events[i]
        triggering_factor = 0
        if i > 0:
            # All the previous time for that i are simply: events[0:i] 
            t_j_vec = events[0:i]
            # Kernel: alpha * beta * exp(-beta * (t_i - t_j))
            kernel_values = alpha * beta * np.exp(-beta * (t_i - t_j_vec))
            triggering_factor = np.sum(kernel_values)
        
        # we add the mu background intensity in t_i
        lambda_t_i = mu + triggering_factor
        sum_log_intensity = sum_log_intensity + np.log(lambda_t_i)
        
    Neg_LL = integral_term - sum_log_intensity
    return Neg_LL


T = df['Hours_since_start'].max()
initial_guess = [0.01, 0.5, 0.1]
bnds = [(1e-5, None), (1e-5, 0.999), (1e-5, None)]
result = minimize(
    nll_hawkes, 
    initial_guess, 
    args=(event_times, T), 
    method='L-BFGS-B',
    bounds=bnds
)

mu_opt, alpha_opt, beta_opt = result.x
print("\nEstimated parameters")
print(f"Background rate (mu):    {mu_opt:.5f} eventi/hour")
print(f"Branching ratio (alpha): {alpha_opt:.5f} (endogeneity of bitcoin market)")
print(f"Decay rate (beta):       {beta_opt:.5f} (decay to normal rate in the market)")
print(f"Optimization succesful: {result.success}")

t_grid = df['Hours_since_start'].values
lambda_t = np.zeros(len(t_grid))
for idx, t in enumerate(t_grid):
    past_events = event_times[event_times < t]
    intensity = mu_opt + np.sum(alpha_opt * beta_opt * np.exp(-beta_opt * (t - past_events)))
    lambda_t[idx] = intensity

df['Hawkes intensity'] = lambda_t
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
ax1.plot(df.index, df['Close'], color='black', alpha=0.7, label='BTC-USD hourly close')
ax1.scatter(df[df['Event']].index, df[df['Event']]['Close'], color='red', s=20, label='Extreme events (> 2%)', zorder=5)
ax1.set_title('Bitcoin (BTC-USD) price and extreme volatility events', fontsize=14)
ax1.set_ylabel('Price (USD)', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax2.plot(df.index, df['Hawkes intensity'], color='purple', linewidth=1.5, label=r'Hawkes conditional intensity $\lambda(t)$')
ax2.axhline(mu_opt, color='gray', linestyle='--', label=r'Background Rate $\mu$')
ax2.set_title('Hawkes process: dynamic market risk', fontsize=14)
ax2.set_xlabel('Date', fontsize=12)
ax2.set_ylabel(r'Intensity $\lambda(t)$', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
