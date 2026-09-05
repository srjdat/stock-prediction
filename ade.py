import datetime

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from data_init import init, walk_forward
import yfinance as yf

"""
    y = Signal array; DataFrame, Array, etc. 
    k = Current time step
    N_ADE = Window size

    a_1(k) = (6 / N_{ADE}^3) * Σ_{K=1}^{N_{ADE}+1} α_K * (N_{ADE} - 2K) * y[k - K]
    Returns a_1k which is the first order derivative at step k. Tells us how fast the signal/trend is changing.

    We have to guard against k being smaller than N_ADE as in the summation we subtract up to N_ADE + 1 from k. If we do not add this safeguard, Python won't tell us the issue since negative indexing is possible. 
"""
def ade_derivative(y, k, N_ADE) -> float:

    if k < N_ADE + 1: 
        return 0.0

    scaling_factor = (6/(N_ADE**3))

    summation: float = 0.0
    for K in range(1, N_ADE+2): 
        a_k = .5 if K == 1 or K == N_ADE+1 else 1
        summation += a_k * (N_ADE-2*K) * y.iloc[k-K]

    a_1k: float = scaling_factor * summation

    return a_1k

"""
    y = Signal array; DataFrame, Array, etc. 
    k = Current time step
    N_ADE = Window size  
    delta_k = Steps into the future 

    Returns the forecasted value for [k + delta_k] (delta_k number of steps into the future).   
"""
def ade_forecast(y, k, N_ADE, delta_k) -> float: 
    a_1k = ade_derivative(y=y, k=k, N_ADE=N_ADE)
    return y.iloc[k] + a_1k * delta_k

def main(): 
    # for testing purposes, not real or anything
    # t = np.arange(1, 301)
    # # sin wave formula = A * sin(2pi * freq * time)
    # trend = 100 + 1 * np.sin(2 * math.pi * .02 * t)
    # noise = np.random.normal(0, .1, size=len(t))
    #
    # noisy_signal = trend + noise


    df = pd.DataFrame(yf.Ticker(ticker='AAPL').history(start='2024-01-01', end='2026-09-4'))
    
    
    # window size for ade got from: Cui, T., Xu, G., Zhou, A., Chen, J., Cook, A., & Wang, Z. (2026). APSO-enhanced algebraic derivative estimation approach for real-time traffic flow prediction on critical road sections during wildfire evacuation. Transportmetrica B: Transport Dynamics, 14(1). https://doi.org/10.1080/21680566.2025.2612243
    N_ADE = 256
    delta_k = 10 # how many days to predict ahead of time 
    ade_preds = []
    ade_preds_time = []
    true_signal = []
    for k in range(N_ADE + 1, len(df) - delta_k): 
        forecast = ade_forecast(y=df['Close'], k=k, N_ADE=N_ADE, delta_k=delta_k)
        ade_preds.append(forecast)
        ade_preds_time.append(df.index[k + delta_k])

    df['ade_pred'] = pd.Series(ade_preds, index=ade_preds_time) # add it to dataframe
    df['naive'] = df['Close'].shift(delta_k)
    # plt.plot(df.index, df['Close'], label='Close Price', alpha=.6)
    # plt.plot(df.index, df['ade_pred'], label='ADE Prediction', linewidth=1)
    # plt.legend()
    # plt.show()

    # error measurement and correlation
    # rsmc^2, rsmc, r^2
    diff = (df['naive'] - df['ade_pred']).dropna()
    rsme = np.sqrt((diff ** 2).mean())
    rsme_2 = rsme ** 2
    
    # find r^2
    mean = df['naive'].mean()
    ss_tot = np.sum((mean - df['naive'])**2)
    ss_res = np.sum(diff**2) 
    r_sqrd = 1 - ss_res/ss_tot

    # mae
    mae = abs(diff).mean()
    print(f"rsme = {rsme} \n mae = {mae} \n r^2 = {r_sqrd}")


if __name__ == "__main__":
    main()
