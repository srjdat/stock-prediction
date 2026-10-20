import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
from data_init import init

def ade_derivative(y, k, N_ADE) -> float:
    """
        Implement Equation (11) as printed in Cui et al. (2026).

        y is a pandas Series, k is the current row, and N_ADE is the
        window parameter for N_ADE + 1 past observations.
        Observations are equally spaced in trading time (sample period 1).

        a_1(k) = (6 / N_ADE**3) * sum(
            alpha_K * (N_ADE - 2*K) * y[k - K]
            for K in range(1, N_ADE + 2)
        )
        alpha_K is 0.5 for K=1 or K=N_ADE+1; otherwise it is 1.
        This uses y[k - 1] through y[k - N_ADE - 1].
        
        The printed equation is reproduced without an indexing correction.
        For constant prices C, it gives -12*C/N_ADE**2, not zero.
    """

    if N_ADE < 1:
        raise ValueError('N_ADE must be a positive integer.')
    if k < N_ADE + 1:
        return 0.0
    if k >= len(y):
        raise IndexError('The current row is outside the price series.')

    scaling_factor = (6/(N_ADE**3))

    summation: float = 0.0
    for K in range(1, N_ADE + 2):
        alpha_K = .5 if K == 1 or K == N_ADE + 1 else 1
        summation += alpha_K * (N_ADE - 2*K) * y.iloc[k - K]

    a_1k: float = scaling_factor * summation

    return a_1k

def ade_forecast(y, k, N_ADE, delta_k) -> float:

    """
        y = Signal array; DataFrame, Array, etc.
        k = Current time step
        N_ADE = Window size
        delta_k = Steps into the future

        Returns the forecasted value for [k + delta_k] (delta_k number of steps into the future).
    """

    a_1k = ade_derivative(y=y, k=k, N_ADE=N_ADE)
    return y.iloc[k] + a_1k * delta_k

def ade():
    df: pd.DataFrame = init(ade = True) # type: ignore
    # window size for ade got from: Cui, T., Xu, G., Zhou, A., Chen, J., Cook, A., & Wang, Z. (2026). APSO-enhanced algebraic derivative estimation approach for real-time traffic flow prediction on critical road sections during wildfire evacuation. Transportmetrica B: Transport Dynamics, 14(1). https://doi.org/10.1080/21680566.2025.2612243
    N_ADE = [10, 20, 40, 65, 100] # how much data we use to predict
    delta = 5 # how many trading days to predict ahead
    prediction_columns = []
    error_columns = []
    comparison_start = max(N_ADE) + 1 + delta

    # true_signal = [] made originally for testing won't be used as of right now
    for n_ade in N_ADE:
        ade_preds = []
        ade_preds_time = []
        for k in range(n_ade + 1, len(df) - delta):
            forecast = ade_forecast(y=df['Close'], k=k, N_ADE=n_ade, delta_k=delta)
            ade_preds.append(forecast)
            ade_preds_time.append(df.index[k + delta])

        # create the columns
        prediction_column = f'ade_pred_{n_ade}'
        error_column = f'ade_error_{n_ade}'
        prediction_columns.append(prediction_column)
        error_columns.append(error_column)
        df[prediction_column] = pd.Series(ade_preds, index=ade_preds_time, dtype=float)
        df[error_column] = df[prediction_column] - df['Close']

        # error measurement and correlation
        # rsmc, r^2, mae
        # Compare every window on the same target dates.
        # df_cleaned = df from largest n_ade we use
        df_cleaned = df.iloc[comparison_start:].dropna(subset=['Close', prediction_column])
        if len(df_cleaned) < 2:
            raise ValueError('Not enough price data to compare the ADE windows.')

        rmse = root_mean_squared_error(y_true=df_cleaned['Close'], y_pred=df_cleaned[prediction_column])
        mae = mean_absolute_error(y_true=df_cleaned['Close'], y_pred=df_cleaned[prediction_column])
        r_sqrd = r2_score(y_true=df_cleaned['Close'], y_pred=df_cleaned[prediction_column])

        print(f"Errors for N_ADE={n_ade}, horizon={delta} trading days")
        print(f"RMSE = {rmse} \n MAE = {mae} \n R^2 = {r_sqrd}")

    df_plot = df.iloc[comparison_start:].dropna(subset=['Close'] + prediction_columns)
    fig, axes = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True,
        gridspec_kw={'height_ratios': [2, 1]},
        layout='constrained'
    )

    axes[0].plot(df_plot.index, df_plot['Close'], label='Actual Price', color='black', alpha=.7)
    for n_ade, prediction_column, error_column in zip(N_ADE, prediction_columns, error_columns):
        line, = axes[0].plot(df_plot.index, df_plot[prediction_column], label=f'N_ADE={n_ade}', linewidth=1)
        axes[1].plot(df_plot.index, df_plot[error_column], label=f'N_ADE={n_ade}', color=line.get_color(), linewidth=1)
    axes[0].set_title(f'NVDA: {delta}-Trading-Day ADE Forecasts')
    axes[0].set_ylabel('Price ($)')
    axes[0].legend()

    axes[1].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[1].set_ylabel('Predicted - actual ($)')
    axes[1].set_xlabel('Target Date')
    axes[1].legend()

    for ax in axes:
        ax.grid(alpha=.25)

    plt.show()
    return df


def main():
    ade()

if __name__ == "__main__":
    main()
