import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
from data_init import init

def ade_derivative(y, k, N_ADE) -> float:
    """
        Implement Equation (11) as printed in Cui et al. (2026).

        Equation we are implementing:
        a_1(k) = (6 / N_{ADE}^3) * Σ_{K=1}^{N_{ADE}+1} α_K * (N_{ADE} - 2K) * y[k - K]

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

def _print_error_metrics(actual, predicted):
    rmse = root_mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    r_sqrd = r2_score(actual, predicted)
    print(f"RMSE = {rmse} \n MAE = {mae} \n R^2 = {r_sqrd}")

def ade():
    df: pd.DataFrame = init(ade = True) # type: ignore
    # window size for ade got from: Cui, T., Xu, G., Zhou, A., Chen, J., Cook, A., & Wang, Z. (2026). APSO-enhanced algebraic derivative estimation approach for real-time traffic flow prediction on critical road sections during wildfire evacuation. Transportmetrica B: Transport Dynamics, 14(1). https://doi.org/10.1080/21680566.2025.2612243
    N_ADE = 60 # how much data we use to predict
    delta = 1 # how many trading days to predict ahead
    comparison_start = N_ADE + 1 + delta

    ade_preds = []
    ade_preds_time = []
    for k in range(N_ADE + 1, len(df) - delta):
        forecast = ade_forecast(y=df['Close'], k=k, N_ADE=N_ADE, delta_k=delta)
        ade_preds.append(forecast)
        ade_preds_time.append(df.index[k + delta])

    prediction_column = f'ade_pred_{delta}'
    error_column = f'ade_error_{delta}'
    df[prediction_column] = pd.Series(ade_preds, index=ade_preds_time, dtype=float)
    df[error_column] = df[prediction_column] - df['Close']

    # Compare every horizon on the same target dates.
    # df_cleaned = df.iloc[comparison_start:].dropna(subset=['Close', prediction_column])
    # if len(df_cleaned) < 2:
    #     raise ValueError('Not enough price data to compare the ADE horizons.')
    #
    # print(f"Errors for N_ADE= {N_ADE}, horizon= {delta} trading days")
    # _print_error_metrics(df_cleaned['Close'], df_cleaned[prediction_column])
    # print(f"Average error for horizon={delta} trading days: {df_cleaned[error_column].mean()}")

    df['shifted_close'] = df['Close'].shift(delta)
    df['shifted_close_error'] = df['shifted_close'] - df['Close']
    ade_label = f'{delta}-day ADE forecast (N_ADE={N_ADE})'
    shift_label = f'Close shifted right {delta} trading day'

    # df_shifted = df.dropna(subset=['Close', 'shifted_close'])
    # print(f"Errors for {shift_label}")
    # _print_error_metrics(df_shifted['Close'], df_shifted['shifted_close'])
    # print(f"Average shifted-close error: {df_shifted['shifted_close_error'].mean()}")

    df_plot = df.iloc[comparison_start:].dropna(subset=['Close', prediction_column, 'shifted_close'])
    if len(df_plot) >= 2:
        print(f"Error comparison on {len(df_plot)} matching target dates")
        for label, price_column, residual_column in (
            (ade_label, prediction_column, error_column),
            (shift_label, 'shifted_close', 'shifted_close_error'),
        ):
            print(f"Errors for {label}")
            _print_error_metrics(df_plot['Close'], df_plot[price_column])
            print(f"Mean signed error = {df_plot[residual_column].mean()}")

        absolute_error_difference = df_plot[error_column].abs() - df_plot['shifted_close_error'].abs()
        print(f"MAE difference (ADE minus shifted close) = {absolute_error_difference.mean():+.6f}")
    else:
        print('At least two matching target dates are needed to compare errors.')

    _, axes = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True,
        gridspec_kw={'height_ratios': [2, 1]},
        layout='constrained'
    )

    axes[0].plot(df_plot.index, df_plot['Close'], label='Actual Price', color='black', alpha=.7)
    ade_line, = axes[0].plot(df_plot.index, df_plot[prediction_column], label=ade_label, linewidth=1)
    axes[1].plot(df_plot.index, df_plot[error_column], label=ade_label, color=ade_line.get_color(), linewidth=1)

    shifted_line, = axes[0].plot(df_plot.index, df_plot['shifted_close'], label=shift_label, linewidth=1)
    axes[1].plot(df_plot.index, df_plot['shifted_close_error'], label=shift_label, color=shifted_line.get_color(), linewidth=1)

    axes[0].set_title('ADE Forecasts and Shifted Close')
    axes[0].set_ylabel('Price ($)')
    axes[0].legend()

    axes[1].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[1].set_ylabel('Difference from actual ($)')
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
