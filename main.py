import pandas as pd
import yfinance as yf
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import datetime

def walk_forward(rows, step_size, train_size, test_size) -> list[tuple]:
    start = 0 # we start at index 0
    return_list = [] # initialize an empty list that we're going to append to

    while (start + train_size + test_size) <= rows:
        train_index = np.arange(start, start+train_size) # train from start to train limit
        test_index = np.arange(start+train_size, start+train_size+test_size) # test form end of train to end of test size

        return_list.append((train_index, test_index)) # list of tuples that have train and test index

        start = start + step_size

    return return_list


def label(df: pd.DataFrame, horizon: int, threshold: float) -> pd.DataFrame:

    new_df = df.copy()
    new_df['Close Tomorrow'] = new_df['Close'].shift(-horizon)

    # 1 if close tomorrow - close / close > threshold else 0 if close tomorrow - close / close < -threshold
    new_df['Difference'] = (new_df['Close Tomorrow'] - new_df['Close']) / new_df['Close']
    new_df['Label'] = (np.select([new_df['Difference'] > threshold, new_df['Difference'] < -threshold], [1, 0], np.nan)) # make the ones in between the threshold nan

    new_df = new_df.dropna(subset=['Label'])
    new_df['Label'] = new_df['Label'].astype(int) # convert it to int after dropping nans because nans to int conversion throws runtime error
    return new_df

def initialize_df(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:

    # start from a year back so we can have 52 week high low and other stuff already loaded in
    start_date_original = datetime.date.strptime(start_date, "%Y-%m-%d")
    start_date = start_date_original.replace(year=start_date_original.year-1) # type: ignore
    df = pd.DataFrame(yf.Ticker(ticker=ticker).history(start=start_date, end=end_date))

    # all these calculations are from https://github.com/srjdat/finance-trader
    df['52wkHigh'] = df.High.rolling(window=252).max()
    df['52wkLow'] = df.Low.rolling(window=252).min()
    df['Distance From High'] = (df.Close - df['52wkHigh']) / df['52wkHigh'] * 100
    df['Distance From Low'] = (df.Close - df['52wkLow']) / df['52wkLow'] * 100

    # moving average
    df["SMA20"] = df.Close.rolling(window=20).mean()
    df["SMA50"] = df.Close.rolling(window=50).mean()

    # bollinger bands
    df['Upper Band'] = 2 * df.Close.rolling(window=20).std() + df['SMA20']
    df['Lower Band'] = df['SMA20'] - 2 * df.Close.rolling(window=20).std()

    # average true range
    # tr = max(high, close_prev) - min(low, close_prev)
    close_prev = df['Close'].shift(1)
    tr1 = pd.concat([df['High'], close_prev], axis=1).max(axis=1)
    tr2 = pd.concat([df['Low'], close_prev], axis=1).min(axis=1)
    true_range = tr1 - tr2

    n = 14
    # instantiate the atr dataframe
    temp = true_range.iloc[0:n].mean() # get the first 14 day average

    # start the atr series
    atr_values = [np.nan] * (n-1) # first 14 is going to be nan
    atr_values.append(temp) # add temp to the 14th index

    # get the rest
    for i in range(n, len(true_range)): # smma
        temp = (temp * (n-1) + true_range.iloc[i]) / n  # yesterday's temp value becomes today's atr value
        atr_values.append(temp)  # add today's temp into atr

    df['ATR'] = pd.Series(data=atr_values, index=true_range.index) # add it into df

    # find the volatility
    df["Daily Change"] = df["Close"].pct_change()
    df["Volatility"] = 100 * (df["Daily Change"].rolling(window=20).std())

    # RVOL
    # find sma 10 for volume
    df['Volume SMA 20'] = df['Volume'].rolling(window=20).mean()
    df['rvol'] = df.Volume/df['Volume SMA 20'].shift(1)

    # find rsi
    daily_change = df["Close"].diff()  # today - yesterday

    # change up and down
    change_up, change_down = daily_change.copy(), daily_change.copy()
    change_up[change_up < 0] = 0  # up = close_now - close_prev down = 0
    change_down[change_down > 0] = 0  # up = 0 down = close_prev - close_now

    # average up and down
    average_up = change_up.rolling(14).mean()  # get average for up
    average_down = change_down.rolling(14).mean().abs() #  get average for down
    df['rsi'] = 100 * average_up / (average_up + average_down)
    # these are the most widely used values (got this from charles schwab youtube video: https://youtu.be/hbcCykbX14U?si=eaaSyrdYvQqW3a8Q)
    oversold = np.full(len(df), 30)  # 1d array with 30 as all the values
    overbought = np.full(len(df), 70)  # 1d array with 70 as all the values

    # MACD
    # ema
    df["EMA12"] = df.Close.ewm(span=12).mean()
    df["EMA26"] = df.Close.ewm(span=26).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["Signal Line"] = df["MACD"].ewm(span=9).mean()
    df["macd hist"] = df["MACD"] - df["Signal Line"]

    # make a new column with tomorrow's open for trades
    df['Tomorrow Open'] = df['Open'].shift(-1)

    # returns over windows
    df['one_day_window'] = (df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1) * 100
    df['one_week_window'] = (df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5) * 100
    df['one_month_window'] = (df['Close'] - df['Close'].shift(21)) / df['Close'].shift(21) * 100
    df['three_month_window'] = (df['Close'] - df['Close'].shift(63)) / df['Close'].shift(63) * 100
    df['six_month_window'] = (df['Close'] - df['Close'].shift(125)) / df['Close'].shift(125) * 100
    df['one_year_window'] = (df['Close'] - df['Close'].shift(252)) / df['Close'].shift(252) * 100

    # make df only from start date to end date
    df = df.iloc[250:len(df)]

    return df # return the dataframe

def main():

    ticker = 'AAPL'
    start_date = '2023-01-01'
    end_date = '2026-07-17'
    df = initialize_df(ticker=ticker, start_date=start_date, end_date=end_date)

    df = label(df=df, horizon=5, threshold=.005)
    df["pos"] = np.arange(len(df)) # create a positional column

    label_df = df['Label']
    features_df = df.drop(columns=['Label', 'Difference', 'Close Tomorrow'])
    results = []

    fold_list = walk_forward(rows=len(features_df), train_size=450, test_size=50, step_size=50)
    for train_index, test_index in fold_list:
        # make the x/y_train/test dataframes
        x_train = features_df.iloc[train_index]
        x_test = features_df.iloc[test_index]
        y_train = label_df.iloc[train_index]
        y_test = label_df.iloc[test_index]

        bst = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1) # create the model
        bst.fit(x_train, y_train) # fit the training data
        preds = bst.predict(x_test) # get the predictions

        # get accuracy score compared to y_test
        accuracy = accuracy_score(y_pred=preds, y_true=y_test)

        results.append(accuracy) # add to results list

    print(results)


if __name__ == "__main__":
    main()
