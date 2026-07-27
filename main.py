import pandas as pd
import yfinance as yf
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import datetime

def walk_forward(rows: int, step_size: int, train_size: int, test_size: int) -> list[tuple]:
    start = 0 # we start at index 0
    return_list = [] # initialize an empty list that we're going to append to

    while (start + train_size + test_size) <= rows:
        train_index = np.arange(0, start+train_size) # train from start to train limit
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
    # some of the calculations have been removed or changed to be percent based from the Close price
    df['52wkHigh'] = df.High.rolling(window=252).max()
    df['52wkLow'] = df.Low.rolling(window=252).min()
    df['Distance From High'] = (df.Close - df['52wkHigh']) / df['52wkHigh'] * 100
    df['Distance From Low'] = (df.Close - df['52wkLow']) / df['52wkLow'] * 100

    # moving average
    df['SMA20'] = df.Close / df.Close.rolling(window=20).mean() - 1
    df["SMA50"] = df.Close / df.Close.rolling(window=50).mean() - 1

    # bollinger bands
    df['Upper Band'] = 2 * df.Close.rolling(window=20).std() + df.Close.rolling(window=20).mean()
    df['Lower Band'] = df.Close.rolling(window=20).mean() - 2 * df.Close.rolling(window=20).std()

    # positions 
    df['bb_position'] = (df['Close'] - df['Lower Band']) / (df['Upper Band'] - df['Lower Band'])
    df['bb_width'] = (df['Upper Band'] - df['Lower Band']) / df.Close.rolling(window=20).mean()

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
    df['normalized ATR'] = df['ATR'] / df['Close']

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

    # MACD
    # ema
    df["EMA12"] = df.Close.ewm(span=12).mean()
    df["EMA26"] = df.Close.ewm(span=26).mean()
    df["MACD"] = (df["EMA12"] - df["EMA26"]) 
    df["Signal Line"] = df["MACD"].ewm(span=9).mean() 
    df["macd hist"] = (df["MACD"] - df["Signal Line"]) 

    # normalize all these 
    df["EMA12"] = df.Close / df['EMA12'] - 1
    df["EMA26"] = df.Close / df['EMA26'] - 1
    df["MACD"] = df['MACD'] / df.Close
    df["Signal Line"] = df['Signal Line'] / df.Close
    df["macd hist"] = (df['macd hist']) / df.Close

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
    start_date = '2020-01-01'
    end_date = '2026-07-17'
    df = initialize_df(ticker=ticker, start_date=start_date, end_date=end_date)

    df = label(df=df, horizon=5, threshold=.005)
    df["pos"] = np.arange(len(df)) # create a positional column

    label_df = df['Label']
    features_df = df.drop(columns=['Label', 'Difference', 'Close Tomorrow', 'pos', 'Close', 'Open', 'High', 'Low', 'Dividends', 'Stock Splits', 'Upper Band', 'Lower Band', '52wkHigh', '52wkLow', 'ATR', ]) # drop a bunch of columns that may contribute to overfitting or aren't useful in this case
    accuracy_list = []
    train_accuracy_list = []
    edge_list = [] # basic model that does whatever most of the data segment does (if mostly it's going up go up, if mostly it's going down go down)
    feature_importance = []

    fold_list = walk_forward(rows=len(features_df), train_size=900, test_size=50, step_size=50)


    bst = XGBClassifier(n_estimators=30, max_depth=2, learning_rate=0.1, subsample=0.7, colsample_bytree=0.7, reg_alpha=1, reg_lambda=1) # create the model
    
    for train_index, test_index in fold_list: # add onto the training data
        # make the x/y_train/test dataframes
        x_train = features_df.iloc[train_index]
        x_test = features_df.iloc[test_index]
        y_train = label_df.iloc[train_index]
        y_test = label_df.iloc[test_index]

        bst.fit(x_train, y_train) # fit the training data
        predictions = bst.predict(x_test) # get the predictions
        feature_importance.append(bst.feature_importances_)

        # predict based on the training data
        predictions_train = bst.predict(x_train)
        accuracy = accuracy_score(y_pred=predictions_train, y_true=y_train)
        train_accuracy_list.append(accuracy)

        # baseline
        preds_baseline = max(y_test.mean(), 1 - y_test.mean()) 

        # get accuracy score compared to y_test
        accuracy = accuracy_score(y_pred=predictions, y_true=y_test)
        edge = accuracy - preds_baseline

        accuracy_list.append(accuracy) # add to results list
        edge_list.append(edge)


    # this is to see which columns are most important for this model
    feature_importance = pd.DataFrame(feature_importance)
    array = feature_importance.mean(axis=0).to_numpy()
    series = pd.Series(array, index=x_train.columns).sort_values(ascending=False)
    # print(series)

    print(f"accuracy list \n{accuracy_list} \n")
    train_accuracy_list = [round(item, 2) for item in train_accuracy_list]
    print(f"train accuracy list \n{train_accuracy_list} \n")
    np.set_printoptions(legacy='1.25')
    edge_list = [round(member, 2) for member in edge_list] # format it to have 2 decimals
    print(f"edge list \n{edge_list} \n")


if __name__ == "__main__":
    main()
