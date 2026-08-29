import torch.nn as nn
import torch
import datetime
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import math

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
    df = df.iloc[252:len(df)]

    return df # return the dataframe

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

class DirectionMLP(nn.Module):
    def __init__(self, n_features, hidden=32, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)  # logit, use BCEWithLogitsLoss
        )
    def forward(self, x):
        return self.net(x)


def neural() -> pd.DataFrame: 
    ticker = 'AAPL'
    start_date = '2020-01-01'
    end_date = '2026-07-17'
    df = initialize_df(ticker=ticker, start_date=start_date, end_date=end_date)
    df = label(df=df, horizon=5, threshold=.005)
    df["pos"] = np.arange(len(df)) # create a positional column
    label_df = df['Label']
    features_df = df.drop(columns=['Label', 'Difference', 'Close Tomorrow', 'pos', 'Close', 'Open', 'High', 'Low', 'Dividends', 'Stock Splits', 'Upper Band', 'Lower Band', '52wkHigh', '52wkLow', 'ATR', ]) # drop a bunch of columns that may contribute to overfitting or aren't useful in this case
    
    fold_list = walk_forward(rows=len(features_df), train_size=450, test_size=50, step_size=50)

    nn_accuracy = []
    baseline_accuracy = []
    edge = []

    for train_index, test_index in fold_list: 
        # make the x/y_train/test dataframes
        x_train = features_df.iloc[train_index]
        x_test = features_df.iloc[test_index]
        y_train = label_df.iloc[train_index]
        y_test = label_df.iloc[test_index]

        val_size = int(len(x_train) * 0.15)  # e.g. last 15% of train
        x_val = x_train[-val_size:]
        y_val = y_train[-val_size:]
        x_train = x_train[:-val_size]
        y_train = y_train[:-val_size]

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        x_val = scaler.transform(x_val)   # after scaler.fit_transform(x_train)
        x_val = torch.FloatTensor(x_val)
        y_val = torch.FloatTensor(y_val.to_numpy(copy=True)).reshape(-1, 1)

        x_train = torch.FloatTensor(x_train)
        x_test = torch.FloatTensor(x_test)
        y_train = torch.FloatTensor(y_train.to_numpy(copy=True)).reshape(-1, 1)
        y_test = torch.FloatTensor(y_test.to_numpy(copy=True)).reshape(-1, 1)

        model = DirectionMLP(n_features=x_train.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        n_pos = (y_train == 1).sum()
        n_neg = (y_train == 0).sum()
        pos_weight = n_neg / n_pos
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_val_loss = math.inf
        patience = 4
        patience_counter = 0

        max_epoch = 20
        for epoch in range(max_epoch): 
            model.train()
            optimizer.zero_grad()      
            y_pred = model(x_train)    
            loss = criterion(y_pred, y_train)  
            loss.backward()           
            optimizer.step()   

            model.eval()
            with torch.no_grad(): 
                y_val_pred = model(x_val)
                val_loss = criterion(y_val_pred, y_val)

            if val_loss < best_val_loss: 
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict()
            else: 
                patience_counter += 1
                if patience_counter > patience: 
                    break


        model.load_state_dict(best_model_state) # type: ignore
        model.eval()
        with torch.no_grad():
            y_test_logits = model(x_test)
            y_test_prob = torch.sigmoid(y_test_logits)    
            y_test_pred = (y_test_prob > 0.5).float()  

        accuracy = (y_test_pred == y_test).float().mean().item()
        nn_accuracy.append(round(accuracy, 2))
        majority_baseline = max(y_test.mean().item(), 1 - y_test.mean().item())
        baseline_accuracy.append(round(majority_baseline, 2))
        edge.append(round((accuracy - majority_baseline), 2))

    # print(f"nn {nn_accuracy}")
    # print(f"baseline {baseline_accuracy}")
    # print(f"edge {edge}")

    return_df = pd.DataFrame(
        {
            'accuracy' : nn_accuracy, 
            'baseline' : baseline_accuracy, 
            'edge' : edge
        }
    )

    return return_df

def main(): 
    neural()

if __name__ == "__main__":
    main()