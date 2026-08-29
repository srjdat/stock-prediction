# Stock Prediction

## Information the model receives 
The model is getting fed the public indicators I have used in my previous projects such as SMA 20/50, EMA 12/26, MACD, Signal Line, Bollinger Bands, Distance from 52 week high/low, ATR, RSI, return over different windows. I have normalized all of them compared to the Close prices of the stock.

The `label()` function creates a label value which is either 1 or 0 which indicates if the stock's close price has increased within a user set threshold over a user set horizon. This method includes a column called "Close Tomorrow" which fetches the next day's close but is omitted from the original dataframe which the model trains on to prevent lookahead bias.  

The `walk_forward()` function has three important parameters: step_size, train_size, and test_size. Step_size is the size the counting index is going to take before moving onto n+1th iteration. Train_size is the amount of data the model is going to be trained on while test_size is what the model is going to be tested on. This function then loops until start + train_size + test_size is greater than row and appends to an empty list a tuple which includes the indices for training and testing and increments the counting variable by the step_size. Every iteration has the previous data alongisde the new data.

I then create x_train, x_test, y_train, and y_test, where x_train and y_train both get their values from the dataframe and are used to train the model. x_test is the information we feed the the model to get its predictions which is compared to y_test and an accuracy score is developed and appended to `accuracy_list`. Alongside this I have an `edge` variable which is a list that detects the difference between xgboost model and a basic model which picks what the majority movement of the stock is. 

## Conclusion
Public indicators (RSI, MACD, moving averages, volatility, momentum windows) that tested on a 5-day horizon don't indicate any meaningful and consistent gap over a baseline predictor. This was uniform across four large-cap, highly liquid stocks (AAPL, MSFT, NVDA, MU). I tested them with the same model settings and the average edge was around 0 or slightly negative. This is inline with the efficient market theory since the indicators used are derived from public history and any pattern is already reflected in the price. 

Model overfitting was looked into and ruled out after fixing feature leakage, normalization, and increasing training data. Train/test accuracy gaps closed reasonably (~.7 vs ~.5), but test performance repeatedly failed to rise above the baseline leading to the conclusion that the limitation is based on the lack of strong and consistent signals provided by the public indicators and not a model issue.  
