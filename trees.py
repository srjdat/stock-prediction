import pandas as pd
import yfinance as yf
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import datetime
from data_init import init, walk_forward



def trees() -> pd.DataFrame:

    df = init()
    label_df = df[0]
    features_df = df[1]
    
    accuracy_list = []
    train_accuracy_list = []
    baseline_accuracy = []
    edge_list = [] # basic model that does whatever most of the data segment does (if mostly it's going up go up, if mostly it's going down go down)
    feature_importance = []

    fold_list = walk_forward(rows=len(features_df), train_size=450, test_size=50, step_size=50)


    bst = XGBClassifier(n_estimators=30, max_depth=2, learning_rate=0.1, subsample=0.7, colsample_bytree=0.7, reg_alpha=1, reg_lambda=1) # create the model
    
    for train_index, test_index in fold_list: # add onto the training data
        # make the x/y_train/test dataframes
        x_train = features_df.iloc[train_index]
        x_test = features_df.iloc[test_index]
        y_train = label_df.iloc[train_index]
        y_test = label_df.iloc[test_index]

        bst.fit(x_train, y_train) # fit the training data
        predictions = bst.predict(x_test) # get the predictions
        print(predictions)
        feature_importance.append(bst.feature_importances_)

        # predict based on the training data
        predictions_train = bst.predict(x_train)
        accuracy = accuracy_score(y_pred=predictions_train, y_true=y_train)
        train_accuracy_list.append(accuracy)

        # baseline
        preds_baseline = max(y_test.mean(), 1 - y_test.mean()) 
        baseline_accuracy.append(round(preds_baseline, 2))

        # get accuracy score compared to y_test
        accuracy = accuracy_score(y_pred=predictions, y_true=y_test)
        edge = accuracy - preds_baseline

        accuracy_list.append(round(accuracy, 2)) # add to results list
        edge_list.append(edge)


    # this is to see which columns are most important for this model
    feature_importance = pd.DataFrame(feature_importance)
    array = feature_importance.mean(axis=0).to_numpy()
    series = pd.Series(array, index=x_train.columns).sort_values(ascending=False) # type: ignore
    # print(series)

    # print(f"accuracy list \n{accuracy_list} \n")
    # train_accuracy_list = [round(item, 2) for item in train_accuracy_list]
    # print(f"train accuracy list \n{train_accuracy_list} \n")
    # np.set_printoptions(legacy='1.25')
    # edge_list = [round(member, 2) for member in edge_list] # format it to have 2 decimals
    # print(f"edge list \n{edge_list} \n")

    return_df = pd.DataFrame(
        {
            'accuracy' : accuracy_list, 
            'baseline' : baseline_accuracy, 
            'edge' : edge_list
        }
    )

    return return_df

def main(): 
    trees()

if __name__ == "__main__":
    main()
