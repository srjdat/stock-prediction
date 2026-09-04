import torch.nn as nn
import torch
import datetime
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import math
from scipy.stats import pearsonr
from data_init import init, walk_forward


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
    df = init()
    label_df = df[0]
    features_df = df[1]
    
    fold_list = walk_forward(rows=len(features_df), train_size=450, test_size=10, step_size=50)

    nn_accuracy = []
    baseline_accuracy = []
    edge = []
    test_majority_list = []
    train_majority_list = []

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

        # print(f"train majority: {max(y_train.mean(), 1-y_train.mean()):.2f}, test majority: {max(y_test.mean(), 1-y_test.mean()):.2f}")
        test_majority_list.append(max(y_test.mean(), 1-y_test.mean()))
        train_majority_list.append(max(y_train.mean(), 1-y_train.mean()))

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
            'edge' : edge,
            'majority_shift' : [test - train for test, train in zip(test_majority_list, train_majority_list)]
        }
    )

    return return_df

def main(): 
    df = neural()

    corr, p_value = pearsonr(df['majority_shift'].astype(float), df['edge'].astype(float))
    print(f"correlation: {corr:.3f}, p-value: {p_value:.3f}")

if __name__ == "__main__":
    main()