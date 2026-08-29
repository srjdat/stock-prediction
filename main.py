import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from nn import neural
from trees import trees
import pandas as pd
import torch
import numpy as np


def main(): 
    # torch.manual_seed(42)
    nn_seed_mean = []
    trees_seed_mean = []
    for i in range(5):
        nn_df = neural()
        trees_df = trees()
        nn_seed_mean.append(round(nn_df.edge, 4))
        trees_seed_mean.append(round(trees_df.edge, 4))
        print(f"run {i+1}")
        print(f"nn edge mean: {round(nn_df.edge.mean(), 4)} \ntrees edge mean: {round(trees_df.edge.mean(), 4)}")
        print(f"nn edge std: {round(nn_df.edge.std(), 4)} \ntrees edge std: {round(trees_df.edge.std(), 4)}")
        print()

    print(f"5 seed nn mean: {round(np.mean(nn_seed_mean), 4)}, 5 seed nn std: {round(np.std(nn_seed_mean), 4)}")
    print(f"5 seed trees mean: {round(np.mean(trees_seed_mean), 4)}, 5 seed trees std: {round(np.std(trees_seed_mean), 4)}")
    # print("        NN                                  Trees")
    # print(pd.concat([nn_df, trees_df], axis=1))
    


if __name__ == "__main__": 
    main()