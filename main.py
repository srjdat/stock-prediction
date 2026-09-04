import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from nn import neural
from trees import trees
import pandas as pd
import torch
import numpy as np
from scipy.stats import pearsonr



def main(): 

    corr_list = []
    edge_list = []
    for i in range(5):
        torch.manual_seed(i)
        df = neural()
        corr, p_val = pearsonr(df['majority_shift'].astype(float), df['edge'].astype(float))
        corr_list.append(corr)
        edge_list.append(df['edge'].mean())
    
    print(f"correlation mean: {np.mean(corr_list)}, correlation std: {np.std(corr_list)}")
    np.set_printoptions(legacy='1.25')
    print(edge_list)
    # print(corr_list)

    

if __name__ == "__main__": 
    main()