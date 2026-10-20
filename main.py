import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from nn import neural
from trees import trees
import pandas as pd
import torch
import numpy as np
from scipy.stats import pearsonr
from ade import ade

def main(): 
    ade()
    

if __name__ == "__main__": 
    main()