import torch
import random
import numpy as np
import pandas as pd
import time
from experiments import make_DF, run_experiments

random.seed(2024)
np.random.seed(2024)
torch.manual_seed(2024)

Stats_Prob = "LR"
t_dim_list = [6, 28]

def step_size(i, norm):
    return norm /torch.sqrt(torch.tensor(i+1))

def loss_function(X,t):
    
    n = len(X)
    
    v = torch.topk(X + t,2).values
    
    return torch.max(v[:,0] - v[:,1])
    
LR_trained_lr_DF = pd.read_pickle("DataFrames/LR_trained_lr_DF.pkl")

LR_DF = make_DF(t_dim_list = t_dim_list, lr_df = LR_trained_lr_DF, Stats_Prob = Stats_Prob)

tic = time.time()
run_experiments(loss_function, LR_DF, 1000, termination=False, Stats_Prob = Stats_Prob)
print((time.time() - tic)/60, " minutes")
LR_DF.to_pickle("DataFrames/LR_DF.pkl")