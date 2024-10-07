import torch
import random
import numpy as np
import pandas as pd
import time
from experiments import make_DF, run_experiments

random.seed(2024)
np.random.seed(2024)
torch.manual_seed(2024)

Stats_Prob = "FM"
t_dim_list = [6, 28]

def step_size(i, norm):
    return norm / torch.sqrt(torch.tensor(i+1))

def loss_function(X,t):
    return torch.sqrt(torch.mean((torch.max(X-t, dim=1).values - torch.min(X-t,dim=1).values)**2))
    
FM_trained_lr_DF = pd.read_pickle("DataFrames/FM_trained_lr_DF.pkl")

FM_DF = make_DF(t_dim_list = t_dim_list, lr_df = FM_trained_lr_DF, Stats_Prob = Stats_Prob)

tic = time.time()
run_experiments(loss_function, FM_DF, 1000, termination=False, Stats_Prob = Stats_Prob)
print((time.time() - tic)/60, " minutes")
FM_DF.to_pickle("DataFrames/FM_DF.pkl")