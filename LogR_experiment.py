import torch
import random
import numpy as np
import pandas as pd
import time
from experiments import make_DF, run_experiments

random.seed(2024)
np.random.seed(2024)
torch.manual_seed(2024)

Stats_Prob = "LogR"
t_dim_list = [12, 56]

def step_size(i, norm):
    return norm /torch.sqrt(torch.tensor(i+1))

def loss_function(X,t,Y):
    N = int(len(t)/2)
    
    h = ((torch.max(X-t[:N], dim=1).values - torch.min(X-t[:N],dim=1).values)) - ((torch.max(X-t[N:], dim=1).values - torch.min(X-t[N:],dim=1).values))
    f = Y*torch.log(1/(1+torch.exp(-h))) + (1-Y)*torch.log(1/(1+torch.exp(h)))

    return -torch.mean(f)
    
LogR_trained_lr_DF = pd.read_pickle("DataFrames/LogR_trained_lr_DF.pkl")

LogR_DF = make_DF(t_dim_list = t_dim_list, lr_df = LogR_trained_lr_DF, Stats_Prob = Stats_Prob)

tic = time.time()
run_experiments(loss_function, LogR_DF, 1000, termination=False, Stats_Prob = Stats_Prob)
print((time.time() - tic)/60, " minutes")
LogR_DF.to_pickle("DataFrames/LogR_DF.pkl")