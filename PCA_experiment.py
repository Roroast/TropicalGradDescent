import torch
import random
import numpy as np
import pandas as pd
import time
from experiments import make_DF, run_experiments

random.seed(2024)
np.random.seed(2024)
torch.manual_seed(2024)

Stats_Prob = "PCA"
t_dim_list = [18, 84]

def step_size(i, norm):
    return norm /torch.sqrt(torch.tensor(i+1))

def loss_function(X,t):
    
    N = int(len(t)/3)
    n = len(X)
    l = torch.empty([3,n])

    l[0,:] = torch.min(X-t[:N], dim = 1).values
    l[1,:] = torch.min(X-t[N:2*N], dim = 1).values
    l[2,:] = torch.min(X-t[2*N:], dim = 1).values
    
    proj_u = torch.maximum(l[0,:, None] + t[:N],torch.maximum(l[1,:, None] + t[N:2*N], l[2,:, None] + t[2*N:]))

    return torch.mean(torch.max(proj_u-X, dim=1).values - torch.min(proj_u-X,dim=1).values)

PCA_trained_lr_DF = pd.read_pickle("DataFrames/PCA_trained_lr_DF.pkl")

PCA_DF = make_DF(t_dim_list = t_dim_list, lr_df = PCA_trained_lr_DF, Stats_Prob = Stats_Prob)

tic = time.time()
run_experiments(loss_function, PCA_DF, 1000, termination=False, Stats_Prob = Stats_Prob)
print((time.time() - tic)/60, " minutes")
PCA_DF.to_pickle("DataFrames/PCA_DF.pkl")