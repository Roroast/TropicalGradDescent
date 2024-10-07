import random
import torch
import numpy as np
from experiments import make_DF, lr_training
import pandas as pd
import time

N_list = [6,28]
t_dim_list = [3*N for N in N_list]
random.seed(2024)
np.random.seed(2024)
torch.manual_seed(2024)
init_lr_list = list(np.logspace(-6, 4, num = 11, base = np.e))
lr_list_DF = pd.DataFrame(init_lr_list, columns = ['lrs'])
Stats_Prob = "PCA"
PCA_lr_DF =  make_DF(t_dim_list, num_exps = 10, lr_df = None, Stats_Prob = Stats_Prob)
PCA_lr_DF = PCA_lr_DF.merge(lr_list_DF, how = "cross")

Stats_Prob = "PCA"
def loss_function(X,t):
    
    N = int(len(t)/3)
    n = len(X)
    l = torch.empty([3,n])

    l[0,:] = torch.min(X-t[:N], dim = 1).values
    l[1,:] = torch.min(X-t[N:2*N], dim = 1).values
    l[2,:] = torch.min(X-t[2*N:], dim = 1).values
    
    proj_u = torch.maximum(l[0,:, None] + t[:N],torch.maximum(l[1,:, None] + t[N:2*N], l[2,:, None] + t[2*N:]))

    return torch.mean(torch.max(proj_u-X, dim=1).values - torch.min(proj_u-X,dim=1).values)

tic = time.time()
lr_training(loss_function, PCA_lr_DF, Stats_Prob = Stats_Prob)
print("PCA lr time: ", (time.time() - tic)/60, " minutes")
PCA_lr_DF['loss'] = PCA_lr_DF['loss_values'].str[-1]

PCA_lr_DF.to_pickle("DataFrames/PCA_lr_DF.pkl")