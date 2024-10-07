import torch
import random
import numpy as np
import pandas as pd
import time
from experiments import make_DF, run_experiments

random.seed(2024)
np.random.seed(2024)
torch.manual_seed(2024)

Stats_Prob = "WD2"
t_dim_list = [6, 28]

def step_size(i, norm):
    return norm /torch.sqrt(torch.tensor(i+1))

def loss_function(X,Y,supp,t):
    # X is input data of dim N
    # Y is output data of dim M
    
    num_data = len(X)
    obj_func = 0
    M = len(supp)
    b = torch.zeros(num_data,M)
    
    for j, row_supp in enumerate(supp):
        rows = t[row_supp]+X[:,row_supp]
        b[:,j] = torch.max(rows, dim = 1).values
    Z = Y-b
    obj_func = torch.mean((torch.max(Z, dim = 1).values - torch.min(Z, dim = 1).values)**2)
    
    return torch.sqrt(obj_func)
    
WD2_trained_lr_DF = pd.read_pickle("DataFrames/WD2_trained_lr_DF.pkl")

WD2_DF = make_DF(t_dim_list = t_dim_list, lr_df = WD2_trained_lr_DF, Stats_Prob = Stats_Prob)

tic = time.time()
run_experiments(loss_function, WD2_DF, 1000, termination=False, Stats_Prob = Stats_Prob)
print((time.time() - tic)/60, " minutes")
WD2_DF.to_pickle("DataFrames/WD2_DF.pkl")