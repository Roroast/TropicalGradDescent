import random
import torch
import numpy as np
from experiments import make_DF, lr_training
import pandas as pd
import time

N_list = [6,28]
t_dim_list = N_list
random.seed(2024)
np.random.seed(2024)
torch.manual_seed(2024)
init_lr_list = list(np.logspace(-6, 4, num = 11, base = np.e))
lr_list_DF = pd.DataFrame(init_lr_list, columns = ['lrs'])
Stats_Prob = "WD2"
WD2_lr_DF =  make_DF(t_dim_list, num_exps = 10, lr_df = None, Stats_Prob = Stats_Prob)
WD2_lr_DF = WD2_lr_DF.merge(lr_list_DF, how = "cross")

Stats_Prob = "WD2"
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

tic = time.time()
lr_training(loss_function, WD2_lr_DF, Stats_Prob = Stats_Prob)
print("WD2 lr time: ", (time.time() - tic)/60, " minutes")
WD2_lr_DF['loss'] = WD2_lr_DF['loss_values'].str[-1]

WD2_lr_DF.to_pickle("DataFrames/WD2_lr_DF.pkl")