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
Stats_Prob = "LR"
LR_lr_DF =  make_DF(t_dim_list, num_exps = 10, lr_df = None, Stats_Prob = Stats_Prob)
LR_lr_DF = LR_lr_DF.merge(lr_list_DF, how = "cross")

Stats_Prob = "LR"
def loss_function(X,t):
    
    n = len(X)
    v = torch.topk(X + t,2).values
    
    return torch.max(v[:,0] - v[:,1])

tic = time.time()
lr_training(loss_function, LR_lr_DF, Stats_Prob = Stats_Prob)
print("LR lr time: ", (time.time() - tic)/60, " minutes")
LR_lr_DF['loss'] = LR_lr_DF['loss_values'].str[-1]

LR_lr_DF.to_pickle("DataFrames/LR_lr_DF.pkl")