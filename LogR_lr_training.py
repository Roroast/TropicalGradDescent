import random
import torch
import numpy as np
from experiments import make_DF, lr_training
import pandas as pd
import time

sig = torch.nn.Sigmoid()

N_list = [6,28]
t_dim_list = [2*N for N in N_list]
random.seed(2024)
np.random.seed(2024)
torch.manual_seed(2024)
init_lr_list = list(np.logspace(-6, 4, num = 11, base = np.e))
lr_list_DF = pd.DataFrame(init_lr_list, columns = ['lrs'])
Stats_Prob = "LogR"
LogR_lr_DF =  make_DF(t_dim_list, num_exps = 10, lr_df = None, Stats_Prob = Stats_Prob)
LogR_lr_DF = LogR_lr_DF.merge(lr_list_DF, how = "cross")

Stats_Prob = "LogR"
def loss_function(X,t,Y):
    N = int(len(t)/2)
    
    h = torch.max(X-t[:N], dim=1).values - torch.min(X-t[:N],dim=1).values - torch.max(X-t[N:], dim=1).values + torch.min(X-t[N:],dim=1).values
    f = Y*torch.log(sig(h)) + (1-Y)*torch.log(sig(-h))

    return -torch.exp(torch.mean(f))

tic = time.time()
lr_training(loss_function, LogR_lr_DF, Stats_Prob = Stats_Prob)
print("LogR lr time: ", (time.time() - tic)/60, " minutes")
LogR_lr_DF['loss'] = LogR_lr_DF['loss_values'].str[-1]

LogR_lr_DF.to_pickle("DataFrames/LogR_lr_DF.pkl")