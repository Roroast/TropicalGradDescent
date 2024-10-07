import random
import torch
import numpy as np
from experiments import make_DF, lr_training
import pandas as pd
import time

N_list = [6,28]
t_dim_list = N_list
random.seed(2023)
np.random.seed(2023)
torch.manual_seed(2023)
init_lr_list = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
wide_lr_list_DF = pd.DataFrame(init_lr_list, columns = ['lrs'])
Stats_Prob = "FW"
wide_FW_lr_DF =  make_DF(t_dim_list, num_exps = 5, lr_df = None, Stats_Prob = Stats_Prob)
wide_FW_lr_DF = wide_FW_lr_DF.merge(wide_lr_list_DF, how = "cross")
Stats_Prob = "FM"
wide_FM_lr_DF =  make_DF(t_dim_list, num_exps = 5, lr_df = None, Stats_Prob = Stats_Prob)
wide_FM_lr_DF = wide_FM_lr_DF.merge(wide_lr_list_DF, how = "cross")
Stats_Prob = "WD2"
wide_WD2_lr_DF =  make_DF(t_dim_list, num_exps = 5, lr_df = None, Stats_Prob = Stats_Prob)
wide_WD2_lr_DF = wide_WD2_lr_DF.merge(wide_lr_list_DF, how = "cross")
Stats_Prob = "WDinf"
wide_WDinf_lr_DF =  make_DF(t_dim_list, num_exps = 5, lr_df = None, Stats_Prob = Stats_Prob)
wide_WDinf_lr_DF = wide_WDinf_lr_DF.merge(wide_lr_list_DF, how = "cross")
t_dim_list = [3*N for N in N_list]
Stats_Prob = "PCA"
wide_PCA_lr_DF =  make_DF(t_dim_list, num_exps = 5, lr_df = None, Stats_Prob = Stats_Prob)
wide_PCA_lr_DF = wide_PCA_lr_DF.merge(wide_lr_list_DF, how = "cross")
t_dim_list = [2*N for N in N_list]
Stats_Prob = "LogR"
wide_LogR_lr_DF =  make_DF(t_dim_list, num_exps = 5, lr_df = None, Stats_Prob = Stats_Prob)
wide_LogR_lr_DF = wide_LogR_lr_DF.merge(pd.DataFrame([0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300], columns = ['lrs']), how = "cross")
t_dim_list = N_list
Stats_Prob = "LR"
wide_LR_lr_DF =  make_DF(t_dim_list, num_exps = 5, lr_df = None, Stats_Prob = Stats_Prob)
wide_LR_lr_DF = wide_LR_lr_DF.merge(wide_lr_list_DF, how = "cross")

Stats_Prob = "FW"
def loss_function(X,t):
    return torch.mean((torch.max(X-t, dim=1).values - torch.min(X-t,dim=1).values))

tic = time.time()
lr_training(loss_function, wide_FW_lr_DF, Stats_Prob=Stats_Prob)
print("FW lr time: ", (time.time() - tic)/60, " minutes")
wide_FW_lr_DF['loss'] = wide_FW_lr_DF['loss_values'].str[-1]

wide_FW_lr_DF.to_pickle("DataFrames/1000_wide_FW_lr_DF.pkl")