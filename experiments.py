import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
from trop_fns import *
import time


def make_DF(t_dim_list, n_list = [10, 100], 
                        N_list = [6, 28], 
                        type_list = ['branching', 'coalescent', 'gaussian'], 
                        grad_list = ["CD", "TD", "SGD", "TropSGD", "Adam", "Adamax", "TropAdamax"], 
                        num_exps = 50, 
                        zero_recentre = None, 
                        one_recentre = None, 
                        lr_df = None,
                        Stats_Prob = "FW"):
    
    #Form data frame columns with sample data details - sample size(data_count), dimensionality(data_dim), sample measure type(data_type), and an empty column for sample data(data)
    n_df = pd.DataFrame(n_list, columns = ['data_count'])
    if Stats_Prob == "LogR":
        N_df = pd.DataFrame(N_list, columns = ['data_dim'])
        empty_df = pd.DataFrame([None], columns = ['recentre'])
        supp_data_empty_df = pd.DataFrame([None], columns = ['supp_data'])
        N_df = N_df.merge(empty_df, how = "cross")
        n_df = n_df.merge(supp_data_empty_df, how = "cross")
        
        for index,row in N_df.iterrows():
            zero_recentre = torch.tensor(np.load(f"Data/{row['data_dim']}_zero_recentre.npy"))
            one_recentre = torch.tensor(np.load(f"Data/{row['data_dim']}_one_recentre.npy"))
            
            N_df.at[index, 'recentre'] = torch.cat((zero_recentre, one_recentre))
        
        for index, row in n_df.iterrows():
            n_df.at[index, 'supp_data'] = torch.tensor([0]*int(row['data_count']/2)+[1]*int(row['data_count']/2))
            
    elif Stats_Prob in ["WD2", "WDinf"]:
        N_df = pd.DataFrame(N_list, columns = ['data_dim'])
        supp_data_empty_df = pd.DataFrame([None], columns = ['supp_data'])
        N_df = N_df.merge(supp_data_empty_df, how = "cross")
        
        for index, row in N_df.iterrows():
            M = np.load(f"Data/{row['data_dim']}_supp.npy")
            supp = [np.where(np.isfinite(row))[0].tolist() for row in M]
            
            N_df.at[index, 'supp_data'] = supp
        
    else:
        N_df = pd.DataFrame(N_list, columns = ['data_dim'])
        supp_data_empty_df = pd.DataFrame([None], columns = ['supp_data'])
        N_df = N_df.merge(supp_data_empty_df, how = "cross")
        
    type_df = pd.DataFrame(type_list, columns = ['data_type'])
    empty_data_df = pd.DataFrame([None], columns = ['data'])
        
    data_df = n_df.merge(N_df, how = "cross").merge(type_df, how = "cross").merge(empty_data_df, how = "cross")
    
    #Filling in the data column with sampled data
    for index, row in data_df.iterrows():
        if row['data_type'] == 'branching':
            data = np.load(f"Data/Norm{row['data_dim']}.0_R100.0.npy")[:row['data_count'],:]
        elif row['data_type'] == 'coalescent':
            data = np.load(f"Data/Coal{row['data_dim']}.0_R100.0.npy")[:row['data_count'],:]
        elif row['data_type'] == 'gaussian': 
            data = np.load(f"Data/Gaus{row['data_dim']}.0_R100.0.npy")[:row['data_count'],:]
        
        data = torch.tensor(data/np.mean(np.max(data,1) - np.min(data,1)))
        data_df.at[index, 'data'] = data
    
    if Stats_Prob == "LogR":
        
        for index, row in data_df.iterrows():
                               
            shift = torch.cat((row['recentre'][:row['data_dim']].repeat(int(row['data_count']/2),1), row['recentre'][row['data_dim']:].repeat(int(row['data_count']/2),1)))
            data_df.at[index, 'data'] = row['data']+shift
                               
    elif Stats_Prob in ["WD2", "WDinf"]:
                               
        for index, row in data_df.iterrows():
            
            L = len(row['supp_data'])
                               
            if row['data_type'] == 'branching':
                small_data = np.load(f"Data/Norm{L}.0_R100.0.npy")[:row['data_count'],:]
            elif row['data_type'] == 'coalescent':
                small_data = np.load(f"Data/Coal{L}.0_R100.0.npy")[:row['data_count'],:]
            elif row['data_type'] == 'gaussian': 
                small_data = np.load(f"Data/Gaus{L}.0_R100.0.npy")[:row['data_count'],:]
                               
            data_df.at[index, 'supp_data'] = [torch.tensor(small_data/np.mean(np.max(small_data,1) - np.min(small_data,1))), row['supp_data']]
    
    exp_df = pd.DataFrame(range(num_exps), columns = ['num_exp'])
    t_dim_df = pd.DataFrame(t_dim_list, columns = ['t_dim'])
    empty_init_df = pd.DataFrame([None], columns = ['t_init'])
                           
    init_df = exp_df.merge(empty_init_df, how = "cross").merge(t_dim_df, how = "cross")
                           
    for index, row in init_df.iterrows():

        init = torch.randn(row['t_dim'])
        init_df.at[index, 't_init'] = init

    N_t_join_df = pd.DataFrame(pd.DataFrame(data={'data_dim': N_list, 't_dim': t_dim_list}))  
                           
    DF = pd.merge(pd.merge(N_t_join_df, data_df, on = 'data_dim'), init_df, on = 't_dim')
                           
    grad_df = pd.DataFrame(grad_list, columns = ['grad'])
    results_df = pd.DataFrame([[None, None, None]], columns = ['loss_values', 't_values', 'steps'])
    time_df = pd.DataFrame([None], columns = ['time taken'])
                           
    if lr_df is None:
        DF = DF.merge(grad_df, how = "cross").merge(results_df, how = "cross")
    else:
        DF = DF.merge(grad_df, how = "cross").merge(results_df, how = "cross").merge(lr_df, on = ['data_dim', 'data_count', 'grad'])
    
    return DF

#Defining Steepest Descent function
def id(i, norm):
    return i + 1

def SD(loss_fn, X, t, lr, num_steps, sample_loss_function = None, grad = "TD", termination = True, tol = 0.02, supp_data = None, Stats_Prob = "FW",verbose = True):
    steps = num_steps
    
    N = len(t)

    if verbose:
        loss_values = []
        t_values = []
#
    if grad == "Adam":
        optimizer = optim.Adam([t], lr=lr)
    elif grad in ["TropAdamax", "Adamax"]:
        optimizer = optim.Adamax([t], lr=lr)
        
    # Perform gradient descent
    for i in range(num_steps):

        if grad in ["SGD", "TropSGD"]:
            if Stats_Prob in ["WD2", "WDinf"]:
                X_sample, idx = torch_subsample(X, 1, index = True)
                Y_sample = supp_data[0][idx]
                loss = sample_loss_function(X_sample, Y_sample, supp_data[1], t)
            elif Stats_Prob == "LogR":
                X_sample, idx = torch_subsample(X, 1, index = True)
                loss = sample_loss_function(X_sample, t, Y=supp_data[idx])
            else:
                X_sample = torch_subsample(X, 1)
                loss = sample_loss_function(X_sample, t)
            if verbose:
                loss_values.append(loss_fn(t).item())
                t_values.append(t.detach().clone().numpy())
        else:
            loss = loss_fn(t)
            if verbose:
                loss_values.append(loss.item())
                t_values.append(t.detach().clone().numpy())
        loss.backward()
        
        #step direction computation
        if not torch.nonzero(t.grad.data).numel():
            continue
        if grad in ["CD", "TD", "TropSGD", "SGD"]:
            if grad in ["CD", "SGD"]:
                step = (t.grad.data)/torch.linalg.norm(t.grad.data, ord = 2)
                norm = torch.linalg.norm(t.grad.data, ord = 2)
            elif grad in ["TD", "TropSGD"]:
                step = (t.grad.data > 0).float()
                norm = torch.max(t.grad.data) - torch.min(t.grad.data)
            t.data = t.data - lr * step * step_size(i, norm)
        elif grad in ["Adam", "Adamax"]:
            optimizer.step()
        elif grad in ["TropAdamax"]:
            t.grad.data = (t.grad.data > 0).float() * (torch.max(t.grad.data) - torch.min(t.grad.data))
            optimizer.step()
        t.grad.data.zero_()

        #convergence tests
        if i>N and termination and verbose:
            if termination == "t_val" and (np.ptp(t_values[-1] - t_values[-N-1]) < tol):
                steps = i
                break
            if termination == "f_val" and (np.abs(loss_values[-1]-loss_values[-N-1]) < tol):
                steps = i
                break
    
    if verbose:
        return steps, loss_values, t_values
    else:
        return steps, [loss_fn(t).item()], [t.detach().clone().numpy()]

def lr_training(loss_function, lr_DF, Stats_Prob = "FW"):
                           
    lr_DF['time'] = ''
    
    for index, row in lr_DF.iterrows():
        
            if Stats_Prob == "LogR":
                def loss_fn(t):
                    return loss_function(row['data'], t, row['supp_data'])
            elif Stats_Prob in ["WD2", "WDinf"]:
                def loss_fn(t):
                    return loss_function(row['data'], row['supp_data'][0], row['supp_data'][1], t)
            else:
                def loss_fn(t):
                    return loss_function(row['data'], t)
            
            GD_t = row['t_init'].clone().detach().requires_grad_(True)
                           
            start = time.time()

            steps,loss,t = SD(loss_fn, row['data'], GD_t, row['lrs'], 1000, sample_loss_function = loss_function, grad = row['grad'], termination=False, supp_data = row['supp_data'],Stats_Prob=Stats_Prob,verbose=False)
                           
            stop = time.time()

            lr_DF.at[index, 'loss_values'] = loss
            lr_DF.at[index, 't_values'] = t
            lr_DF.at[index, 'steps'] = steps
            lr_DF.at[index, 'time'] = stop - start

def run_experiments(loss_function, DF, num_steps, termination=True, Stats_Prob = "FW"):
    
        for index, row in DF.iterrows():
            
            if Stats_Prob == "LogR":
                def loss_fn(t):
                    return loss_function(row['data'], t, row['supp_data'])
            elif Stats_Prob in ["WD2", "WDinf"]:
                def loss_fn(t):
                    return loss_function(row['data'], row['supp_data'][0], row['supp_data'][1], t)
            else:
                def loss_fn(t):
                    return loss_function(row['data'], t)
            
            GD_t = row['t_init'].clone().detach().requires_grad_(True)
                
            start = time.time()

            steps,loss_vals,t_vals = SD(loss_fn, row['data'], GD_t, row['lrs'], num_steps, sample_loss_function = loss_function, grad = row['grad'], termination=termination, supp_data = row['supp_data'], Stats_Prob=Stats_Prob,verbose = False)

            stop = time.time()
            
            DF.at[index, 'time'] = stop - start
            DF.at[index, 'loss_values'] = loss_vals
            DF.at[index, 't_values'] = t_vals
            DF.at[index, 'steps'] = steps
                           
def error(loss_values,global_soln = 0.8, relative = True):
    if relative:
        return np.log(np.array(loss_values) - global_soln*0.99) - np.log(np.abs(global_soln)*0.99)
    else:
        return np.log(np.array(loss_values) - global_soln + 0.0001)
                           
def lr_experiment_analysis(lr_DF, relative = True):
    
    solution_DF = lr_DF.copy()
    solution_DF['optimal_val'] = solution_DF.apply(lambda row: min(row['loss_values']), axis=1)
    solution_DF['optimal_t'] = solution_DF.apply(lambda row: row['t_values'][np.argmin(row['loss_values'])], axis=1)
    solution_DF = solution_DF.groupby(['data_dim', 'data_count', 'data_type'])['optimal_val'].agg([('min', 'min')]).reset_index()
    
    log_error_DF = lr_DF[['data_dim', 'data_count', 'data_type', 'grad', 'loss_values', 'lrs']].merge(solution_DF, on = ['data_dim', 'data_count', 'data_type'])
    if not relative :
        log_error_DF['log_error']  = log_error_DF[['loss_values', 'min']].apply(lambda x: error(x['loss_values'], x['min'], relative = False), axis=1)
    else:
        log_error_DF['log_error']  = log_error_DF[['loss_values', 'min']].apply(lambda x: error(x['loss_values'], x['min']), axis=1)
        
    mean_std_DF = log_error_DF.groupby(['data_dim', 'data_count', 'data_type', 'grad', 'lrs'])['log_error'].agg([('mean', 'mean'), ('std', 'std')]).reset_index()
    mean_std_DF
        
    return mean_std_DF
                           
def experiment_analysis(DF, Stats_Prob = "FW"):
    
    DF['loss'] = DF['loss_values'].str[-1]
    mean_std_DF = DF.groupby(['data_dim', 'data_count', 'data_type', 'grad'])['loss'].agg([('mean', 'mean'), ('std', 'std')]).reset_index()
    mean_std_DF = mean_std_DF.pivot(index=['data_dim', 'data_count', 'data_type'], columns='grad', values=['mean', 'std']).reset_index()

    solution_DF = DF.copy()
    solution_DF['optimal_val'] = solution_DF.apply(lambda row: min(row['loss_values']), axis=1)
    solution_DF['optimal_t'] = solution_DF.apply(lambda row: row['t_values'][np.argmin(row['loss_values'])], axis=1)
    solution_DF = solution_DF.groupby(['data_dim', 'data_count', 'data_type'])['optimal_val'].agg([('min', 'min')]).reset_index()

    log_error_DF = DF[['data_dim', 'data_count', 'data_type', 'grad', 'loss', 'loss_values']].merge(solution_DF, on = ['data_dim', 'data_count', 'data_type'])
    if Stats_Prob == "LR":
        log_error_DF['log_error_values']  = log_error_DF[['loss_values', 'min']].apply(lambda x: error(x['loss_values'], x['min'], relative = False), axis=1)
    else:
        log_error_DF['log_error_values']  = log_error_DF[['loss_values', 'min']].apply(lambda x: error(x['loss_values'], x['min']), axis=1)

    log_error_DF['log_error'] = log_error_DF['log_error_values'].str[-1]
        
    return mean_std_DF, solution_DF, log_error_DF
                                                      
def CDF(log_error_DF, 
        datadims =[[6,10], [6,100], [28, 10], [28, 100]],
        datatypes = ['branching', 'coalescent', 'gaussian'],
        grad_types = ["CD", "TD", "SGD", "TropSGD", "Adam", "Adamax", "TropAdamax"], 
        save_plot = True,
        Stats_Prob = "FW"):
    
#     for frac_step in frac_steps:
#         log_error_DF[str(frac_step)] = log_error_DF['log_error'].apply(extract_frac_steps, frac_step = frac_step)
        
    datadim_count = len(datadims)
    datatype_count = len(datatypes)
    fig, axs = plt.subplots(datatype_count, datadim_count, figsize = (8*datadim_count,8*datatype_count))
    #fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    ax = fig.add_subplot(111,frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    fig.suptitle("CDF of Gradient Methods for %s Loss"%Stats_Prob,fontsize = 45)
    
    x_max = np.nanmax(log_error_DF[[(grad in grad_types) for grad in log_error_DF['grad']]]['log_error'].apply(np.nanmax))
    x_min = np.nanmin(log_error_DF[[(grad in grad_types) for grad in log_error_DF['grad']]]['log_error'].apply(np.nanmin))
    
    for j, datatype in enumerate(datatypes):
        
        axs[j,0].set_ylabel('CDF for %s Data' %datatype.capitalize(), fontsize = 25)
        
        for i, datadim in enumerate(datadims):
            
            axs[j, i].grid()
            axs[j, i].set_xlim([x_min- (x_max-x_min)*0.05,x_max+ (x_max-x_min)*0.05])
            
            for grad in grad_types:
                if grad == "TropSGD":
            
                    axs[j, i].ecdf(np.array(log_error_DF[(log_error_DF['data_dim'] == datadim[0])
                                                &(log_error_DF['data_count'] == datadim[1])
                                                &(log_error_DF['data_type'] == datatype)
                                                &(log_error_DF['grad'] == grad)
                                                        ]['log_error']), label = "TSGD")
                elif grad == "TropAdamax":
            
                    axs[j, i].ecdf(np.array(log_error_DF[(log_error_DF['data_dim'] == datadim[0])
                                                &(log_error_DF['data_count'] == datadim[1])
                                                &(log_error_DF['data_type'] == datatype)
                                                &(log_error_DF['grad'] == grad)
                                                        ]['log_error']), label = "TrAdamax")
                else:
            
                    axs[j, i].ecdf(np.array(log_error_DF[(log_error_DF['data_dim'] == datadim[0])
                                                &(log_error_DF['data_count'] == datadim[1])
                                                &(log_error_DF['data_type'] == datatype)
                                                &(log_error_DF['grad'] == grad)
                                                        ]['log_error']), label = grad)
                
                axs[j,i].tick_params(axis='both', labelsize=20)
            
    axs[-1, -1].legend(prop={'size': 20})
            
    for i, datadim in enumerate(datadims):
        
        axs[0,i].set_title("Solving %d by %d Data" %(datadim[0], datadim[1]), fontsize = 30)
        
        axs[-1,i].set_xlabel('Log Relative Error', fontsize = 25)
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.93)
    
    if save_plot:
        plt.savefig('Figures/%s_CDF_plot.png'%(Stats_Prob))

def latex_table(log_error_DF):
    mean_log_error_DF = log_error_DF.groupby(['data_dim', 'data_count', 'data_type', 'grad'])['log_error'].agg([('mean', 'mean'), ('std', 'std')]).reset_index()
    mean_log_error_DF = mean_log_error_DF.pivot(index=['data_dim', 'data_count', 'data_type'], columns='grad', values=['mean', 'std']).reset_index()
    #mean_log_error_DF.loc[-1] = ['mean'].append(mean_log_error_DF['mean'].mean(axis=0))

    mean_log_error_DF['dataset'] = mean_log_error_DF['data_dim'].astype('string')+"$times$"+mean_log_error_DF['data_count'].astype('string')+' '+mean_log_error_DF['data_type'].str.capitalize()+' Data'
    for grad in ['CD', 'TD', 'SGD', 'TropSGD', 'Adam', 'Adamax', 'TropAdamax']:
        mean_log_error_DF[grad] = mean_log_error_DF['mean', grad].map(lambda x: f'{x:.2f}')

    print(mean_log_error_DF[['dataset', 'CD', 'TD', 'SGD', 'TropSGD', 'Adam', 'Adamax', 'TropAdamax']].to_latex(index = False))
    
    return mean_log_error_DF