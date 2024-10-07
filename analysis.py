import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import math


#Run experiments
def experiment_analysis(DF):
    
    
    mean_std_DF = DF.groupby(['data_dim', 'data_count', 'data_type', 'grad'])['loss'].agg([('mean', 'mean'), ('std', 'std')]).reset_index()
    mean_std_DF = mean_std_DF.pivot(index=['data_dim', 'data_count', 'data_type'], columns='grad', values=['mean', 'std']).reset_index()

    solution_DF = DF.copy()
    solution_DF['optimal_val'] = solution_DF.apply(lambda row: min(row['loss_values']), axis=1)
    solution_DF['optimal_t'] = solution_DF.apply(lambda row: row['t_values'][np.argmin(row['loss_values'])], axis=1)
    solution_DF = solution_DF.groupby(['data_dim', 'data_count', 'data_type'])['optimal_val'].agg([('min', 'min')]).reset_index()
        
    return mean_std_DF, solution_DF

def extract_frac_steps(loss_vals, frac_step):
    k = len(loss_vals)
    return loss_vals[int(k*frac_step)-1]

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
        
        axs[j,0].set_ylabel('Log Relative Error CDF for %s Data' %datatype.capitalize(), fontsize = 25)
        
        for i, datadim in enumerate(datadims):
            
            axs[j, i].grid()
            axs[j, i].set_xlim([x_min- (x_max-x_min)*0.05,x_max+ (x_max-x_min)*0.05])
            
            for grad in grad_types:
            
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
        
        
#Logplot of error for successful iterations
def error(loss_values,global_soln = 0.8, relative = True):
    if relative:
        return np.log(np.array(loss_values) - global_soln*0.99) - np.log(np.abs(global_soln)*0.99)
    else:
        return np.log(np.array(loss_values) - global_soln + 0.0001)

def log_error_plot(DF, 
                    solution_DF, 
                    datadims =[[6,10], [6,100], [28, 10], [28, 100]],
                    datatypes = ['branching', 'coalescent', 'gaussian'],
                    grads = ["CD", "TD", "SGD", "TropSGD", "Adam", "Adamax", "TropAdamax"], 
                    save_plot = True,
                    zero_solns = False,
                    Stats_Prob = "FW"):
    
    datadim_count = len(datadims)
    datatype_count = len(datatypes)
    fig, axs = plt.subplots(datatype_count, datadim_count, figsize = (8*datadim_count,8*datatype_count))
    #fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    ax = fig.add_subplot(111,frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    fig.suptitle("Minimising %s Loss"%Stats_Prob,fontsize = 45)
    #fig.tight_layout(rect=[0, 0, 1, 2])
    
    log_error_DF = DF[['data_dim', 'data_count', 'data_type', 'grad', 'loss', 'loss_values']].merge(solution_DF, on = ['data_dim', 'data_count', 'data_type'])
    if Stats_Prob == "LR":
        log_error_DF['log_error_values']  = log_error_DF[['loss_values', 'min']].apply(lambda x: error(x['loss_values'], x['min'], relative = False), axis=1)
    else:
        log_error_DF['log_error_values']  = log_error_DF[['loss_values', 'min']].apply(lambda x: error(x['loss_values'], x['min']), axis=1)
        
    log_error_DF['log_error'] = log_error_DF['log_error_values'].str[-1]
    
    for j, datadim in enumerate(datadims):
        
        axs[0,j].set_title("Solving %d by %d Data" %(datadim[0], datadim[1]), fontsize = 30)
        axs[-1,j].set_xlabel('Step Count', fontsize = 25)
        
        for i, datatype in enumerate(datatypes):
        
            sub_DF = log_error_DF[(log_error_DF['data_dim'] == datadim[0])&(log_error_DF['data_count'] == datadim[1])&(log_error_DF['data_type'] == datatype)]

            y_max = np.nanmax(sub_DF['log_error_values'].apply(np.nanmax))
            y_min = np.nanmin(sub_DF['log_error_values'].apply(np.nanmin))
            
            axs[i,j].set_ylim([y_min,y_max])
            
            axs[i,j].grid()

            for grad in grads:

                sub_sub_DF = sub_DF[(sub_DF['grad'] == grad)]
                axs[i,j].plot(np.mean(sub_sub_DF['log_error_values']), label = grad)

                #axs[i,j].plot(range(0,num_steps,10), mean_log_loss_values[j,:num_steps:10], "blue", linewidth = 2.5)
                
            axs[i, j].legend(prop={'size': 15})
            
    for i, datatype in enumerate(datatypes):
        axs[i,0].set_ylabel('Mean Log Error for %s Data' %datatype.capitalize(), fontsize = 25)
            
    fig.tight_layout()
    fig.subplots_adjust(top=0.93)
    
    if save_plot:
        plt.savefig('Figures/%s_Log_Error_plot.png'%(Stats_Prob))
        
        
    return log_error_DF

def lr_experiment_analysis(lr_DF, relative = True):
    
    solution_DF = lr_DF.copy()
    solution_DF['optimal_val'] = solution_DF.apply(lambda row: min(row['loss_values']), axis=1)
    solution_DF['optimal_t'] = solution_DF.apply(lambda row: row['t_values'][np.argmin(row['loss_values'])], axis=1)
    solution_DF = solution_DF.groupby(['data_dim', 'data_count', 'data_type'])['optimal_val'].agg([('min', 'min')]).reset_index()
    print(solution_DF)
    
    log_error_DF = lr_DF[['data_dim', 'data_count', 'data_type', 'grad', 'loss', 'lrs']].merge(solution_DF, on = ['data_dim', 'data_count', 'data_type'])
    if not relative :
        log_error_DF['log_error']  = log_error_DF[['loss', 'min']].apply(lambda x: error(x['loss'], x['min'], relative = False), axis=1)
    else:
        log_error_DF['log_error']  = log_error_DF[['loss', 'min']].apply(lambda x: error(x['loss'], x['min']), axis=1)
    print(log_error_DF)
        
    mean_std_DF = log_error_DF.groupby(['data_dim', 'data_count', 'data_type', 'grad', 'lrs'])['log_error'].agg([('mean', 'mean'), ('std', 'std')]).reset_index()
    mean_std_DF
        
    return mean_std_DF

def plot_lr_results(mean_std_DFs, DF_names, 
                    grad_list = ["CD", "TD", "SGD", "TropSGD", "Adam", "Adamax", "TropAdamax"], 
                    datasets = [[6,10],[6,100],[28,10],[28,100]], 
                    search = "wide",
                    save_name = None):
    
    trained_lr_dfs = [pd.DataFrame({'data_dim' : [], 'data_count' : [], 'grad' : [], 'lr' : []}) for DF_name in DF_names]
    
    grad_count = len(grad_list)
    problem_count = len(DF_names)
    print([8*grad_count,8*problem_count])
    fig, axs = plt.subplots(problem_count, grad_count, figsize = (8*grad_count,4*problem_count))
    ax = fig.add_subplot(111,frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.title("Mean Log Relative Error Against Learning Rate",fontsize = 45)
    fig.tight_layout(rect=[0, 0, 1, 2])
    
    for k,mean_std_DF in enumerate(mean_std_DFs):
        
        mini = min(mean_std_DF['mean'])
        maxi = max(mean_std_DF['mean'])
        if math.isnan(maxi):
            maxi = 5
        if math.isnan(mini):
            mini = -5
        
        y_lims = [mini, maxi]
        print(y_lims)
        axs[k,0].set_ylabel(f'{DF_names[k]} Loss', fontsize = 35)
        
        for j, dataset in enumerate(datasets):
            
            colour = ['b', 'g', 'r', 'c'][j]

            for i, grad in enumerate(grad_list):

                axs[k,i].set_ylim(y_lims)

                sub_DF = mean_std_DF[(mean_std_DF['grad'] == grad) & (mean_std_DF['data_dim'] == dataset[0]) & (mean_std_DF['data_count'] == dataset[1])]# & (mean_std_DF['data_type'] == dataset[2])
                
                sub_sub_DF = sub_DF[sub_DF['data_type'] == 'branching']
                axs[k,i].plot(np.log(sub_sub_DF['lrs']), sub_sub_DF['mean'],
                                  #sub_sub_DF['std'], 
                                  label = str(dataset[0])+'x'+str(dataset[1]), color = colour)
                    
                for dt in ['coalescent', 'gaussian']:
                    sub_sub_DF = sub_DF[sub_DF['data_type'] == dt]
                    axs[k,i].plot(np.log(sub_sub_DF['lrs']), sub_sub_DF['mean'],
                                   color = colour)

                if search == "narrow":
                    min_df = sub_DF.groupby(['data_dim', 'data_count', 'grad', 'lrs'])['mean'].agg([('mean', 'mean')]).reset_index()
                    min_lr = min_df.loc[min_df['mean'].idxmin(),'lrs']

                #axs[j,i].axvline(x=np.log(min_lr), label = f"lr = {min_lr:.2f}")

                    trained_lr_dfs[k].loc[len(trained_lr_dfs[k])] = [dataset[0], dataset[1], grad, min_lr]
                    
    axs[0,0].legend(prop={'size': 20})
                
    for i, grad in enumerate(grad_list):
        
        axs[0,i].set_title(grad, fontsize = 35)
        axs[-1,i].set_xlabel('ln(Learning Rate)', fontsize = 35)
            
    fig.tight_layout(rect=[0, 0, 1, 2])
    #fig.subplots_adjust(top=1)

    if save_name is not None:
        plt.savefig('Figures/%s_LR_training_plot.png'%(save_name), bbox_inches='tight')
        
    return trained_lr_dfs

def latex_table(log_error_DF):
    mean_log_error_DF = log_error_DF.groupby(['data_dim', 'data_count', 'data_type', 'grad'])['log_error'].agg([('mean', 'mean'), ('std', 'std')]).reset_index()
    mean_log_error_DF = mean_log_error_DF.pivot(index=['data_dim', 'data_count', 'data_type'], columns='grad', values=['mean', 'std']).reset_index()
    #mean_log_error_DF.loc[-1] = ['mean'].append(mean_log_error_DF['mean'].mean(axis=0))

    mean_log_error_DF['dataset'] = mean_log_error_DF['data_dim'].astype('string')+"$times$"+mean_log_error_DF['data_count'].astype('string')+' '+mean_log_error_DF['data_type'].str.capitalize()+' Data'
    for grad in ['CD', 'TD', 'SGD', 'TropSGD', 'Adam', 'Adamax', 'TropAdamax']:
        mean_log_error_DF[grad] = mean_log_error_DF['mean', grad].map(lambda x: f'{x:.2f}')

    print(mean_log_error_DF[['dataset', 'CD', 'TD', 'SGD', 'TropSGD', 'Adam', 'Adamax', 'TropAdamax']].to_latex(index = False))
    
    return mean_log_error_DF
