import torch
import numpy as np


def torch_subsample(X,k, index = False):
    perm = torch.randperm(X.size(0))
    idx = perm[:k]
    
    if index:
        return X[idx], idx
    else:
        return X[idx]
    
def step_size(i, norm):
    return norm /torch.sqrt(torch.tensor(i+1))

def transpose_trop_mult(a, b, args=False):
    # we calculate (ab)_{ij} = sum a_jk b_ik
    if a.shape[1] == b.shape[1]:
        if args:
            # We calculate c_{ijk} = a_{jk} + b_{ik}
            c = a[np.newaxis, :, :] + b[:, np.newaxis, :]
            # We find the maximal k
            index = np.nanargmax(c, axis=2)
            prod = np.array([[c[i, j, index[i, j]] for j in range(a.shape[0])] for i in range(b.shape[0])])
            return prod, index
        else:
            prod = np.nanmax(a[np.newaxis, :, :] + b[:, np.newaxis, :], axis=2)
            return prod
    else:
        print('Matrix shapes do not match')

def rand_supp(N, M):
    #Generating a random map
    permutation = np.random.permutation(range(N)).tolist()
    coords = list(range(M)) + list(np.random.choice(range(M), N-M))

    supp = [[] for n in range(M)]

    for n in range(N):
        supp[coords[n]].append(permutation[n])
    
    return supp

def SuppToMatrix(supp, N, M):
    Map = np.full((M, N), np.nan)
    for i in range(M):
        for j in supp[i]:
            Map[i, j] = 0
    return Map


##################################Pytorch Part
# my_inf = torch.const(-1e^11)
def trop_mul(a, supp, Mat):
    # this function returns the tropical multiplication of matrix
    M = len(supp)
    b = torch.zeros(M)
    for i in range(M):
        row_supp = sorted(supp[i])
        num_supp = len(row_supp)
        row = torch.zeros(num_supp)
        for j in range(num_supp):
            row[j] = Mat[row_supp[j]]+a[row_supp[j]]
        b[i] = torch.max(row)
    return b

# Define the objective function
def objective_function(X, Y, supp, Mat, p):
    # X is input data of dim N
    # Y is output data of dim M
    
    num_data = len(X)
    obj_func = 0
    
    for i in range(num_data):
        Z = Y[i,:]-trop_mul(X[i,:], supp, Mat)
        obj_func += torch.sum((torch.max(Z) - torch.min(Z))**p)/num_data
    
    return obj_func

# Define the objective function
def objective_function_infty(X, Y, supp, Mat):
    # X is input data of dim N
    # Y is output data of dim M
    
    num_data = len(X)
    obj_func = 0
    norms = torch.zeros(len(X))
    
    for i in range(num_data):
        Z = Y[i,:]-trop_mul(X[i,:], supp, Mat)
        norms[i] = (torch.max(Z) - torch.min(Z))
    obj_func = torch.max(norms)
    
    return obj_func

# Define the identity matrix w.r.t given supp
def IdProj(X, supp):
    M = len(supp)
    [n, N] = X.size()
    Y = torch.zeros(n, M)
    IdMat = torch.zeros(N)
    for i in range(n):
        Y[i,:] = trop_mul(X[i,:], supp, IdMat)
        
    return Y 

def trop_dist(x,y):

    return torch.max(x-y, dim=1).values - torch.min(x-y,dim=1).values

def trop_dist_1d(x,y):
    return torch.max(x-y) - torch.min(x-y)

def matrix_upgrade(supp):
    K = len(supp)
    L = len([j for i in supp for j in i])
    
    big_supp = [[0]*(K) for _ in range(K)]
    new_supp = []
    
    for i in range(K):
        for j in range(i,K):
            big_supp[i][j] = []
            big_supp[j][i] = []
            for a in supp[j]:
                for b in supp[i]:
                    r = max(a,b)
                    s = min(a,b)
                    
                    big_supp[i][j].append(int(r-s-1+s*(L-1)- (s-1)*s/2))
                    big_supp[j][i].append(int(r-s-1+s*(L-1)- (s-1)*s/2))
                    
            if i<j:
                new_supp.append(big_supp[i][j])
            
    return new_supp

def proj(X):
    proj_mat = torch.tensor([[1, -0.5, -0.5],[0, np.sqrt(3)/2, -np.sqrt(3)/2]],dtype = torch.double)
    return torch.matmul(proj_mat, X)/torch.sqrt(torch.tensor(1.5))

def rev_proj(X):
    rev_proj_mat = torch.tensor([[1, 0],[-1/2, np.sqrt(3)/2],[-1/2, -np.sqrt(3)/2]],dtype = torch.double)
    return torch.t(torch.matmul(rev_proj_mat, X)/torch.sqrt(torch.tensor(1.5)))