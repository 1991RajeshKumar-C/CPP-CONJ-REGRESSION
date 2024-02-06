import numpy as np
import random
import os

os.system('clear')

A = np.random.normal(0, 0.1, (7, 15)) 
x = np.zeros((15,1), dtype=float)
spars = 2
Thr = 1/(2*spars - 1)
x[2] = 1.0
x[0] = 1.0 
max_cor = -np.inf
for each in range(A.shape[1]):
    for each1 in range(A.shape[1]):
        if each is not each1:
            temp = np.abs(A[:, each].T @ A[:, each1])
            if temp > max_cor:
                max_cor = temp

y = A @ x
y_glob = y

def swap(a, b):
    a, b = b, a
    return a, b

def hardthres(x_est, supp_vec):
    if len(supp_vec) == 0:
        supp_vec_temp = np.argsort(np.abs(x_est.squeeze()))[:(A.shape[1] - spars)]
        supp_vec = np.argsort(np.abs(x_est.squeeze()))[(A.shape[1] - spars):]
    else:
        supp_vec_temp = np.argsort(np.abs(x_est.squeeze()))[:(spars)]
    x_est = np.delete(x_est, supp_vec_temp)
    x_est = x_est.reshape(-1,1)
    if len(supp_vec) == spars:
        supp_vec = supp_vec
    else:
        supp_vec = np.delete(supp_vec, supp_vec_temp)
    return x_est, supp_vec

def suppfill(supp_vec, x_est):
    if (A.shape[1] > len(supp_vec)):
        x_est1 = np.zeros((A.shape[1], 1))
        for ele in range(len(supp_vec)):
            x_est1[supp_vec[ele]] = x_est[ele]
        x_est = x_est1
    return x_est

def CoSaMP(y, y_glob, res):
    for j in range(100):
        temp = A.T @ y
        supp_vec = np.argsort(-temp.squeeze())[:(2*spars)]
        vec = A[:, supp_vec]
        x_est = np.linalg.inv(vec.T @ vec) @ vec.T @ y_glob
        # hard thresholding
        x_est, supp_vec = hardthres(x_est, supp_vec)
        res_temp = y_glob - A[:, supp_vec] @ x_est
        y = res_temp
        res_temp = res_temp.T @ res_temp
        if res_temp < res or np.abs(res - res_temp) < 0.000001:
            break; 
        else:
            res = res_temp

    
    x_est = suppfill(supp_vec, x_est)
    return x_est

def basic(y):
    temp = A.T @ y
    supp_vec = np.argsort(-temp.squeeze())[:(spars)]
    vec = A[:, supp_vec]
    x_est = np.linalg.inv(vec.T @ vec) @ vec.T @ y
    x_est = suppfill(supp_vec, x_est)

    return x_est

def itr_thres(y, res):
    x_est = np.zeros((A.shape[1],1))
    for j in range(100):
        temp = x_est + A.T @ y
        # hard thresholding
        x_est, supp_vec = hardthres(temp, supp_vec=[])
        res_temp = y_glob - A[:, supp_vec] @ x_est
        y = res_temp
        res_temp = res_temp.T @ res_temp
        x_est = suppfill(supp_vec, x_est)
        if res_temp < res or np.abs(res - res_temp) < 0.000001:
            break; 
        else:
            res = res_temp

    return x_est

def hard_itr_thres(y, res):
    x_est = np.zeros((A.shape[1],1))
    for j in range(100):
        temp = x_est + A.T @ y
        # hard thresholding
        _, supp_vec = hardthres(temp, supp_vec=[])
        vec = A[:, supp_vec]
        x_est = np.linalg.inv(vec.T @ vec) @ vec.T @ y

        res_temp = y_glob - A[:, supp_vec] @ x_est
        y = res_temp
        res_temp = res_temp.T @ res_temp
        x_est = suppfill(supp_vec, x_est)
        if res_temp < res or np.abs(res - res_temp) < 0.000001:
            break; 
        else:
            res = res_temp

    return x_est


ncol = A.shape[1]
ind1 = []
res = 0.000001
for j in range(ncol):
    ind = []
    maxv = -np.inf
    for i in range(ncol):
        if i not in ind1:
            temp = np.abs(A[:, i].T @ y)
            if temp > maxv:
                _, maxv = swap(temp, maxv)
                ind.append(i)
    ind1.append(ind[-1])
    vec = A[:, ind1]
    x_est = np.linalg.inv(vec.T @ vec) @ vec.T @ y_glob
    res_temp = y_glob - A[:, ind1] @ x_est
    y = res_temp
    res_temp = res_temp.T @ res_temp
    if res_temp < res or np.abs(res - res_temp) < 0.00001:
        break; 
    else:
        res = res_temp

x_est = suppfill(ind1, x_est)


x_cosamp_est = CoSaMP(y, y_glob, res)
x_basic = basic(y)
x_itrthres = itr_thres(y, res)
x_hardthres = hard_itr_thres(y, res)
# print(x_itrthres)
print("Coherance: ", max_cor)
print(f"Coherance < (1/(2*sparse - 1)), Coherance: {max_cor}, RHS: {Thr}")
print(np.hstack([x, x_est, x_cosamp_est, x_basic, x_itrthres, x_hardthres]))
# print(x_est)