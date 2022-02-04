from linear_cg import *
from scipy.io import loadmat

import numpy as np

data = loadmat(r'C:\Users\sgalee2\Documents\GitHub\benchmarking-preconditioners-for-gp\MATLAB Files\protein.mat')['data']
np.random.shuffle(data)
data = torch.Tensor(data)
N = data.shape[0]

# make train/val/test START LOW, CAN HAPPILY ACCEPT N = 10,000
n_train = 12000
train_x, train_y = data[:n_train, :-1], data[:n_train, -1:]
test_x, test_y = data[n_train:, :-1], data[n_train:, -1:]

#form covariance matrix and add noise
k = gpytorch.kernels.RBFKernel()
K = k(train_x).evaluate()
C = K + 1e-4*torch.eye(n_train)

#compute the exact solution and time it
t1 = time()
v_sol = torch.inverse(C) @ train_y
t2 = time()
time_sol = t2 - t1
print("Exact sol in:",time_sol,"s.")

#perform our CG run and time it, computing the error
t1 = time()
cg_ = linear_cg()
run_1 = cg_(C, train_y)
t2 = time()
time_cg = t2 - t1
error = run_1[0] - v_sol
err_ = torch.norm(v_sol - run_1[0])

print("Solution with",err_.item(),"error in",time_cg,"s.")

print()