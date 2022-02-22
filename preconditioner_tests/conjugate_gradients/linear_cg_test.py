from linear_cg import *

n = 1000
x = torch.linspace(0.,10.,n).reshape(n,1)
perm = torch.randperm(n)
y_true = torch.sin(x)
noise = torch.normal(0.0,0.5,size=[n,1])
y = y_true + noise

x, y = x[perm], y[perm]

k = gpytorch.kernels.RBFKernel()
K = k(x, x).evaluate().detach()
C = K + 1e0*torch.eye(n)

t1 = time()
x_sol = torch.inverse(C) @ y
t2 = time()
time_sol = t2 - t1

cg_ = linear_cg()

t1 = time()
run_1 = cg_(C, y, tol=10e-2)
t2 = time()
time_cg = t2 - t1

t1 = time()
L = pivoted_cholesky(K, 20)
precon = torch.inverse(L @ L.T + 1e0*torch.eye(n))
run_2 = cg_(C, y, pmvm=precon, tol=10e-2)
t2 = time()
time_pcg = t2 - t1

err1 = x_sol - run_1[0]
err2 = x_sol - run_2[0]

print(time_sol)
print(torch.norm(err1), run_1[2], time_cg)
print(torch.norm(err2), run_2[2], time_pcg)