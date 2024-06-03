import numpy as np 
import matplotlib.pyplot as plt
import plots
import os 
import sparse_solvers
import solvers

T = 5
Ts = 1
dx = 0.025
d = 4 #This is the reduced dimension, so d+1 experts

u = sparse_solvers.sparse_reduced(T,d,dx,tol=0.01)
X,m = sparse_solvers.sparse_mesh(T,d,dx)
 
if d == 1:
    u_true = solvers.true_2d(X[:,0]*dx,0)
if d == 2:
    u_true = solvers.true_3d(X[:,0]*dx,X[:,1]*dx,0)
if d == 3:
    #u_true = solvers.true_4d(X[:,0]*dx,X[:,1]*dx,X[:,2]*dx,0)
    u_true = solvers.alt_true_4d(X[:,0]*dx,X[:,1]*dx,X[:,2]*dx,0)
if d <= 3:
    mask = X[:,0]*dx <= 1
    error = np.max(np.abs(u[mask] - u_true[mask]))
    print('Error = ',error)

strats = sparse_solvers.strategy_optimality_reduced(u,T,d,dx,Ts)
v_comb = sparse_solvers.comb(d+1)
solvers.opt_strats_table(strats)

if d >= 5:
    plt.figure(figsize=(15,5))
else:
    plt.figure()
plt.axvline(x = np.sum(v_comb*2**np.arange(d+1)[::-1]), color = 'r', linestyle='--')
solvers.plot_strats(strats)
plots.savefig('figures/reduced_strats_%.2f_%d_%.5f.pdf'%(T,d+1,dx),axis=True,grid=True)
#plt.show()

