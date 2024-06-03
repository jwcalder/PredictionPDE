import solvers
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import plots

plt.figure()
T = 5
alpha = 0.25

#Now solve the reduced n=1 dimensional problem
errs = []
dx_vals = [0.1,0.05,0.01]
for dx in dx_vals:
    X,n,m = solvers.mesh_1d(T,dx)
    u = solvers.solver_2d_reduced(T,dx,tol=0.01)
    u_true = solvers.true_2d(X,0)
    errs += [np.max(np.abs(u[np.abs(X) < 1]-u_true[np.abs(X) < 1]))]

plt.plot(dx_vals,errs,'o-',label='$n=2$ experts')
a,b = np.polyfit(np.log(dx_vals),np.log(errs),1)
print('2d',a)
    
#Now solve the reduced n=2 dimensional problem
errs = []
dx_vals = [0.1,0.05,0.01]
for dx in dx_vals:
    X,Y,n,m = solvers.mesh_2d(T,dx)
    u = solvers.solver_3d_reduced(T,dx,tol=0.01)
    u_true = solvers.true_3d(X,Y,0)
    mask = (np.abs(X) < 1) & (np.abs(Y) < 1)
    errs += [np.max(np.abs(u[mask]-u_true[mask]))]

plt.plot(dx_vals,errs,'s-',label='$n=3$ experts')
a,b = np.polyfit(np.log(dx_vals),np.log(errs),1)
print('3d',a)
 
#Now solve the reduced n=2 dimensional problem
errs = []
dx_vals = [0.1,0.05,0.025]
for dx in dx_vals:
    X,Y,Z,n,m = solvers.mesh_3d(T,dx)
    u = solvers.solver_4d_reduced(T,dx,tol=0.01)
    u_true = solvers.true_4d(X,Y,Z,0)
    mask = (np.abs(X) < 1) & (np.abs(Y) < 1) & (np.abs(Z) < 1)
    errs += [np.max(np.abs(u[mask]-u_true[mask]))]

plt.plot(dx_vals,errs,'^-',label='$n=4$ experts')
a,b = np.polyfit(np.log(dx_vals),np.log(errs),1)
print('4d',a)
 
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Grid resolution $h$')
plt.ylabel('Error')
plt.xlim((0.1/0.75,0.01*0.75))
a = 2
plt.xticks((0.01,0.03,0.1))
plt.legend()
plots.savefig('figures/convergence_rates.pdf',axis=True,grid=True)
plt.show()
