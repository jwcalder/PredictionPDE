import solvers
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import plots

T = 5
dx = 0.01
alpha = 0.25

#Solve reduced 2 dimensional problem
X,Y,n,m = solvers.mesh_2d(T,dx)
u = solvers.solver_3d_reduced(T,dx,tol=0.01)
u_true = solvers.true_3d(X,Y,0)

#Compute optimal strategies
strats = solvers.strategy_optimality_3d_reduced(u,T,dx,alpha)
plt.figure()
plt.axvline(x = 2, color = 'r')
solvers.plot_strats(strats)
plots.savefig('figures/3d_reduced_strats_%.2f_%.5f.pdf'%(T,dx),axis=True,grid=True)
print(strats)

#Restrict to interior
Xs = solvers.restrict_2d(X,m,alpha)
Ys = solvers.restrict_2d(Y,m,alpha)
us = solvers.restrict_2d(u,m,alpha)
us_true = solvers.restrict_2d(u_true,m,alpha)

#Plot numerical solution, true solution, and difference
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(Xs, Ys, us, cmap=cm.coolwarm,linewidth=0, antialiased=False)
plots.savefig('figures/3d_reduced_numerical_%.2f_%.5f.pdf'%(T,dx),axis=True)
plt.title('Numerical')

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(Xs, Ys, us_true, cmap=cm.coolwarm,linewidth=0, antialiased=False)
plots.savefig('figures/3d_reduced_true_%.2f_%.5f.pdf'%(T,dx),axis=True)
plt.title('True')

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(Xs, Ys, us_true-us, cmap=cm.coolwarm,linewidth=0, antialiased=False)
plots.savefig('figures/3d_reduced_diff_%.2f_%.5f.pdf'%(T,dx),axis=True,pad_inches=0.3)
plt.title('Difference')

plt.show()

