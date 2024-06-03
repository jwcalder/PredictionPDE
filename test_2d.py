import solvers
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import plots

T = 5
dx = 0.01
alpha = 0.25

#First solve on whole domain
X,Y,n,m = solvers.mesh_2d(T,dx)
u = solvers.solver_2d(T,dx,tol=0.01)
u_true = solvers.true_2d(X,Y)

#Compute optimal strategies
strats = solvers.strategy_optimality_2d(u,T,dx,alpha)
plt.figure()
plt.axvline(x = 2, color = 'r')
plt.axvline(x = 1, color = 'r')
solvers.plot_strats(strats)
plots.savefig('figures/2d_strats_%.2f_%.5f.pdf'%(T,dx),axis=True,grid=True)
print(strats)

#Restrict to interior
Xs = solvers.restrict_2d(X,m,alpha)
Ys = solvers.restrict_2d(Y,m,alpha)
us = solvers.restrict_2d(u,m,alpha)
us_true = solvers.restrict_2d(u_true,m,alpha)

#Plot numerical solution, true solution, and difference
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(Xs, Ys, us, cmap=cm.coolwarm,linewidth=0, antialiased=False)
plots.savefig('figures/2d_numerical_%.2f_%.5f.pdf'%(T,dx),axis=True)
plt.title('Numerical')

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(Xs, Ys, us_true, cmap=cm.coolwarm,linewidth=0, antialiased=False)
plots.savefig('figures/2d_true_%.2f_%.5f.pdf'%(T,dx),axis=True)
plt.title('True')

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(Xs, Ys, us_true-us, cmap=cm.coolwarm,linewidth=0, antialiased=False)
plots.savefig('figures/2d_diff_%.2f_%.5f.pdf'%(T,dx),axis=True,pad_inches=0.3)
plt.title('Difference')

#Now solve the reduced n=1 dimensional problem
X,n,m = solvers.mesh_1d(T,dx)
u = solvers.solver_2d_reduced(T,dx,tol=0.01)
u_true = solvers.true_2d(X,0)

#Plot numerical solution, true solution, and difference
plt.figure()
plt.plot(X,u,label='Numerical')
plt.plot(X,u_true,'--',label='True')
plt.legend()
plots.savefig('figures/2d_reduced_%.2f_%.5f.pdf'%(T,dx),axis=True, grid=True)

plt.figure()
plt.plot(X,u-u_true,label='Difference')
plt.legend()
plots.savefig('figures/2d_reduced_diff_%.2f_%.5f.pdf'%(T,dx),axis=True, grid=True)

plt.show()

