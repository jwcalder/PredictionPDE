import numpy as np
import matplotlib.pyplot as plt 
import plots

T = 5
D = [5,6,7,8,9,10]
DX = [0.025,0.05,0.1,0.2,0.25,0.35]

for i in range(len(D)):
    d = D[i]-1
    dx = DX[i]
    fname = 'sparse_solutions/sol_%.2f_%d_%.5f.npy'%(T,d,dx)
    u = np.load(fname)
    m = len(u)
    print('%d,%.3f,%.0E,%.0E'%(d,dx,m,(2*T/dx)**d))
