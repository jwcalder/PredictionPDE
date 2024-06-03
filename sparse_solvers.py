import numpy as np 
import matplotlib.pyplot as plt
import plots
import os 
import solvers
import itertools

def comb(d):
    v = np.zeros(d,dtype=int)
    v[::-2] = 1
    if d%2:
        v = 1-v
    return v

def proj_sector_2d(X,Y,u=None):
    X,Y = X.T.flatten(),Y.T.flatten()
    ind = (X >= Y) & (Y >= 0)
    SX = X[ind]
    SY = Y[ind]
    DX = np.vstack((SX,SY)).T
    if u is None:
        return DX
    else:
        Du = u.T.flatten()
        Du = Du[ind]
        return DX,Du

def proj_sector_3d(X,Y,Z,u=None):
    X,Y,Z = np.swapaxes(X,0,2).flatten(),np.swapaxes(Y,0,2).flatten(),np.swapaxes(Z,0,2).flatten()
    ind = (X >= Y) & (Y >= Z) & (Z >= 0)
    SX = X[ind]
    SY = Y[ind]
    SZ = Z[ind]
    DX = np.vstack((SX,SY,SZ)).T
    if u is None:
        return DX
    else:
        Du = np.swapaxes(u,0,2).flatten()
        Du = Du[ind]
        return DX,Du

def proj_sector_4d(X,Y,Z,W,u=None):
    DX = np.swapaxes(np.swapaxes(X,0,3),1,2).flatten()
    DY = np.swapaxes(np.swapaxes(Y,0,3),1,2).flatten()
    DZ = np.swapaxes(np.swapaxes(Z,0,3),1,2).flatten()
    DW = np.swapaxes(np.swapaxes(W,0,3),1,2).flatten()
    ind = (DX >= DY) & (DY >= DZ) & (DZ >= DW) & (DW >= 0)
    SX = DX[ind]
    SY = DY[ind]
    SZ = DZ[ind]
    SW = DW[ind]
    DX = np.vstack((SX,SY,SZ,SW)).T
    if u is None:
        return DX
    else:
        Du = np.swapaxes(np.swapaxes(W,0,3),1,2).flatten()
        Du = Du[ind]
        return DX,Du

def sparse_mesh(T,d,dx):
    '''Set up sparse mesh in arbitrary dimension'''
    m = int(T/dx)
    T = dx*m

    fname = 'meshes/mesh_%.2f_%d_%.5f.npy'%(T,d,dx)

    if os.path.exists(fname):
        X = np.load(fname)
    else:
        X = [np.zeros(d,dtype=int)]
        final = np.ones(d,dtype=int)*m
        while np.sum(X[-1] == final) < d:
            p = X[-1].copy()
            p[0] += 1
            for i in range(d-1):
                if p[i] == m+1:
                    p[i+1] += 1 
            for i in range(d-1,-1,-1):
                if p[i] == m+1:
                    p[i] = p[i+1]
            X += [p]
        X = np.array(X)
        np.save(fname,X)

    return X,m

def find_pts(Xidx,Yidx):
    '''Indexes sparse array by linear index via binary search'''

    N = Xidx.shape[0]
    num = int(np.log(N)/np.log(2) + 1)
    stops = 2**np.arange(num-1,-1,-1)
    idx = stops[0]*np.ones(Yidx.shape[0],dtype=int)
    for i in range(1,num):
        q = np.minimum(idx[:5],N-1)
        ind1 = Xidx[np.minimum(idx,N-1)] < Yidx
        ind2 = Xidx[np.minimum(idx,N-1)] > Yidx
        idx += stops[i]*ind1
        idx -= stops[i]*ind2

    idx[Yidx==0] = 0
    return np.minimum(idx,N-1)

def binary_vectors(d):
    '''All binary vectors in dimension d'''

    return list(itertools.product([0, 1], repeat=d))


def D2v(u,v,X,interior,Xidx,b,dx):
    '''Computes second derivative in direction v'''

    d = X.shape[1]

    #Forward u(x+v)
    Y = np.sort(X[interior,:] + v,axis=1)[:,::-1]
    Yidx = Y@b
    idx1 = find_pts(Xidx,Yidx)
    upv = u[idx1]
    print(np.mean(Xidx[idx1] == Yidx),end=',')

    #Backward u(x-v)
    Y = np.sort(X[interior,:] - v,axis=1)[:,::-1]
    ind_xi = np.min(Y,axis=1) < 0
    i_xi = np.argmin(Y[ind_xi,:],axis=1)
    xi = np.ones((len(i_xi),d),dtype=int)
    xi[range(len(i_xi)),i_xi] = 2
    Y[ind_xi,:] += xi
    Yidx = Y@b
    idx2 = find_pts(Xidx,Yidx)
    umv = u[idx2] - dx*ind_xi
    print(np.mean(Xidx[idx2] == Yidx))

    #Second derivative
    uvv = (upv + umv - 2*u[interior])/(dx**2)

def D2v_stencil(v,X,interior,Xidx,b):
    '''Computes second derivative stencil in direction v'''

    d = X.shape[1]

    #Forward u(x+v)
    Y = np.sort(X[interior,:] + v,axis=1)[:,::-1]
    Yidx = Y@b
    idx1 = find_pts(Xidx,Yidx)

    #Backward u(x-v)
    Y = np.sort(X[interior,:] - v,axis=1)[:,::-1]
    ind_xi = np.min(Y,axis=1) < 0
    i_xi = np.argmin(Y[ind_xi,:],axis=1)
    xi = np.ones((len(i_xi),d),dtype=int)
    xi[range(len(i_xi)),i_xi] = 2
    Y[ind_xi,:] += xi
    Yidx = Y@b
    idx2 = find_pts(Xidx,Yidx)


    return idx1,idx2,ind_xi

def sparse_stencil(T,d,dx):
    '''Sets up sparse stencil'''

    fname = 'sparse_solutions/stencil_%.2f_%d_%.5f.npz'%(T,d,dx)
    if os.path.exists(fname):
        print('Loading stencils...')
        M = np.load(fname)
        idx1 = M['idx1']
        idx2 = M['idx2']
        ind_xi = M['ind_xi']
    else:

        #Load mesh
        X,m = sparse_mesh(T,d,dx)

        #Set up stencils
        b = (m+1)**np.arange(d)
        interior = X[:,0] < m
        Xidx = X@b

        #Binary vectors
        B = binary_vectors(d)

        #Setting up stencils
        idx1 = np.zeros((len(B),np.sum(interior)),dtype=int)
        idx2 = np.zeros((len(B),np.sum(interior)),dtype=int)
        ind_xi = np.zeros((len(B),np.sum(interior)),dtype=bool)
        print('Setting up stencils...')
        for i,v in enumerate(B):
            print(v)
            idx1[i,:],idx2[i,:],ind_xi[i,:] = D2v_stencil(v,X,interior,Xidx,b)
        np.savez_compressed(fname,idx1=idx1,idx2=idx2,ind_xi=ind_xi)

    return idx1,idx2,ind_xi

def sparse_reduced(T,d,dx,tol=1e-1):
    '''Sparse solver using reduced dimension problem in positive sector'''

    fname = 'sparse_solutions/sol_%.2f_%d_%.5f.npy'%(T,d,dx)

    #Mesh and initialization
    X,m = sparse_mesh(T,d,dx)
    if os.path.exists(fname):
        u = np.load(fname)
    else:
        u = np.max(X,axis=1)*dx
    g = np.max(X,axis=1)*dx
    interior = X[:,0] < m

    #Get stencil
    idx1,idx2,ind_xi = sparse_stencil(T, d, dx)

    #Initialize
    dt = dx**2/(1 + dx**2)
    err = 1 
    i = 0
    print('Solving PDE...')
    while err > tol*dx**2:
        F = 0.5*np.max((u[idx1] + u[idx2] - dx*ind_xi - 2*u[interior])/(dx)**2,axis=0)
        err = np.max(np.abs(u[interior] - F - g[interior]))
        print(i,err/dx**2)
        u[interior] = (1-dt)*u[interior] + dt*(F + g[interior])
        i += 1
    
    np.save(fname,u)
    return u

def strategy_optimality_reduced(u,T,d,dx,Ts):
    '''Computes optimality of strategies in dimension reduced domain'''

    idx1,idx2,ind_xi = sparse_stencil(T, d, dx)
    X,m = sparse_mesh(T,d,dx)
    interior = X[:,0] < m
    mask = X[interior,0]*dx <= Ts

    #Compute second derivatives and max
    uvv = (u[idx1] + u[idx2] - dx*ind_xi - 2*u[interior])/(dx)**2
    F = np.max((u[idx1] + u[idx2] - dx*ind_xi - 2*u[interior])/(dx)**2,axis=0)
    optvv = np.maximum(uvv[:,mask],0)/F[mask]
    meanvv = np.mean(optvv,axis=1)
    minvv = np.min(optvv,axis=1)
    maxvv = np.max(optvv,axis=1)
    orgvv = optvv[:,0]
    #print(orgvv.shape)
    #print('minF=',np.min(F[mask]))

    #svv = np.mean(np.maximum(uvv[:,mask],0)/F[mask],axis=1) #Old
    #svv = np.mean(np.maximum(uvv[:,mask],0),axis=1)/np.mean(F[mask]) #New
    #svv = np.min(np.maximum(uvv[:,mask],0)/F[mask],axis=1) #Old

    #Binary vectors
    B = binary_vectors(d+1)

    strat = {}
    for k,v in enumerate(B):
        vstr = solvers.to_str(v)
        if v[-1] == 1:
            w = np.ones(d+1) - v
        else:
            w = v
        i = int(np.sum(2**np.arange(d)[::-1]*w[:-1]))
        #strat[vstr] = svv[i]
        strat[vstr] = [meanvv[i],minvv[i],maxvv[i],orgvv[i]]

    return strat

