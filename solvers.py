import numpy as np 
import matplotlib.pyplot as plt
import plots
import os 

v3d = [(0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)]
v4d = [(0,0,0,0),(0,0,0,1),(0,0,1,0),(0,0,1,1),(0,1,0,0),(0,1,0,1),(0,1,1,0),(0,1,1,1),
       (1,0,0,0),(1,0,0,1),(1,0,1,0),(1,0,1,1),(1,1,0,0),(1,1,0,1),(1,1,1,0),(1,1,1,1)]

def neg(v):
    return (1-v[0],1-v[1],1-v[2],1-v[3])

def to_str(v):
    return str(v).replace(',','').replace('(','').replace(')','').replace(' ','')

def restrict_3d(u,m,alpha):
    '''Restrict a grid function to a box smaller by alpha'''

    m1 = m-int(alpha*m)
    m2 = m+int(alpha*m)
    return u[m1:m2,m1:m2,m1:m2]


def restrict_2d(u,m,alpha):
    '''Restrict a grid function to a box smaller by alpha'''

    m1 = m-int(alpha*m)
    m2 = m+int(alpha*m)
    return u[m1:m2,m1:m2]

def restrict_1d(u,m,alpha):
    '''Restrict a grid function to a box smaller by alpha'''

    m1 = m-int(alpha*m)
    m2 = m+int(alpha*m)
    return u[m1:m2]

def mesh_1d(T,dx):
    '''Set up 1D mesh'''

    #Set up mesh
    m = int(T/dx)
    T = dx*m
    n = 2*m+1
    X = np.arange(-m,m+1)*dx

    return X,n,m

def mesh_2d(T,dx):
    '''Set up 2D mesh'''

    #Set up mesh
    m = int(T/dx)
    T = dx*m
    n = 2*m+1
    x = np.arange(-m,m+1)*dx
    y = np.arange(-m,m+1)*dx
    X,Y = np.meshgrid(x,y,indexing='ij')

    return X,Y,n,m

def mesh_3d(T,dx):
    '''Set up 3D mesh'''

    #Set up mesh
    m = int(T/dx)
    T = dx*m
    n = 2*m+1
    x = np.arange(-m,m+1)*dx
    y = np.arange(-m,m+1)*dx
    z = np.arange(-m,m+1)*dx
    X,Y,Z = np.meshgrid(x,y,z,indexing='ij')

    return X,Y,Z,n,m

def mesh_4d(T,dx):
    '''Set up 4D mesh'''

    #Set up mesh
    m = int(T/dx)
    T = dx*m
    n = 2*m+1
    x = np.arange(-m,m+1)*dx
    y = np.arange(-m,m+1)*dx
    z = np.arange(-m,m+1)*dx
    w = np.arange(-m,m+1)*dx
    X,Y,Z,W = np.meshgrid(x,y,z,w,indexing='ij')

    return X,Y,Z,W,n,m



def true_2d(x,y):
    '''True solution in 2D'''

    x1 = np.maximum(x,y)
    x2 = np.minimum(x,y)
    u_true = x1 + (1/(2*np.sqrt(2)))*np.exp(np.sqrt(2)*(x2-x1))
    return u_true

def true_3d(x,y,z):
    '''True solution in 2D'''

    x1 = np.maximum(np.maximum(x,y),z)
    x3 = np.minimum(np.minimum(x,y),z)
    x2 = x + y + z - x1 - x3
    
    u_true = x1 + (1/(2*np.sqrt(2)))*np.exp(np.sqrt(2)*(x2-x1)) + (1/(6*np.sqrt(2)))*np.exp(np.sqrt(2)*(2*x3-x2-x1)) 

    return u_true

def true_4d(x,y,z,w):
    '''True solution in 4D'''

    #Formulas below are not defined at specific points where arctanh/arctan are undefined
    #Adding noise makes sure we don't hit those points
    #Function has a C^{2,1} extension, so the returned result is correct up to roughly 1e-15
    x += np.random.randn(1)*1e-15
    y += np.random.randn(1)*1e-15
    z += np.random.randn(1)*1e-15
    w += np.random.randn(1)*1e-15

    #Sort 4 coordinates
    a1 = np.maximum(x,y)
    a2 = np.minimum(x,y)
    b1 = np.maximum(z,w)
    b2 = np.minimum(z,w)

    x1 = np.maximum(a1,b1)
    c1 = np.minimum(a1,b1)
    x4 = np.minimum(a2,b2)
    c2 = np.maximum(a2,b2)

    x2 = np.maximum(c1,c2)
    x3 = np.minimum(c1,c2)
    
    r2 = np.sqrt(2)
    u_true =  x1 - (r2/4)*np.sinh(r2*(x1-x2))
    a  = (r2/2)*np.arctan(np.exp((x4+x3-x2-x1)/r2))*np.cosh((x4-x3+x2-x1)/r2)
    a *= np.cosh((-x4+x3+x2-x1)/r2)*np.cosh((-x4-x3+x2+x1)/r2)
    b  = (r2/2)*np.arctanh(np.exp((x4+x3-x2-x1)/r2))*np.sinh((x4-x3+x2-x1)/r2)
    b *= np.sinh((-x4+x3+x2-x1)/r2)*np.sinh((-x4-x3+x2+x1)/r2)
    u_true += a + b

    return u_true

def alt_true_4d(x,y,z,w):
    '''Alternate formula for true solution in 4D'''

    #Sort 4 coordinates
    a1 = np.maximum(x,y)
    a2 = np.minimum(x,y)
    b1 = np.maximum(z,w)
    b2 = np.minimum(z,w)

    x1 = np.maximum(a1,b1)
    c1 = np.minimum(a1,b1)
    x4 = np.minimum(a2,b2)
    c2 = np.maximum(a2,b2)

    x2 = np.maximum(c1,c2)
    x3 = np.minimum(c1,c2)
    
    r2 = np.sqrt(2)
    ir2 = 1/r2

    u_true = x1 + ir2*np.exp(r2*(x2-x1)) + ir2*np.exp(-r2*(2*x2-x3-x4))
    u_true += ir2*(1/3)*np.exp(r2*(2*x3-x1-x2)) + ir2*(7/6)*np.exp(-ir2*(x1+x2+x3-3*x4))
    return u_true

def opt_strats_table(strats):
    opt = np.array(list(strats.values()))
    bi = np.array(list(strats.keys()))
    m = len(opt)
    a = np.arange(m>>1)[:,None]
    b = bi[:m>>1,None]
    c = opt[:m>>1,:]
    opt = np.hstack((a,b,c))
    ind = np.argsort(opt[:,3])[::-1]
    opt = opt[ind,:]

    print('Strategy & Binary & Mean Optimality & Min Optimality & Max Optimality & Origin Optimality \\\\')
    for i in range(opt.shape[0]):
        r = opt[i,:]
        st = ''
        for s in list(r):
            st += s + ' & '
        st = st[:-2] + '\\\\'
        print(st)

def plot_strats(strats):

    opt = np.array(list(strats.values()))
    #m = np.maximum(len(strats)>>1,4)
    m = len(strats)>>1
    #plt.plot(np.ones(m),'--',c='black')
    plt.plot(opt[:m,0],'o-',label='Mean Optimality')
    plt.plot(opt[:m,1],'s--',label='Min Optimality')
    plt.plot(opt[:m,2],'^--',label='Max Optimality')
    if m >= 64:
        if m == 64:
            plt.xticks(np.arange(m>>1,m,dtype=int))
        plt.xlim((-0.5+(m>>1),m))
        plt.ylim((0.5,1.05))
    else:
        plt.xticks(np.arange(m,dtype=int))
    plt.xlabel('Strategy')
    plt.ylabel('Optimality')
    plt.legend()

def strategy_optimality_3d_reduced(u,T,dx,alpha):
    '''Computes optimality of strategies on 3D restricted and dimension reduced domain'''

    X,Y,n,m = mesh_2d(T,dx)
    us = restrict_2d(u,m,alpha)
    Xs = restrict_2d(X,m,alpha)
    Ys = restrict_2d(Y,m,alpha)
    mask = (Xs[1:-1,1:-1] >= Ys[1:-1,1:-1]) & (Ys[1:-1,1:-1] >= 0)

    uxx = (us[2:,1:-1] + us[:-2,1:-1] - 2*us[1:-1,1:-1])/(dx*dx)
    uyy = (us[1:-1,2:] + us[1:-1,:-2] - 2*us[1:-1,1:-1])/(dx*dx)
    uvv = (us[2:,2:] + us[:-2,:-2] - 2*us[1:-1,1:-1])/(dx*dx)
    uoo = us[1:-1,1:-1]*0.0

    F = np.maximum(np.maximum(np.maximum(uxx,uyy),uvv),uoo)
    sxx = np.mean(np.maximum(uxx[mask],0)/F[mask])
    syy = np.mean(np.maximum(uyy[mask],0)/F[mask])
    svv = np.mean(np.maximum(uvv[mask],0)/F[mask])

    strat = {'000':0.0,'001':svv,'010':syy,'011':sxx,'100':sxx,'101':syy,'110':svv,'111':0.0}

    return strat

def strategy_optimality_4d_reduced(u,T,dx,alpha):
    '''Computes optimality of strategies on 4D restricted and dimension reduced domain'''

    X,Y,Z,n,m = mesh_3d(T,dx)
    us = restrict_3d(u,m,alpha)
    Xs = restrict_3d(X,m,alpha)
    Ys = restrict_3d(Y,m,alpha)
    Zs = restrict_3d(Z,m,alpha)

    F = us[1:-1,1:-1,1:-1]*0.0
    for v in v3d[1:]:
        F = np.maximum(diff_3d(us,v,dx),F)

    strat = {}
    mask = (Xs[1:-1,1:-1,1:-1] >= Ys[1:-1,1:-1,1:-1]) & (Ys[1:-1,1:-1,1:-1] >= Zs[1:-1,1:-1,1:-1]) & (Zs[1:-1,1:-1,1:-1] >= 0)
    for v in v4d:
        if v[3] == 1:
            uvv = diff_3d(us,neg(v),dx)
        else:
            uvv = diff_3d(us,v,dx)
        svv = np.mean(np.maximum(uvv[mask],0)/F[mask])
        strat[to_str(v)] = svv

    return strat

def strategy_optimality_2d(u,T,dx,alpha):
    '''Computes optimality of strategies on 2D restricted domain'''

    X,Y,n,m = mesh_2d(T,dx)
    us = restrict_2d(u,m,alpha)
    Xs = restrict_2d(X,m,alpha)
    Ys = restrict_2d(Y,m,alpha)
    mask = Xs[1:-1,1:-1] >= Ys[1:-1,1:-1]

    uxx = (us[2:,1:-1] + us[:-2,1:-1] - 2*us[1:-1,1:-1])/(dx*dx)
    uyy = (us[1:-1,2:] + us[1:-1,:-2] - 2*us[1:-1,1:-1])/(dx*dx)
    uvv = (us[2:,2:] + us[:-2,:-2] - 2*us[1:-1,1:-1])/(dx*dx)
    uoo = us[1:-1,1:-1]*0.0

    F = np.maximum(np.maximum(np.maximum(uxx,uyy),uvv),uoo)
    sxx = np.mean(np.maximum(uxx[mask],0)/F[mask])
    syy = np.mean(np.maximum(uyy[mask],0)/F[mask])
    svv = np.mean(np.maximum(uvv[mask],0)/F[mask])

    strat = {'00':0.0,'01':syy,'10':sxx,'11':svv}

    return strat

def solver_2d(T,dx,u0=None,tol=1e-1):
    '''2D solver in full dimensions'''

    fname = 'solutions/2d_%.2f_%.5f.npy'%(T,dx)
    X,Y,n,m = mesh_2d(T,dx)

    #Set up payoff, u and true solution
    g = np.maximum(X,Y)
    if u0 is None:
        if os.path.exists(fname):
            u = np.load(fname)
        else:
            u = np.maximum(X,Y)
    else:
        u = u0.copy()

    dt = dx**2/(1 + dx**2)
    err = 1 
    i = 0
    while err > tol*dx**2:
        uxx = (u[2:,1:-1] + u[:-2,1:-1] - 2*u[1:-1,1:-1])/(dx*dx)
        uyy = (u[1:-1,2:] + u[1:-1,:-2] - 2*u[1:-1,1:-1])/(dx*dx)
        uvv = (u[2:,2:] + u[:-2,:-2] - 2*u[1:-1,1:-1])/(dx*dx)
        uoo = u[1:-1,1:-1]*0.0

        F = 0.5*np.maximum(np.maximum(np.maximum(uxx,uyy),uvv),uoo)
        err = np.max(np.abs(u[1:-1,1:-1] - F - g[1:-1,1:-1]))
        print(i,err/dx**2)
        u[1:-1,1:-1] = (1-dt)*u[1:-1,1:-1] + dt*(F + g[1:-1,1:-1])
        i += 1

    np.save(fname,u)
    return u

def solver_2d_reduced(T,dx,u0=None,tol=1e-1):
    '''2D solver in reduced dimensions (n=1)'''

    fname = 'solutions/2d_reduced_%.2f_%.5f.npy'%(T,dx)
    X,n,m = mesh_1d(T,dx)

    #Set up payoff, u and true solution
    g = np.maximum(X,0)
    if u0 is None:
        if os.path.exists(fname):
            u = np.load(fname)
        else:
            u = np.maximum(X,0)
    else:
        u = u0.copy()

    dt = dx**2/(1 + dx**2)
    err = 1 
    i = 0
    while err > tol*dx**2:
        uxx = (u[2:] + u[:-2] - 2*u[1:-1])/(dx*dx)
        uoo = u[1:-1]*0.0

        F = 0.5*np.maximum(uxx,uoo)
        err = np.max(np.abs(u[1:-1] - F - g[1:-1]))
        print(i,err/dx**2)
        u[1:-1] = (1-dt)*u[1:-1] + dt*(F + g[1:-1])
        i += 1

    np.save(fname,u)
    return u

def solver_3d_reduced(T,dx,u0=None,tol=1e-1):
    '''3D solver in reduced dimensions (n=2)'''

    fname = 'solutions/3d_reduced_%.2f_%.5f.npy'%(T,dx)
    X,Y,n,m = mesh_2d(T,dx)

    #Set up payoff, u and true solution
    g = np.maximum(np.maximum(X,Y),0)
    if u0 is None:
        if os.path.exists(fname):
            u = np.load(fname)
        else:
            u = np.maximum(np.maximum(X,Y),0)
    else:
        u = u0.copy()

    dt = dx**2/(1 + dx**2)
    err = 1 
    i = 0
    while err > tol*dx**2:
        uxx = (u[2:,1:-1] + u[:-2,1:-1] - 2*u[1:-1,1:-1])/(dx*dx)
        uyy = (u[1:-1,2:] + u[1:-1,:-2] - 2*u[1:-1,1:-1])/(dx*dx)
        uvv = (u[2:,2:] + u[:-2,:-2] - 2*u[1:-1,1:-1])/(dx*dx)
        uoo = u[1:-1,1:-1]*0.0

        F = 0.5*np.maximum(np.maximum(np.maximum(uxx,uyy),uvv),uoo)
        err = np.max(np.abs(u[1:-1,1:-1] - F - g[1:-1,1:-1]))
        print(i,err/dx**2)
        u[1:-1,1:-1] = (1-dt)*u[1:-1,1:-1] + dt*(F + g[1:-1,1:-1])
        i += 1

    np.save(fname,u)
    return u

def diff_3d(u,v,dx):
    '''Computes centered difference for second derivative in binary direction v'''
    
    n = u.shape[0]
    return (u[1+v[0]:n-1+v[0],1+v[1]:n-1+v[1],1+v[2]:n-1+v[2]] + u[1-v[0]:n-1-v[0],1-v[1]:n-1-v[1],1-v[2]:n-1-v[2]] - 2*u[1:-1,1:-1,1:-1])/(dx**2)

def solver_4d_reduced(T,dx,u0=None,tol=1e-1):
    '''4D solver in reduced dimensions (n=3)'''

    fname = 'solutions/4d_reduced_%.2f_%.5f.npy'%(T,dx)
    X,Y,Z,n,m = mesh_3d(T,dx)

    #Set up payoff, u and true solution
    g = np.maximum(np.maximum(np.maximum(X,Y),Z),0)
    if u0 is None:
        if os.path.exists(fname):
            u = np.load(fname)
        else:
            u = np.maximum(np.maximum(np.maximum(X,Y),Z),0)
    else:
        u = u0.copy()

    dt = dx**2/(1 + dx**2)
    err = 1 
    i = 0
    while err > tol*dx**2:
        F = u[1:-1,1:-1,1:-1]*0.0
        for v in v3d[1:]:
            F = np.maximum(diff_3d(u,v,dx),F)
        F *= 0.5
        err = np.max(np.abs(u[1:-1,1:-1,1:-1] - F - g[1:-1,1:-1,1:-1]))
        print(i,err/dx**2)
        u[1:-1,1:-1,1:-1] = (1-dt)*u[1:-1,1:-1,1:-1] + dt*(F + g[1:-1,1:-1,1:-1])
        i += 1

    np.save(fname,u)
    return u



        

        

