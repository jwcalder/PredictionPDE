import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D
from skimage.transform import rescale
import numpy as np

#General plot settings
legend_fontsize = 16
label_fontsize = 18
fontsize = 16
#matplotlib.rcParams.update({'font.size': fontsize})
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Helvetica"],
    "font.size": 16,
    "axes.axisbelow": True})

styles = ['^b-','or-','dg-','sk-','pm-','xc-','*y-']

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728','#9467bd', '#8c564b', '#e377c2', '#7f7f7f','#bcbd22', '#17becf']
markers = ['^','o','d','s','p','x','*']

def imsave(fname, img, scale=1, gray=True, order=0):
    if scale != 1:
        rimg = rescale(img, (scale,scale), order=order)
    else:
        rimg = img
    plt.imsave(fname,rimg,cmap='gray')


def savefig(fname, dim=2, axis=False,grid=False,square=False,pad_inches=0.01):

    if square:
        plt.axis('equal')
        #ax.set_aspect('equal', 'box')
    if grid:
        plt.grid(True)

    if not axis:
        plt.axis('off')
        plt.gca().set_axis_off()
    #plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    if dim == 2:
        if not axis:
            plt.margins(0,0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
    if dim == 3:
        plt.margins(0,0,0)
        if not axis:
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.gca().zaxis.set_major_locator(plt.NullLocator())

    plt.savefig(fname,bbox_inches='tight',pad_inches=pad_inches)

def plot(x,y,labels=None,markers=False,ylog=False,xlabel=None,ylabel=None):

    plt.figure()
    for i,data in enumerate(y):
        if markers:
            if labels is None:
                plt.plot(x,data,c=colors[i],marker=markers[i],linestyle='-')
            else:
                plt.plot(x,data,c=colors[i],marker=markers[i],linestyle='-',label=labels[i])
        else:
            if labels is None:
                plt.plot(x,data,c=colors[i],linestyle='-')
            else:
                plt.plot(x,data,c=colors[i],linestyle='-',label=labels[i])
    if labels is not None:
        plt.legend(fontsize=legend_fontsize)
    if ylog:
        plt.yscale('log')
    if xlabel is not None:
        plt.xlabel(xlabel,fontsize=label_fontsize)
    if ylabel is not None:
        plt.ylabel(ylabel,fontsize=label_fontsize)

def surf(f,xlim,ylim,zlim=None,grid=20,fig=None,ax=None):
    #fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    if fig is None:
        fig = plt.figure()
    #ax = Axes3D(fig, computed_zorder=False)
    if ax is None:
        ax = plt.axes(projection ='3d', computed_zorder=False)
    
    # Make data.
    X = np.linspace(xlim[0], xlim[1], grid)
    Y = np.linspace(ylim[0], ylim[1], grid)
    X, Y = np.meshgrid(X, Y)

    # Plot the surface.
    for fcn in f:
        surf = ax.plot_surface(X, Y, fcn(X,Y), cmap=cm.coolwarm,zorder=1)

    # Customize the z axis.
    if zlim is not None:
        ax.set_zlim(zlim[0],zlim[1])
    #ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    #ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    #fig.colorbar(surf, shrink=0.5, aspect=5)

    return fig,ax

def plot_region(X,L,clf,alpha=0.5,cmap='Paired',cp=np.array([0.5,2.5,6.5,8.5,10.5,11.5]),markers = ['o','s','D','^','v','p'],vmin=0,vmax=12,fname=None,markersize=75,linewidths=1.25,markerlinewidths=1,res=0.01,train_pts=None):

    plt.figure()
    x,y = X[:,0],X[:,1]
    xmin, xmax = np.min(x),np.max(x)
    ymin, ymax = np.min(y),np.max(y)
    f =0.1*np.maximum(np.max(np.abs(x)),np.max(np.abs(y)))
    xmin -= f
    ymin -= f
    xmax += f
    ymax += f
    c = cp[L]
    c_u = np.unique(c)
    
    for i,color in enumerate(c_u):
        sub = c==color
        plt.scatter(x[sub],y[sub],zorder=2,c=c[sub],cmap=cmap,edgecolors='black',vmin=vmin,vmax=vmax,linewidths=markerlinewidths,marker=markers[i],s=markersize)
        if train_pts is not None:
            plt.scatter(x[sub & train_pts],y[sub & train_pts],zorder=2,c=np.ones(np.sum(sub&train_pts))*5.5,cmap=cmap,edgecolors='black',vmin=vmin,vmax=vmax,linewidths=markerlinewidths,marker=markers[i],s=markersize)


    X,Y = np.mgrid[xmin:xmax:0.01,ymin:ymax:0.01]
    points = np.c_[X.ravel(),Y.ravel()]
    z = clf.predict(points)
    z = z.reshape(X.shape)
    plt.contourf(X, Y, cp[z],alpha=alpha,cmap=cmap,antialiased=True,vmin=vmin,vmax=vmax)

    X,Y = np.mgrid[xmin:xmax:res,ymin:ymax:res]
    points = np.c_[X.ravel(),Y.ravel()]
    if len(np.unique(c)) == 2:

        if hasattr(clf, "decision_function"):
            z = clf.decision_function(points)
        else:
            z = clf.predict_proba(points)
            z = z[:,0] - z[:,1] + 1e-15
        z = z.reshape(X.shape)
        plt.contour(X, Y, z, [0], colors='black',linewidths=linewidths,antialiased=True)
    else:
        z = clf.predict(points)
        z = z.reshape(X.shape)
        plt.contour(X, Y, z, colors='black',linewidths=linewidths,antialiased=True)
    plt.xlim((xmin,xmax))
    plt.ylim((ymin,ymax))
    if fname is None:
        plt.show()
    else:
        savefig(fname,axis=True,pad_inches=0.01)


#def scatter(X,labels,cp=np.array([0.5,2.5,6.5,8.5,10.5,11.5]),markers = ['o','s','D','^','v','p'],vmin=0,vmax=12,):
#    plt.scatter(X[:,0],X[:,1], c=cp[labels],s=30,edgecolor='black',linewidths=1,cmap='Paired',vmin=vmin,vmax=vmax)


def scatter_means(means,num_wide=2):
    if num_wide==2:
        s = 300
    elif num_wide==3:
        s = 450
    else:
        s = 100
    plt.scatter(means[:,0],means[:,1], c='r',marker='*',s=s,edgecolors='black',linewidths=1)


def scatter(X,L,cmap='Paired',cp=np.array([0.5,2.5,6.5,8.5,10.5,11.5]),markers = ['o','s','D','^','v','p'],vmin=0,vmax=12,markersize=50,linewidths=1,markerlinewidths=1,res=0.01,num_wide=2):

    if num_wide == 2:
        markersize = 50
    elif num_wide == 3:
        markersize = 75
    else:
        markersize = 30

    #plt.figure()
    x,y = X[:,0],X[:,1]
    xmin, xmax = np.min(x),np.max(x)
    ymin, ymax = np.min(y),np.max(y)
    fx = 0.1*np.max(np.abs(x))
    fy = 0.1*np.max(np.abs(y))
    xmin -= fx
    ymin -= fy
    xmax += fx
    ymax += fy
    c = cp[L]
    c_u = np.unique(c)
    
    for i,color in enumerate(c_u):
        sub = c==color
        plt.scatter(x[sub],y[sub],c=c[sub],cmap=cmap,edgecolors='black',vmin=vmin,vmax=vmax,linewidths=markerlinewidths,marker=markers[i],s=markersize)
    plt.xlim((xmin,xmax))
    plt.ylim((ymin,ymax))
    
















