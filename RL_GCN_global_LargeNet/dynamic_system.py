"""Simulation coupled chua system with different network topology"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
import network_topology
from mpl_toolkits.mplot3d import Axes3D
import time


def runge_kutta(f, tspan, Y0, Nt):
    '''f deviation function;
    tspan 2*1 vector tspan(1)start time tspan(2) end time;
    Y0 is a columb vector, Y0(i) is the initial value of i-th variable;
    Nt is the number of t'''
    Nvar = Y0.shape[1]  #number of variables
    dt = (tspan[1]-tspan[0])/(Nt-1)
    Y = np.zeros((Nvar, Nt))   #initial value of Y
    Y[:,0] = Y0
    t = list(np.linspace(tspan[0], tspan[1], Nt))
    for i in range(0, Nt-1):
        K1 = f(Y[:,i], t[i])
        K2 = f(Y[:,i]+(K1*dt/2).flatten('F'), t[i]+dt/2.0)
        K3 = f(Y[:,i]+(K2*dt/2).flatten('F'), t[i]+dt/2.0)
        K4 = f(Y[:,i]+(K3*dt/2).flatten('F'), t[i]+dt)
        Y[:,i+1] = Y[:,i] + dt/6*(K1+2*K2+2*K3+K4).flatten('F')
    return Y


def function_g(n, A, sub_n, H, X):
    """coupling function g(x,y)
    n--> node n
    sub_n --> dimension x11
    A--> adjacent matrix of node n x1"""
    arrow_n = A[n]
    neighbours = np.nonzero(arrow_n)[0] # neighbours of node n
    degree_n = np.sum(A,axis=1)[n]
    g1 = sum([np.dot(H[sub_n], X[i]) for i in list(neighbours)])
    g = g1 - degree_n*(np.dot(H[sub_n], X[n]))
    return g


def f_func(Y, t, A):
    """A -- adjacent"""
    YF= Y.flatten('F')
    step = 3
    X = [YF[i:i + step] for i in range(0, len(YF), step)]  # variable
    a = 1
    b = 3
    c = 1
    d = 5
    r = 0.06
    s = 4.0
    I = 3.2
    coup =1/30
    Nvar1 = len(list(YF)) #number of variables
    Y1 = np.zeros((Nvar1, 1))
    H = np.array([[1, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0]])
    for i in range(Nvar1):
        a1 = i//step  # X1
        b1 = i%step   # X11
        coupling_g = function_g(a1, A, b1, H, X)
        if b1 == 0:
            Y1[i, 0] = X[a1][1] - a*X[a1][0]**3 + b*X[a1][0]**2 - X[a1][2]+ I + coup*(coupling_g)
        elif b1 == 1:
            Y1[i, 0] = c -d*X[a1][0]**2 - X[a1][1] + coup*(coupling_g)
        elif b1 == 2:
            Y1[i, 0] = r*(s*(X[a1][0]+1.6)-X[a1][2]) + coup*(coupling_g)
    return Y1


def keplerRK4(A):
    Dimension = 3  # 3 dimension
    N_node = A.shape[1]  # number of nodes
    Y0 = np.zeros((1, Dimension*N_node))
    for i in range(Dimension*N_node):
        random.seed(i)
        Y0[0, i] = i/100
    tspan = [0, 5000]
    Nt = 50000
    f = lambda Y,t:f_func(Y,t,A)
    Y = runge_kutta(f, tspan, Y0, Nt)
    # f1 = lambda Y1, t: f_func(Y1, t, AA)
    # Y1 = runge_kutta(f1, tspan, Y0, Nt)




    c1 = [Y[3][i] - Y[0][i] for i in range(0, len(Y[5]))]
    c2 = [Y[4][i] - Y[1][i] for i in range(0, len(Y[5]))]
    c3 = [Y[5][i] - Y[2][i] for i in range(0, len(Y[5]))]
    #print(c1)
    plt.plot(list(np.linspace(tspan[0], tspan[1], Nt)), c1, label=r'$x_{2}^{(1)}-x_{1}^{(1)}$')
    plt.plot(list(np.linspace(tspan[0], tspan[1], Nt)), c2, label=r'$x_{2}^{(2)}-x_{1}^{(2)}$')
    plt.plot(list(np.linspace(tspan[0], tspan[1], Nt)), c3, label=r'$x_{2}^{(3)}-x_{1}^{(3)}$')
    plt.xlabel('t')
    plt.legend()
    plt.savefig('cat_syn.png')
    plt.show()








start = time.time()
A = network_topology.A1  # desyn
# A = network_topology.A2 #synchronizetion graph
keplerRK4(A)
end = time.time()
print('time cost:'+str(end-start))