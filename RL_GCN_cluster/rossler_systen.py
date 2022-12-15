"""Simulation coupled rossler system with different network topology"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
from mpl_toolkits.mplot3d import Axes3D
import torch


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
    a = 0.2
    b = 0.2
    c = 5.7
    coup =0.05
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
            Y1[i, 0] = -X[a1][1] - X[a1][2] + coup*(coupling_g)
        elif b1 == 1:
            Y1[i, 0] = X[a1][0] + a*X[a1][1] + coup*(coupling_g)
        elif b1 == 2:
            Y1[i, 0] = b + X[a1][2]*(X[a1][0]-c) + coup*(coupling_g)
    return Y1


def keplerRK4(A, AA):
    Dimension = 3  # 3 dimension
    N_node = A.shape[1]  # number of nodes
    Y0 = np.zeros((1, Dimension*N_node))
    for i in range(Dimension*N_node):
        random.seed(i)
        Y0[0, i] = i/100.0
    tspan = [0, 200]
    Nt = 2000
    f = lambda Y,t:f_func(Y,t,A)
    Y = runge_kutta(f, tspan, Y0, Nt)
    f1 = lambda Y1, t: f_func(Y1, t, AA)
    Y1 = runge_kutta(f1, tspan, Y0, Nt)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot(Y1[0], Y1[1], Y1[2])
    ax.plot(Y1[3], Y1[4], Y1[5])
    plt.savefig('syn.jpg')
    plt.show()

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot(Y[0], Y[1], Y[2])
    ax.plot(Y[3], Y[4], Y[5])
    plt.savefig('desyn.jpg')
    plt.show()

    c1 = [Y[3][i] - Y[0][i] for i in range(0, len(Y[5]))]
    c2 = [Y[4][i] - Y[1][i] for i in range(0, len(Y[5]))]
    c3 = [Y[5][i] - Y[2][i] for i in range(0, len(Y[5]))]
    plt.plot(list(np.linspace(tspan[0], tspan[1], Nt)), c1, label=r'$x_{2}^{(1)}-x_{1}^{(1)}$')
    plt.plot(list(np.linspace(tspan[0], tspan[1], Nt)), c2, label=r'$x_{2}^{(2)}-x_{1}^{(2)}$')
    plt.plot(list(np.linspace(tspan[0], tspan[1], Nt)), c3, label=r'$x_{2}^{(3)}-x_{1}^{(3)}$')
    plt.xlabel('t')
    plt.legend()
    plt.savefig('node12desynch.jpg')
    plt.show()

    c1 = [Y1[3][i] - Y1[0][i] for i in range(0, len(Y[5]))]
    c2 = [Y1[4][i] - Y1[1][i] for i in range(0, len(Y[5]))]
    c3 = [Y1[5][i] - Y1[2][i] for i in range(0, len(Y[5]))]
    plt.plot(list(np.linspace(tspan[0], tspan[1], Nt)), c1, label=r'$x_{2}^{(1)}-x_{1}^{(1)}$')
    plt.plot(list(np.linspace(tspan[0], tspan[1], Nt)), c2, label=r'$x_{2}^{(2)}-x_{1}^{(2)}$')
    plt.plot(list(np.linspace(tspan[0], tspan[1], Nt)), c3, label=r'$x_{2}^{(3)}-x_{1}^{(3)}$')
    plt.xlabel('t')
    plt.legend()
    plt.savefig('node12synch.jpg')
    plt.show()


    # fig3 = plt.figure(figsize=(20, 10))
    # #fig3 = plt.figure()
    # plt.subplot(2, 3, 1)
    # c1 = [Y[3][i] - Y[0][i] for i in range(0, len(Y[5]))]
    # c2 = [Y[4][i] - Y[1][i] for i in range(0, len(Y[5]))]
    # c3 = [Y[5][i] - Y[2][i] for i in range(0, len(Y[5]))]
    # plt.plot(list(np.linspace(tspan[0], tspan[1], Nt)), c1, label=r'$x_{2}^{(1)}-x_{1}^{(1)}$')
    # plt.plot(list(np.linspace(tspan[0], tspan[1], Nt)), c2, label=r'$x_{2}^{(2)}-x_{1}^{(2)}$')
    # plt.plot(list(np.linspace(tspan[0], tspan[1], Nt)), c3, label=r'$x_{2}^{(3)}-x_{1}^{(3)}$')
    # plt.xlabel('t')
    # plt.legend()
    #
    # plt.subplot(2, 3, 2)
    # c1 = [Y[9][i] - Y[6][i] for i in range(0, len(Y[5]))]
    # c2 = [Y[10][i] - Y[7][i] for i in range(0, len(Y[5]))]
    # c3 = [Y[11][i] - Y[8][i] for i in range(0, len(Y[5]))]
    # plt.plot(list(np.linspace(tspan[0], tspan[1], Nt)), c1, label=r'$x_{4}^{(1)}-x_{3}^{(1)}$')
    # plt.plot(list(np.linspace(tspan[0], tspan[1], Nt)), c2, label=r'$x_{4}^{(2)}-x_{3}^{(2)}$')
    # plt.plot(list(np.linspace(tspan[0], tspan[1], Nt)), c3, label=r'$x_{4}^{(3)}-x_{3}^{(3)}$')
    # plt.xlabel('t')
    # plt.legend()
    #
    # plt.subplot(2, 3, 3)
    # c1 = [Y[15][i] - Y[12][i] for i in range(0, len(Y[5]))]
    # c2 = [Y[16][i] - Y[13][i] for i in range(0, len(Y[5]))]
    # c3 = [Y[17][i] - Y[14][i] for i in range(0, len(Y[5]))]
    # plt.plot(list(np.linspace(tspan[0], tspan[1], Nt)), c1, label=r'$x_{6}^{(1)}-x_{5}^{(1)}$')
    # plt.plot(list(np.linspace(tspan[0], tspan[1], Nt)), c2, label=r'$x_{6}^{(2)}-x_{5}^{(2)}$')
    # plt.plot(list(np.linspace(tspan[0], tspan[1], Nt)), c3, label=r'$x_{6}^{(3)}-x_{5}^{(3)}$')
    # plt.xlabel('t')
    # plt.legend()
    #
    # plt.subplot(2, 3, 4)
    # c1 = [Y1[3][i] - Y1[0][i] for i in range(0, len(Y1[5]))]
    # c2 = [Y1[4][i] - Y1[1][i] for i in range(0, len(Y1[5]))]
    # c3 = [Y1[5][i] - Y1[2][i] for i in range(0, len(Y1[5]))]
    # plt.plot(list(np.linspace(tspan[0], tspan[1], Nt)), c1, label=r'$x_{2}^{(1)}-x_{1}^{(1)}$')
    # plt.plot(list(np.linspace(tspan[0], tspan[1], Nt)), c2, label=r'$x_{2}^{(2)}-x_{1}^{(2)}$')
    # plt.plot(list(np.linspace(tspan[0], tspan[1], Nt)), c3, label=r'$x_{2}^{(3)}-x_{1}^{(3)}$')
    # plt.xlabel('t')
    # plt.legend()
    #
    # plt.subplot(2, 3, 5)
    # c1 = [Y1[9][i] - Y1[6][i] for i in range(0, len(Y1[5]))]
    # c2 = [Y1[10][i] - Y1[7][i] for i in range(0, len(Y1[5]))]
    # c3 = [Y1[11][i] - Y1[8][i] for i in range(0, len(Y1[5]))]
    # plt.plot(list(np.linspace(tspan[0], tspan[1], Nt)), c1, label=r'$x_{4}^{(1)}-x_{3}^{(1)}$')
    # plt.plot(list(np.linspace(tspan[0], tspan[1], Nt)), c2, label=r'$x_{4}^{(2)}-x_{3}^{(2)}$')
    # plt.plot(list(np.linspace(tspan[0], tspan[1], Nt)), c3, label=r'$x_{4}^{(3)}-x_{3}^{(3)}$')
    # plt.xlabel('t')
    # plt.legend()
    #
    # plt.subplot(2, 3, 6)
    # c1 = [Y1[15][i] - Y1[12][i] for i in range(0, len(Y1[5]))]
    # c2 = [Y1[16][i] - Y1[13][i] for i in range(0, len(Y1[5]))]
    # c3 = [Y1[17][i] - Y1[14][i] for i in range(0, len(Y1[5]))]
    # plt.plot(list(np.linspace(tspan[0], tspan[1], Nt)), c1, label=r'$x_{6}^{(1)}-x_{5}^{(1)}$')
    # plt.plot(list(np.linspace(tspan[0], tspan[1], Nt)), c2, label=r'$x_{6}^{(2)}-x_{5}^{(2)}$')
    # plt.plot(list(np.linspace(tspan[0], tspan[1], Nt)), c3, label=r'$x_{6}^{(3)}-x_{5}^{(3)}$')
    # plt.xlabel('t')
    # plt.legend()
    #
    # plt.savefig('node12desynch.pdf')
    # plt.show()


def draw_graph(state):
    '''state --> edge_index'''
    G1 = nx.Graph()
    for i in range(state.shape[1]):
        G1.add_edge(int(state[0][i]), int(state[1][i]))
    #nx.draw_networkx(G1)
    return G1
state1 = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                      [1, 2, 3, 4, 5, 6, 7, 8, 9, 1]])
state2 = torch.tensor([[0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                      [1, 4, 2, 3, 4, 5, 6, 7, 8, 9, 1]])

state3 = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 9, 9, 9, 9, 9, 7, 7, 7, 7, 8, 6, 6,
          6, 5, 5, 3, 3, 4, 1, 9, 7, 8, 6, 5, 3, 4, 7, 9, 8, 3, 8, 6, 5, 2, 3, 4,
          5, 8, 2, 4, 5, 3, 4, 2, 4, 4, 2, 2],
         [1, 9, 7, 8, 6, 5, 3, 4, 7, 9, 8, 3, 8, 6, 5, 2, 3, 4, 5, 8, 2, 4, 5, 3,
         4, 2, 4, 4, 2, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 9, 9, 9, 9, 9, 7,
          7, 7, 7, 8, 6, 6, 6, 5, 5, 3, 3, 4]])
A = draw_graph(state2)
AA = draw_graph(state3)
A = nx.to_numpy_array(A)
AA = nx.to_numpy_array(AA)
#A = network_topology.A # dessyn
#AA = network_topology.AA # syn
keplerRK4(A, AA)