import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy
from scipy import linalg

G = nx.barabasi_albert_graph(60, 10, seed=2)  # BA network
df = pd.read_csv('bn_cat.txt', delimiter=' ', header=None)
#df[1] = df[1].apply(lambda x: x.split('\\')[0])
#print(df)
G_N = nx.Graph()
for index, row in df.iterrows():
    S = row[0]
    T = row[1]
    G_N.add_edge(S, T)

nx.write_gexf(G_N,'cat_brain.gexf' )
print(G_N.nodes)
print(len(G_N.nodes))

def topology_matrix(GG):
    lap_mat = nx.laplacian_matrix(GG).todense()
    A2 = nx.to_numpy_array(GG)  # adjacent matrix of network
    eigval= linalg.eigvals(lap_mat)
    eigval.sort()
    #print(eigval)
    eig_max = round(eigval[-1], 4)
    eig_min = round(eigval[1], 4)
    print('max eigen:' + str(eig_max))
    print('min eigen:' + str(eig_min))
    return A2

A1 = topology_matrix(G_N)

# action sequence
list1 =[[9, 63], [39, 13], [20, 13], [31, 24], [35, 63], [64, 0], [42, 13], [9, 13], [14, 63], [0, 13], [63, 24], [18, 48], [29, 20], [22, 0], [6, 63], [9, 1], [15, 24], [35, 22], [25, 0], [14, 24], [4, 23], [35, 13], [3, 31], [0, 7], [40, 1], [60, 13], [57, 20], [52, 63], [26, 56], [40, 21], [6, 1], [5, 24], [42, 24]]



for i in range(len(list1)):
    a, b = list1[i][0], list1[i][1]
    if G_N.has_edge(a, b) ==1:
        G_N.remove_edge(a, b)
    else:
        G_N.add_edge(a, b)

lap_mat = nx.laplacian_matrix(G_N).todense()
eigval= linalg.eigvals(lap_mat)
eigval.sort()
#print(eigval)
eig_max = round(eigval[-1], 4)
eig_min = round(eigval[1], 4)
print('max eigen:' + str(eig_max))
print('min eigen:' + str(eig_min))
A2 = nx.to_numpy_array(G_N)
print(A2)