from node2vec import Node2Vec
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.utils import to_networkx, from_networkx
from torch_geometric.nn import GCNConv

def extreme_eigenvalues(state):
    '''extreme eigenvalues of laplacian matrix'''
    graph = draw_graph(state)
    laplacian_spec = nx.laplacian_spectrum(graph)
    laplacian_spec.sort()
    min_eigen, max_eigen = laplacian_spec[1], laplacian_spec[-1]
    print(min_eigen, max_eigen)
    return min_eigen, max_eigen


def draw_graph(state):
    '''state --> edge_index'''
    G1 = nx.Graph()
    for i in range(state.shape[1]):
        G1.add_edge(int(state[0][i]), int(state[1][i]))
    nx.draw_networkx(G1)
    plt.show()
    return G1

def difference_two_graph(G1, G2):
    edge_list1 = list(G1.edges())
    edge_list2 = list(G2.edges())
    # print(edge_list1, edge_list2)
    # print(len(edge_list1))
    a = 0
    for i in range(len(edge_list1)):
        # print(edge_list1[i])
        if G2.has_edge(edge_list1[i][0], edge_list1[i][1]) == 1:
            a += 1
    b = 0
    for i in range(len(edge_list2)):
        if G1.has_edge(edge_list2[i][0], edge_list2[i][1]) == 1:
            b += 1
    print(a, b)
    return a+b




a = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
         3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8,
         8, 8, 8, 8, 9, 9, 9, 9, 9],
        [1, 2, 3, 4, 5, 0, 3, 2, 6, 7, 0, 1, 8, 5, 7, 4, 2, 3, 0, 1, 2, 4, 7, 5,
         8, 0, 2, 3, 8, 9, 7, 0, 2, 3, 7, 9, 1, 9, 8, 6, 1, 2, 3, 4, 5, 8, 2, 3,
         4, 6, 7, 9, 4, 5, 6, 8, 9]])
G1 = draw_graph(a)
extreme_eigenvalues(a)

G2 = nx.Graph()
for i in range(9):
    G2.add_edge(i, i+1)
nx.draw_networkx(G2)
plt.show()
difference_two_graph(G1, G2)
