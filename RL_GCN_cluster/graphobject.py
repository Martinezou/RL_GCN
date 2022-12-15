import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import networkx as nx
from torch_geometric.utils import to_networkx, from_networkx
import torch


def graph_to_tensor(G1):
    edge_list = list(G1.edges())
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_index1 = edge_index[0, :]
    edge_index2 = edge_index[1, :]
    edge_index1 = torch.stack([edge_index2, edge_index1], dim=0)
    edge_index = torch.cat((edge_index, edge_index1), dim=1)
    # print(edge_index)
    return edge_index


def extreme_eigenvalues(state):
    '''extreme eigenvalues of laplacian matrix'''
    graph = draw_graph(state)
    laplacian_spec = nx.laplacian_spectrum(graph)
    laplacian_spec.sort()
    laplacian_spec = [round(i, 4) for i in laplacian_spec]
    min_eigen, max_eigen = laplacian_spec[1], laplacian_spec[-1]
    return min_eigen, max_eigen, laplacian_spec

def get_keys(d, value):
    return [k for k,v in d.items() if v==value]

def draw_graph(state):
    '''state --> edge_index'''
    # print(state)
    G1 = nx.Graph()
    for i in range(state.shape[1]):
        G1.add_edge(int(state[0][i]), int(state[1][i]))
    #nx.draw_networkx(G1)
    return G1

def quotient_graph(state, partition):
    G1 = draw_graph(state)
    num_nodes = G1.number_of_nodes()
    P = np.zeros((num_nodes, len(partition)))
    for i in range(len(partition)):
        for j in range(len(partition[i])):  # [[0, 1, 2, 4, 5], [3]]
            P[partition[i][j], i] = 1
    P = np.mat(P)
    lap_mat = nx.laplacian_matrix(G1).todense()
    LQ = (P.T * P).I * P.T * lap_mat * P
    laplacian_spec = np.linalg.eig(LQ)[0]
    laplacian_spec = [round(i, 4) for i in laplacian_spec]
    laplacian_spec.sort()
    # print(laplacian_spec)
    return laplacian_spec

def EEP(state, node_cluster):
    G1 = draw_graph(state)
    num_node = G1.number_of_nodes()
    cluster = list(node_cluster.values())
    Ncluster = max(cluster)+1  #number of class
    neighbours = []
    for i in range(num_node):
        neighbours.append(list(G1.neighbors(i))) #neighbours of each node
    neighbour_class = [[] for i in range(num_node)]
    for i in range(num_node):
        for j in range(len(neighbours[i])):
            neighbour_class[i].append(node_cluster[neighbours[i][j]]) #neighbour class of each node

    class_nodes = []
    for i in range(Ncluster):
        class_nodes.append(get_keys(node_cluster, i))
    for i in range(len(neighbour_class)):
        nodeclass = node_cluster[i]
        for j in neighbour_class[i]:
            if j == nodeclass:
                while j in neighbour_class[i]:
                    neighbour_class[i].remove(j)
    #print(neighbour_class)
    list1 = [[] for i in range(len(class_nodes))]
    for i in range(len(class_nodes)):
        if len(class_nodes[i]) == 1:
            list1[i].append(0)
        else:
            for j in range(1, len(class_nodes[i])):
                a = neighbour_class[class_nodes[i][0]] == neighbour_class[class_nodes[i][j]]
                if a==1:
                    list1[i].append(0)
                else:
                    list1[i].append(1)
    list2 = []
    for i in range(len(list1)):
        if sum(list1[i]) == 0:
            list2.append(0)
        else:
            list2.append(1)
    a = sum(list2)   # number of cluster not satisfy EEP
    #a = Ncluster - a
    if a == 0:
        b = 1
    else:
        b = 0    # graph not satisfy EEP
    return b, a


class GraphEnv(gym.Env):
    def __init__(self):
        self.num_node = 6  #number of node in graph
        self.action_space = self.num_node #action is devided to two steps: starting node
        # and ending node, the action space of each step equals to the number of nodes
        self.observation_space = spaces.Box(low=0, high=1000, shape=(1,))
        self.state = None
        self.steps = 0
        self._max_episode_steps = 500
        # bounds for eigenvalues
        self.low_eigenvalue = 0.02
        self.high_eigenvalue = 5.5
        self.coupling = 1
        self.node_cluster = {0: 0, 1: 0, 2: 0, 3: 1, 4: 0, 5: 0}
        self.node_cluster1 = torch.tensor([[0], [0], [0],
                                           [1],
                                           [0], [0]])
        self.partition = [[0, 1, 2, 4, 5],
                          [3]]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def stepGraph(self, action, state):
        G1 = draw_graph(state)
        starting, ending = action[0], action[1]
        a = 0 # whether action is valid
        if starting != ending:
            if G1.has_edge(starting, ending) ==1 and G1.degree(starting)>1 and G1.degree(ending)>1:
                G1.remove_edge(starting, ending)
                a = 1
                if nx.is_connected(G1) == 0:
                    G1.add_edge(starting, ending)
                    a = 0
            elif G1.has_edge(starting, ending) ==0:
                G1.add_edge(starting, ending)
                a = 1
        else:
            a = 0

        self.state = graph_to_tensor(G1)
        return self.state, a

    def step(self, action, state):
        self.state, a = self.stepGraph(action, state)
        #print(self.state)
        min_eig, max_eig, laplacian_spec = extreme_eigenvalues(self.state)
        #print(spec)
        EEP1, Num_EEP = EEP(self.state, self.node_cluster)
        def eigen_criterion():
            if self.coupling*min_eig<self.low_eigenvalue or self.coupling*max_eig>self.high_eigenvalue:
                done1 = 1
            else:
                done1 = 0
            if EEP1 ==1:
                quotient_spec = quotient_graph(state, self.partition)
                #spec = [i for i in laplacian_spec if i not in quotient_spec]
                for i in range(len(laplacian_spec)):
                    for j in laplacian_spec:
                        if j in quotient_spec:
                            laplacian_spec.remove(j)
                            quotient_spec.remove(j)
                spec = laplacian_spec
                if len(spec)>0:
                    if self.low_eigenvalue < self.coupling*spec[0] and self.coupling*spec[-1]< self.high_eigenvalue:
                        done2 = 1
                    else:
                        done2 = 0
                else:
                    done2 = 0
            else:
                done2 = 0
            return done1, done2
        done1, done2=eigen_criterion()
        done = done1 and done2
        done = bool(done)
        #done = bool(EEP1)
        if a == 0:
            reward =0
        else:
            if not done:
                if EEP1 == 0:
                    reward = -Num_EEP
                else:
                    reward = -1
            else:
                reward = 100
        return self.state, reward, done, {}, a



    def reset(self):
        G = nx.Graph()
        num_node = self.num_node
        for i in range(num_node-1):
            G.add_edge(i, i+1)

        self.state = graph_to_tensor(G)
        # print(self.state)
        return self.state