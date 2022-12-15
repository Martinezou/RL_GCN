import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import networkx as nx
from torch_geometric.utils import to_networkx, from_networkx
import torch
import network_topology


def graph_to_tensor(G1):
    edge_list = list(G1.edges())
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_index1 = edge_index[0, :]
    edge_index2 = edge_index[1, :]
    edge_index1 = torch.stack([edge_index2, edge_index1], dim=0)
    edge_index = torch.cat((edge_index, edge_index1), dim=1)
    return edge_index


def extreme_eigenvalues(state):
    '''extreme eigenvalues of laplacian matrix'''
    graph = draw_graph(state)
    laplacian_spec = nx.laplacian_spectrum(graph)
    laplacian_spec.sort()
    laplacian_spec = [round(i, 4) for i in laplacian_spec]
    min_eigen, max_eigen = laplacian_spec[1], laplacian_spec[-1]
    return min_eigen, max_eigen


def draw_graph(state):
    '''state --> edge_index'''
    G1 = nx.Graph()
    for i in range(state.shape[1]):
        G1.add_edge(int(state[0][i]), int(state[1][i]))
    #nx.draw_networkx(G1)
    return G1


class GraphEnv(gym.Env):
    def __init__(self):
        self.num_node = 65  #number of node in graph
        self.action_space = 65 #action is devided to two steps: starting node
        # and ending node, the action space of each step equals to the number of nodes
        self.observation_space = spaces.Box(low=0, high=1000, shape=(1,))
        self.state = None
        self.steps = 0
        self._max_episode_steps = 500
        # bounds for eigenvalues
        self.low_eigenvalue = 6.9
        self.high_eigenvalue = 10000

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
        min_eig, max_eig = extreme_eigenvalues(self.state)

        done = min_eig>self.low_eigenvalue and max_eig<self.high_eigenvalue
        done = bool(done)
        #print(a)
        if a == 0:
            reward = 0
        else:
            if not done:
                if min_eig < self.low_eigenvalue and max_eig > self.high_eigenvalue:
                    reward = -2.0
                elif self.high_eigenvalue > min_eig > self.low_eigenvalue and max_eig > self.high_eigenvalue:
                    reward = 1.0/(max_eig-self.high_eigenvalue+1)-1
                    #reward = -1
                elif min_eig < self.low_eigenvalue and self.low_eigenvalue < max_eig < self.high_eigenvalue:
                    reward = 1.0/(self.low_eigenvalue+1-min_eig)-1
                    #reward = -1
                else:
                    reward = -2.0

            else:
                reward = 100
        #print(reward)
        return self.state, reward, done, {}, a

    def reset(self):
        # G = nx.Graph()
        # num_node = self.num_node
        # for i in range(num_node-1):
        #     G.add_edge(i, i+1)
        # G.add_edge(0, 9)
        G = network_topology.G_N

        self.state = graph_to_tensor(G)
        return self.state