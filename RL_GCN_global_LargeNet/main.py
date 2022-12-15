"""cat brain network synchronization based on reforcement learning"""

import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from node2vec import Node2Vec
from graphobject import GraphEnv
import graphobject
from torch_geometric.utils import to_networkx, from_networkx
import torch.nn as nn
import torch.nn.functional as F
import rl_utils
import time


def node_to_vector(state, feature1_dim):
    G1 = graphobject.draw_graph(state)
    node2vec = Node2Vec(G1, feature1_dim, quiet=True, seed=0)
    model = node2vec.fit(window=5, min_count=1)
    vectorlist = [model.wv.get_vector(str(i)) for i in list(G1.nodes)]
    state_feature = torch.tensor(vectorlist)
    return state_feature


def add_feature(state_feature, action1):
    '''add feature to node embedding vector according to choosing action'''

    Nnodes = env.num_node
    action_feature = torch.zeros([Nnodes, 1]) # convert action 1 to feature of nodes,
    # combine with node embedding
    action_feature[int(action1)][0] = 1
    #print(state_feature)
    #print(action_feature)
    state_feature1 = torch.cat((state_feature, action_feature), 1)
    return state_feature1


class GCN_Actor1(torch.nn.Module):
    def __init__(self, feature1_dim, hidden1_dim, hidden2_dim):
        super().__init__()
        torch.manual_seed(12)
        self.conv1 = GCNConv(feature1_dim, hidden1_dim)
        self.conv2 = GCNConv(hidden1_dim, hidden2_dim)
        self.classifier = Linear(hidden2_dim, 1)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        out = self.classifier(h)
        out = F.softmax(torch.unbind(out, dim=1)[0])
        return out


class GCN_Actor2(torch.nn.Module):
    def __init__(self, feature2_dim, hidden1_dim, hidden2_dim):
        """feature_dim1 = feature_dim+1"""
        super().__init__()
        torch.manual_seed(123)
        self.conv1 = GCNConv(feature2_dim, hidden1_dim)
        self.conv2 = GCNConv(hidden1_dim, hidden2_dim)
        self.classifier = Linear(hidden2_dim, 1)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        out = self.classifier(h)
        out = F.softmax(torch.unbind(out, dim=1)[0])
        return out


class GCN_ValueNet(torch.nn.Module):
    def __init__(self, feature1_dim, hidden1_dim, hidden2_dim, action_dim):
        super().__init__()
        torch.manual_seed(1234)
        self.conv1 = GCNConv(feature1_dim, hidden1_dim)
        self.conv2 = GCNConv(hidden1_dim, hidden2_dim)
        self.conv3 = GCNConv(hidden2_dim, 1)
        self.conv4 = nn.Linear(action_dim, 1)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()
        h = h.reshape(-1)

        out = self.conv4(h)
        #print('out:'+str(out))
        return out


class ActorCritic:
    def __init__(self, feature1_dim, feature2_dim, hidden1_dim,
                 hidden2_dim, action_dim, actor1_lr, actor2_lr,
                 critic_lr, gamma, device):
        self.actor1 = GCN_Actor1(feature1_dim, hidden1_dim, hidden2_dim).to(device)
        self.actor2 = GCN_Actor2(feature2_dim, hidden1_dim, hidden2_dim).to(device)
        self.critic = GCN_ValueNet(feature1_dim, hidden1_dim, hidden2_dim, action_dim)
        self.actor1_optimizer = torch.optim.Adam(self.actor1.parameters(), lr=actor1_lr)
        self.actor2_optimizer = torch.optim.Adam(self.actor2.parameters(), lr=actor2_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.device = device

    def take_action(self, state):
        # action_feature, edge_index, action1 = add_feature(state)[0], \
        #                                       add_feature(state)[1], add_feature(state)[2]
        state_feature1 = node_to_vector(state, feature1_dim)
        state = state.type(torch.LongTensor)
        probs1 = self.actor1(state_feature1, state)
        action_dist1 = torch.distributions.Categorical(probs1)
        action1 = action_dist1.sample()
        action1 = action1.item()

        state_feature2 = add_feature(state_feature1, action1)
        probs2 = self.actor2(state_feature2, state)  #probability of each action
        action_dist2 = torch.distributions.Categorical(probs2)
        action2 = action_dist2.sample()
        action2 = action2.item()
        #print(action2)
        action = []
        action.append(int(action1))
        action.append(int(action2))
        return action, state_feature1, state_feature2

    def update(self, transition_dict):
        states = transition_dict['states']
        actions = torch.tensor(transition_dict['actions'])
        actions1, actions2 = torch.unbind(actions, dim=1)
        actions1 = torch.reshape(actions1, (actions1.shape[0], 1))
        actions2 = torch.reshape(actions2, (actions2.shape[0], 1))

        states_feature1 = transition_dict['state_feature1']
        states_feature2 = transition_dict['state_feature2']

        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1,1).to(self.device)
        next_states = transition_dict['next_states']
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)


        next_states_feature = [node_to_vector(next_states[i], feature1_dim) for i in range(len(next_states))]
        next_rewards = [self.gamma*self.critic(next_states_feature[i], next_states[i])*(1-dones[i])
                       for i in range(len(next_states))]
        next_rewards = torch.tensor(next_rewards, dtype=torch.float).view(-1, 1).to(self.device)
        td_target = rewards + next_rewards

        td_target1 = [self.critic(states_feature1[i], states[i]) for i in range(len(states))]
        td_target1 = torch.tensor(td_target1, dtype=torch.float).view(-1, 1).to(self.device)
        td_delta = td_target - td_target1


        log_actor1 = [self.actor1(states_feature1[i], states[i]) for i in range(len(states))]
        log_actor1 = torch.stack(log_actor1)
        log_probs1 = torch.log(log_actor1.gather(1, actions1))


        log_actor2 = [self.actor2(states_feature2[i], states[i]) for i in range(len(states))]
        log_actor2 = torch.stack(log_actor2)
        log_probs2 = torch.log(log_actor2.gather(1, actions2))

        actor_loss1 = torch.mean(-log_probs1 * td_delta.detach())
        actor_loss2 = torch.mean(-log_probs2 * td_delta.detach())
        critic = [self.critic(states_feature1[i], states[i]) for i in range(len(states))]
        critic = torch.stack(critic)
        critic_loss = torch.mean(F.mse_loss(critic, td_target))

        self.actor1_optimizer.zero_grad()
        self.actor2_optimizer.zero_grad()
        actor_loss1.backward()
        actor_loss2.backward()
        critic_loss.backward()

        self.actor1_optimizer.step()
        self.actor2_optimizer.step()
        self.critic_optimizer.step()
        #print(self.critic_optimizer)

for i in range(1):
    start = time.time()
    actor1_lr = 1e-2
    actor2_lr = 1e-2
    critic_lr = 1e-1
    num_episodes = 100
    feature1_dim = 10  #8
    feature2_dim = feature1_dim+1
    hidden1_dim = 20   #10
    hidden2_dim = 10   #10
    gamma = 0.98
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    env = GraphEnv()
    action_dim = env.action_space
    agent = ActorCritic(feature1_dim, feature2_dim, hidden1_dim,
                 hidden2_dim, action_dim, actor1_lr, actor2_lr,
                 critic_lr, gamma, device)
    return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('returns')
    plt.title('Actor-Critic on graph')
    plt.savefig('brain_gcn.pdf')
    plt.show()




