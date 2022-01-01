import math
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Network, self).__init__()

        # Feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU())

        # Value layer
        self.value_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,1))

        # Advantage layer
        self.advantage_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,output_size))

    def forward(self, inputs):
        feature = self.feature_layer(inputs)
        value = self.value_layer(feature)
        advantage = self.advantage_layer(feature)

        q = value + advantage - torch.mean(advantage, dim=-1, keepdim=True)
        return q


class NoNoisyNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):  # (state_dimension, hidden_size, num_actions)
        super(NoNoisyNet, self).__init__()

        # model
        self.model = Network(input_size, hidden_size, output_size).to(device)
        # target model
        self.target_model = Network(input_size, hidden_size, output_size).to(device)
        # first sync
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        # hyper parameters
        #self.n_step = n_step
        #self.gamma = 0.9 ** self.n_step
        self.gamma = 0.9
        self.reg_l2 = 1e-3
        self.max_norm = 10
        self.target_update_period = 50
        lr = 0.002

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    # 更新网络
    def update_network(self):
        # update target network
        self.target_model.load_state_dict(self.model.state_dict())

    def Variable(self, x):
        return Variable(x, requires_grad=False).to(device)

    # 在AgentDQN的train/train_iter函数中被调用
    def singleBatch(self, batch):
        self.optimizer.zero_grad()
        loss = 0

        # each example in a batch: [s, a, r, s_prime, term]
        s = self.Variable(torch.FloatTensor(batch[0]))  # size: (16,213)
        a = self.Variable(torch.LongTensor(batch[1]))  # size: (16,1)
        r = self.Variable(torch.FloatTensor([batch[2]]))  # size: (1,16,1)
        s_prime = self.Variable(torch.FloatTensor(batch[3]))  # size: (16,213)

        q = self.model(s)  # size: (16,31)
        q_prime = self.target_model(s_prime)

        # double dqn td_error
        q_a = torch.gather(q, 1, a)
        td_target = r.squeeze_(0) + torch.mul(
            q_prime.gather(1, self.model(s_prime).argmax(dim=1, keepdim=True)).detach(), self.gamma).to(device)
        td_error = td_target - q_a

        loss += td_error.pow(2).sum()  # Loss Function是td-error的均方误差
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.max_norm)
        self.optimizer.step()

    def predict(self, inputs):  # 输入是representation，一个numpy.hstack的矩阵
        inputs = self.Variable(torch.from_numpy(inputs).float())
        a = self.model(inputs)
        a = a.detach().cpu().data.numpy()[0]
        return a

    ################################################################################
    #    Debug Functions
    ################################################################################
    def save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)
        print("model saved.")

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        print("model loaded.")