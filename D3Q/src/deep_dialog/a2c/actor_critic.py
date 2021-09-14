import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np

use_cuda = torch.cuda.is_available()


class ValueNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ValueNetwork, self).__init__()
        self.model = nn.Sequential(nn.Linear(input_size, hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(hidden_size, hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(hidden_size, 1))

    def forward(self, inputs):
        return self.model(inputs)


class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.model = nn.Sequential(nn.Linear(input_size, hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(hidden_size, hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(hidden_size, output_size),
                                   nn.Softmax(dim=1))

    def forward(self, inputs):
        out = self.model(inputs)
        dist = Categorical(out)
        return dist


class A2C(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(A2C, self).__init__()

        # init models
        self.value = ValueNetwork(input_size, hidden_size)
        self.target_value = ValueNetwork(input_size, hidden_size)  # 更新方式是否正确，更新步数 重点理清！
        self.policy = PolicyNetwork(input_size, hidden_size, output_size)

        # first sync
        self.target_value.load_state_dict(self.value.state_dict())  # target_value的使用，貌似没有使用target_V

        # hyper parameters
        self.gamma = 0.9
        self.max_norm = 10  # 尝试多种值
        self.lr = 0.02  # 尝试多种值

        # optimizer
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=self.lr)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

        # self.to(device)
        if use_cuda:
            self.cuda()

    def update_fixed_target_network(self):
        self.target_value.load_state_dict(self.value.state_dict())

    def Variable(self, x):
        return Variable(x, requires_grad=False).cuda() if use_cuda else Variable(x, requires_grad=False)

    def run(self, batch):
        ########## load batch ##########
        s = self.Variable(torch.FloatTensor(batch[0]))  # size: (16,213)
        a = self.Variable(torch.LongTensor(batch[1]))  # size: (16,1)
        r = self.Variable(torch.FloatTensor([batch[2]]))  # size: (1,16,1)
        s_prime = self.Variable(torch.FloatTensor(batch[3]))  # size: (16,213)

        ########## Value Part ##########
        self.value.train()
        self.value_optimizer.zero_grad()

        v = self.value(s)  # size: (16,1)
        v_prime = self.target_value(s_prime)  # size: (16,1)

        td_error = torch.squeeze(r.squeeze_(0) + torch.mul(v_prime, self.gamma) - v,
                                 dim=1)  # r.squeeze_(0) (16,1)  td_error size: (16,1)--> (16)
        value_loss = td_error.pow(2).sum()
        value_loss.backward()
        clip_grad_norm_(self.value.parameters(), self.max_norm)
        self.value_optimizer.step()

        ########## Policy Part ##########
        self.policy.train()
        self.policy_optimizer.zero_grad()

        dist = self.policy(s)  # size: (16,31)
        action = dist.sample()
        log_probs = - dist.log_prob(action)  # size: (16)

        actor_loss = torch.mean(log_probs * td_error.detach())
        actor_loss.backward()
        clip_grad_norm_(self.policy.parameters(), self.max_norm)
        self.policy_optimizer.step()

    def predict_a(self, inputs):
        self.policy.eval()

        inputs = self.Variable(torch.from_numpy(inputs).float())
        dist = self.policy(inputs)  # size: (1,31)
        a = dist.sample().item()
        return a  # 返回的是动作索引,类型为常数

    def save_model(self, model_path):
        torch.save(self.policy.state_dict(), model_path)
        torch.save(self.value.state_dict(), model_path)
        print("model saved.")

    def load_model(self, model_path):
        self.policy.load_state_dict(torch.load(model_path))
        self.value.load_state_dict(torch.load(model_path))
        print("model loaded.")
