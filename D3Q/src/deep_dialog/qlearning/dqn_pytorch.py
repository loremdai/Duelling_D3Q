'''
created on Mar 08, 2018
@author: Shang-Yu Su
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
import numpy as np

use_cuda = torch.cuda.is_available()


class network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(network, self).__init__()
        self.model = nn.Sequential(nn.Linear(input_size, hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(hidden_size, output_size))

    def forward(self, inputs):
        return self.model(inputs)


class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):   # (state_dimension, hidden_size, num_actions)
        super(DQN, self).__init__()

        # model
        self.model = network(input_size, hidden_size, output_size)
        # target model
        self.target_model = network(input_size, hidden_size, output_size)
        # first sync
        self.target_model.load_state_dict(self.model.state_dict())

        # hyper parameters
        self.gamma = 0.9
        self.reg_l2 = 1e-3
        self.max_norm = 1
        self.target_update_period = 100
        lr = 0.001

        self.optimizer = optim.RMSprop(self.model.parameters(), lr=lr)

        self.batch_count = 0

        # self.to(device)
        if use_cuda:
            self.cuda()

    # 更新目标网络（将model的参数载入到target_model的参数）
    def update_fixed_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def Variable(self, x):
        return Variable(x, requires_grad=False).cuda() if use_cuda else Variable(x, requires_grad=False)

    # 在AgentDQN的train/train_iter函数中被调用
    def singleBatch(self, batch):   # batch的大小为(5,16)
        self.optimizer.zero_grad()
        loss = 0

        # each example in a batch: [s, a, r, s_prime, term]
        s = self.Variable(torch.FloatTensor(batch[0]))  # size: (1,16)
        a = self.Variable(torch.LongTensor(batch[1]))   # size: (1,16)
        r = self.Variable(torch.FloatTensor([batch[2]]))    # size: (1,16)
        s_prime = self.Variable(torch.FloatTensor(batch[3]))    # size: (1,16)

        q = self.model(s)
        q_prime = self.target_model(s_prime)    # 目标值网络

        # the batch style of (td_error = r + self.gamma * torch.max(q_prime) - q[a])  TD误差部分
        td_error = r.squeeze_(0) + torch.mul(torch.max(q_prime, 1)[0], self.gamma).unsqueeze(1) - torch.gather(q, 1, a)
        loss += td_error.pow(2).sum()   # Loss Function是td-error的均方误差

        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.max_norm)
        self.optimizer.step()

    def predict(self, inputs):  # 输入是representation，一个numpy.hstack的矩阵
        inputs = self.Variable(torch.from_numpy(inputs).float())
        return self.model(inputs).cpu().data.numpy()[0]


    ################################################################################
    #    Debug Functions
    ################################################################################
    def save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)
        print("model saved.")

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        print("model loaded.")
