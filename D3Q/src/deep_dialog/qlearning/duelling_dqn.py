'''
created on Mar 08, 2018
@author: Shang-Yu Su
'''
import math
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

use_cuda = torch.cuda.is_available()


class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.

    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter

    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        """Initialization."""
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        self.register_buffer(
            "weight_epsilon", torch.Tensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )

    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.

        We don't use separate statements on train / eval mode.
        It doesn't show remarkable difference of performance.
        """
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )

    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        x = torch.randn(size)

        return x.sign().mul(x.abs().sqrt())


# class Network(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(Network, self).__init__()
#         self.feature_layer = nn.Sequential(nn.Linear(input_size, hidden_size),
#                                            nn.ReLU())
#         self.value_layer = nn.Sequential(nn.Linear(hidden_size, hidden_size),
#                                          nn.ReLU(),
#                                          nn.Linear(hidden_size, 1))
#         self.advantage_layer = nn.Sequential(nn.Linear(hidden_size, hidden_size),
#                                              nn.ReLU(),
#                                              nn.Linear(hidden_size, output_size))
#
#     def forward(self, inputs):
#         out = self.feature_layer(inputs)
#         value = self.value_layer(out)
#         advantage = self.advantage_layer(out)
#         q = value + advantage - torch.mean(advantage, dim=-1, keepdim=True)
#         return q

class Network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Network, self).__init__()
        # common feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )

        # value layer
        self.value_hid_layer = NoisyLinear(hidden_size, hidden_size)
        self.value_layer = NoisyLinear(hidden_size, 1)

        # set advantage layer
        self.advantage_hid_layer = NoisyLinear(hidden_size, hidden_size)
        self.advantage_layer = NoisyLinear(hidden_size, output_size)

    def forward(self, inputs):
        feature = self.feature_layer(inputs)
        value = self.value_layer(F.relu(self.value_hid_layer(feature)))
        advantage = self.advantage_layer(F.relu(self.advantage_hid_layer(feature)))
        q = value + advantage - torch.mean(advantage, dim=-1, keepdim=True)
        return q

    def reset_noise(self):
        """Reset all noisy layers."""
        self.value_hid_layer.reset_noise()
        self.value_layer.reset_noise()
        self.advantage_hid_layer.reset_noise()
        self.advantage_layer.reset_noise()


class DuellingDQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):  # (state_dimension, hidden_size, num_actions)
        super(DuellingDQN, self).__init__()

        # model
        self.model = Network(input_size, hidden_size, output_size)
        # target model
        self.target_model = Network(input_size, hidden_size, output_size)
        # first sync
        self.target_model.load_state_dict(self.model.state_dict())

        # hyper parameters
        self.gamma = 0.9
        self.reg_l2 = 1e-3
        self.max_norm = 1
        self.target_update_period = 100
        lr = 0.002

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # self.to(device)
        if use_cuda:
            self.cuda()

    # 更新目标网络（将model的参数载入到target_model的参数）
    def update_fixed_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def Variable(self, x):
        return Variable(x, requires_grad=False).cuda() if use_cuda else Variable(x, requires_grad=False)

    # 在AgentDuellingDQN的train/train_iter函数中被调用
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

        # the batch style of (td_error = r + self.gamma * torch.max(q_prime) - q[a])  TD误差部分
        td_error = r.squeeze_(0) + torch.mul(torch.max(q_prime, 1)[0], self.gamma).unsqueeze(1) - torch.gather(q, 1, a)
        loss += td_error.pow(2).sum()  # Loss Function是td-error的均方误差
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.max_norm)
        self.optimizer.step()

        # NoisyNet settings: (reset noise)
        self.model.reset_noise()
        self.target_model.reset_noise()

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
