import math
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NoisyLinear(nn.Module):
    def __init__(self, input_size, output_size, std_init=0.5):
        super(NoisyLinear, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(output_size, input_size))
        self.weight_sigma = nn.Parameter(
            torch.Tensor(output_size, input_size)
        )
        self.register_buffer(
            "weight_epsilon", torch.Tensor(output_size, input_size)
        )

        self.bias_mu = nn.Parameter(torch.Tensor(output_size))
        self.bias_sigma = nn.Parameter(torch.Tensor(output_size))
        self.register_buffer("bias_epsilon", torch.Tensor(output_size))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """implement factorized gaussian noise"""
        mu_range = 1 / math.sqrt(self.input_size)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.input_size)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.output_size)
        )

    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.input_size)
        epsilon_out = self.scale_noise(self.output_size)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
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


class Network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Network, self).__init__()

        # common feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU())

        # value layer
        self.value_hid_layer = NoisyLinear(hidden_size, hidden_size)
        self.value_layer = NoisyLinear(hidden_size, 1)

        # advantage layer
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


class NoDDQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_step):  # (state_dimension, hidden_size, num_actions)
        super(NoDDQN, self).__init__()

        # model
        self.model = Network(input_size, hidden_size, output_size).to(device)
        # target model
        self.target_model = Network(input_size, hidden_size, output_size).to(device)
        # first sync
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        # hyper parameters
        self.n_step = n_step
        self.gamma = 0.9 ** self.n_step
        self.reg_l2 = 1e-3
        self.max_norm = 10
        # self.target_update_period = 10
        lr = 0.001

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    # 更新网络
    def update_network(self):
        # update target network
        self.target_model.load_state_dict(self.model.state_dict())
        # NoisyNet settings: (reset noise)
        self.model.reset_noise()
        self.target_model.reset_noise()

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

        # No DDQN (original) td_error
        td_error = r.squeeze_(0) + torch.mul(torch.max(q_prime, 1)[0], self.gamma).unsqueeze(1) - torch.gather(q, 1, a)
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
