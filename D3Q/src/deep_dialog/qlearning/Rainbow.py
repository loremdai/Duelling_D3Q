import math
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

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
    def __init__(self, input_size, hidden_size, output_size, atom_size, z):
        super(Network, self).__init__()

        # Categorical DQN init
        self.output_size = output_size
        self.atom_size = atom_size
        self.z = z

        # common feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU())

        # value layer
        self.value_hid_layer = NoisyLinear(hidden_size, hidden_size)
        self.value_layer = NoisyLinear(hidden_size, atom_size)

        # advantage layer
        self.advantage_hid_layer = NoisyLinear(hidden_size, hidden_size)
        self.advantage_layer = NoisyLinear(hidden_size, output_size * atom_size)

    def forward(self, x):
        p = self.compute_prob(x)
        q = torch.sum(p * self.z, dim=2)
        return q  # size: (16,31)

    def compute_prob(self, inputs):
        feature = self.feature_layer(inputs)
        value = self.value_layer(F.relu(self.value_hid_layer(feature))).view(-1, 1, self.atom_size)
        # size: (16,1,atom_size)
        advantage = self.advantage_layer(F.relu(self.advantage_hid_layer(feature))).view(-1, self.output_size,
                                                                                         self.atom_size)
        # size: (16,output_size,atom_size)
        q_atoms = value + advantage - torch.mean(advantage, dim=1, keepdim=True)

        prob = F.softmax(q_atoms, dim=-1)
        prob = prob.clamp(min=1e-3)  # 防止除数为0的情况
        return prob  # size: (16,31,atom_size=51)

    def reset_noise(self):
        """Reset all noisy layers."""
        self.value_hid_layer.reset_noise()
        self.value_layer.reset_noise()
        self.advantage_hid_layer.reset_noise()
        self.advantage_layer.reset_noise()


class Rainbow(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, v_min=0.0, v_max=200.0,
                 atom_size=51):  # (state_dimension, hidden_size, num_actions)
        super(Rainbow, self).__init__()

        # Categorical DQN init
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.z = torch.linspace(start=self.v_min, end=self.v_max, steps=self.atom_size).to(device)

        # model
        self.model = Network(input_size, hidden_size, output_size, atom_size, z=self.z).to(device)
        # target model
        self.target_model = Network(input_size, hidden_size, output_size, atom_size, z=self.z).to(device)
        # first sync
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        # hyper parameters
        self.gamma = 0.9
        self.reg_l2 = 1e-3
        self.max_norm = 10.0
        self.target_update_period = 100
        lr = 0.001

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    # 更新目标网络（将model的参数载入到target_model的参数）
    def update_fixed_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # 更新网络
    def update_network(self):
        # update target network
        self.target_model.load_state_dict(self.model.state_dict())
        # NoisyNet settings: (reset noise)
        self.model.reset_noise()
        self.target_model.reset_noise()

    def Variable(self, x):
        return Variable(x, requires_grad=False).to(device)

    # 在文件AgentDQN的train/train_iter函数中被调用
    def singleBatch(self, batch):
        self.optimizer.zero_grad()

        # each example in a batch: [s, a, r, s_prime, term]
        s = self.Variable(torch.FloatTensor(batch[0]))  # size: (16,213)
        a = self.Variable(torch.LongTensor(batch[1]))  # size: (16,1)
        r = self.Variable(torch.FloatTensor([batch[2]]))  # size: (1,16,1)
        s_prime = self.Variable(torch.FloatTensor(batch[3]))  # size: (16,213)

        # Compute Categorical DQN loss
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            next_action = self.model(s_prime).argmax(1)  # size: (16)   instrument Double DQN
            next_dist = self.target_model.compute_prob(s_prime)  # size: (16,31,51)
            next_dist = next_dist[range(16), next_action]  # size: (16,51)     p{x_(t+1),a^*}

            t_z = torch.clamp((r.squeeze_(0) + self.gamma * self.z), min=self.v_min, max=self.v_max)  # size: (16,51)
            b = (t_z - self.v_min) / delta_z  # size: (16,51)
            l = b.floor().long()  # size: (16,51)
            u = b.ceil().long()  # size: (16,51)

            offset = (torch.linspace(start=0, end=((16 - 1) * self.atom_size), steps=16).long()
                      .unsqueeze(1).expand(16, self.atom_size).to(device))  # size: (16,51)

            proj_dist = torch.zeros(next_dist.size(), device=device)  # size: (16,51)

            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )  # m_l  size: (16,51)

            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )  # m_u  size: (16,51)

        dist = self.model.compute_prob(s)   # size: (16,31,51)
        log_p = torch.log(dist[range(16), a])     # size: (16,51)
        loss = -(proj_dist * log_p).sum(1).mean()   # cross-entropy term of the KL divergence

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
