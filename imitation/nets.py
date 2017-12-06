import torch
import torch.nn as nn
from torch.nn import functional as F
from model.utils import log_normal_density


class Discriminator(nn.Module):
    """D(x, a) """
    def __init__(self, obs_space, action_space):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(obs_space + action_space, 64)

        self.fc2 = nn.Linear(64, 128)
        self.logits = nn.Linear(128, 1)

    def forward(self, observation, action):
        x = torch.cat([observation, action], dim=1)
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        x = F.tanh(x)
        logits = self.logits(x)
        return F.sigmoid(logits).view(-1), logits.view(-1)


class Policy(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(Policy, self).__init__()
        self.act_fc1 = nn.Linear(obs_dim, 64)
        self.act_fc2 = nn.Linear(64, 128)
        self.mu = nn.Linear(128, action_dim)
        self.mu.weight.data.mul_(0.1)
        # torch.log(std)
        self.logstd = nn.Parameter(torch.zeros(action_dim))

        # value network
        self.value_fc1 = nn.Linear(obs_dim, 64)
        self.value_fc2 = nn.Linear(64, 128)
        self.value_fc3 = nn.Linear(128, 1)
        self.value_fc3.weight.data.mul(0.1)

    def forward(self, x):
        action = self.act_fc1(x)
        action = F.tanh(action)
        action = self.act_fc2(action)
        action = F.tanh(action)
        mean = self.mu(action)

        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        action = torch.normal(mean, std)
        logprob = log_normal_density(x=action, mean=mean, std=std, log_std=logstd)

        value = self.value_fc1(x)
        value = self.value_fc2(value)
        value = self.value_fc3(value)

        return value, action, logprob, mean

    def evaluate_actions(self, x, action):
        value, _, _, mean = self.forward(x)
        logstd = self.logstd
        std = torch.exp(logstd)
        logprob = log_normal_density(action, mean, log_std=logstd, std=std)
        # E[-log(\pi(a|s))]
        casual_entropy = -logprob.sum(-1).mean()
        return value, logprob, casual_entropy


# if __name__ == '__main__':
#     from torch.autograd import Variable
#     import numpy as np
#     discriminator = Discriminator(2, 2).cuda()
#     obs = Variable(torch.FloatTensor(np.random.randn(2, 2))).cuda()
#     action = Variable(torch.randn(2, 2)).cuda()
#     prob, logits = discriminator(obs, action)
#     print(prob)
#     print(logits)
#
#     c1 = nn.BCEWithLogitsLoss()
#     c2 = nn.BCELoss()
#     print(c2(prob.view(-1), Variable(torch.zeros(2)).cuda()))
#     print(c1(logits.view(-1), Variable(torch.zeros(2)).cuda()))
