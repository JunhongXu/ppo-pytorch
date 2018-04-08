import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from ppo_single.model.utils import log_normal_density


class CNNPolicy(nn.Module):
    def __init__(self, obs_space, action_space):
        super(CNNPolicy, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1)

        self.linear1 = nn.Linear(32 * 7 * 7, 512)

        self.critic_linear = nn.Linear(512, 1)

        self.mu = nn.Linear(512, action_space)
        self.logstd = nn.Parameter(torch.zeros(action_space))

    def forward(self, x):
        """
            returns value estimation, action, log_action_prob
        """
        # action
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = x.view(-1, 32 * 7 * 7)
        x = self.linear1(x)
        x = F.relu(x)

        mean = self.mu(x)  # N, num_actions
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        action = torch.normal(mean, std)

        # value
        v = self.critic_linear(x)

        # action prob on log scale
        logprob = log_normal_density(action, mean, std=std, log_std=logstd)
        return v, action, logprob, mean

    def evaluate_actions(self, x, action):
        v, _, _, mean = self.forward(x)
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        # evaluate
        logprob = log_normal_density(action, mean, log_std=logstd, std=std)
        dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + logstd
        dist_entropy = dist_entropy.sum(-1).mean()
        return v, logprob, dist_entropy


class MLPPolicy(nn.Module):
    def __init__(self, obs_space, action_space):
        super(MLPPolicy, self).__init__()
        # action network
        self.act_fc1 = nn.Linear(obs_space, 64)
        self.act_fc2 = nn.Linear(64, 128)
        self.mu = nn.Linear(128, action_space)
        self.mu.weight.data.mul_(0.1)
        # torch.log(std)
        self.logstd = nn.Parameter(torch.zeros(action_space))

        # value network
        self.value_fc1 = nn.Linear(obs_space, 64)
        self.value_fc2 = nn.Linear(64, 128)
        self.value_fc3 = nn.Linear(128, 1)
        self.value_fc3.weight.data.mul(0.1)

    def forward(self, x):
        """
            returns value estimation, action, log_action_prob
        """
        # action
        act = self.act_fc1(x)
        act = F.tanh(act)
        act = self.act_fc2(act)
        act = F.tanh(act)
        mean = self.mu(act)  # N, num_actions
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        action = torch.normal(mean, std)

        # value
        v = self.value_fc1(x)
        v = F.tanh(v)
        v = self.value_fc2(v)
        v = F.tanh(v)
        v = self.value_fc3(v)

        # action prob on log scale
        logprob = log_normal_density(action, mean, std=std, log_std=logstd)
        return v, action, logprob, mean

    def evaluate_actions(self, x, action):
        v, _, _, mean = self.forward(x)
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        # evaluate
        logprob = log_normal_density(action, mean, log_std=logstd, std=std)
        dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + logstd
        dist_entropy = dist_entropy.sum(-1).mean()
        return v, logprob, dist_entropy


if __name__ == '__main__':
    from torch.autograd import Variable

    net = MLPPolicy(3, 2)

    observation = Variable(torch.randn(2, 3))
    v, action, logprob, mean = net.forward(observation)
    print(v)
    print(action)
    print(logprob)
    print(mean)
    print(net.logstd)
