import numpy as np


class Rollouts(object):
    def __init__(self, obs_space, action_space, max_len):
        self.observations = np.empty(max_len, obs_space)
        self.actions = np.empty(max_len, action_space)
        self.logprobs = np.empty(max_len, 1)
        self.rewards = np.empty(max_len)
        self.returns = np.empty(max_len)
        self.index = 0

    def sample(self, batch_size):
        if self.index > batch_size:
            samples = np.arange(0, self.index)
        else:
            samples = np.random.randint(0, self.index, batch_size)
            
    def insert(self, observation, action, logprob, reward):
        pass

    def compute_returns(self):
        pass
