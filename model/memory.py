import numpy as np
import torch


class Rollouts(object):
    def __init__(self, obs_space, action_space, max_len):
        self.observations = np.empty(max_len, obs_space).astype(np.float32)
        self.actions = np.empty(max_len, action_space).astype(np.float32)
        self.logprobs = np.empty(max_len, 1).astype(np.float32)
        self.rewards = np.empty(max_len).astype(np.float32)
        self.returns = np.empty(max_len).astype(np.float32)
        self.values = np.empty(max_len).astype(np.float32)
        self.index = 0

    def sample(self, batch_size):
        if self.index > batch_size:
            samples = np.arange(0, self.index)
        else:
            samples = np.random.randint(0, self.index, batch_size)
        obs = torch.from_numpy(self.observations[samples])
        actions = torch.from_numpy(self.actions[samples])
        logprobs = torch.from_numpy(self.logprobs[samples])
        rewards = torch.from_numpy(self.rewards[samples])
        returns = torch.from_numpy(self.returns[samples])
        values = torch.from_numpy(self.values[samples])
        return obs, actions, logprobs, rewards, returns, values

    def insert(self, observation, action, logprob, reward):
        """
            insert one sample trajectory into the memory
        """

    def compute_returns(self):
        pass
