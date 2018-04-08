import torch
from ray.rllib.optimizers.sample_batch import SampleBatch
from torch.autograd import Variable
from ray.rllib.optimizers.policy_evaluator import PolicyEvaluator
from agent import MLPPolicy
import numpy as np
import torch.optim as optim

# TODO: Make a minimum example


def to_variable(np_var, use_cuda=True):
    var = Variable(torch.from_numpy(np_var)).float()
    if use_cuda:
        return var.cuda()
    else:
        return var


def calculate_returns(rewards, dones, last_value, gamma=0.99):
    returns = np.zeros(rewards.shape[0] + 1)
    returns[-1] = last_value
    dones = 1 - dones
    for i in reversed(range(rewards.shape[0])):
        returns[i] = gamma * returns[i+1] * dones[i] + rewards[i]
    return returns


class TorchPPOEvaluator(PolicyEvaluator):
    """
        Policy Evaluator is the class that does the following:
        1. sample: sample the rollouts in the environment provided
        2. compute_gradients/apply_gradients: to update the model

        The evaluator is being used by PolicyOptimizer object. A PolicyOptimizer
        object may contain multiple evaluator to sample rollouts and update gradients.

        The exact behaviour of how to sample/update is implemented differently in PolicyOptimizer,
        e.g. AsyncOptimizer does the asynchronous gradient-based optimization (A3C).
    """
    def __init__(self, config, env_creator):
        self.config = config
        self.env = env_creator(self.config['env_config'])
        self.action_dim = self.env.action_space.n
        self.obs_dim = self.env.observation_space.shape
        self.net = MLPPolicy(self.obs_dim, self.action_dim)

        if self.config['cuda']:
            self.net.cuda()

        self.net.train()
        # initialize gym
        self.obs = self.env.reset()

        # initialize torch optimizer
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.config['lr'])

    def sample(self):
        """sample rollouts from the environment, being called in step in PolicyOptimizer"""
        observations, rewards, actions, logprobs, dones, values = [], [], [], [], [], []
        done = False
        for _ in range(self.config['min_steps_per_rollout']):
            value, action, logprob, mean = self.net.forward(to_variable(self.obs[np.newaxis]))
            action = action.cpu().data.numpy() if self.config['cuda'] else action.data.numpy()
            next_obs, reward, done, _ = self.env.step(action)

            if self.config['cuda']:
                # torch has an additional dimension for batch size, so we need to select that batch
                value, logprob, mean = value.data.cpu().numpy()[0], logprob.data.cpu().numpy()[0], \
                                       mean.data.cpu().numpy()[0]
            else:
                value, logprob, mean = value.data.numpy()[0], logprob.data.numpy()[0], \
                                       mean.data.numpy()[0]

            observations.append(self.obs)
            actions.append(action)
            rewards.append(reward)
            logprobs.append(logprob)
            values.append(value)
            dones.append(done)

            self.obs = next_obs

            if done:
                # reset the environment
                self.obs = self.env.reset()

        if done:
            last_value = 0.0
        else:
            # bootstrap, we only need the last value to do this
            value, action, logprob, mean = self.net.forward(to_variable(self.obs[np.newaxis]))

            if self.config['cuda']:
                # torch has an additional dimension for batch size, so we need to select that batch
                value, = value.data.cpu().numpy()[0]
            else:
                value, = value.data.numpy()[0]
            last_value = value

        # same as old/ppo.py
        observations = np.asarray(observations)
        rewards = np.asarray(rewards)
        logprobs = np.asarray(logprobs)
        dones = np.asarray(dones)
        values = np.asarray(values)
        actions = np.asarray(actions)
        returns = calculate_returns(rewards, dones, last_value, self.config['gamma'])
        return SampleBatch(
            {'observations': observations, 'rewards': rewards, 'logprobs': logprobs, 'dones': dones,
             'values': values, 'actions': actions, 'returns': returns
             }

        )

    def compute_gradients(self, samples):
        """Not used in this evaluator, all done in compute apply"""
        return None, {}

    def compute_apply(self, samples):
        # TODO do the PPO update
        pass

    def apply_gradients(self, grads):
        """Not used in this evaluator, all done in compute apply"""
        pass

    def set_weights(self, weights):
        pass

    def get_weights(self):
        pass