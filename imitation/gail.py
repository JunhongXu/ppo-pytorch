import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.optim import Adam
from model.ppo import ppo_update, calculate_returns
from imitation.nets import Policy
import pickle
import gym
from imitation.nets import Discriminator
import random


class GAIL(object):
    def __init__(self, env, discriminator, policy, d_lr, p_lr, entropy=0., exp_path='expert.pkl'):
        self.discriminator = discriminator
        self.policy = policy
        self.entropy = entropy
        self.env = env
        self.exp_trajs = []
        self.policy_trajs = []
        # we use a stable version of BCELoss. BCELoss takes probability BCEWithLogitsLoss takes
        # logits
        self.bce_loss = nn.BCEWithLogitsLoss()

        # optimizers
        self.d_optim = Adam(self.discriminator.parameters(), lr=d_lr)
        self.p_optim = Adam(self.policy.parameters(), lr=p_lr)

        with open(exp_path, 'rb') as f:
            self.exp_trajs = pickle.load(f)

    def run_policy(self, max_frames=2048, epoch=None):
        """
            run the policy nframes times in the environment
            and record (observations, rewards, actions, logprobs, dones, values)
        """
        nframes = 0
        observations, rewards, actions, logprobs, dones, values = [], [], [], [], [], []
        total_true_reward = 0
        while nframes < max_frames:
            done = False
            obs = self.env.reset()
            while not done:
                if epoch is not None and epoch % 300 == 0 and epoch != 0:
                    self.env.render()
                obs = Variable(torch.FloatTensor(obs[np.newaxis]), volatile=True).float().cuda()
                value, action, logprob, _ = self.policy(obs)

                # q(s, a) = log(D(s, pi(a|s)))
                reward, _ = self.discriminator(obs, action)
                print('\rProbability for this action is the policy %.4f, the reward is' % reward.data[0], -np.log(reward.data[0]), flush=True, end='')
                reward = -torch.log(reward)

                value, action, logprob, reward = value.cpu().data.numpy()[0, 0], action.cpu().data.numpy()[0], \
                                                 logprob.cpu().data.numpy()[0], reward.cpu().data.numpy()[0]

                next_obs, true_reward, done, _ = self.env.step(action)

                observations.append(obs.data.cpu().numpy()[0])
                dones.append(done)
                actions.append(action)
                rewards.append(reward)
                values.append(value)
                logprobs.append(logprob)
                nframes += 1
                obs = next_obs
                total_true_reward += true_reward

        if done:
            last_value = 0.0
        else:
            obs = Variable(torch.from_numpy(obs[np.newaxis])).float().cuda()
            value, action, logprob, mean = policy(obs)
            last_value = value.data[0][0]
        observations = np.asarray(observations)
        rewards = np.asarray(rewards)
        logprobs = np.asarray(logprobs)
        dones = np.asarray(dones)
        values = np.asarray(values)
        actions = np.asarray(actions)
        returns = calculate_returns(rewards, dones, last_value)
        print('\n', total_true_reward)
        self.policy_trajs = (observations, actions, logprobs, returns, values, rewards)

    def update_discriminator(self, batch_size=1500):
        """
            update the discriminator using GAN loss function: 0 is the expert, 1 is the policy
            50 state-action paris are sampled
        """
        # sample batch_size of expert trajectories! problem: samples are not the same!!!
        exp_obs, exp_action = random.sample(self.exp_trajs[0], batch_size), random.sample(self.exp_trajs[1], batch_size)
        policy_observations, policy_actions, _, _, _, _ = self.policy_trajs
        nsamples = len(exp_obs)

        exp_obs = Variable(torch.from_numpy(np.stack(exp_obs))).float().cuda()
        exp_action = Variable(torch.from_numpy(np.stack(exp_action))).float().cuda()
        policy_obs = Variable(torch.from_numpy(policy_observations)).float().cuda()
        policy_action = Variable(torch.from_numpy(policy_actions)).float().cuda()

        # train on expert dataset
        prob, logits = self.discriminator(exp_obs, exp_action)
        e_loss = self.bce_loss(logits, Variable(torch.zeros(nsamples)).cuda())

        # train on policy dataset
        prob, logits = self.discriminator(policy_obs, policy_action)
        p_loss = self.bce_loss(logits, Variable(torch.ones(policy_obs.size(0))).cuda())

        # backpropgate
        loss = e_loss + p_loss
        self.d_optim.zero_grad()
        loss.backward()
        self.d_optim.step()
        print('Discriminator loss is %.4f' % loss.data[0])

    def update_policy(self, batch_size=64, nupdates=3):
        """perform ppo updates"""
        observations, actions, logprobs, returns, values, rewards = self.policy_trajs
        memory = (observations, actions, logprobs, returns[:-1], values)
        ppo_update(self.policy, self.p_optim, batch_size=batch_size,
                   memory=memory, nupdates=nupdates, coeff_entropy=self.entropy)


if __name__ == '__main__':
    env = gym.make('MountainCarContinuous-v0')
    policy = Policy(env.observation_space.shape[0], env.action_space.shape[0])
    policy.cuda()
    discriminator = Discriminator(env.observation_space.shape[0], env.action_space.shape[0])
    discriminator.cuda()

    gail = GAIL(env, discriminator, policy, 1e-4, 5e-4, entropy=1e-4)
    for i in range(0, 10000):
        gail.run_policy(max_frames=5000, epoch=i)
        gail.update_discriminator()
        # gail.run_policy()
        gail.update_policy()
