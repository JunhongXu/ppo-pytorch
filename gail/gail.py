import numpy as np
from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.optim import Adam
from model.ppo import ppo_update
from gail.dataset import build_d_dataset





class GAIL(object):
    def __init__(self, env, discriminator, policy, d_lr, p_lr, entropy=0, exp_path='./trajectories'):
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

        # TODO: get path from expert path

    def run_policy(self, max_frames=2048):
        """
            run the policy nframes times in the environment
            and record (observations, rewards, actions, logprobs, dones, values)
        """
        nframes = 0
        observations, rewards, actions, logprobs, dones, values = [], [], [], [], [], []
        while nframes < max_frames:
            done = False
            obs = self.env.reset()
            while not done:
                obs = Variable(torch.FloatTensor(obs[np.newaxis]), volatile=True).float().cuda()
                value, action, logprob, _ = self.policy(obs)

                # q(s, a) = log(D(s, pi(a|s)))
                reward, _ = self.discriminator(obs, action)
                reward = torch.log(reward).data.cpu().numpy()[0]

                value, action, logprob, reward = value.data.cpu().numpy()[0, 0], action.data.cpu().numpy()[0], \
                                                 logprob.data.cpu().numpy()[0], reward.data.cpu().numpy()[0, 0]

                next_obs, _, done, _ = self.env.step(action)

                observations.append(obs.data.cpu().numpy()[0])
                dones.append(done)
                actions.append(action)
                rewards.append(reward)
                values.append(value)
                logprobs.append(logprob)
                nframes += 1
                obs = next_obs

        self.policy_trajs = (observations, rewards, actions, logprobs, dones, values)

    def update_discriminator(self):
        """
            update the discriminator using GAN loss function: 0 is the expert, 1 is the policy
        """
        d_dataset = build_d_dataset(self.exp_trajs, self.policy_trajs, nbatch=128)
        for index, (exp_obs, exp_action, policy_obs, policy_action) in enumerate(d_dataset):
            nsamples = exp_obs.size(0)
            exp_obs = Variable(exp_obs).cuda()
            exp_action = Variable(exp_action).cuda()
            policy_obs = Variable(policy_obs).cuda()
            policy_action = Variable(policy_action).cuda()

            # train on expert dataset
            prob, logits = self.discriminator(exp_obs, exp_action)
            e_loss = self.bce_loss(logits, Variable(torch.zeros(nsamples)).cuda())

            # train on policy dataset
            prob, logits = self.discriminator(policy_obs, policy_action)
            p_loss = self.bce_loss(logits, Variable(torch.ones(nsamples)).cuda())

            # backpropgate
            loss = e_loss + p_loss
            self.d_optim.zero_grad()
            loss.backward()
            self.d_optim.step()
            print('\rDiscriminator loss is %.4f' % loss.data[0], flush=True, end='')

    def update_policy(self, batch_size=64, nupdates=3):
        """perform ppo updates"""
        ppo_update(self.policy, self.p_optim, batch_size=batch_size,
                   memory=self.policy_trajs, nupdates=nupdates)


