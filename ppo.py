import torch
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from net import MLPPolicy
import gym
from torch.optim import Adam
import torch.nn as nn
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import os
import tensorboardX


env_name = 'BipedalWalker-v2'
coeff_entropy = 0.003
lr = 2e-4
mini_batch_size = 64
horizon = 2048
nupdates = 5
nepoch = 40000
clip_value = 0.2
policy_path = '{}/lr_{}/coeff_entropy_{}/batch_size_{}/horizon_{}'. \
    format(env_name, lr, coeff_entropy, mini_batch_size, horizon)
writer = tensorboardX.SummaryWriter(policy_path+'/log/')


def calculate_returns(rewards, dones, last_value, gamma=0.99):
    returns = np.zeros(rewards.shape[0] + 1)
    returns[-1] = last_value
    dones = 1 - dones
    for i in reversed(range(rewards.shape[0])):
        returns[i] = gamma * returns[i+1] * dones[i] + rewards[i]
    return returns


def ppo_update(policy, optimizer, batch_size, memory, nupdates, epoch):
    obs, actions, logprobs, returns, values = memory
    advantages = returns - values
    advantages = (advantages - advantages.mean()) / advantages.std()

    for update in range(nupdates):
        sampler = BatchSampler(SubsetRandomSampler(list(range(advantages.shape[0]))), batch_size=batch_size,
                               drop_last=False)
        nbatches = len(sampler)
        for i, index in enumerate(sampler):
            sampled_obs = Variable(torch.from_numpy(obs[index])).float().cuda()
            sampled_actions = Variable(torch.from_numpy(actions[index])).float().cuda()
            sampled_logprobs = Variable(torch.from_numpy(logprobs[index])).float().cuda()
            sampled_returns = Variable(torch.from_numpy(returns[index])).float().cuda()
            sampled_advs = Variable(torch.from_numpy(advantages[index])).float().cuda()

            new_value, new_logprob, dist_entropy = policy.evaluate_actions(sampled_obs, sampled_actions)

            sampled_logprobs = sampled_logprobs.view(-1, 1)
            ratio = torch.exp(new_logprob - sampled_logprobs)

            sampled_advs = sampled_advs.view(-1, 1)
            surrogate1 = ratio * sampled_advs
            surrogate2 = torch.clamp(ratio, 1 - clip_value, 1 + clip_value) * sampled_advs   # 0.2 for now
            policy_loss = -torch.min(surrogate1, surrogate2).mean()

            sampled_returns = sampled_returns.view(-1, 1)
            value_loss = F.mse_loss(new_value, sampled_returns)

            loss = policy_loss + value_loss - coeff_entropy * dist_entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar('ppo/value_loss', value_loss.data[0], global_step=(epoch + update) * nbatches + i)
            writer.add_scalar('ppo/policy_loss', policy_loss.data[0], global_step=(epoch + update) * nbatches + i)
            writer.add_scalar('ppo/entropy', dist_entropy.data[0], global_step=(epoch + update) * nbatches + i)



def generate_trajectory(env, policy, max_step, is_render=False):
    """generate a batch of examples using policy"""
    nstep = 0
    obs = env.reset()
    done = False
    observations, rewards, actions, logprobs, dones, values = [], [], [], [], [], []
    while not (nstep == max_step):
        if done:
            obs = env.reset()
        if is_render:
            env.render()
        obs = Variable(torch.from_numpy(obs[np.newaxis])).float().cuda()

        value, action, logprob, mean = policy(obs)
        value, action, logprob = value.data.cpu().numpy()[0], action.data.cpu().numpy()[0], \
                                 logprob.data.cpu().numpy()[0]

        next_obs, reward, done, _ = env.step(action)
        observations.append(obs.data.cpu().numpy()[0])
        rewards.append(reward)
        logprobs.append(logprob)
        dones.append(done)
        values.append(value[0])
        actions.append(action)

        obs = next_obs
        nstep += 1

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
    print(rewards.sum())
    return observations, actions, logprobs, returns, values, rewards


if __name__ == '__main__':
    """Somthing learned: need to update multiple times for ppo!!!! not just 1 time!!!!!!"""

    env = gym.make(env_name)
    policy = MLPPolicy(env.observation_space.shape[0], env.action_space.shape[0])
    policy.cuda()
    opt = Adam(policy.parameters(), lr=lr)
    mse = nn.MSELoss()

    if not os.path.exists(policy_path):
        os.makedirs(policy_path)

    for e in range(nepoch):
        if e % 100 == 0 and e != 0:
            is_render = True
        else:
            is_render = False
        observations, actions, logprobs, returns, values, rewards = generate_trajectory(env, policy, horizon, is_render=is_render)
        writer.add_scalar('ppo/mean_reward', np.mean(rewards), global_step=e)
        memory = (observations, actions, logprobs, returns[:-1], values)
        ppo_update(policy, opt, mini_batch_size, memory, nupdates, e)
        # save every epoch
        torch.save(policy.state_dict(), policy_path+'/policy.pth')
