import torch
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from net import MLPPolicy
import gym
from torch.optim import Adam
import torch.nn as nn


def calculate_returns(rewards, dones, last_value, gamma=0.99):
    returns = np.zeros(rewards.shape[0] + 1)
    returns[-1] = last_value
    dones = 1 - dones
    for i in reversed(range(rewards.shape[0])):
        returns[i] = gamma * returns[i+1] * dones[i] + rewards[i]
    return returns


def ppo_update(policy, optimizer, batch_size, memory, nupdates):
    obs, actions, logprobs, returns, values = memory
    advantages = returns - values
    advantages = (advantages - advantages.mean()) / advantages.std()
    nstep = obs.shape[0]//batch_size
    for update in range(nupdates):
        for step in range(nstep):
            index = np.random.randint(0, obs.shape[0], batch_size)
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
            surrogate2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * sampled_advs   # 0.2 for now
            policy_loss = -torch.min(surrogate1, surrogate2).mean()

            sampled_returns = sampled_returns.view(-1, 1)
            value_loss = F.mse_loss(new_value, sampled_returns)

            loss = policy_loss + value_loss - 0.002 * dist_entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    print('value loss', value_loss.data[0])
    print('policy loss', policy_loss.data[0])
    print('distribution entropy', dist_entropy.data[0])

def generate_trajectory(env, policy, max_step, is_render=False):
    """generate a single trajectory using policy return the experiences"""
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
        # print(action, value)
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
    return observations, actions, logprobs, returns, values


if __name__ == '__main__':
    """Somthing learned: need to update multiple times for ppo!!!! not just 1 time!!!!!!"""
    env = gym.make('BipedalWalkerHardcore-v2')
    policy = MLPPolicy(env.observation_space.shape[0], env.action_space.shape[0])
    policy.cuda()
    opt = Adam(policy.parameters(), lr=1e-4)
    mse = nn.MSELoss()
    for e in range(10000):
        if e % 100 == 0 and e != 0:
            is_render = True
        else:
            is_render = False
        observations, actions, logprobs, returns, values = generate_trajectory(env, policy, 2048, is_render=is_render)
        memory = (observations, actions, logprobs, returns[:-1], values)
        ppo_update(policy, opt, 64, memory, 5)
