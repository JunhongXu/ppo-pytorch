from model.net import MLPPolicy
from model.ppo import generate_trajectory, ppo_update
import gym
from torch.optim import Adam
from torch.nn import MSELoss
import tensorboardX
import numpy as np
import torch
from model.utils import enjoy

if __name__ == '__main__':
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
    train = False

    env = gym.make(env_name)
    policy = MLPPolicy(env.observation_space.shape[0],
                       env.action_space.shape[0])
    policy.cuda()
    policy.eval()

    if train:
        optimizer = Adam(lr=lr, params=policy.parameters())
        mse = MSELoss()
        writer = tensorboardX.SummaryWriter(policy_path + '/log/')
        # start training
        for e in range(nepoch):
            if e % 100 == 0 and e != 0:
                is_render = True
            else:
                is_render = False
            # generate trajectories
            observations, actions, logprobs, returns, values, rewards = \
                generate_trajectory(env, policy, horizon, is_render=is_render)
            writer.add_scalar('ppo/mean_reward', np.mean(rewards),
                              global_step=e)
            memory = (observations, actions, logprobs, returns[:-1], values)

            # update using ppo
            ppo_update(policy, optimizer, mini_batch_size, memory, nupdates, e,
                       coeff_entropy=coeff_entropy)
            # save every epoch
            torch.save(policy.state_dict(), policy_path + '/policy.pth')
    else:
        policy.load_state_dict(torch.load(policy_path + '/policy.pth'))
        enjoy(policy, env, save_path=policy_path + '/temp',
              save_video=True, nepochs=5)
