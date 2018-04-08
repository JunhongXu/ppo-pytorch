import numpy as np
from gym.wrappers import Monitor
import torch
from torch.autograd import Variable


def log_normal_density(x, mean, log_std, std):
    """returns guassian density given x on log scale"""
    variance = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * variance) - 0.5 *\
        np.log(2 * np.pi) - log_std
    log_density = log_density.sum(dim=1, keepdim=True)
    return log_density


def enjoy(policy, env, save_path=None, save_video=False, obs_fn=None,
          nepochs=100):
    """
        Enjoy and flush your result using Monitor class.
    """
    if save_video:
        assert save_path is not None, 'A path to save videos must be provided!'
    policy.cuda()
    policy.eval()
    if save_video:
        env = Monitor(env, directory=save_path)

    for e in range(0, 100):
        done = False
        obs = env.reset()
        episode_rwd = 0
        while not done:
            env.render()
            if obs_fn is not None:
                obs = obs_fn(obs)
            obs = Variable(torch.from_numpy(obs[np.newaxis])).float().cuda()
            value, action, logprob, mean = policy(obs)
            action = action.data[0].cpu().numpy()
            obs, reward, done, _ = env.step(action)
            episode_rwd += reward
        print('Episode reward is', episode_rwd)
