from model.net import MLPPolicy
from model.ppo import generate_trajectory, ppo_update
from torch.optim import Adam
from torch.nn import MSELoss
import numpy as np
import torch
from torcs.gym_torcs import TorcsEnv
import os


def convert_obs(obs):
    """
        Convert torcs observations into a 35-dim vector.
    """
    speedX = obs.speedX
    speedY = obs.speedY
    speedZ = obs.speedZ
    focus = obs.focus
    z = obs.z
    trackPos = obs.trackPos
    track = obs.track
    rpm = obs.rpm
    angle = obs.angle
    wheelSpinVel = obs.wheelSpinVel
    obs = np.concatenate([[speedX], [speedY], [speedZ], [trackPos], focus,
                          track, [z], [rpm], [angle], wheelSpinVel])
    return obs


if __name__ == '__main__':
    # hyper-parameters
    coeff_entropy = 0.00001
    lr = 5e-4
    mini_batch_size = 64
    horizon = 1024
    nupdates = 10
    nepoch = 40000
    clip_value = 0.2
    train = True
    # initialize env
    env_name = 'torcs_game'
    env = TorcsEnv(throttle=True, brake=True, vision=True)

    policy = MLPPolicy(35, action_space=env.action_space.shape[0])
    policy.cuda()
    if os.path.exists('policy.pth'):
        policy.load_state_dict(torch.load('policy.pth'))
        print('Loading complete!')
    if train:
        optimizer = Adam(lr=lr, params=policy.parameters())
        mse = MSELoss()

        # start training
        for e in range(nepoch):
            # generate trajectories
            observations, actions, logprobs, returns, values, rewards = \
                generate_trajectory(env, policy, horizon, is_render=False,
                                    obs_fn=convert_obs, progress=True)
            print('Episode %s reward is %s' % (e, rewards.sum()))
            memory = (observations, actions, logprobs, returns[:-1], values)
            # update using ppo
            policy_loss, value_loss, dist_entropy =\
                ppo_update(
                    policy, optimizer, mini_batch_size, memory, nupdates,
                    coeff_entropy=coeff_entropy, clip_value=clip_value
                )
            print('\nEpisode: {}'.format(e))
            print('Total reward {}'.format(rewards.sum()))
            print('Entroy', dist_entropy)
            print('Policy loss', policy_loss)
            print('Value loss', value_loss)
            torch.save(policy.state_dict(), 'torcs/policy.pth')
