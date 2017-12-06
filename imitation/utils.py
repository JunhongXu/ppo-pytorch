from torch.autograd import Variable
import torch
import numpy as np
import gym
from model.net import MLPPolicy
import pickle


def expert_trajectories(env, expert, num_frames=5000):
    """Generate expert trajectories using the trained PPO agent"""
    nframes = 0
    exp_actions = []
    exp_obs = []
    stop = False
    while True:
        done = False
        obs = env.reset()
        rewards = 0
        while not done:
            # env.render()
            exp_obs.append(obs)
            obs = Variable(torch.FloatTensor(obs[np.newaxis]), volatile=True).float().cuda()
            _, action, _, _ = expert(obs)
            action = action.data.cpu().numpy()[0]
            exp_actions.append(action)
            obs, reward, done, _ = env.step(action)
            nframes += 1
            rewards += reward
            if nframes == num_frames:
                stop = True
                break
        if stop:
            break
    expert_traj = (exp_obs, exp_actions)
    # because we are using low-dim states, we can pickle them
    with open('expert.pkl', 'wb') as f:
        pickle.dump(expert_traj, f)


if __name__ == '__main__':
    # env = gym.make('MountainCarContinuous-v0')
    # policy = MLPPolicy(env.observation_space.shape[0],
    #                    env.action_space.shape[0])
    # policy.cuda()
    # policy.load_state_dict(torch.load('policy.pth'))
    # expert_trajectories(env, policy)
    with open('expert.pkl', 'rb') as f:
        data = pickle.load(f)
    print(np.stack(data[0]))
