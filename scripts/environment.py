#!/usr/bin/env python3

# Outside Library Imports
import numpy as np
import gym
from gym import spaces
import os
import sys
import time
import torch

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO

# Written Imports
from env_interface import EnvironmentInterface


class CrowdEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CrowdEnv, self).__init__()
        # ROS Components
        self.rate = 20 # Hz
        self.dt = 1/self.rate
        self.num_agents = 3 # Desired Number of ORCA Agents

        self.interface = EnvironmentInterface(rate=self.rate,
                                              num_agents=self.num_agents)
        # Env Paramters
        self.last_dist_reward = 0
        self.done = False
        self.reward = 0
        self.k = 5
        self.img_size = (1, 800,1280) # Intel RealSense D435 image size
        self.goal_pos = 50*np.random.rand(2,) - 25
        self.step_loss = 10
        self.action_space = spaces.Box(low=-np.inf, high=np.inf,shape=(self.k*7,))
        # Actions are going to be 3d position, quat i.e.
        # 7 by k time  steps, currently choosing k to be
        # 5; hyperparameter to pick
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Dict({
        'image':spaces.Box(low=0, high=255, shape=self.img_size, dtype=np.uint8),
        'goal_pos': spaces.Box(low=-25, high=25, shape=(2,), dtype=np.float64),
        'robot_state': spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float64)
        }) # Contains image data from onboard camera, global position of goal, global pose of robot


    def step(self, action):
        # update state, base link linear and angular position and velocity
        # Add collision subscriber, check contacts on any links other than
        # calf
        # if collision checks if anything collided with anything else, which
        # includes base link and falls over
        #
        action = np.reshape(action, (self.k, -1))
        img, goal_pos, robot_state, reward, done, dist = self.interface.step(action,
            self.dt*np.arange(self.k))
        # img = np.array([img])
        # min = np.min(img)
        # max = np.max(img)
        #
        # range = max - min
        # if range == 0:
        #     if max == 0:
        #         img = img/1
        #     else:
        #         img = img/max
        # else:
        #     img = ((img - min)/(range)) * 255.0
        # img = np.round(img).astype(np.uint8)
        # print(f'env step {img.shape}')
        # print(goal_pos, goal_pos.shape)
        # print(f"reward for step {reward}")
        if done:
            assert(dist is None)
            obs = {
            'image': img,
            'goal_pos': goal_pos,
            'robot_state': robot_state
            }
            info = {}
            self.reward += reward
            self.done = done
            # obs = spaces.Dict(obs)
            return obs, reward, done, info
        else:
            dist_reward = 5*np.exp(0.5 - dist)
            reward += 10 * (dist_reward - self.last_dist_reward)
            self.last_dist_reward = dist_reward
            reward -= self.step_loss
            self.reward += reward
            self.done = done
            obs = {
            'image': img,
            'goal_pos': goal_pos,
            'robot_state': robot_state
            }
            # Rewards progress towards goal more than distance
            info = {}
            return obs, reward, done, info
            # torque reward, position reward,

            # calculate torque commands based on action
            # self.robot.
            # Get robot state i.e. position, velocity, orientation,
        # Calculate reward
        # if collision with agent reward is -1000
        # else if reached goal, reward is 1/dist to goal, take into account smoothness of state change, way to quantify how achievable a calculated trajectory is based on robot dynamics --> should be taken care of in the step function, plus a small constant negative reward to minimize wasting time


    def reset(self):
        print(f'Episode Reward: {self.reward}')
        img, goal_pos, robot_state = self.interface.reset()
        self.done = False
        self.reward = 0
        self.last_dist_reward = 0
        # img = np.array([img])
        # min = np.min(img)
        # max = np.max(img)
        # range = max - min
        # if range == 0:
        #     if max == 0:
        #         img = img/1
        #     else:
        #         img = img/max
        # else:
        #     img = ((img - min)/(range)) * 255.0
        # img = np.round(img).astype(np.uint8)
        # print(f'env step {img.shape}')

        # obs = {
        # 'image':spaces.Box(low=0, high=255, shape=self.img_size, dtype=np.uint8),
        # 'goal_pos': spaces.Box(low=-25, high=25, shape=(2,)),
        # 'robot_state': spaces.Box(low=-np.inf, high=np.inf, shape=(13,))
        # }
        obs = {
        'image': img,
        'goal_pos': goal_pos,
        'robot_state': robot_state
        }
        # print(f"obs shape {obs['goal_pos'].shape} {goal_pos.shape}")
        # print(f"obs shape {obs['image'].shape} {img.dtype}")
        # print(f"obs shape {obs['robot_state'].dtype} {robot_state.dtype}")

        # obs['image'] = img
        # obs['goal_pos'] = goal_pos
        # obs['robot_state'] = robot_state
        # print(obs)
        return obs


    def render(self, mode='human'):
        pass # Stable Baselines doesn't have to worry about any rendering


    def close (self):
        rospy.signal_shutdown('')
        pass


if __name__ == '__main__':
    torch.cuda.set_per_process_memory_fraction(0.8, 0)
    torch.cuda.empty_cache()
    models_dir = f"models/{int(time.time())}/"
    logdir = f"logs/{int(time.time())}/"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    env = CrowdEnv()
    check_env(env)
    model = PPO("MultiInputPolicy", env, verbose=2, n_steps=16, batch_size=16,
        tensorboard_log=logdir)

    TIMESTEPS = 100
    iters = 0
    while True:
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        print('log function called')
        model.save(f"{models_dir}/{TIMESTEPS*iters}")
