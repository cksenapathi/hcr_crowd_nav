#!/usr/bin/env python3

# Outside Library Imports
import numpy as np
import gym
from gym import spaces
import os
import sys

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
        self.img_size = (3,720,1280) # Intel RealSense D435 image size
        self.goal_pos = 50*np.random.rand(2,) - 25
        self.step_loss = 1
        self.action_space = spaces.Box(low=-np.inf, high=np.inf,shape=(7,
            self.k)) # Actions are going to be 3d position, quat i.e.
                     # 7 by k time  steps, currently choosing k to be
                     # 5; hyperparameter to pick
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = {
        'img':spaces.Box(low=0, high=255, shape=self.img_size, dtype=np.uint8),
        'goal_pos': spaces.Box(low=-25, high=25, shape=(2,1)),
        'robot_state': spaces.Box(low=-np.inf, high=np.inf, shape=(13,1))
        } # Contains image data from onboard camera, global position of goal, global pose of robot


    def step(self, action):
        # update state, base link linear and angular position and velocity
        # Add collision subscriber, check contacts on any links other than
        # calf
        # if collision checks if anything collided with anything else, which
        # includes base link and falls over
        #
        img, goal_pos, robot_state, reward, done, dist = self.interface.step(action,
            self.dt*np.arange(self.k))
        if done:
            assert(dist is None)
            obs['img'] = img
            obs['goal_pos'] = goal_pos
            obs['robot_state'] = robot_state
            return obs, reward, done, info
        else:
            dist_reward = 7*np.exp(0.5 - dist)
            reward += 10 * (dist_reward - self.last_dist_reward)
            # Rewards progress towards goal more than distance
            # torque reward, position reward,

            # calculate torque commands based on action
            # self.robot.
            # Get robot state i.e. position, velocity, orientation,
        # Calculate reward
        # if collision with agent reward is -1000
        # else if reached goal, reward is 1/dist to goal, take into account smoothness of state change, way to quantify how achievable a calculated trajectory is based on robot dynamics --> should be taken care of in the step function, plus a small constant negative reward to minimize wasting time
        info = {}
        return obs, reward, done, info


    def reset(self):
        img, goal_pos, robot_state = self.interface.reset()
        self.done = False
        self.reward = 0

        obs['img'] = img
        obs['goal_pos'] = goal_pos
        obs['robot_state'] = robot_state

        return obs


    def render(self, mode='human'):
        pass # Stable Baselines doesn't have to worry about any rendering


    def close (self):
        pass


if __name__ == '__main__':
    env = CrowdEnv()
    env.reset()
