#!/usr/bin/env python3

import rvo2 as rvo
import numpy as np
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose, Point, Quaternion, Vector3, Twist


class RVO():

    def __init__(self, num_agents=3, speed=1.5, rad=.1, dist=1.5, k=5, dt=1/20):
        self.num_agents = num_agents
        self.speed = speed
        self.rad = rad
        self.dist = dist
        self.k = k
        self.dt = dt
        self.rvo_sim = rvo.PyRVOSimulator(self.dt, 1.5, 2, self.k*self.dt, self.k*self.dt,
            self.rad, self.speed, (0,0))
        self.agent_nums = np.zeros((self.num_agents, 1))
        self.orca_pos = np.zeros((self.num_agents, 2))
        self.orca_vel = np.zeros((self.num_agents, 2))
        self.agent_names = []

    def set_sim(self):
        model_states = []
        self.orca_pos = 30*np.random.rand(*self.orca_pos.shape) - 15
        norm_pos = np.divide(self.orca_pos, np.linalg.norm(self.orca_pos, axis=1)[:, np.newaxis])
        self.orca_vel = -self.speed * norm_pos # Moves all agents back towards center
        print(f'rvo file set_sim looking at orca vel {self.orca_vel}')
        for i in range(self.num_agents):
            self.agent_nums[i] = self.rvo_sim.addAgent(tuple(self.orca_pos[i,:]),
                velocity=tuple(self.orca_vel[i,:]))
            self.agent_names.append(str(self.agent_nums[i]))
            state = ModelState()
            state.model_name = str(self.agent_nums[i])
            state.pose = Pose(Point(self.orca_pos[i,0], self.orca_pos[i,1], 0),
                            Quaternion(0,0,0,1))
            state.twist = Twist(Vector3(self.orca_vel[i,0], self.orca_vel[i,1], 0),
                            Vector3(0, 0, 0))
            state.reference_frame = "world"
            model_states.append((self.agent_nums[i], state))
        return model_states

    def step(self):
        model_states = []
        self.rvo_sim.doStep()
        for i, num in enumerate(self.agent_nums):
            self.orca_pos[i,:] = self.rvo_sim.getAgentPosition(num)
            self.orca_vel[i,:] = self.rvo_sim.getAgentVelocity(num)
            state = ModelState()
            state.model_name = str(self.agent_nums[i])
            state.pose = Pose(Point(self.orca_pos[i,0], self.orca_pos[i,1], 0),
                            Quaternion(0,0,0,1))
            state.twist = Twist(Vector3(self.orca_vel[i,0], self.orca_vel[i,1], 0),
                            Vector3(0, 0, 0))
            state.reference_frame = "world"
            model_states.append((self.agent_names[i], state))
        return model_states
        # get new position and new velocity
        # publish new
