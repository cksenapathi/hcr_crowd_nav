#!/usr/bin/env python3

# Outside Library Imports
import numpy as np
import gym
from gym import spaces
import os
import sys

# Written Imports
from robot import Robot
from rvo import RVO

# ROS Imports
import rospy
from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SpawnModel, DeleteModel
from sensor_msgs.msg import Image
from unitree_legged_msgs.msg import MotorCmd

# class Environment:
#     def __init__(self, arg):
#         self.episodes = []
#         self.goal_pos = None
#         self.robot_pos = None
#         self.agent = RLAgent()
#
#     def reset():
#         self.episo

class CrowdEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CrowdEnv, self).__init__()
        # ROS Components
        self.rate = 20 # Hz
        self.dt = 1/self.rate
        self.reset_world = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        self.joint_controllers = []
        # self.camera_sub = rospy.Subscriber('real_sense/image_data/raw', Image,
        #     self._cam_sub_cb)
        rospy.init_node('env_node', anonymous=True)

        # Env Paramters
        self.last_dist_reward = 0
        self.done = False
        self.reward = 0
        self.k = 5
        self.img_size = (3,720,1280) # Intel RealSense D435 image size
        self.goal_pos = 50*np.random.rand(2,) - 25
        self.time_step_loss = 1

        # ORCA Parameters
        self.rvo_sim = RVO()
        self.orca_agent_pub = rospy.Publisher('gazebo/set_model_state',
            ModelState, queue_size=10)

        # Robot Definitions
        self.robot = Robot()
        joint_topics = ['a1_gazebo/'+name[:-5]+'controller/command' for name in
            self.robot.joint_names]
        self.joint_pubs = [rospy.Publisher(topic),MotorCmd, queue_size=10) for
            topic in joint_topics]
        self.curr_state = np.zeros((37,1))


        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7, self.k)) # Actions are going to be 3d position, quat i.e. 7 by k time  steps, currently choosing k to be 5; hyperparameter to pick
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = {
        'img':spaces.Box(low=0, high=255, shape=self.img_size, dtype=np.uint8),
        'goal_pos': spaces.Box(low=-25, high=25, shape=(2,1)),
        'robot_state': spaces.Box(low=-np.inf, high=np.inf, shape=(13,1))
        } # Contains image data from onboard camera, global position of goal, global pose of robot


    def step(self, action):
        # update state, base link linear and angular position and velocity
        # Add collision subscriber, check contacts on any links other than calf
        # if collision checks if anything collided with anything else, which includes
        #
        if collision:
            reward = -1000
            self.done = True
        else:
            u, cost = self.robot.calc_command(self.curr_state, action,
                self.dt*np.arange(self.k))
            for i, pub in enumerate(self.joint_pubs):
                cmd = MotorCmd()
                cmd.tau = u[i]
                cmd.position = 2.146e9
                cmd.velocity = 16000.
                cmd.Kp = 0
                cmd.Kd = 0
                pub.publish(cmd)

            # calculate torque commands based on action
            # self.robot.
            # Get robot state i.e. position, velocity, orientation,
        # Calculate reward
        # if collision with agent reward is -1000
        # else if reached goal, reward is 1/dist to goal, take into account smoothness of state change, way to quantify how achievable a calculated trajectory is based on robot dynamics --> should be taken care of in the step function, plus a small constant negative reward to minimize wasting time
        return observation, reward, done, info


    def reset(self):
        try:
            self.reset_world() # Resets time and model positions
        except Exception as e:
            print(e)
            exit(-1)
        self.agent_names = self.rvo_sim.agent_names
        self.rvo_sim = RVO()
        spawner = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
        # pose = Pose(Point(0,0,0), Quaternion(0,0,0,1))
        file = open(os.getcwd()+"/urdf/agent.urdf", 'r').read()
        model_states = self.rvo_sim.set_sim()
        for i in range(self.rvo_sim.num_agents):
            agent_num, agent_state = model_states[i]
            # Spawns in desired position
            spawner(str(agent_num), file, '/orca', agent_state.pose, "world")
            # Publishes desired velocity
            self.orca_agent_pub.publish(agent_state)
            rospy.sleep(self.dt)
            self.orca_agent_pub.publish(agent_state)

            # Add orca agent to rvo sim
            # add cylinder to gazebo
            # set initial state randomly
        # Reset RVO agents https://answers.ros.org/question/337065/what-is-the-correct-way-to-spawn-a-model-to-gazebo-using-a-python-script/
        # Reset quad state ros gazebo service set model state
        self.goal_pos = 50*np.random.rand(2,) - 25
        self.robot_state = None
        # Reset done to false
        self.done = False
        self.reward = 0

        obs = {'img':None, 'goal_pos':None, 'robot_state':None}
        obs['img'] = np.zeros(self.img_size)
        obs['goal_pos'] = self.goal_pos
        obs['robot_state'] = self.robot_state

        return obs  # reward, done, info can't be included


    def render(self, mode='human'):
        pass


    def close (self):
        pass


if __name__ == '__main__':
    env = CrowdEnv()
    env.reset()
