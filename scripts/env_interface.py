#!/usr/bin/env python3

# Class to hold all publishers and subscribers so data is passed through here
# SB3 just interacts with CrowdEnv Class, and data is passed from this class to
#   that class -- robot and rvo also interact only with this class
# Serves as ROS Wrapper for CrowdEnv
import os
# import sys

import numpy as np
import rospy

from rvo import RVO
from robot import Robot

from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SpawnModel, DeleteModel, GetModelState
from sensor_msgs.msg import JointState
from unitree_legged_msgs.msg import MotorCmd
from stereo_msgs.msg import DisparityImage
from geometry_msgs.msg import WrenchStamped
from cv_bridge import CvBridge


class EnvironmentInterface:
    def __init__(self, rate=20, num_agents=3):
        rospy.init_node('env_node', anonymous=True)
        self.rate = rate
        self.dt = 1/self.rate
        self.rate = rospy.Rate(rate)

        self.rvo = RVO()
        self.orca_agent_pub = rospy.Publisher('gazebo/set_model_state',
            ModelState, queue_size=10)

        self.robot = Robot()
        joint_topics = ['a1_gazebo/'+name[:-5]+'controller/command' for name in
            self.robot.joint_names]
        self.joint_pubs = [rospy.Publisher(topic ,MotorCmd, queue_size=10) for
            topic in joint_topics]
        self.curr_state = np.zeros(37,)

        self.contact = 0
        contact_names = ['FL_hip', 'FL_thigh', 'FR_hip',
                         'FR_thigh', 'RL_hip', 'RL_thigh',
                         'RR_hip', 'RR_thigh', 'trunk']
        self.contact_subs = [rospy.Subscriber('/visual/'+name+'_contact/the_force',
            WrenchStamped, self.contact_cb) for name in contact_names]

        self.reset_world = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.delete_model = rospy.ServiceProxy('/gazebo/delete_model',
            DeleteModel)
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state',
            GetModelState)
        self.camera_sub = rospy.Subscriber('/a1_gazebo/stereo/disparity', DisparityImage,
            self.img_cb)
        self.bridge = CvBridge()
        self.joint_state_sub = rospy.Subscriber('/a1_gazebo/joint_states',
            JointState, self.joint_state_cb)
        self.img = np.zeros((1, 800, 1280), dtype=np.uint8)

        self.goal_pos = 50*np.random.rand(2,) - 25
        pass


    def joint_state_cb(self, msg):
        self.curr_state[7:19] = np.array([np.asarray(msg.position)])
        self.curr_state[25:37] = np.array([np.asarray(msg.velocity)])


    def img_cb(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg.image, desired_encoding="32FC1")
        min = np.min(img)
        max = np.max(img)
        range = max - min
        if range == 0:
            if max == 0:
                img = img/1
            else:
                img = img/max
        else:
            img = ((img - min)/(range)) * 255.0
        img = np.round(img).astype(np.uint8)
        self.img = np.array([img])
        # print(f'img shape {self.img.shape}')

    def contact_cb(self, msg):
        self.contact += np.abs(msg.wrench.force.x)
        self.contact += np.abs(msg.wrench.force.y)
        self.contact += np.abs(msg.wrench.force.z)


    def step(self, wp, time):
        # Check collision with contact subscribers
        collision = False
        collision = self.contact > 0
        self.rate.sleep()
        dist_from_goal = np.linalg.norm(self.curr_state[:2] - self.goal_pos)
        if collision:
            reward = -1000
            done = True
            self.contact = 0
            return (self.img, self.goal_pos, np.hstack((self.curr_state[:7],
                self.curr_state[19:25])), reward, done, None)
        elif dist_from_goal <= 0.5:
            reward = 1000
            done = True
            self.contact = 0
            return (self.img, self.goal_pos, np.hstack((self.curr_state[:7],
                self.curr_state[19:25])), reward, done, None)
        else:
            done = False
            model_states = self.rvo.step()
            # spawner = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
            try:
                for i in range(self.rvo.num_agents):
                    agent_num, agent_state = model_states[i]
                    # Spawns in desired position
                    # spawner(str(agent_num), file, '/orca', agent_state.pose, "world")
                    # Publishes desired velocity
                    self.orca_agent_pub.publish(agent_state)
                    rospy.sleep(self.dt)
                    self.orca_agent_pub.publish(agent_state)
            except Exception as e:
                rospy.logerr("Spawning and Setting ORCA Agent State Failed %s\nenv_int 95", e)
            trq_msgs, cost = self.robot.calc_command(self.curr_state, wp, time)
            if len(trq_msgs)==0:
                return (self.img, self.goal_pos, np.hstack((self.curr_state[:7],
                        self.curr_state[19:25])), -cost, True, None)
            assert len(trq_msgs) == len(self.joint_pubs)
            for msg, pub in zip(trq_msgs, self.joint_pubs):
                pub.publish(msg)
            self.rate.sleep()
            self.contact = 0
            return (self.img, self.goal_pos, np.hstack((self.curr_state[:7],
                self.curr_state[19:25])), -cost, done, dist_from_goal)
        pass


    def get_curr_pose(self):
        try:
            resp = self.get_model_state('a1_gazebo', 'ground_plane')
        except Exception as e:
            rospy.logerr("Getting Model State Failed %s", e)
            exit(-1)

        self.curr_state[0:3] = np.asarray([resp.pose.position.x,
                                           resp.pose.position.y,
                                           resp.pose.position.z])

        self.curr_state[3:7] = np.asarray([resp.pose.orientation.x,
                                           resp.pose.orientation.y,
                                           resp.pose.orientation.z,
                                           resp.pose.orientation.w])

        self.curr_state[19:22] = np.asarray([resp.twist.linear.x,
                                            resp.twist.linear.y,
                                            resp.twist.linear.z])

        self.curr_state[22:25] = np.asarray([resp.twist.angular.x,
                                             resp.twist.angular.y,
                                             resp.twist.angular.z])


    def reset(self):
        self.goal_pos = 50*np.random.rand(2,) - 25
        agent_names = self.rvo.agent_names
        try:
            for name in agent_names:
                # rospy.logwarn("deleting model name: %s", name)
                resp = self.delete_model(name)
                # rospy.loginfo("deleted model %s", name)
        except rospy.ServiceException as e:
            rospy.logerr('Deleting Previous Models Failed %s', e)
            exit(-1)
        try:
            self.reset_world()
        except rospy.ServiceException as e:
            rospy.logerr('Resetting Simulation Failed %s', e)

        self.rvo = RVO()
        spawner = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
        agent_file = open(os.getcwd()+"/urdf/agent.urdf", 'r').read()
        agent_states = self.rvo.set_sim()
        try:
            for i in range(self.rvo.num_agents):
                agent_num, agent_state = agent_states[i]
                # Spawns in desired position
                spawner(str(agent_num), agent_file, '/orca', agent_state.pose, "world")
                # Publishes desired velocity
                self.orca_agent_pub.publish(agent_state)
                rospy.sleep(self.dt)
                self.orca_agent_pub.publish(agent_state)
        except Exception as e:
            rospy.logerr("Spawning and Setting ORCA Agent State Failed %s\nenv_int 157", e)

        start = rospy.get_rostime()
        while(rospy.get_rostime() - start <= rospy.Duration(1)):
            self.rate.sleep()
            # Allows a couple of iterations for pubs to get new data
            # from gazebo
        self.contact = 0
        self.get_curr_pose()

        return (self.img, self.goal_pos, np.hstack((self.curr_state[:7],
            self.curr_state[19:25])))
