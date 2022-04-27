#!/usr/bin/env python3

# Class to hold all publishers and subscribers so data is passed through here
# SB3 just interacts with CrowdEnv Class, and data is passed from this class to
#   that class -- robot and rvo also interact only with this class
# Serves as ROS Wrapper for CrowdEnv
import numpy as np
import rospy

from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelState, GetModelState
from gazebo_msgs.srv import SpawnModel, DeleteModel
from sensor_msgs.msg import JointState
from unitree_legged_msgs.msg import MotorCmd
from stereo_msgs.msg import DisparityImage


class EnvironmentInterface:
    def __init__(self, rate=20, num_agents=3):
        self.dt = 1/self.rate
        self.rate = rospy.Rate(rate)

        self.rvo = RVO()
        self.orca_agent_pub = rospy.Publisher('gazebo/set_model_state',
            ModelState, queue_size=10)

        self.robot = Robot()
        joint_topics = ['a1_gazebo/'+name[:-5]+'controller/command' for name in
            self.robot.joint_names]
        self.joint_pubs = [rospy.Publisher(topic),MotorCmd, queue_size=10) for
            topic in joint_topics]
        self.curr_state = np.zeros((37,1))

        self.reset_world = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.delete_model = rospy.ServiceProxy('/gazebo/delete_model',
            DeleteModel)
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state',
            GetModelState)
        self.camera_sub = rospy.Subscriber('real_sense/image_data/raw', Image,
            self._cam_sub_cb)
        self.joint_state_sub = rospy.Subscriber('/a1_gazebo/joint_states',
            JointState, self.joint_state_cb)

        self.goal_pos = 50*np.random.rand(2,1) - 25
        pass


    def joint_state_cb(self, msg):
        self.curr_state[13:25] = np.asarray(msg.position)
        self.curr_state[25:37] = np.asarray(msg.velocity)


    def img_cb(self, msg):
        self.img = np.asarray(msg.data)


    def step(self, wp, time):
        # Check collision with contact subscribers
        self.rate.sleep()
        dist_from_goal = np.linalg.norm(self.curr_state[:2] - self.goal_pos)
        if collision:
            reward = -1000
            done = True
            return (self.img, self.goal_pos, self.curr_state[:13], reward, done, None)
        elif dist_from_goal <= 0.5:
            reward = 1000
            done = True
            return return (self.img, self.goal_pos, self.curr_state[:13], reward, done, None)
        else:
            done = False
            trq_msgs, cost = self.robot.calc_command(self.curr_state, wp, time)
            assert len(trq_msgs) == len(self.joint_pubs)
            for msg, pub in zip(trq_msgs, self.joint_pubs):
                pub.publish(msg)
            self.rate.sleep()
            return (self.img, self.goal_pos, self.curr_state[:13], cost, done, dist_from_goal)
        pass


    def get_curr_pose(self):
        try:
            resp = self.get_model_state('a1_gazebo', 'ground_plane')
        except Exception as e:
            rospy.logerr("Getting Model State Failed %s", e)

        self.curr_state[0:3] = np.asarray(resp.pose.position)
        self.curr_state[3:7] = np.asarray(resp.pose.orientation)
        self.curr_state[7:10] = np.asarray(resp.twist.linear)
        self.curr_state[10:13] = np.asarray(resp.twist.angular)


    def reset(self):
        try:
            for name in self.rvo_sim.agent_names:
                resp = self.delete_model(name)
        except rospy.ServiceException as e:
            rospy.logerr('Deleting Previous Models Failed %s', e)
            exit(-1)
        try:
            self.reset_world()
        except rospy.ServiceException as e:
            rospy.logerr('Resetting Simulation Failed %s', e)

        self.rvo_sim = RVO()
        spawner = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
        agent_file = open(os.getcwd()+"/urdf/agent.urdf", 'r').read()
        agent_states = self.rvo_sim.set_sim()
        try:
            for i in range(self.rvo_sim.num_agents):
                agent_num, agent_state = model_states[i]
                # Spawns in desired position
                spawner(str(agent_num), file, '/orca', agent_state.pose, "world")
                # Publishes desired velocity
                self.orca_agent_pub.publish(agent_state)
                rospy.sleep(self.dt)
                self.orca_agent_pub.publish(agent_state)
        except Exception as e:
            rospy.logerr("Spawning and Setting ORCA Agent State Failed %s", e)

        start = rospy.get_rostime()
        while(rospy.get_rostime() - start <= rospy.Duration(1)):
            self.rate.sleep()
            # Allows a couple of iterations for pubs to get new data
            # from gazebo

        self.get_curr_pose()

        return (self.img, self.goal_pos, self.curr_state[:13])
