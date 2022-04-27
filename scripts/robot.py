from pinocchio_robot_system import PinocchioRobotSystem
import numpy as np
import os
import sys
from cvxpy import *
from util import quaternion_multiply



class Robot:
    def __init__(self, model_path=None, dt = 0.05):
        if model_path is not None:
            self.robot = PinocchioRobotSystem(model_path)
            sys.path.append(os.getcwd())
        else:
            # Robot Components
            urdf_path = os.getcwd() + "/urdf/a1.urdf"
            self.robot = PinocchioRobotSystem(urdf_path,os.getcwd(), False,False)
            sys.path.append(os.getcwd())
            self.robot._joint_pos_limit[:-1,:] = self.robot._joint_pos_limit[1:,:]
            self.robot._joint_pos_limit[-1, :] = self.robot._joint_pos_limit[2,:]
            self.robot._config_robot(urdf_path, os.getcwd())
            self.joint_names = [name for name in self.robot._model.names if name.startswith(('F', 'R'))]
            self.robot.update_system(np.zeros(3,), np.array([0, 0, 0, 1]),
            np.zeros(3,), np.zeros(3,),
            np.zeros(3,), np.array([0, 0, 0, 1]),
            np.zeros(3,), np.zeros(3,),
            dict(zip(self.joint_names,np.zeros(self.robot._n_a))),
            dict(zip(self.joint_names,np.zeros(self.robot._n_a))))

            self.U = np.append(np.zeros((self.robot._n_a, 6)), np.eye(self.robot._n_a), axis=1)


    def update_system(self,
                      base_com_pos,
                      base_com_quat,
                      base_com_lin_vel,
                      base_com_ang_vel,
                      base_joint_pos,
                      base_joint_quat,
                      base_joint_lin_vel,
                      base_joint_ang_vel,
                      joint_pos,
                      joint_vel,
                      b_cent=False):

        self.robot.update_system(base_com_pos, base_com_quat, base_com_lin_vel,
        base_com_ang_vel, base_joint_pos, base_joint_quat, base_joint_lin_vel,
        base_joint_ang_vel, dict(zip(self.joint_names, joint_pos),
        dict(zip(self.joint_names, joint_vel))))
        # link "trunk" vs com stuff

    def contact_jacobian(self):
        return np.vstack((self.robot.get_link_jacobian('FL_foot')[:3,:],
                             self.robot.get_link_jacobian('FR_foot')[:3,:],
                             self.robot.get_link_jacobian('RL_foot')[:3,:],
                             self.robot.get_link_jacobian('RR_foot')[:3,:]))


    def discrete_dynamics(self, x_i, u_i, dt):
        x_dot = np.zeros_like(x_i) #x[:19] is q, x[19:] is q_dot
        x_dot[0:3] = x_i[19:22] # linear velocity
        # Requires scalar last quaternions of angular velocity
        x_dot[3:7] = .5 * quaternion_multiply(np.hstack(x_i[22:25], [0]), x_i[3:7])
        x_dot[7:19] = x_i[25:] # joint angular velocity

        A_inv = np.linalg.pinv(self.robot.get_mass_matrix())
        # Calculating second derivative based on whole body EOM
        x_dot[19:] = -A_inv*(self.robot.get_coriolis() + self.robot.get_gravity())\
            + A_inv*u_i[:self.robot._n_a] + A_inv*np.dot(contact_jacobian(), u_i[self.robot._n_a:])

        return x_i + x_dot * dt # Assuming linearized dynamics for short enough time step


    def calc_command(self, curr_state, wp: np.ndarray, time: np.ndarray):
        # Update Pinocchio system
        self.update_system(curr_state[0:3], curr_state[3:7], curr_state[7:10],
        curr_state[10:13], curr_state[0:3], curr_state[3:7], curr_state[7:10],
        curr_state[10:13], curr_state[13:25], curr_state[25:37])

        pos = wp[:3, :]
        quat = wp[3:7, :]

        start = time[0]
        dt = time[1] - time[0]
        end = time[-1]
        duration = end - start
        assert(len(wp) == len(time))
        vel = np.gradient(pos, time)
        # acc = np.gradient(vel, time)
        N = len(pos)
        n_a = self.robot._n_a


        x = Variable((self.robot._n_q+self.robot._n_q_dot, N+1))
        u = Variable((self.robot._n_a+4, N)) # joint torques plus contact forces
        x_0 = np.array([curr_state[0:7], curr_state[13:25], curr_state[7:13], curr_state[25:37]]).T

        Qu = np.eye(self._robot._n_a)
        hip_trq_cost = 1.2
        shoulder_trq_cost = 1.0
        knee_trq_cost = 1.0
        Qu[0::3] = hip_trq_cost
        Qu[1::3] = shoulder_trq_cost
        Qu[2::3] = knee_trq_cost

        Q_pos = 10*np.eye(3)
        Q_quat = 7*np.eye(4)
        Q_ang_vel = np.diag([3,3,1])
        constraints = [x[:,0] == x_0]
        objective = 0
        for i in range(N):           # position cost                  # orientation cost
            objective += quad_form(x[:3,i]-pos[:,i],Q_pos) + quad_form(x[3:7,i]-quat[:,i],Q_quat)\
                + quad_form(x[10:13,i], Q_ang_vel) + quad_form(u[:,i], Qu)

                    # angular velocity cost           # torque cost
            constraints += [x[:,i+1] == self.discrete_dynamics(x[:,i], u[:,i], dt)]
            constraints += [self.robot._joint_trq_limit[:,0] <= u[:n_a,i], u[:n_a,i] <= self.robot._joint_trq_limit[:,1]]
            constraints += [self.robot._joint_pos_limit[:,0] <= x[7:19,i], x[7:19,i] <= self.robot._joint_pos_limit[:,1]]
            constraints += [self.robot._joint_vel_limit[:,0] <= x[25:37,i], x[25:37,i] <= self.robot._joint_vel_limit[:,1]]
        # objective += Add terminal cost
        Problem(Minimize(objective), constraints).solve(solver=OSQP)
        print('calc command objective type and value {} {}'.format(type(objective), objective))
        return u[:,0].value, objective.value

            # s = (time[i] - start)/duration
            # - https://osqp.org/docs/examples/mpc.html#cvxpy
            # TODO: Check crowd_nav README for information on todo

        # terminal cost is based on error from achieved state and desired terminal state
            # cost based on euclidean distance between state vector
        # transition cost is based on torques, difference in desired acceleration vs actual acceleration, smoothness of joint torques, smoothness of reaction forces
        # cost is quadratized
        # constraints need to be quadratized
        # dynamics constraints are linearized by taylor polynomials
        # need to go back to notes for wb mpc
        # Task list
        # - follow com
        # - minimize angular acceleration --> stabilizes camera readings
            # Build PyTorch Model

if __name__ == '__main__':
    print(os.getcwd())

    robot = Robot()
    print(f'Initialized robot with a1 urdf')
    print(robot.joint_names)
    print('printing joint pos limit')
    print(robot.robot._joint_pos_limit)
    print('printing joint vel limit')
    print(robot.robot._joint_vel_limit)
    print('printing joint trq limit')
    print(robot.robot._joint_trq_limit)
    print('q')
    print(f'{robot.robot.get_q()}\n{robot.robot.get_q().shape}')
    print('q_dot')
    print(f'{robot.robot.get_q_dot()}\n{robot.robot.get_q_dot().shape}')
    print(robot.robot.get_com_pos())
    print()
    print(robot.robot.get_mass_matrix().shape)
    print(f'{robot.robot.get_coriolis()}  \n{robot.robot.get_coriolis().shape}')
    g = robot.robot.get_gravity()
    print(f'{g} \n{g.shape}')
    print('mass matrix inverse')
    print(np.linalg.inv(robot.robot.get_mass_matrix()))

    print(f"FR_foot link iso \n{robot.robot.get_link_jacobian('FR_foot')}\n {robot.robot.get_link_jacobian('FR_foot').shape}")

# Model will take in sequence of stereo image data
# Return Pose and Twist set points for x time steps
# Same number of time steps for ORCA agents and RL agent
# PyPnC to convert from waypoints to joint commands
# Trajectory following WB-MPC
