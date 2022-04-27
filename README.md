## hcr_crowd_nav
# TODO:
- Utilizing Unitree ROS Simulation --> using A1 for this project
  - https://github.com/unitreerobotics/unitree_ros
  - Add stereo camera to Unitree A1
    - Stereo camera added, mess with link pose to make sure camera is facing forward
    - `ROS_NAMESPACE=a1_gazebo/stereo rosrun stereo_image_proc stereo_image_proc`
    - `rosrun image_view stereo_view stereo:=a1_gazebo/stereo image:=image_rect_color`
    - camera neeeds to be moved to the left half the distance of the camera distance
- ORCA implementation
  - Installing Python bindings for RVO2-3D from https://github.com/mtreml/Python-RVO2-3D
- Installing pytorch-gpu from source to work with ROS from https://github.com/pytorch/pytorch#from-source
- Create Environment Class
  - Randomize robot starting position and goal position
  - Add ORCA agents to Environment class with different groups and respective goals
- Create CoM to Joint Space conversion
- Add terrain

# Dependencies:
  - PyPnC,
  - Stable Baselines 3, installed with pip
  - PyTorch
  - Pinocchio, can be installed through ROS
  - Unitree_ROS
    - unitree_controller
    - a1_description
    - unitree_legged_msgs
    - unitree_legged_control
  - RVO2-3D

# Loop Ordering:
- Start environment
  - Random starting and goal position
  - ORCA agents with goal and initial state
  - Add robot as agent
  - Start time
- Take visual frames, robot position, and goal position as input
- Cost based on smoothness of trajectory, time taken to reach goal, smoothness of joint torque values
- Calculate trajectory based on system dynamics
- Calculate estimate position waypoints and change in waypoint positions over time
- Convert CoM Trajectory to joint Space
