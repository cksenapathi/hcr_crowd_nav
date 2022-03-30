## hcr_crowd_nav
# TODO:
- ORCA implementation
- Add stereo camera
- Create Environment Class
  - Randomize robot starting position and goal position
  - Add ORCA agents to Environment class with different groups and respective goals
- Create CoM to Joint Space conversion
- Add terrain

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
