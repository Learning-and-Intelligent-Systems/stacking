# Trajectory Information ROS message

string type

# Joint path data
stacking_ros/RobotConfig[] joint_path
float64 joint_path_speed

# Grasp data
geometry_msgs/Pose grasp

# Move to/from touch data
bool use_wrist_camera
stacking_ros/RobotConfig q_start
stacking_ros/RobotConfig q_end