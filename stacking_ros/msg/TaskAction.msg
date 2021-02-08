# Task Action ROS Message Definition
# This message has sufficient fields to capture the parameters of all
# actions in the PDDL domain (e.g. move, move_holding, pick, place)

string type
string obj1
string obj2
geometry_msgs/Pose grasp
geometry_msgs/Pose pose1
geometry_msgs/Pose pose2
stacking_ros/RobotConfig q1
stacking_ros/RobotConfig q2
stacking_ros/TrajInfo[] trajectories