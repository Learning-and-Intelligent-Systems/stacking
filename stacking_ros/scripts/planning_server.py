#!/usr/bin/env python3.7
"""
ROS Action Server for PDDLStream Planning
"""

import time
import rospy
import actionlib
import pb_robot
import numpy as np
from stacking_ros.msg import TaskPlanAction, TaskPlanResult, TaskAction
from tamp.misc import get_pddl_block_lookup, get_pddlstream_info, setup_panda_world
from tamp.ros_utils import task_plan_to_ros
from pddlstream.algorithms.focused import solve_focused
from pddlstream.utils import INF


class PlanningServer():
    def __init__(self, blocks, block_init_xy_poses=None, use_platform=False):

        # Start up a robot simulation for planning
        self._planning_client_id = pb_robot.utils.connect(use_gui=False)
        self.plan()
        self.robot = pb_robot.panda.Panda()
        self.robot.arm.hand.Open()
        self.saved_world = pb_robot.utils.WorldSaver()

        # Initialize the world
        self.pddl_blocks, self.platform_table, self.platform_leg, self.table, self.frame, self.wall = \
            setup_panda_world(self.robot, blocks, block_init_xy_poses, use_platform=use_platform) 
        self.fixed = [self.platform_table, self.platform_leg, self.table, self.frame, self.wall]
        self.pddl_block_lookup = get_pddl_block_lookup(blocks, self.pddl_blocks)

        # Create the ROS services
        rospy.init_node("planning_server")
        self.planning_service = actionlib.SimpleActionServer(
            "/get_plan", TaskPlanAction, 
            execute_cb=self.find_plan, auto_start=False)
        self.planning_service.start()
        print("Planning server ready!")

        rospy.spin()


    def plan(self):
        self.state = 'plan'
        pb_robot.aabb.set_client(self._planning_client_id)
        pb_robot.body.set_client(self._planning_client_id)
        pb_robot.collisions.set_client(self._planning_client_id)
        pb_robot.geometry.set_client(self._planning_client_id)
        pb_robot.grasp.set_client(self._planning_client_id)
        pb_robot.joint.set_client(self._planning_client_id)
        pb_robot.link.set_client(self._planning_client_id)
        pb_robot.panda.set_client(self._planning_client_id)
        pb_robot.planning.set_client(self._planning_client_id)
        pb_robot.utils.set_client(self._planning_client_id)
        pb_robot.viz.set_client(self._planning_client_id)


    def get_initial_pddl_state(self, conf=None):
        """
        Get the PDDL representation of the world between experiments. This
        method assumes that all blocks are on the table. We will always "clean
        up" an experiment by moving blocks away from the platform after an
        experiment.
        """
        fixed = [self.table, self.platform_table, self.platform_leg, self.frame]
        if conf is None:
            conf = pb_robot.vobj.BodyConf(self.robot, self.robot.arm.GetJointValues())
        init = [('CanMove',),
                ('Conf', conf),
                ('AtConf', conf),
                ('HandEmpty',)]

        self.table_pose = pb_robot.vobj.BodyPose(self.table, self.table.get_base_link_pose())
        init += [('Pose', self.table, self.table_pose), ('AtPose', self.table, self.table_pose)]

        for body in self.pddl_blocks:
            print(type(body), body)
            pose = pb_robot.vobj.BodyPose(body, body.get_base_link_pose())
            init += [('Graspable', body),
                    ('Pose', body, pose),
                    ('AtPose', body, pose),
                    ('Block', body),
                    ('On', body, self.table),
                    ('Supported', body, pose, self.table, self.table_pose)]

        if not self.platform_table is None:
            self.platform_pose = pb_robot.vobj.BodyPose(self.platform_table, self.platform_table.get_base_link_pose())
            init += [('Pose', self.platform_table, self.platform_pose), ('AtPose', self.platform_table, self.platform_pose)]
            init += [('Block', self.platform_table)]
        init += [('Table', self.table)]
        return init

    
    def unpack_goal(self, ros_goal):
        """ 
        Convert TaskPlanGoal ROS message to PDDLStream compliant 
        initial conditions and goal specification
        """
        fixed_objs = [f for f in self.fixed]
        
        # Unpack block initial poses
        for elem in ros_goal.blocks:
            blk = self.pddl_block_lookup[elem.name]
            pos = [elem.pose.position.x, elem.pose.position.y, elem.pose.position.z]
            orn = [elem.pose.orientation.x, elem.pose.orientation.y, 
                   elem.pose.orientation.z, elem.pose.orientation.w]
            blk.set_base_link_pose((pos, orn))
            if elem.fixed:
                fixed_objs.append(blk)

        # Unpack initial robot configuration
        conf = pb_robot.vobj.BodyConf(self.robot, ros_goal.robot_config.angles)

        # Create the initial PDDL state
        pddl_init = self.get_initial_pddl_state(conf=conf)

        # Extend the initial state with information about the goal
        additional_init = []
        for elem in ros_goal.goal:
            if elem.type == "AtPose":
                pos = [elem.pose.position.x, elem.pose.position.y, elem.pose.position.z]
                orn = [elem.pose.orientation.x, elem.pose.orientation.y, 
                       elem.pose.orientation.z, elem.pose.orientation.w]
                goal_blk_pose = pb_robot.vobj.BodyPose(blk, (pos, orn))
                additional_init.extend([
                    ("Pose", blk, goal_blk_pose),
                    ("Supported", blk, goal_blk_pose, self.table, self.table_pose)
                ])
        pddl_init.extend(additional_init)

        # Finally unpack the goal specification
        pddl_goal = ["and",]
        for elem in ros_goal.goal:
            blk = self.pddl_block_lookup[elem.target_obj]
            if elem.type == "AtPose":
                pddl_goal.append((elem.type, blk, goal_blk_pose))
            elif elem.type == "On":
                try:
                    base = self.pddl_block_lookup[elem.base_obj]
                except:
                    base = self.table
                pddl_goal.append((elem.type, blk, base))
        
        return pddl_init, tuple(pddl_goal), fixed_objs



    def find_plan(self, ros_goal):
        """ Main PDDLStream planning function """
        print("Planning...")
        self.plan()
        self.robot.arm.hand.Open()

        init, goal, fixed_objs = self.unpack_goal(ros_goal)
        print(f"\nINITIAL CONDITIONS\n{init}\n")
        print(f"\nGOAL SPECIFICATION\n{goal}\n")

        # Get PDDLStream planning information
        self.pddl_info = get_pddlstream_info(self.robot,
                                             fixed_objs,
                                             self.pddl_blocks,
                                             add_slanted_grasps=False,
                                             approach_frame='global')

        # Run PDDLStream focused solver
        start = time.time()
        pddlstream_problem = tuple([*self.pddl_info, init, goal])
        plan, _, _ = solve_focused(pddlstream_problem,
                                success_cost=INF,
                                max_skeletons=2,
                                search_sample_ratio=1.,
                                max_time=INF)
        duration = time.time() - start
        self.saved_world.restore()
        print('Planning Complete: Time %f seconds' % duration)
        print(f"\nFINAL PLAN\n{plan}\n")

        # Package and return the result
        result = task_plan_to_ros(plan)
        print(result)
        self.planning_service.set_succeeded(result, text="Planning complete")


if __name__=="__main__":
    from block_utils import get_adversarial_blocks
    blocks = get_adversarial_blocks(num_blocks=4)
    s = PlanningServer(blocks)
