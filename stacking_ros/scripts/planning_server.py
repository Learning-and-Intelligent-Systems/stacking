#!/usr/bin/env python3
"""
ROS Server for PDDLStream Planning

This server holds an action server for requesting a plan synchronously,
as well as a mode to continue planning towers and pass along plans from a
buffer upon request from a service client.
"""

import os
import dill
import time
import numpy
import rospy
import psutil
import argparse
import pb_robot
import pybullet as pb
from random import getrandbits
from multiprocessing import Process, Manager
from pddlstream.algorithms.focused import solve_focused
from pddlstream.utils import INF
from stacking_ros.msg import TaskPlanAction, TaskPlanResult, TaskAction
from stacking_ros.srv import (GetPlan, GetPlanResponse, 
    SetPlanningState, SetPlanningStateResponse)
from tamp.misc import (get_pddl_block_lookup, print_planning_problem, 
                       setup_panda_world, ExecuteActions, load_blocks)
from tamp.pddlstream_utils import get_pddlstream_info, pddlstream_plan
from tamp.ros_utils import (pose_to_transform, ros_to_pose,
    ros_to_transform, task_plan_to_ros)
from tf.transformations import quaternion_multiply


class PlanningServer():
    def __init__(self, blocks, block_init_xy_poses=None, 
                 max_tries=1, sim_failure_prob=0.0, 
                 alternate_orientations=False, multiprocessing=True,
                 use_platform=False, use_vision=False):

        # Start up a robot simulation for planning
        self._planning_client_id = pb_robot.utils.connect(use_gui=False)
        self.set_pybullet_client()
        pb_robot.utils.set_default_camera()
        self.robot = pb_robot.panda.Panda()
        self.robot.arm.hand.Open()

        # Initialize general attributes
        self.max_tries = max_tries
        self.use_vision = use_vision
        self.alternate_orientations = alternate_orientations
        self.sim_failure_prob = sim_failure_prob
        self.multiprocessing = multiprocessing
        if self.multiprocessing:
            self.proc_manager = Manager()

        # Initialize the world
        self.pddl_blocks, self.platform_table, self.platform_leg, self.table, self.frame, self.wall = \
            setup_panda_world(self.robot, blocks, block_init_xy_poses, use_platform=use_platform)
        self.fixed = [self.platform_table, self.platform_leg, self.table, self.frame, self.wall]
        self.pddl_block_lookup = get_pddl_block_lookup(blocks, self.pddl_blocks)

        # Initialize planning variables
        self.planning = False
        self.plan_buffer = []
        self.planning_active = False
        self.cancel_planning = False
        self.plan_complete = False
        self.new_block_states = []
        self.goal_block_states = []

        # Create the ROS services
        rospy.init_node("planning_server")
        self.get_latest_plan_service = rospy.Service(
            "/get_latest_plan", GetPlan, self.get_latest_plan)
        self.reset_service = rospy.Service(
            "/reset_planning", SetPlanningState, self.reset_planning)
        print("Planning server ready!")


    def run_planning_loop(self):
        """ Starts main planning loop """
        while not rospy.is_shutdown():

            # If no planning has been received, just keep waiting
            if not self.cancel_planning and \
              (not self.planning_active or (self.planning_active and self.plan_complete)):
                # print("Waiting for client ...")
                rospy.sleep(1)
            # Otherwise, plan until failure or cancellation
            else:
                self.reset_planning_state()
                self.plan_from_goals()


    def set_pybullet_client(self):
        """ Sets PyBullet client to this planning server """
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


    def get_initial_pddl_state(self):
        """
        Get the PDDL representation of the world between experiments. This
        method assumes that all blocks are on the table. We will always "clean
        up" an experiment by moving blocks away from the platform after an
        experiment.
        """
        fixed = [self.table, self.platform_table, self.platform_leg, self.frame]
        robot_config = self.robot.arm.GetJointValues()
        conf = pb_robot.vobj.BodyConf(self.robot, robot_config)
        print('Initial config:', robot_config)
        init = [('CanMove',),
                ('Conf', conf),
                ('StartConf', conf),
                ('AtConf', conf)]

        # Get the grasp state
        if self.latest_body_grasp is None:
            init += [('HandEmpty',)]
        else:
            init += [('AtGrasp', self.latest_body_grasp.body, self.latest_body_grasp)]

        self.table_pose = pb_robot.vobj.BodyPose(self.table, self.table.get_base_link_pose())
        init += [('Pose', self.table, self.table_pose), 
                 ('AtPose', self.table, self.table_pose)]

        for body in self.pddl_blocks:
            print(type(body), body)
            pose = pb_robot.vobj.BodyPose(body, body.get_base_link_pose())
            init += [('Graspable', body),
                     ('Pose', body, pose),
                     ('AtPose', body, pose),
                     ('Block', body)]
            if (self.latest_body_grasp is None) or (body != self.latest_body_grasp.body):
                init += [('On', body, self.table),
                         ('Supported', body, pose, self.table, self.table_pose)]

        if not self.platform_table is None:
            self.platform_pose = pb_robot.vobj.BodyPose(self.platform_table, self.platform_table.get_base_link_pose())
            init += [('Pose', self.platform_table, self.platform_pose), 
                     ('AtPose', self.platform_table, self.platform_pose)]
            init += [('Block', self.platform_table)]
        init += [('Table', self.table)]
        return init


    def plan_from_goals(self):
        """ Executes plan for a set of goal states """
        # Get the home poses
        self.home_poses = {}
        for blk, base, pose, stack in self.goal_block_states:
            if not stack:
                self.home_poses[blk.get_name()] = pose

        # Plan for all goals sequentially
        for blk, base, pose, stack in self.goal_block_states:
            if self.cancel_planning:
                return

            # Unpack the goal states into PDDLStream
            init = self.get_initial_pddl_state()

            fixed_objs = self.fixed + [b for b in self.pddl_blocks if b != blk]
            if base == self.table:
                pose_orig = blk.get_base_link_pose()
                pose_obj = pb_robot.vobj.BodyPose(blk, pose)

                if not stack and self.alternate_orientations:
                    init += [("Reset",)]
                    pose_goal = ("AtHome", blk)
                else:
                    init += [("Pose", blk, pose_obj),
                             ("Supported", blk, pose_obj, self.table, self.table_pose)]
                    pose_goal = ("AtPose", blk, pose_obj)

                goal = ("and", ("On", blk, self.table), pose_goal)
            else:
                rel_tform = pose_to_transform(pose)
                init += [("RelPose", blk, base, rel_tform)]
                goal = ("and", ("On", blk, base))

            # Plan
            if self.multiprocessing:
                ret_dict = self.proc_manager.dict()
                p = Process(target=self.pddlstream_plan,
                            args=(ret_dict, init, goal, fixed_objs, self.max_tries, getrandbits(32)))
                p.start()
                while p.is_alive():
                    if self.cancel_planning:
                        print("Killing planning process")
                        parent = psutil.Process(p.pid)
                        for child in parent.children(recursive=True):
                            child.kill()
                        parent.kill()
                    rospy.sleep(3)
                if "plan" in ret_dict:
                    plan = dill.loads(ret_dict["plan"])
                else:
                    print("No plan returned")
                    return 
            else:
                ret_dict = {}
                plan = self.pddlstream_plan(ret_dict, init, goal, fixed_objs, self.max_tries)

            if self.cancel_planning:
                print("Discarding latest plan")
                self.plan_buffer = []
            elif plan is not None:
                print(f"Simulating plan")
                self.simulate_plan(plan)
                if self.cancel_planning:
                    print("Discarding latest plan")
                    self.plan_buffer = []
            else:
                print(f"No plan found to place {blk}")
                self.goal_block_states = []
                self.planning_active = False

            # Convert the plan to a ROS message
            result = GetPlanResponse()
            result.plan = task_plan_to_ros(plan)
            result.planning_active = self.planning_active
            self.plan_buffer.append(result)
            
        # Set the completion flag if the plan succeeded until the end
        self.plan_complete = True


    def get_latest_plan(self, request):
        """ Extracts the latest action plan from the plan buffer """
        result = None
        while result is None:
            if len(self.plan_buffer) > 0:
                print("Popped plan from buffer!")
                result = self.plan_buffer.pop(0)
            else:
                rospy.sleep(1)
        return result


    def reset_planning(self, ros_request):
        """
        Clears any existing planning buffer and sets the initial
        block and robot states based on the request information
        """
        self.plan_buffer = []
        print("\n\nReset request received!\n\n")

        # Get the new initial poses of blocks based on the execution world
        self.new_block_states = []
        for elem in ros_request.init_state:
            blk = self.pddl_block_lookup[elem.name]
            pos = [elem.pose.position.x, elem.pose.position.y, elem.pose.position.z]
            orn = [elem.pose.orientation.x, elem.pose.orientation.y,
                elem.pose.orientation.z, elem.pose.orientation.w]
            self.new_block_states.append((blk, (pos,orn)))

        # Get the new goal poses of blocks based on the plan
        self.goal_block_states = []
        for elem in ros_request.goal_state:
            blk = self.pddl_block_lookup[elem.name]
            try:
                base = self.pddl_block_lookup[elem.base_obj]
            except:
                base = self.table
            pos = [elem.pose.position.x, elem.pose.position.y, elem.pose.position.z]
            orn = [elem.pose.orientation.x, elem.pose.orientation.y,
                elem.pose.orientation.z, elem.pose.orientation.w]
            stack = elem.stack
            self.goal_block_states.append((blk, base, (pos,orn), stack))

        # Get the robot configuration
        self.latest_robot_config = ros_request.robot_config.angles

        # Get the held block, if any
        if len(ros_request.held_block.name) == 0:
            self.latest_body_grasp = None
        else:
            held_block = self.pddl_block_lookup[ros_request.held_block.name]
            T = ros_to_transform(ros_request.held_block.pose)
            self.latest_body_grasp = pb_robot.vobj.BodyGrasp(held_block, T, self.robot.arm)

        self.cancel_planning = True
        self.planning_active = True
        self.plan_complete = False
        return SetPlanningStateResponse()


    def reset_planning_state(self):
        """ Resets the state of planning (blocks and robot arm) """
        self.plan_buffer = []
        self.cancel_planning = False
        for blk, pose in self.new_block_states:
            blk.set_base_link_pose(pose)
            print(f"Repositioning {blk}")
        robot_config = [q for q in self.latest_robot_config]
        self.robot.arm.SetJointValues(robot_config)
        print("Reset robot joint angles")
        print("Planning state reset!")


    def pddlstream_plan(self, return_dict, init, goal, fixed_objs, max_tries=1, random_seed=None):
        """ Plans using PDDLStream and the necessary specifications """
        num_tries = 0
        plan = None
        cost = 0.0
        found_plan = False
        while (not found_plan) and (num_tries < max_tries):
            print(f"\n\nPlanning try {num_tries}...\n\n")
            saved_world = pb_robot.utils.WorldSaver()
            start = time.time()
            
            # print_planning_problem(init, goal, fixed_objs)

            # Simulate planning failures (for testing)
            if random_seed is not None:
                numpy.random.seed(random_seed)
            if numpy.random.random() < self.sim_failure_prob:
                print("Simulated planning failure!")
                rospy.sleep(5)
            else:
                # Get PDDLStream planning information
                pddl_info = get_pddlstream_info(self.robot,
                                                fixed_objs,
                                                self.pddl_blocks,
                                                add_slanted_grasps=True,
                                                approach_frame="global",
                                                use_vision=self.use_vision,
                                                home_poses=self.home_poses)
        
                # Run PDDLStream focused solver
                try:
                    plan, cost = pddlstream_plan(pddl_info, init, goal, 
                                                 search_sample_ratio=1.0, max_time=INF)
                except Exception as e:
                    print("Failed planning")
                    print(e)
                    pass                  
            duration = time.time() - start

            if plan is not None:
                found_plan = True
            num_tries += 1
            saved_world.restore()

        print(f"\nFINAL PLAN:\n{plan}\nCOST: {cost}\n")
        if self.multiprocessing:
            return_dict["plan"] = dill.dumps(plan)
        return plan


    def simulate_plan(self, plan):
        """ Simulates an action plan """
        if plan is None:
            return
        self.set_pybullet_client()
        self.robot.arm.hand.Open()
        ExecuteActions(plan, real=False, pause=True, wait=False, prompt=False)
        print("Plan simulated")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--blocks-file', default='learning/domains/towers/final_block_set_10.pkl', type=str)
    parser.add_argument('--num-blocks', type=int, default=4)
    parser.add_argument('--max-tries', type=int, default=1)
    parser.add_argument('--use-vision', default=False, action='store_true')
    parser.add_argument('--alternate-orientations', default=False, action='store_true')
    parser.add_argument('--sim-failure-prob', type=float, default=0.0)
    args = parser.parse_args()

    blocks = load_blocks(fname=args.blocks_file,
                            num_blocks=args.num_blocks)
    block_init_xy_poses = None
    
    s = PlanningServer(blocks, 
                       max_tries=args.max_tries,
                       multiprocessing=True,
                       use_vision=args.use_vision, 
                       alternate_orientations=args.alternate_orientations,
                       sim_failure_prob=args.sim_failure_prob)
    s.run_planning_loop()
