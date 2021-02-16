import numpy
import pybullet as p
import sys
import time

from copy import deepcopy

import pyquaternion
import pb_robot
import tamp.primitives

from actions import PlaceAction, make_platform_world
from block_utils import get_adversarial_blocks, rotation_group, ZERO_POS, \
                        Quaternion, get_rotated_block, Pose, add_noise, \
                        Environment, Position
from pddlstream.algorithms.constraints import PlanConstraints, WILD
from pddlstream.algorithms.focused import solve_focused
from pddlstream.language.stream import StreamInfo
from pddlstream.utils import INF
from pybullet_utils import transformation
from tamp.misc import setup_panda_world, get_pddl_block_lookup, \
                      get_pddlstream_info, print_planning_problem, \
                      ExecuteActions, ExecutionFailure


class PandaAgent:
    def __init__(self, blocks, noise=0.00005, block_init_xy_poses=None,
                 teleport=False, use_platform=False, use_vision=False, real=False,
                 use_action_server=False, use_learning_server=False):
        """
        Build the Panda world in PyBullet and set up the PDDLStream solver.
        The Panda world should in include the given blocks as well as a
        platform which can be used in experimentation.
        :param teleport: Debugging parameter used to skip planning while moving
                         blocks around this world.
        :param use_platform: Boolean stating whether to include the platform to
                             push blocks off of or not.
        :param use_vision: Boolean stating whether to use vision to detect blocks.
        :param use_action_server: Boolean stating whether to use the separate
                                  ROS action server to do planning.
        :param use_learning_server: Boolean stating whether to host a ROS service
                                    server to drive planning from active learning script.

        If you are using the ROS action server, you must start it in a separate terminal:
            rosrun stacking_ros planning_server.py
        """
        self.real = real
        self.use_vision = use_vision
        self.use_action_server = use_action_server
        self.use_learning_server = use_learning_server

        # Setup PyBullet instance to run in the background and handle planning/collision checking.
        self._planning_client_id = pb_robot.utils.connect(use_gui=False)
        self.plan()
        pb_robot.utils.set_default_camera()
        self.robot = pb_robot.panda.Panda()
        self.robot.arm.hand.Open()
        self.belief_blocks = blocks
        self.pddl_blocks, self.platform_table, self.platform_leg, self.table, self.frame, self.wall = setup_panda_world(self.robot,
                                                                                                                        blocks,
                                                                                                                        block_init_xy_poses,
                                                                                                                        use_platform=use_platform)
        self.fixed = [self.platform_table, self.platform_leg, self.table, self.frame, self.wall]
        self.pddl_block_lookup = get_pddl_block_lookup(blocks, self.pddl_blocks)

        # Setup PyBullet instance that only visualizes plan execution. State needs to match the planning instance.
        poses = [b.get_base_link_pose() for b in self.pddl_blocks]
        poses = [Pose(Position(*p[0]), Quaternion(*p[1])) for p in poses]
        self._execution_client_id = pb_robot.utils.connect(use_gui=True)
        self.execute()
        pb_robot.utils.set_default_camera()
        self.execution_robot = pb_robot.panda.Panda()
        self.execution_robot.arm.hand.Open()
        setup_panda_world(self.execution_robot, blocks, poses, use_platform=use_platform)

        # Set up ROS plumbing if using features that require it
        if self.use_vision or self.use_action_server or real:
            import rospy
            rospy.init_node("panda_agent")

        # Create an arm interface
        if real:
            from franka_interface import ArmInterface
            self.real_arm = ArmInterface()

        # Set initial poses of all blocks and setup vision ROS services.
        if self.use_vision:
            from panda_vision.srv import GetBlockPosesWorld, GetBlockPosesWrist
            rospy.wait_for_service('get_block_poses_world')
            rospy.wait_for_service('get_block_poses_wrist')
            self._get_block_poses_world = rospy.ServiceProxy('get_block_poses_world', GetBlockPosesWorld)
            self._get_block_poses_wrist = rospy.ServiceProxy('get_block_poses_wrist', GetBlockPosesWrist)
            self._update_block_poses()

        # Start ROS clients and servers as needed
        self.last_obj_held = None
        if self.use_action_server:
            from stacking_ros.srv import GetPlan, SetPlanningState, PlanTower
            from tamp.ros_utils import goal_to_ros, ros_to_task_plan

            print("Waiting for planning server...")
            rospy.wait_for_service("get_latest_plan")
            self.goal_to_ros = goal_to_ros
            self.ros_to_task_plan = ros_to_task_plan
            self.init_state_client = rospy.ServiceProxy(
                "/reset_planning", SetPlanningState)
            self.get_plan_client = rospy.ServiceProxy(
                "/get_latest_plan", GetPlan)
            if self.use_learning_server:
                self.learning_server = rospy.Service(
                    "/plan_tower", PlanTower, self.plan_and_execute_tower)
                print("Learning server started!")
            print("Done!")
        else:
            self.planning_client = None

        self.pddl_info = get_pddlstream_info(self.robot,
                                             self.fixed,
                                             self.pddl_blocks,
                                             add_slanted_grasps=False,
                                             approach_frame='global',
                                             use_vision=self.use_vision)

        self.noise = noise
        self.teleport = teleport
        self.txt_id = None
        self.plan()

    def _add_text(self, txt):
        self.execute()
        pb_robot.viz.remove_all_debug()
        self.txt_id = pb_robot.viz.add_text(txt, position=(0, 0.25, 0.75), size=2)
        self.plan()

    def execute(self):
        self.state = 'execute'
        pb_robot.aabb.set_client(self._execution_client_id)
        pb_robot.body.set_client(self._execution_client_id)
        pb_robot.collisions.set_client(self._execution_client_id)
        pb_robot.geometry.set_client(self._execution_client_id)
        pb_robot.grasp.set_client(self._execution_client_id)
        pb_robot.joint.set_client(self._execution_client_id)
        pb_robot.link.set_client(self._execution_client_id)
        pb_robot.panda.set_client(self._execution_client_id)
        pb_robot.planning.set_client(self._execution_client_id)
        pb_robot.utils.set_client(self._execution_client_id)
        pb_robot.viz.set_client(self._execution_client_id)

    def plan(self):
        if self.use_action_server:
            return
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

    def _update_block_poses(self):
        """ Use the global world cameras to update the positions of the blocks """
        try:
            resp = self._get_block_poses_world()
            named_poses = resp.poses
        except:
            import sys
            print('Service call to get block poses failed. Exiting.')
            sys.exit()

        for pddl_block_name, pddl_block in self.pddl_block_lookup.items():
            for named_pose in named_poses:
                print(pddl_block_name, named_pose.block_id)
                if named_pose.block_id in pddl_block_name:
                    pose = named_pose.pose.pose
                    # Skip changes the pose of objects in storage.
                    if pose.position.x < 0.05:
                        continue
                    position = (pose.position.x, pose.position.y, pose.position.z)
                    orientation = (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
                    self.execute()
                    pddl_block.set_base_link_pose((position, orientation))
                    if not self.use_action_server:
                        self.plan()
                        pddl_block.set_base_link_pose((position, orientation))

        # After loading from vision, objects may be in collision. Resolve this.
        for _, pddl_block in self.pddl_block_lookup.items():
            if pb_robot.collisions.body_collision(pddl_block, self.table):
                print('Collision with table and block:', pddl_block.readableName)
                position, orientation = pddl_block.get_base_link_pose()
                stable_z = pb_robot.placements.stable_z(pddl_block, self.table)
                position = (position[0], position[1], stable_z)
                self.execute()
                pddl_block.set_base_link_pose((position, orientation))
                self.plan()
                pddl_block.set_base_link_pose((position, orientation))

        # Resolve from low to high blocks.
        current_poses = [b.get_base_link_pose() for b in self.pddl_blocks]
        block_ixs = range(len(self.pddl_blocks))
        block_ixs = sorted(block_ixs, key=lambda ix: current_poses[ix][0][2], reverse=False)
        for ix in range(len(block_ixs)):
            bottom_block = self.pddl_blocks[block_ixs[ix]]
            for jx in range(ix+1, len(block_ixs)):
                top_block = self.pddl_blocks[block_ixs[jx]]

                if pb_robot.collisions.body_collision(bottom_block, top_block):
                    print('Collision with bottom %s and top %s:' % (bottom_block.readableName, top_block.readableName))
                    position, orientation = top_block.get_base_link_pose()
                    stable_z = pb_robot.placements.stable_z(top_block, bottom_block)
                    position = (position[0], position[1], stable_z)
                    self.execute()
                    top_block.set_base_link_pose((position, orientation))
                    self.plan()
                    top_block.set_base_link_pose((position, orientation))


    def _get_initial_pddl_state(self):
        """
        Get the PDDL representation of the world between experiments. This
        method assumes that all blocks are on the table. We will always "clean
        up" an experiment by moving blocks away from the platform after an
        experiment.
        """
        fixed = [self.table, self.platform_table, self.platform_leg, self.frame]
        conf = pb_robot.vobj.BodyConf(self.robot, self.robot.arm.GetJointValues())
        print('Initial configuration:', conf.configuration)
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

    def simulate_action(self, action, block_ix, T=50, vis_sim=False, vis_placement=False):
        """
        Perform the given action to with the given block. An observation
        should be returned in the reference frame of the platform.
        :param action: Place action which describes the relative pose of the block to the platform surface.
        :param real_block: Belief representation of the block to perform the action on.
        :param T: How many timesteps to simulate the block falling for.
        :param vis_sim: Ununsed.
        :return: (action, T, end_pose) End pose should be TODO: what frame?
        """
        assert(self.platform_table is not None)
        real_block = self.belief_blocks[block_ix]
        pddl_block = self.pddl_blocks[block_ix]

        original_pose = pddl_block.get_base_link_pose()

        # Set up the PDDLStream problem for the placing the given block on the
        # platform with the specified action.
        self.pddl_info = get_pddlstream_info(self.robot,
                                             self.fixed,
                                             self.pddl_blocks,
                                             add_slanted_grasps=False,
                                             approach_frame='gripper',
                                             use_vision=self.use_vision)
        init = self._get_initial_pddl_state()

        #  Figure out the correct transformation matrix based on the action.
        real_block.set_pose(Pose(ZERO_POS, Quaternion(*action.rot.as_quat())))
        rotated_block = get_rotated_block(real_block)

        x = action.pos[0]
        y = action.pos[1]
        z = self.platform_table.get_dimensions()[2]/2. + rotated_block.dimensions[2]/2 #+ 1e-5
        tform = numpy.array([[1., 0., 0., x],
                             [0., 1., 0., y],
                             [0., 0., 1., z],
                             [0., 0., 0., 1.]])
        tform[0:3, 0:3] = action.rot.as_matrix()

        # Code to visualize where the block will be placed.
        if vis_placement:
            surface_tform = pb_robot.geometry.tform_from_pose(self.platform_table.get_base_link_pose())
            body_tform = surface_tform@tform
            length, lifeTime = 0.2, 0.0

            pos, quat = pb_robot.geometry.pose_from_tform(body_tform)
            new_x = transformation([length, 0.0, 0.0], pos, quat)
            new_y = transformation([0.0, length, 0.0], pos, quat)
            new_z = transformation([0.0, 0.0, length], pos, quat)

            p.addUserDebugLine(pos, new_x, [1,0,0], lifeTime=lifeTime)
            p.addUserDebugLine(pos, new_y, [0,1,0], lifeTime=lifeTime)
            p.addUserDebugLine(pos, new_z, [0,0,1], lifeTime=lifeTime)

        init += [('RelPose', pddl_block, self.platform_table, tform)]
        goal = ('On', pddl_block, self.platform_table)

        # Solve the PDDLStream problem.
        print('Init:', init)
        print('Goal:', goal)

        if not self.teleport:
            self._solve_and_execute_pddl(init, goal, search_sample_ratio=1000)
        else:
            goal_pose_fn = tamp.primitives.get_stable_gen_block()
            goal_pose = goal_pose_fn(pddl_block,
                                     self.platform_table,
                                     self.platform_pose,
                                     tform)[0].pose
            self.teleport_block(pddl_block, goal_pose)

        # Execture the action.
        # TODO: Check gravity compensation in the arm.

        self.step_simulation(T)
        end_pose = self._get_observed_pose(pddl_block, action)
        observation = (action, T, end_pose)
        self.step_simulation(500-T)

        # Put block back in original position.

        # TODO: Check if block is on the table or platform to start.
        self.pddl_info = get_pddlstream_info(self.robot,
                                             self.fixed,
                                             self.pddl_blocks,
                                             add_slanted_grasps=True,
                                             approach_frame='gripper',
                                             use_vision=self.use_vision)

        init = self._get_initial_pddl_state()
        goal_pose = pb_robot.vobj.BodyPose(pddl_block, original_pose)
        init += [('Pose', pddl_block, goal_pose),
                 ('Supported', pddl_block, goal_pose, self.table, self.table_pose)]
        goal = ('and', ('AtPose', pddl_block, goal_pose),
                       ('On', pddl_block, self.table))

        # Solve the PDDLStream problem.
        print('Init:', init)
        print('Goal:', goal)

        if not self.teleport:
            success = self._solve_and_execute_pddl(init, goal, max_time=100., search_sample_ratio=1000)
            if not success:
                print('Plan failed: Teleporting block to intial position.')
                self.teleport_block(pddl_block, original_pose)
        else:
            self.teleport_block(pddl_block, original_pose)

        return observation

    def teleport_block(self, block, pose):
        self.execute()
        block.set_base_link_pose(pose)
        self.plan()
        block.set_base_link_pose(pose)


    def simulate_tower_parallel(self, tower, vis, T=2500, real=False, base_xy=(0., 0.5)):
        """
        Simulates a tower stacking and unstacking by requesting plans from a separate planning server
        """

        for block in tower:
            print('Block:', block.name)
            print('Pose:', block.pose)
            print('Dims:', block.dimensions)
            print('CoM:', block.com)
            print('-----')
        if self.use_vision:
            self._update_block_poses()

        # Set up the list of original poses and order of blocks in the tower
        self.moved_blocks = set()
        original_poses = [b.get_base_link_pose() for b in self.pddl_blocks]
        tower_pddl = [self.pddl_block_lookup[b.name] for b in tower]
        tower_block_order = [self.pddl_blocks.index(b) for b in tower_pddl]

        # Package up the ROS message
        from stacking_ros.msg import BodyInfo
        from stacking_ros.srv import SetPlanningStateRequest
        from tamp.ros_utils import block_init_to_ros, pose_to_ros, pose_tuple_to_ros, transform_to_ros
        ros_req = SetPlanningStateRequest()
        # Initial poses and robot configuration
        if self.real:
            ros_req.robot_config.angles = self.real_arm.convertToList(arm.joint_angles())
        else:
            ros_req.robot_config.angles = self.robot.arm.GetJointValues()
        ros_req.init_state = block_init_to_ros(self.pddl_blocks)
        # Goal poses
        # TODO: Set base block to be rotated in its current position.
        base_block = self.pddl_block_lookup[tower[0].name]
        base_pos = (base_xy[0], base_xy[1], tower[0].pose.pos.z)
        base_pose = (base_pos, tower[0].rotation)
        base_pose = pb_robot.vobj.BodyPose(base_block, base_pose)
        base_block_ros = BodyInfo()
        base_block_ros.name = base_block.readableName
        base_block_ros.stack = True
        pose_to_ros(base_pose, base_block_ros.pose)
        ros_req.goal_state.append(base_block_ros)
        # Now loop through the other tower blocks
        for b_ix in range(1, len(tower)):
            bottom_block = tower[b_ix-1]
            bottom_pose = (bottom_block.pose.pos, bottom_block.rotation)
            bottom_tform = pb_robot.geometry.tform_from_pose(bottom_pose)
            top_block = tower[b_ix]
            top_pose = (top_block.pose.pos, top_block.rotation)
            top_tform = pb_robot.geometry.tform_from_pose(top_pose)

            rel_tform = numpy.linalg.inv(bottom_tform)@top_tform
            top_pddl = self.pddl_block_lookup[top_block.name]
            bottom_pddl = self.pddl_block_lookup[bottom_block.name]

            block_ros = BodyInfo()
            block_ros.name = top_pddl.readableName
            block_ros.base_obj = bottom_pddl.readableName
            transform_to_ros(rel_tform, block_ros.pose)
            block_ros.is_rel_pose = True
            block_ros.stack = True
            ros_req.goal_state.append(block_ros)
        # Finally, tack on the tower resetting steps
        for ix in reversed(tower_block_order):
            blk, pose = self.pddl_blocks[ix], original_poses[ix]
            goal_pose = pb_robot.vobj.BodyPose(blk, pose)
            block_ros = BodyInfo()
            block_ros.name = blk.readableName
            block_ros.stack = False
            pose_to_ros(goal_pose, block_ros.pose)
            ros_req.goal_state.append(block_ros)

        # Execute the stacking plan
        success, stack_stable, reset_stable, num_success, fatal = \
            self.execute_plans_from_server(ros_req, real, T, stack=True)
        print(f"Completed tower stack with success: {success}, stable: {stack_stable}")
        if reset_stable:
            print(f"Completed tower reset stable: {reset_stable}")

        # If we have a nonfatal failure, replan from new state, removing successful goals
        while (not success and not fatal):
            print(f"Got recoverable failure. Replanning from step index {num_success}.")
            if self.real:
                ros_req.robot_config.angles = self.real_arm.convertToList(arm.joint_angles())
            else:
                ros_req.robot_config.angles = self.robot.arm.GetJointValues()
            ros_req.init_state = block_init_to_ros(self.pddl_blocks)
            ros_req.goal_state = ros_req.goal_state[num_success:]
            if isinstance(self.last_obj_held, pb_robot.vobj.BodyGrasp):
                ros_req.held_block.name = self.last_obj_held.body.readableName
                transform_to_ros(self.last_obj_held.grasp_objF, ros_req.held_block.pose)
            success, stack_stable, reset_stable, num_success, fatal = \
                self.execute_plans_from_server(ros_req, real, T, stack=True)
            print(f"Completed tower stack with success: {success}, stable: {stack_stable}")
            if reset_stable:
                print(f"Completed tower reset stable: {reset_stable}")

        # If the full tower did not succeed, send out another reset plan
        try:
            if not (stack_stable and reset_stable):
                if self.use_vision and not stack_stable:
                    self._update_block_poses()

                # Instruct a reset plan
                print("Resetting blocks...")
                current_poses = [b.get_base_link_pose() for b in self.pddl_blocks]
                block_ixs = range(len(self.pddl_blocks))
                block_ixs = sorted(block_ixs, key=lambda ix: current_poses[ix][0][2], reverse=True)
                ros_req = SetPlanningStateRequest()
                ros_req.init_state = block_init_to_ros(self.pddl_blocks)
                if self.real:
                    ros_req.robot_config.angles = self.real_arm.convertToList(arm.joint_angles())
                else:
                    ros_req.robot_config.angles = self.robot.arm.GetJointValues()
                for ix in block_ixs:
                    blk, pose = self.pddl_blocks[ix], original_poses[ix]
                    if blk in self.moved_blocks:
                        goal_pose = pb_robot.vobj.BodyPose(blk, pose)
                        block_ros = BodyInfo()
                        block_ros.name = blk.readableName
                        block_ros.stack = False
                        pose_to_ros(goal_pose, block_ros.pose)
                        ros_req.goal_state.append(block_ros)

                # Execute the reset plan
                success, _, reset_stable, num_success, fatal = \
                    self.execute_plans_from_server(ros_req, real, T, stack=False)
                print(f"Completed tower reset with success: {success}, stable: {reset_stable}")

                # If we have a nonfatal failure, replan from new state, removing successful goals
                while (not success and not fatal):
                    print(f"Got recoverable failure. Replanning from step index {num_success}.")
                    if self.real:
                        ros_req.robot_config.angles = self.real_arm.convertToList(arm.joint_angles())
                    else:
                        ros_req.robot_config.angles = self.robot.arm.GetJointValues()
                    ros_req.init_state = block_init_to_ros(self.pddl_blocks)
                    ros_req.goal_state = ros_req.goal_state[num_success:]
                    success, _, reset_stable, num_success, fatal = \
                    self.execute_plans_from_server(ros_req, real, T, stack=False)
                    print(f"Completed tower reset with success: {success}, stable: {reset_stable}")

        except Exception as e:
            print("Planning/execution failed during tower reset.")
            print(e)

        # Return the final planning state
        return success, stack_stable


    def validate_ros_plan(self, ros_resp, tgt_block):
        """ Validates a ROS plan to move a block against the expected target block name """
        if len(ros_resp.plan) == 0:
            return True
        else:
            plan_blocks = [t.obj1 for t in ros_resp.plan if t.type == "pick"]
            if len(plan_blocks) > 0:
                plan_block = plan_blocks[0]
            else:
                return False
            print(f"Received plan to move {plan_block} and expected to move {tgt_block}")
            return (tgt_block == plan_block)


    def execute_plans_from_server(self, ros_req, real=False, T=2500, stack=True):
        """
        Executes plans received from planning server
        Returns:
            success : Flag for whether the plan execution succeeded
            stack_stable : Flag for whether stacking a stable tower was successful
            reset_stable : Flag for whether resetting a tower was successful
            num_success : Progress (in number of steps) of successful tasks
            fatal : Flag for whether the error was fatal (True) or recoverable (False)
        """
        self.init_state_client.call(ros_req)

        stack_stable = False
        reset_stable = False
        num_success = 0
        planning_active = True
        num_steps = len(ros_req.goal_state)
        while num_success < num_steps:
            try:
                # Wait for a valid plan
                plan = []
                while len(plan) == 0 and planning_active:
                    time.sleep(1)
                    ros_resp = self.get_plan_client.call()
                    if not ros_resp.planning_active:
                        print("Planning ended on server side")
                        return False, stack_stable, reset_stable, num_success, False
                    tgt_block = ros_req.goal_state[num_success].name
                    if self.validate_ros_plan(ros_resp, tgt_block):
                        plan = self.ros_to_task_plan(ros_resp, self.execution_robot, self.pddl_block_lookup)

                print("\nGot plan:")
                print(plan)

                # Once we have a plan, execute it
                self.execute()
                ExecuteActions(plan, real=real, pause=True, wait=False, prompt=False, obstacles=[f for f in self.fixed if f is not None], sim_fatal_failure_prob=0.0, sim_recoverable_failure_prob=0.0)

                # Manage the moved blocks (add to the set when stacking, remove when unstacking)
                query_block = self.pddl_block_lookup[ros_req.goal_state[num_success].name]
                desired_pose = query_block.get_base_link_pose()
                if query_block not in self.moved_blocks:
                    self.moved_blocks.add(query_block)
                else:
                    self.moved_blocks.remove(query_block)

                # Check stability
                if not real:
                    self.step_simulation(T, vis_frames=False)
                #input('Press enter to check stability.')
                stable = self.check_stability(real, query_block, desired_pose)
                #input('Continue?')

                # Manage the success status of the plan
                if stable == 0.:
                    print("Unstable after execution!")
                    return True, stack_stable, reset_stable, num_success, False
                else:
                    num_success += 1
                    if stack and num_success == num_steps/2:
                        print("Completed tower stack!")
                        stack_stable = True
                    elif num_success == num_steps:
                        print("Completed tower reset!")
                        reset_stable = True
                        return True, stack_stable, reset_stable, num_success, False

            except ExecutionFailure as e:
                print("Planning/execution failed.")
                print(e)
                self.last_obj_held = e.obj_held
                return False, stack_stable, reset_stable, num_success, e.fatal


    def plan_and_execute_tower(self, ros_req, base_xy=(0.5, -0.3)):
        """ Service callback function to plan and execute a tower """
        from stacking_ros.srv import PlanTowerResponse
        from tamp.ros_utils import ros_to_tower
        tower = ros_to_tower(ros_req.tower_info)
        success, stable = self.simulate_tower_parallel(tower, True, real=self.real, base_xy=base_xy)
        resp = PlanTowerResponse()
        resp.success = success
        resp.stable = stable
        return resp


    def simulate_tower(self, tower, vis, T=2500, real=False, base_xy=(0., 0.5), save_tower=False):
        """
        :param tower: list of belief blocks that are rotated to have no
                      orientation in the tower. These are in the order of
                      the tower starting at the base.
        """
        for block in tower:
            print('Block:', block.name)
            print('Pose:', block.pose)
            print('Dims:', block.dimensions)
            print('CoM:', block.com)
            print('-----')

        if self.use_vision:
            self._update_block_poses()

        moved_blocks = set()
        original_poses = [b.get_base_link_pose() for b in self.pddl_blocks]

        # starting state of the PDDL planning problem. includes block poses
        # and robot configuration
        init = self._get_initial_pddl_state()
        goal_terms = []

        # TODO: Set base block to be rotated in its current position.

        # before we begin planning, figure out the pose of the block that will
        # be the base of the tower
        base_block = self.pddl_block_lookup[tower[0].name]
        base_pos = (base_xy[0], base_xy[1], tower[0].pose.pos.z)
        base_pose = (base_pos, tower[0].rotation)
        # I believe that this adds that pose to the planning problem, and says that
        # a block placed at that pose will satisfy the `Supported` predicate
        base_pose = pb_robot.vobj.BodyPose(base_block, base_pose)
        init += [('Pose', base_block, base_pose),
                 ('Supported', base_block, base_pose, self.table, self.table_pose)]
        # And this specifies that we would like the base block to end up at the
        # the base pose
        goal_terms.append(('AtPose', base_block, base_pose))
        goal_terms.append(('On', base_block, self.table))
        # all the objects other than the base block are fixed
        fixed_objs = self.fixed + [b for b in self.pddl_blocks if b != base_block]
        # get the pddl problem spec
        self.pddl_info = get_pddlstream_info(self.robot,
                                             fixed_objs,
                                             self.pddl_blocks,
                                             add_slanted_grasps=False,
                                             approach_frame='global',
                                             use_vision=self.use_vision)

        # NOTE(izzy): this codeblock was getting called three times in this function, so I decided to
        # remove the redundancy. this should get a refactor sometime soon tho
        def _solve_and_execute_pddl_subroutine(init, goal_terms, real, fixed_objs, reset=True):
            goal = tuple(['and'] + goal_terms)
            if not self.use_action_server:
                plan_found = self._solve_and_execute_pddl(init, goal, real=real, search_sample_ratio=1.)
                if not plan_found: return False
            else:
                self.execute()
                has_plan = False
                while not has_plan:
                    plan = self._request_plan_from_server(
                        init, goal, fixed_objs, reset=reset)
                    if len(plan) > 0:
                        has_plan = True
                    else:
                        time.sleep(1)
                ExecuteActions(plan, real=real, pause=True, wait=False, obstacles=[f for f in self.fixed if f is not None])

            return True

        # this first section set up a PDDL instance to move the base block into the
        # desired configuration. now we solve that PDDL problem and execute the plan
        if not self.teleport:
            if not _solve_and_execute_pddl_subroutine(init, goal_terms, real, fixed_objs, reset=True):
                return False, None
        else:
            self.teleport_block(base_block, base_pose.pose)

        moved_blocks.add(base_block)
        stable = self.check_stability(real, base_block, base_pose.pose[0])
        poses = [base_pose]
        # TODO: Calculate each blocks pose relative to the block beneath.
        for b_ix in range(1, len(tower)):
            bottom_block = tower[b_ix-1]
            bottom_pose = (bottom_block.pose.pos, bottom_block.rotation)
            bottom_tform = pb_robot.geometry.tform_from_pose(bottom_pose)
            top_block = tower[b_ix]
            top_pose = (top_block.pose.pos, top_block.rotation)
            top_tform = pb_robot.geometry.tform_from_pose(top_pose)

            rel_tform = numpy.linalg.inv(bottom_tform)@top_tform
            top_pddl = self.pddl_block_lookup[top_block.name]
            bottom_pddl = self.pddl_block_lookup[bottom_block.name]

            init = self._get_initial_pddl_state()
            goal_terms = []

            fixed_objs = self.fixed + [b for b in self.pddl_blocks if b != top_pddl]
            self.pddl_info = get_pddlstream_info(self.robot,
                                                 fixed_objs,
                                                 self.pddl_blocks,
                                                 add_slanted_grasps=False,
                                                 approach_frame='global',
                                                 use_vision=self.use_vision)
            init += [('RelPose', top_pddl, bottom_pddl, rel_tform)]
            goal_terms.append(('On', top_pddl, bottom_pddl))

            moved_blocks.add(top_pddl)

            if not self.teleport:
                if not _solve_and_execute_pddl_subroutine(init, goal_terms, real, fixed_objs, reset=False):
                    return False, None
            else:
                get_pose = tamp.primitives.get_stable_gen_block()
                pose = get_pose(top_pddl, bottom_pddl, poses[-1], rel_tform)[0]
                poses.append(pose)
                self.teleport_block(top_pddl, pose.pose)

            # Execute the block placement.
            desired_pose = top_pddl.get_base_link_pose()
            if not real:
                self.step_simulation(T, vis_frames=False)
            # TODO: Check if the tower was stable, stop construction if not.
            #input('Press enter to stability.')
            stable = self.check_stability(real, top_pddl, desired_pose)
            #input('Continue?')
            if stable == 0.:
                break

        if not real:
            self.step_simulation(T, vis_frames=False)
        if self.use_vision and not stable:
            input('Update block poses after tower?')
            self._update_block_poses()

        # Reset Environment. Need to handle conditions where the blocks are still a stable tower.
        # As a heuristic for which block to reset first, do it in order of their z-values.
        current_poses = [b.get_base_link_pose() for b in self.pddl_blocks]
        block_ixs = range(len(self.pddl_blocks))
        block_ixs = sorted(block_ixs, key=lambda ix: current_poses[ix][0][2], reverse=True)

        for ix in block_ixs:
            b, pose = self.pddl_blocks[ix], original_poses[ix]
            if b not in moved_blocks: continue

            goal_pose = pb_robot.vobj.BodyPose(b, pose)
            fixed_objs = self.fixed + [obj for obj in self.pddl_blocks if obj != b]
            self.pddl_info = get_pddlstream_info(self.robot,
                                                 fixed_objs,
                                                 self.pddl_blocks,
                                                 add_slanted_grasps=False,
                                                 approach_frame='global',
                                                 use_vision=self.use_vision)

            init = self._get_initial_pddl_state()
            init += [('Pose', b, goal_pose),
                     ('Supported', b, goal_pose, self.table, self.table_pose)]
            goal_terms = []
            goal_terms.append(('AtPose', b, goal_pose))
            goal_terms.append(('On', b, self.table))

            if not _solve_and_execute_pddl_subroutine(init, goal_terms, real, fixed_objs, reset=True):
                return False, None

        return True, stable

    def check_stability(self, real, block_pddl, desired_pose):
        if self.use_vision:
            # Get pose of blocks using wrist camera.
            try:
                poses = self._get_block_poses_wrist().poses
            except:
                print('Service call to get block poses failed during check stability. Exiting.')
                sys.exit()

            # Check if pose is close to desired_pose.
            visible = False
            for named_pose in poses:
                if named_pose.block_id in block_pddl.readableName:
                    visible = True
                    pose = named_pose.pose.pose

                    des_pos = desired_pose[0]
                    obs_pos = (pose.position.x, pose.position.y, pose.position.z)
                    print('[Check Stability] Desired Pos:', des_pos)
                    print('[Check Stability] Detected Pos:', obs_pos)
                    # First check if the pose is too far away.
                    dist = numpy.linalg.norm(numpy.array(obs_pos)-numpy.array(des_pos))
                    print(f'[Check Stability] Position Distance (>0.04): {dist}')
                    if dist > 0.04:
                        return 0.
                    # Also check that the block is flat on the table.
                    orn = desired_pose[1]
                    obs_orn = pyquaternion.Quaternion(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z)
                    des_orn = pyquaternion.Quaternion(orn[3], orn[0], orn[1], orn[2])
                    angle = (des_orn.inverse*obs_orn).angle
                    angle = numpy.abs(numpy.rad2deg(angle))
                    print(f'[Check Stability] Orientation Distance (> 15): {angle}')
                    if angle > 15:
                        return 0.

            # If block isn't visible, return 0.
            if not visible:
                return 0.

        else:
            end_pose = block_pddl.get_base_link_point()
            dist = numpy.linalg.norm(numpy.array(end_pose) - numpy.array(desired_pose[0]))
            print(f"Distance is {dist}")
            print(f"Block dimensions are {block_pddl.get_dimensions()}")
            if dist > 0.01:
                print('Unstable!')
                return 0.
        return 1.


    def step_simulation(self, T, vis_frames=False, lifeTime=0.1):
        p.setGravity(0, 0, -10, physicsClientId=self._execution_client_id)
        p.setGravity(0, 0, -10, physicsClientId=self._planning_client_id)

        q = self.robot.get_joint_positions()

        for _ in range(T):
            p.stepSimulation(physicsClientId=self._execution_client_id)
            p.stepSimulation(physicsClientId=self._planning_client_id)

            self.execute()
            self.execution_robot.set_joint_positions(self.robot.joints, q)
            self.plan()
            self.robot.set_joint_positions(self.robot.joints, q)

            time.sleep(1/2400.)

            if vis_frames:
                length = 0.1
                for pddl_block in self.pddl_blocks:
                    pos, quat = pddl_block.get_pose()
                    new_x = transformation([length, 0.0, 0.0], pos, quat)
                    new_y = transformation([0.0, length, 0.0], pos, quat)
                    new_z = transformation([0.0, 0.0, length], pos, quat)

                    p.addUserDebugLine(pos, new_x, [1,0,0], lineWidth=3, lifeTime=lifeTime, physicsClientId=self._execution_client_id)
                    p.addUserDebugLine(pos, new_y, [0,1,0], lineWidth=3, lifeTime=lifeTime, physicsClientId=self._execution_client_id)
                    p.addUserDebugLine(pos, new_z, [0,0,1], lineWidth=3, lifeTime=lifeTime, physicsClientId=self._execution_client_id)

    def _get_observed_pose(self, pddl_block, action):
        """
        This pose should be relative to the base of the platform leg to
        agree with the simulation. The two block representations will have
        different orientation but their positions should be the same.
        """
        block_transform = pddl_block.get_base_link_transform()
        platform_transform = self.platform_leg.get_base_link_transform()
        platform_transform[2,3] -= self.platform_leg.get_dimensions()[2]/2.

        rel_transform = numpy.linalg.inv(platform_transform)@block_transform
        end_pose = pb_robot.geometry.pose_from_tform(rel_transform)
        # TODO: Add noise to the observation.

        end_pose = Pose(Position(*end_pose[0]), Quaternion(*end_pose[1]))
        end_pose = add_noise(end_pose, self.noise*numpy.eye(3))

        return end_pose

    def _solve_and_execute_pddl(self, init, goal, real=False, max_time=INF, search_sample_ratio=0.):
        self._add_text('Planning block placement')
        self.robot.arm.hand.Open()
        saved_world = pb_robot.utils.WorldSaver()

        self.plan()
        start = time.time()

        constraints = PlanConstraints(skeletons=self._get_regrasp_skeleton(),
                                  exact=True)

        pddlstream_problem = tuple([*self.pddl_info, init, goal])
        plan, _, _ = solve_focused(pddlstream_problem,
                                #constraints=constraints,
                                success_cost=numpy.inf,
                                max_skeletons=2,
                                search_sample_ratio=search_sample_ratio,
                                max_time=max_time)
        duration = time.time() - start
        print('Planning Complete: Time %f seconds' % duration)
        # TODO: Try planner= argument https://github.com/caelan/pddlstream/blob/stable/pddlstream/algorithms/downward.py

        self._add_text('Executing block placement')
        # Execute the PDDLStream solution to setup the world.
        if plan is None:
            input("Planning failed.")
            return False
        else:
            saved_world.restore()
            self.execute()
            ExecuteActions(plan, real=real, pause=True, wait=False, obstacles=[f for f in self.fixed if f is not None])
            self.plan()
            ExecuteActions(plan, real=False, pause=False, wait=False, obstacles=[f for f in self.fixed if f is not None])
            return True


    def _request_plan_from_server(self, init, goal, fixed_objs, reset=True, real=False):
        print('Requesting block placement plan from server...')
        # Package up the ROS action goal
        ros_goal = self.goal_to_ros(init, goal, fixed_objs)
        ros_goal.reset = reset
        print_planning_problem(init, goal, fixed_objs)

        # Call the planning action server
        self.planning_client.send_goal(ros_goal)
        self.planning_client.wait_for_result() # TODO: Blocking for now
        result = self.planning_client.get_result()
        # print(result)

        # Unpack the ROS message
        plan = self.ros_to_task_plan(result, self.execution_robot, self.pddl_block_lookup)
        return plan


    # TODO: Try this again.
    def _get_regrasp_skeleton(self):
        no_regrasp = []
        no_regrasp += [('move_free', [WILD, '?q0', WILD])]
        no_regrasp += [('pick', ['?b0', WILD, WILD, '?g0', '?q0', '?q1', WILD])]
        no_regrasp += [('move_holding', ['?q1', '?q2', '?b0', '?g0', WILD])]
        no_regrasp += [('place', ['?b0', WILD, WILD, WILD, '?g0', '?q2', WILD, WILD])]

        regrasp = []
        regrasp += [('move_free', [WILD, '?rq0', WILD])]
        regrasp += [('pick', ['?rb0', WILD, WILD, '?rg0', '?rq0', '?rq1', WILD])]
        regrasp += [('move_holding', ['?rq1', '?rq2', '?rb0', '?rg0', WILD])]
        regrasp += [('place', ['?rb0', WILD, WILD, WILD, '?rg0', '?rq2', '?rq3', WILD])]
        regrasp += [('move_free', ['?rq3', '?rq4', WILD])]
        regrasp += [('pick', ['?rb0', WILD, WILD, '?rg1', '?rq4', '?rq5', WILD])]
        regrasp += [('move_holding', ['?rq5', '?rq6', '?rb0', '?rg1', WILD])]
        regrasp += [('place', ['?rb0', WILD, WILD, WILD, '?rg1', '?rq6', WILD, WILD])]

        return [no_regrasp, regrasp]


class PandaClientAgent:
    """
    Lightweight client to call a PandaAgent as a service for active learning
    """

    def __init__(self):
        import rospy
        rospy.init_node("panda_client")
        from stacking_ros.srv import PlanTower
        print("Waiting for Panda Agent server...")
        rospy.wait_for_service("/plan_tower")
        print("Done")
        self.client = rospy.ServiceProxy(
            "/plan_tower", PlanTower)


    def simulate_tower(self, tower, vis, real=False):
        """ Call the PandaAgent server """
        from stacking_ros.srv import PlanTowerRequest
        from tamp.ros_utils import tower_to_ros, ros_to_tower
        request = PlanTowerRequest()
        request.tower_info = tower_to_ros(tower)
        response = self.client.call(request)
        return response.success, response.stable
