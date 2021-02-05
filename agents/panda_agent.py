import numpy
import pybullet as p
import time

from copy import deepcopy

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
                      get_pddlstream_info, print_planning_problem, ExecuteActions


class PandaAgent:
    def __init__(self, blocks, noise=0.00005, use_platform=False, block_init_xy_poses=None, teleport=False, use_vision=False, use_action_server=False):
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

        If you are using the ROS action server, you must start it in a separate terminal:
            rosrun stacking_ros planning_server.py
        """
        self.use_vision = use_vision
        self.use_action_server = use_action_server

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
        if self.use_vision or self.use_action_server:
            import rospy
            #rospy.init_node("panda_agent")

        # Set initial poses of all blocks and setup vision ROS services.
        if self.use_vision:
            from panda_vision.srv import GetBlockPosesWorld
            rospy.wait_for_service('get_block_poses_world')
            self._get_block_poses_world = rospy.ServiceProxy('get_block_poses_world', GetBlockPosesWorld)
            self._update_block_poses()

        # Start ROS action client
        if self.use_action_server:
            rospy.init_node("panda_agent")
            import actionlib
            from stacking_ros.msg import TaskPlanAction
            from stacking_ros.srv import GetPlan, SetPlanningState
            from tamp.ros_utils import goal_to_ros, ros_to_task_plan
            self.goal_to_ros = goal_to_ros
            self.ros_to_task_plan = ros_to_task_plan
            self.init_state_client = rospy.ServiceProxy(
                "/reset_planning", SetPlanningState)
            self.get_plan_client = rospy.ServiceProxy(
                "/get_latest_plan", GetPlan)
            self.planning_client = actionlib.SimpleActionClient(
                "/get_plan", TaskPlanAction)
            print("Waiting for planning server...")
            self.planning_client.wait_for_server()
            print("Done!")
        else:
            self.planning_client = None

        self.pddl_info = get_pddlstream_info(self.robot,
                                             self.fixed,
                                             self.pddl_blocks,
                                             add_slanted_grasps=False,
                                             approach_frame='global')

        self.noise = noise
        self.teleport = teleport
        self.txt_id = None
        self.plan()
        
        self.initial_world = pb_robot.utils.WorldSaver()
        
    def reset(self):
        self.initial_world.restore()

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
                                             approach_frame='gripper')
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
                                             approach_frame='gripper')

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
        """

        for block in tower:
            print('Block:', block.name)
            print('Pose:', block.pose)
            print('Dims:', block.dimensions)
            print('CoM:', block.com)
            print('-----')
        if self.use_vision:
            self._update_block_poses()

        self.moved_blocks = set()
        original_poses = [b.get_base_link_pose() for b in self.pddl_blocks]

        # Package up the ROS message
        from stacking_ros.msg import BodyInfo
        from stacking_ros.srv import SetPlanningStateRequest
        from tamp.ros_utils import pose_to_ros, pose_tuple_to_ros, transform_to_ros
        ros_req = SetPlanningStateRequest()
        # Initial poses
        for blk in self.pddl_blocks:
            ros_block = BodyInfo()
            ros_block.name = blk.readableName
            pose_tuple_to_ros(blk.get_base_link_pose(), ros_block.pose)
            ros_req.init_state.append(ros_block)
        # Goal poses
        # TODO: Set base block to be rotated in its current position.
        base_block = self.pddl_block_lookup[tower[0].name]
        base_pos = (base_xy[0], base_xy[1], tower[0].pose.pos.z)
        base_pose = (base_pos, tower[0].rotation)
        base_pose = pb_robot.vobj.BodyPose(base_block, base_pose)
        base_block_ros = BodyInfo()
        base_block_ros.name = base_block.readableName
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
            ros_req.goal_state.append(block_ros)

        success = self.execute_plans_from_server(ros_req, real, T)
        print(f"Completed tower stack with success: {success}")

        # Instruct a reset plan
        print("Resetting blocks...")
        current_poses = [b.get_base_link_pose() for b in self.pddl_blocks]
        block_ixs = range(len(self.pddl_blocks))
        block_ixs = sorted(block_ixs, key=lambda ix: current_poses[ix][0][2], reverse=True)

        ros_req = SetPlanningStateRequest()
        # Initial poses
        for blk in self.pddl_blocks:
            ros_block = BodyInfo()
            ros_block.name = blk.readableName
            pose_tuple_to_ros(blk.get_base_link_pose(), ros_block.pose)
            ros_req.init_state.append(ros_block)
        # Goal poses
        for ix in block_ixs:
            blk, pose = self.pddl_blocks[ix], original_poses[ix]
            if blk in self.moved_blocks:
                goal_pose = pb_robot.vobj.BodyPose(blk, pose)
                ros_block = BodyInfo()
                ros_block.name = blk.readableName
                pose_to_ros(goal_pose, ros_block.pose)
                ros_req.goal_state.append(ros_block)
        
        success = self.execute_plans_from_server(ros_req, real, T)
        print(f"Completed tower reset with success: {success}")


    def execute_plans_from_server(self, ros_req, real=False, T=2500):
        """ Executes plans received from planning server """
        self.init_state_client.call(ros_req)

        success = False
        num_success = 0
        planning_active = True
        while num_success < len(ros_req.goal_state):
            # Wait for a valid plan
            plan = []
            while len(plan) == 0 and planning_active:
                time.sleep(3)
                ros_resp = self.get_plan_client.call()
                planning_active = ros_resp.planning_active
                plan = self.ros_to_task_plan(ros_resp, self.execution_robot, self.pddl_block_lookup)
                if not planning_active:
                    print("Planning ended on server side")
                    return success
            print("\nGot plan:")
            print(plan)

            # Once we have a plan, execute it
            self.execute()
            ExecuteActions(plan, real=real, pause=True, wait=False, prompt=False)

            # Check if the plan failed; if so, exit
            query_block = self.pddl_block_lookup[ros_req.goal_state[num_success].name]
            self.moved_blocks.add(query_block)
            desired_pose = query_block.get_point()
            if not real:
                self.step_simulation(T, vis_frames=False)
            end_pose = query_block.get_point()
            if numpy.linalg.norm(numpy.array(end_pose) - numpy.array(desired_pose)) > 0.01:
                print("Unstable after execution!")
                return success
            else:
                num_success += 1
                if num_success == len(ros_req.goal_state):
                    success = True
        return success


    def simulate_tower(self, tower, vis, T=2500, real=False, base_xy=(0., 0.5), save_tower=False, solve_joint=False):
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

        init = self._get_initial_pddl_state()
        goal_terms = []

        stable = 1.

        # TODO: Set base block to be rotated in its current position.
        base_block = self.pddl_block_lookup[tower[0].name]
        base_pos = (base_xy[0], base_xy[1], tower[0].pose.pos.z)
        base_pose = (base_pos, tower[0].rotation)

        base_pose = pb_robot.vobj.BodyPose(base_block, base_pose)
        init += [('Pose', base_block, base_pose),
                 ('Supported', base_block, base_pose, self.table, self.table_pose)]
        goal_terms.append(('AtPose', base_block, base_pose))
        goal_terms.append(('On', base_block, self.table))

        fixed_objs = self.fixed + [b for b in self.pddl_blocks if b != base_block]
        self.pddl_info = get_pddlstream_info(self.robot,
                                             fixed_objs,
                                             self.pddl_blocks,
                                             add_slanted_grasps=False,
                                             approach_frame='global')
        if not solve_joint:
            if not self.teleport:
                goal = tuple(['and'] + goal_terms)
                if not self.use_action_server:
                    plan_found = self._solve_and_execute_pddl(init, goal, search_sample_ratio=1.)
                    if not plan_found: return False, None
                else:
                    self.execute()
                    has_plan = False
                    while not has_plan:
                        plan = self._request_plan_from_server(
                            init, goal, fixed_objs, reset=True)
                        if len(plan) > 0:
                            has_plan = True
                        else:
                            time.sleep(1)
                    ExecuteActions(plan, real=real, pause=True, wait=False)
            else:
                self.teleport_block(base_block, base_pose.pose)
        moved_blocks.add(base_block)

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

            if not solve_joint:
                init = self._get_initial_pddl_state()
                goal_terms = []

            fixed_objs = self.fixed + [b for b in self.pddl_blocks if b != top_pddl]
            self.pddl_info = get_pddlstream_info(self.robot,
                                                 fixed_objs,
                                                 self.pddl_blocks,
                                                 add_slanted_grasps=False,
                                                 approach_frame='global')
            init += [('RelPose', top_pddl, bottom_pddl, rel_tform)]
            goal_terms.append(('On', top_pddl, bottom_pddl))

            moved_blocks.add(top_pddl)
            if not solve_joint:
                if not self.teleport:
                    goal = tuple(['and'] + goal_terms)
                    if not self.use_action_server:
                        plan_found = self._solve_and_execute_pddl(init, goal, search_sample_ratio=1.)
                        if not plan_found: return False, None
                    else:
                        has_plan = False
                        while not has_plan:
                            self.execute()
                            plan = self._request_plan_from_server(
                                init, goal, fixed_objs, reset=False)
                            if len(plan) > 0:
                                has_plan = True
                            else:
                                time.sleep(1)
                        ExecuteActions(plan, real=real, pause=True, wait=False)
                else:
                    get_pose = tamp.primitives.get_stable_gen_block()
                    pose = get_pose(top_pddl, bottom_pddl, poses[-1], rel_tform)[0]
                    poses.append(pose)
                    self.teleport_block(top_pddl, pose.pose)

                # Execute the block placement.
                desired_pose = top_pddl.get_point()
                if not real:
                    self.step_simulation(T, vis_frames=False)
                # TODO: Check if the tower was stable, stop construction if not.
                end_pose = top_pddl.get_point()
                if numpy.linalg.norm(numpy.array(end_pose) - numpy.array(desired_pose)) > 0.01:
                    print('Unstable!')
                    stable = 0.
                    break

        if solve_joint:
            goal = tuple(['and'] + goal_terms)
            if not self.use_action_server:
                plan_found = self._solve_and_execute_pddl(init, goal, search_sample_ratio=1.)
                if not plan_found: return False, None
            else:
                self.execute()
                has_plan = False
                while not has_plan:
                    plan = self._request_plan_from_server(
                        init, goal, fixed_objs, reset=True)
                    if len(plan) > 0:
                        has_plan = True
                    else:
                        time.sleep(1)
                ExecuteActions(plan, real=real, pause=True, wait=False)
        if not real:
            self.step_simulation(T, vis_frames=False)
        if self.use_vision:
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
                                                 approach_frame='global')

            init = self._get_initial_pddl_state()
            init += [('Pose', b, goal_pose),
                     ('Supported', b, goal_pose, self.table, self.table_pose)]
            goal_terms = []
            goal_terms.append(('AtPose', b, goal_pose))
            goal_terms.append(('On', b, self.table))

            goal = tuple(['and'] + goal_terms)
            if not self.use_action_server:
                plan_found = self._solve_and_execute_pddl(init, goal, search_sample_ratio=1.)
                if not plan_found: return False, None
            else:
                self.execute()
                has_plan = False
                while not has_plan:
                    plan = self._request_plan_from_server(
                        init, goal, fixed_objs, reset=True)
                    if len(plan) > 0:
                        has_plan = True
                    else:
                        time.sleep(1)
                ExecuteActions(plan, real=real, pause=True, wait=False)

        return True, stable

    def step_simulation(self, T, vis_frames=False):
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
                length, lifeTime = 0.1, 0.1
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
            print("Planning failed.")
            return False
        else:
            saved_world.restore()
            self.execute()
            ExecuteActions(plan, real=real, pause=True, wait=False)
            self.plan()
            ExecuteActions(plan, real=False, pause=False, wait=False)
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
