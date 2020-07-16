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
from pddlstream.algorithms.focused import solve_focused
from pddlstream.utils import INF
from pybullet_utils import transformation
from tamp.misc import setup_panda_world, get_pddlstream_info, ExecuteActions


class PandaAgent:
    def __init__(self, blocks, noise, teleport=False):
        """
        Build the Panda world in PyBullet and set up the PDDLStream solver.
        The Panda world should in include the given blocks as well as a 
        platform which can be used in experimentation.
        :param teleport: Debugging parameter used to skip planning while moving
                         blocks around this world.
        """
        # TODO: Check that having this as client 0 is okay when interacting 
        # with everything else.
        self.client_id = pb_robot.utils.connect(use_gui=True)
        pb_robot.utils.set_default_camera()

        self.robot = pb_robot.panda.Panda()
        self.robot.arm.hand.Open()

        self.belief_blocks = blocks

        self.pddl_blocks, self.platform_table, self.platform_leg, self.table = setup_panda_world(self.robot, blocks)
        self.pddl_info = get_pddlstream_info(self.robot, [self.table, self.platform_table, self.platform_leg], self.pddl_blocks)

        self.noise = noise
        self.teleport = teleport

    def _get_initial_pddl_state(self):
        """
        Get the PDDL representation of the world between experiments. This
        method assumes that all blocks are on the table. We will always "clean
        up" an experiment by moving blocks away from the platform after an 
        experiment.
        """
        fixed = [self.table, self.platform_table, self.platform_leg]
        conf = pb_robot.vobj.BodyConf(self.robot, self.robot.arm.GetJointValues())
        init = [('CanMove',),
                ('Conf', conf),
                ('AtConf', conf),
                ('HandEmpty',)]

        for body in self.pddl_blocks:
            print(type(body), body)
            pose = pb_robot.vobj.BodyPose(body, body.get_base_link_pose())
            init += [('Graspable', body),
                    ('Pose', body, pose),
                    ('AtPose', body, pose),
                    ('Block', body),
                    ('On', body, self.table)]


        self.table_pose = pb_robot.vobj.BodyPose(self.table, self.table.get_base_link_pose())
        init += [('Pose', self.table, self.table_pose), ('AtPose', self.table, self.table_pose)]

        self.platform_pose = pb_robot.vobj.BodyPose(self.platform_table, self.platform_table.get_base_link_pose())
        init += [('Pose', self.platform_table, self.platform_pose), ('AtPose', self.platform_table, self.platform_pose)]

        init += [('Table', self.table), ('Block', self.platform_table)]
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
        real_block = self.belief_blocks[block_ix]
        pddl_block = self.pddl_blocks[block_ix]

        original_pose = pddl_block.get_base_link_pose()

        # Set up the PDDLStream problem for the placing the given block on the
        # platform with the specified action.
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
            self._solve_and_execute_pddl(init, goal, max_time=30., search_sample_ratio=1000)
        else:
            goal_pose_fn = tamp.primitives.get_stable_gen_block()
            goal_pose = goal_pose_fn(pddl_block, 
                                     self.platform_table, 
                                     self.platform_pose,
                                     tform)[0].pose
            pddl_block.set_base_link_pose(goal_pose)

        # Execture the action. 
        # TODO: Check gravity compensation in the arm.
        p.setGravity(0,0,-10)
        for tx in range(500):
            p.stepSimulation()
            time.sleep(0.01)

            # Save the result of the experiment after T steps of simulation.
            if tx == T-1:
                end_pose = self._get_observed_pose(pddl_block, action)
                observation = (action, T, end_pose)
        
        # Put block back in original position.

        # TODO: Check if block is on the table or platform to start.         self.pddl_info = get_pddlstream_info(self.robot, [self.table, self.platform], self.pddl_blocks)
        self.pddl_info = get_pddlstream_info(self.robot, [self.table, self.platform_table, self.platform_leg], self.pddl_blocks)

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
            success = self._solve_and_execute_pddl(init, goal, max_time=30., search_sample_ratio=1000)
            if not success:
                print('Plan failed: Teleporting block to intial position.')
                pddl_block.set_base_link_pose(original_pose)
        else:
            pddl_block.set_base_link_pose(original_pose)

        return observation

    def simulate_tower(self, tower, vis, T, save_tower=False):
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

        init = self._get_initial_pddl_state()
        goal_terms = []

        # Unrotate all blocks and build a map to PDDL. (i.e., use the block.rotation for orn)
        pddl_block_lookup = {}
        for block in tower:
            for pddl_block in self.pddl_blocks:
                if block.name in pddl_block.get_name():
                    pddl_block_lookup[block] = pddl_block

        # TODO: Set base block to be rotated in its current position.
        base_block = pddl_block_lookup[tower[0]]
        base_pos = base_block.get_base_link_pose()[0]
        table_z = self.table_pose.pose[0][2] + 1e-5
        base_pose = ((base_pos[0], base_pos[1], table_z + tower[0].pose.pos.z), tower[0].rotation)

        print(base_pose)
        # base_block.set_base_link_pose(base_pose)
        input('Continue?')

        base_pose = pb_robot.vobj.BodyPose(pddl_block, base_pose)
        init += [('Pose', base_block, base_pose),
                 ('Supported', base_block, base_pose, self.table, self.table_pose)]
        goal_terms.append(('AtPose', base_block, base_pose)) 
        goal_terms.append(('On', base_block, self.table))

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
            top_pddl = pddl_block_lookup[top_block]
            bottom_pddl = pddl_block_lookup[bottom_block]

            init = self._get_initial_pddl_state()
            init += [('RelPose', top_pddl, bottom_pddl, rel_tform)]
            goal_terms = []
            goal_terms.append(('On', top_pddl, bottom_pddl))
            # get_pose = tamp.primitives.get_stable_gen_block()
            # pose = get_pose(top_pddl, bottom_pddl, poses[-1], rel_tform)[0]
            # poses.append(pose)
            # #print(base_pose.pose)
            # print(pose.pose)
            # base_block.set_base_link_pose(base_pose.pose)
            # top_pddl.set_base_link_pose(pose.pose)
            # input('Continue?')

            # TODO: Build PDDL Goal Spec with RelPose between all blocks.
            goal = tuple(['and'] + goal_terms)
            self._solve_and_execute_pddl(init, goal, search_sample_ratio=1000.)


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

    def _solve_and_execute_pddl(self, init, goal, max_time=INF, search_sample_ratio=0.):
        self.robot.arm.hand.Open()
        saved_world = pb_robot.utils.WorldSaver()

        pddlstream_problem = tuple([*self.pddl_info, init, goal])
        plan, _, _ = solve_focused(pddlstream_problem, 
                                   success_cost=numpy.inf, 
                                   search_sample_ratio=search_sample_ratio,
                                   max_time=max_time)

        # Execute the PDDLStream solution to setup the world.
        if plan is None:
            print("No plan found")
            return False
        else:
            # TODO: Have this execute instead of prompt for input.
            saved_world.restore()
            ExecuteActions(self.robot.arm, plan, pause=False)
            return True
