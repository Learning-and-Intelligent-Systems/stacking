import numpy
import pybullet as p
import time

import pb_robot
import tamp.primitives

from actions import PlaceAction
from block_utils import get_adversarial_blocks, rotation_group, ZERO_POS, \
                        Quaternion, get_rotated_block, Pose
from pddlstream.algorithms.focused import solve_focused
from pybullet_utils import transformation
from tamp.misc import setup_panda_world, get_pddlstream_info, ExecuteActions


class PandaAgent:
    def __init__(self, blocks):
        """
        Build the Panda world in PyBullet and set up the PDDLStream solver.
        The Panda world should in include the given blocks as well as a 
        platform which can be used in experimentation.
        """
        # TODO: Check that having this as client 0 is okay when interacting 
        # with everything else.
        self.client_id = pb_robot.utils.connect(use_gui=True)
        pb_robot.utils.set_default_camera()

        self.robot = pb_robot.panda.Panda()
        self.robot.arm.hand.Open()

        self.belief_blocks = blocks

        self.pddl_blocks, self.platform, self.table = setup_panda_world(self.robot, blocks)
        self.pddl_info = get_pddlstream_info(self.robot,[self.table, self.platform], self.pddl_blocks)

    def _get_initial_pddl_state(self):
        """
        Get the PDDL representation of the world between experiments. This
        method assumes that all blocks are on the table. We will always "clean
        up" an experiment by moving blocks away from the platform after an 
        experiment.
        """
        fixed = [self.table, self.platform]
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

        for surface in fixed:
            pose = pb_robot.vobj.BodyPose(surface, surface.get_base_link_pose())
            init += [('Pose', surface, pose), ('AtPose', surface, pose)]
        init += [('Table', self.table), ('Block', self.platform)]
        return init

    def simulate_action(self, action, real_block, T=50, vis_sim=False, vis_placement=False):
        """
        Perform the given action to with the given block. An observation 
        should be returned in the reference frame of the platform.
        :param action: Place action which describes the relative pose of the block to the platform surface.
        :param real_block: Belief representation of the block to perform the action on.
        :param T: How many timesteps to simulate the block falling for.
        :param vis_sim: Ununsed.
        :return: (action, T, end_pose) End pose should be TODO: what frame?
        """
        #real_block = self.belief_blocks[block_ix]
        pddl_block = self.pddl_blocks[0]

        # Set up the PDDLStream problem for the placing the given block on the
        # platform with the specified action.
        init = self._get_initial_pddl_state()

        #  Figure out the correct transformation matrix based on the action.
        real_block.set_pose(Pose(ZERO_POS, Quaternion(*action.rot.as_quat())))
        rotated_block = get_rotated_block(real_block)
        
        x = action.pos[0]
        y = action.pos[1]
        z = self.platform.get_dimensions()[2]/2 + rotated_block.dimensions[2]/2 + 1e-5
        tform = numpy.array([[1., 0., 0., x],
                             [0., 1., 0., y],
                             [0., 0., 1., z],
                             [0., 0., 0., 1.]])
        tform[0:3, 0:3] = action.rot.as_matrix()

        # Code to visualize where the block will be placed.
        if vis_placement:
            surface_tform = pb_robot.geometry.tform_from_pose(self.platform.get_base_link_pose())
            body_tform = surface_tform@tform
            length, lifeTime = 0.2, 0.0
            
            pos, quat = pb_robot.geometry.pose_from_tform(body_tform)
            new_x = transformation([length, 0.0, 0.0], pos, quat)
            new_y = transformation([0.0, length, 0.0], pos, quat)
            new_z = transformation([0.0, 0.0, length], pos, quat)

            p.addUserDebugLine(pos, new_x, [1,0,0], lifeTime=lifeTime)
            p.addUserDebugLine(pos, new_y, [0,1,0], lifeTime=lifeTime)
            p.addUserDebugLine(pos, new_z, [0,0,1], lifeTime=lifeTime)
        
        init += [('RelPose', pddl_block, self.platform, tform)]
        goal = ('On', pddl_block, self.platform)

        # Solve the PDDLStream problem.
        print('Init:', init)
        print('Goal:', goal)

        self.robot.arm.hand.Open()
        saved_world = pb_robot.utils.WorldSaver()

        pddlstream_problem = tuple([*self.pddl_info, init, goal])
        plan, cost, evaluations = solve_focused(pddlstream_problem, success_cost=numpy.inf, search_sample_ratio=1000.)

        # Execute the PDDLStream solution to setup the world.
        if plan is None:
            print("No plan found")
        else:
            # TODO: Have this execute instead of prompt for input.
            saved_world.restore()
            input("Execute?")

            ExecuteActions(self.robot.arm, plan)

        # TODO: Execture the action.

        # TODO: Move the block back to the other side of the table.

    def simulate_tower(self):
        pass

def test_place_action(agent, blocks, block_ix):
    """
    Test method to try placing the given blocks on the platform.
    """
    for r in list(rotation_group())[4:]:
        action = PlaceAction(pos=None,
                             rot=r,
                             block=blocks[block_ix])
        agent.simulate_action(action, blocks[block_ix])
        break

def test_placement_ik(agent, blocks):
    """
    To make sure that the platform is in a good position, make sure the
    IK is feasible for some grasp position.
    """
    get_block_pose = tamp.primitives.get_stable_gen_block([agent.table, agent.platform])
    get_grasp = tamp.primitives.get_grasp_gen(agent.robot)
    get_ik = tamp.primitives.get_ik_fn(agent.robot, [agent.platform, agent.table])
    
    for r in list(rotation_group()):
        r = list(rotation_group())[4]
        action = PlaceAction(pos=None,
                             rot=r,
                             block=blocks[0])
        blocks[0].set_pose(Pose(ZERO_POS, Quaternion(*action.rot.as_quat())))
        rotated_block = get_rotated_block(blocks[0])
        x = action.pos[0]
        y = action.pos[1]
        z = agent.platform.get_dimensions()[2]/2 + rotated_block.dimensions[2]/2 + 1e-5
        tform = numpy.array([[1., 0., 0., x],
                             [0., 1., 0., y],
                             [0., 0., 1., z],
                             [0., 0., 0., 1.]])
        tform[0:3, 0:3] = action.rot.as_matrix()

        platform_pose = pb_robot.vobj.BodyPose(agent.platform, 
                                               agent.platform.get_base_link_pose())
        
        start_pose = pb_robot.vobj.BodyPose(agent.pddl_blocks[0], 
                                            agent.pddl_blocks[0].get_base_link_pose())
        placement_pose = get_block_pose(agent.pddl_blocks[0], 
                              agent.platform, 
                              platform_pose, 
                              tform)[0]

        ik_found = False
        for grasp in get_grasp(agent.pddl_blocks[0]):
            ik_start = get_ik(agent.pddl_blocks[0], start_pose, grasp[0])
            ik_placement = get_ik(agent.pddl_blocks[0], placement_pose, grasp[0])
            
            if ik_start is not None:
                print('Y', end='')
            else:
                print('N', end='')
            
            if ik_placement is not None:
                ik_found = True
                print('Y', end=' ')
            else:
                print('N', end=' ')

        if ik_found:
            print('Found IK.')
        else:
            print('No IK.')

        break

def test_table_pose_ik(agent, blocks):
    """
    To make sure that the platform is in a good position, make sure the
    IK is feasible for some grasp position.
    """
    get_block_pose = tamp.primitives.get_stable_gen_table([agent.table, agent.platform])
    get_grasp = tamp.primitives.get_grasp_gen(agent.robot)
    get_ik = tamp.primitives.get_ik_fn(agent.robot, [agent.platform, agent.table])
    
    for r in list(rotation_group())[5:]:
        
        table_pose = pb_robot.vobj.BodyPose(agent.table, 
                                            agent.table.get_base_link_pose())
        
        start_pose = pb_robot.vobj.BodyPose(agent.pddl_blocks[0], 
                                            agent.pddl_blocks[0].get_base_link_pose())
        placement_pose = next(get_block_pose(agent.pddl_blocks[0], 
                                             agent.table, 
                                             table_pose, rotation=r))[0]

        ik_found = False
        for grasp in get_grasp(agent.pddl_blocks[0]):
            ik_start = get_ik(agent.pddl_blocks[0], start_pose, grasp[0])
           
            
            if ik_start is not None:
                print('Y', end='')
                agent.robot.arm.SetJointValues(ik_start[0].configuration)
                import time
                time.sleep(5)
            else:
                print('N', end='')
                #continue
            ik_placement = get_ik(agent.pddl_blocks[0], placement_pose, grasp[0])
            if ik_placement is not None:
                ik_found = True
                print('Y', end=' ')
                agent.robot.arm.SetJointValues(ik_placement[0].configuration)
                import time
                time.sleep(5)
            else:
                print('N', end=' ')
        if ik_found:
            print('Found IK.')
        else:
            print('No IK.')


def visualize_grasps(agent, blocks):
    """
    Test method to visualize the grasps. Helps check for potential problems
    due to collisions with table or underspecified grasp set.
    """
    get_block_pose = tamp.primitives.get_stable_gen_table([agent.table, agent.platform])
    get_grasp = tamp.primitives.get_grasp_gen(agent.robot)
    get_ik = tamp.primitives.get_ik_fn(agent.robot, [agent.platform, agent.table])
    
    for r in list(rotation_group())[5:]:
        table_pose = pb_robot.vobj.BodyPose(agent.table, 
                                            agent.table.get_base_link_pose())
        pose = agent.pddl_blocks[0].get_base_link_pose()
        pose = ((pose[0][0], pose[0][1], pose[0][2] + 0.2), pose[1])
        start_pose = pb_robot.vobj.BodyPose(agent.pddl_blocks[0], 
                                            pose)
        agent.pddl_blocks[0].set_base_link_pose(pose)
        ix = 0
        for grasp in list(get_grasp(agent.pddl_blocks[0])):
            ik_start = get_ik(agent.pddl_blocks[0], start_pose, grasp[0])
            import time
            if ik_start is not None:
                print(ix, 'Y')
                agent.robot.arm.SetJointValues(ik_start[0].configuration)
                import time
                time.sleep(2)
            else:
                print(ix, 'N')
            ix += 1



def test_placement_on_platform(agent):
    """
    Tests setting up a PDDL problem with each block on the platform.
    Intended to ensure the given domain/stream can complete the experiment
    setup.
    """
    init = agent._get_initial_pddl_state()
    
    # Create the problem for one block on the platform.
    tform = numpy.array([[1., 0., 0., 0.],
                         [0., 1., 0., 0.],
                         [0., 0., 1., 0.1],
                         [0., 0., 0., 1.]])
    init += [('RelPose', agent.pddl_blocks[1], agent.platform, tform)]
    goal = ('On', agent.pddl_blocks[1], agent.platform)

    print('Blocks:', [b.get_name() for b in agent.pddl_blocks])
    print('Init:', init)
    print('Goal:', goal)
    agent.robot.arm.hand.Open()
    saved_world = pb_robot.utils.WorldSaver()

    pddlstream_problem = tuple([*agent.pddl_info, init, goal])
    plan, cost, evaluations = solve_focused(pddlstream_problem, success_cost=numpy.inf)

    if plan is None:
        print("No plan found")
    else:
        saved_world.restore()
        input("Execute?")
        ExecuteActions(agent.robot.arm, plan)

if __name__ == '__main__':
    # TODO: Test the panda agent.
    blocks = get_adversarial_blocks()

    agent = PandaAgent(blocks)
    #visualize_grasps(agent, blocks)
    #test_table_pose_ik(agent, blocks)
    test_placement_ik(agent, blocks)
    input('Continue?')
    test_place_action(agent, blocks, 0)
    #test_placement_on_platform(agent)

    time.sleep(5.0)
    pb_robot.utils.disconnect()