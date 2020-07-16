import numpy
import pybullet as p
import time

from copy import deepcopy

import pb_robot
import tamp.primitives

from actions import PlaceAction, make_platform_world
from agents.panda_agent import PandaAgent
from block_utils import get_adversarial_blocks, rotation_group, ZERO_POS, \
                        Quaternion, get_rotated_block, Pose, add_noise, \
                        Environment, Position 
from particle_belief import ParticleBelief
from pddlstream.algorithms.focused import solve_focused
from pddlstream.utils import INF
from pybullet_utils import transformation
from tamp.misc import setup_panda_world, get_pddlstream_info, ExecuteActions
from tower_planner import TowerPlanner


def test_observations(blocks, block_ix):
    """
    Test method to try placing the given blocks on the platform.
    """
    agent = PandaAgent(blocks, NOISE)
    for r in list(rotation_group())[0:]:
        action = PlaceAction(pos=None,
                             rot=r,
                             block=blocks[block_ix])
        obs = agent.simulate_action(action, block_ix, T=50)
        
        # TODO: Check that the simulated observation agrees with the true observation.
        particle_block = deepcopy(blocks[block_ix])
        world = make_platform_world(particle_block, action)
        env = Environment([world], vis_sim=False)

        if False:
            env.step(action=action)
            length, lifeTime = 0.2, 0.0
            pos = end_pose = world.get_pose(world.objects[1])[0]
            quat = action.rot.as_quat()
            new_x = transformation([length, 0.0, 0.0], pos, quat)
            new_y = transformation([0.0, length, 0.0], pos, quat)
            new_z = transformation([0.0, 0.0, length], pos, quat)

            p.addUserDebugLine(pos, new_x, [1,0,0], lifeTime=lifeTime)
            p.addUserDebugLine(pos, new_y, [0,1,0], lifeTime=lifeTime)
            p.addUserDebugLine(pos, new_z, [0,0,1], lifeTime=lifeTime)
            input('Continue')

        for _ in range(50):
            env.step(action=action)
        
        end_pose = world.get_pose(world.objects[1])

        

        print('Simulated Pose:', end_pose)
        print('TAMP Pose:', obs[2])

        input('Continue?')

def test_place_action(blocks, block_ix):
    """
    Test method to try placing the given blocks on the platform.
    """
    agent = PandaAgent(blocks, NOISE)
    for r in list(rotation_group())[0:]:
        
        action = PlaceAction(pos=None,
                             rot=r,
                             block=blocks[block_ix])
        agent.simulate_action(action, block_ix)
        # p.disconnect()
        # break

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

def check_ungraspable_block(agent):

    platform_pose = agent.platform.get_base_link_pose()
    for pddl_block in agent.pddl_blocks:
        block_pose = pddl_block.get_base_link_pose()


def test_return_to_start(blocks, n_placements=5, rot_ix=0, block_ix=1):
    """
    Let a block fall off the platform and check that we can successfully 
    pick it up and return it to the starting position.
    """
    numpy.random.seed(10)
    rot = list(rotation_group())[rot_ix]
    for _ in range(n_placements):
        # Create agent.
        agent = PandaAgent(blocks)
        original_pose = agent.pddl_blocks[block_ix].get_base_link_pose()
        # Create a random action.
        new_dims = numpy.abs(rot.apply(blocks[block_ix].dimensions))
        place_pos = new_dims*(-0.5*numpy.random.rand(3))
        x, y, _ = place_pos + numpy.array(agent.platform.get_dimensions())/2
        action = PlaceAction(pos=None,
                             rot=rot,
                             block=blocks[block_ix])

        # Teleport block to platform.
        blocks[block_ix].set_pose(Pose(ZERO_POS, Quaternion(*action.rot.as_quat())))
        rotated_block = get_rotated_block(blocks[block_ix])
        platform_pose = agent.platform.get_base_link_pose()
        platform_tform = pb_robot.geometry.tform_from_pose(platform_pose)
        z = agent.platform.get_dimensions()[2]/2 + rotated_block.dimensions[2]/2 + 1e-5
        tform = numpy.array([[1., 0., 0., action.pos[0]],
                             [0., 1., 0., action.pos[1]],
                             [0., 0., 1., z],
                             [0., 0., 0., 1.]])
        tform[0:3, 0:3] = action.rot.as_matrix()
        body_tform = platform_tform@tform
        pose = pb_robot.geometry.pose_from_tform(body_tform)        

        agent.pddl_blocks[block_ix].set_base_link_pose(pose)

        # Execute action.
        p.setGravity(0,0,-10)
        for _ in range(500):
            p.stepSimulation()
            time.sleep(0.01)

        check_ungraspable_block(agent)

        # Solve PDDL Problem.
        pddl_block = agent.pddl_blocks[block_ix]
        init = agent._get_initial_pddl_state()
        goal_pose = pb_robot.vobj.BodyPose(pddl_block, original_pose)
        init += [('Pose', pddl_block, goal_pose),
                 ('Supported', pddl_block, goal_pose, agent.table, agent.table_pose)]
        goal = ('and', ('AtPose', pddl_block, goal_pose), 
                       ('On', pddl_block, agent.table))
        
        # Solve the PDDLStream problem.
        print('Init:', init)
        print('Goal:', goal)
        agent._solve_and_execute_pddl(init, goal)

        p.disconnect()

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

def test_tower_simulation(blocks):
    agent = PandaAgent(blocks, NOISE)

    for b_ix, block in enumerate(blocks):
        belief = ParticleBelief(block, 
                                N=200, 
                                plot=False, 
                                vis_sim=False,
                                noise=NOISE)
        block.com_filter = belief.particles

    tp = TowerPlanner()
    tallest_tower = tp.plan(blocks, num_samples=10)

    # and visualize the result
    agent.simulate_tower(tallest_tower, vis=True, T=250)

NOISE=0.00005
if __name__ == '__main__':
    # TODO: Test the panda agent.
    blocks = get_adversarial_blocks()

    # agent = PandaAgent(blocks)
    # visualize_grasps(agent, blocks)
    #test_table_pose_ik(agent, blocks)
    # test_placement_ik(agent, blocks)
    # input('Continue?')
    #test_place_action(blocks, 3)
    #test_observations(blocks, 1)
    #test_return_to_start(blocks, rot_ix=3)
    #test_placement_on_platform(agent)
    test_tower_simulation(blocks)
    time.sleep(5.0)
    pb_robot.utils.disconnect()