import numpy
import pb_robot
import tamp.misc as misc
import pybullet as p

DEBUG_FAILURE = False 

def get_grasp_gen(robot):
    # I opt to use TSR to define grasp sets but you could replace this
    # with your favorite grasp generator
    def gen(body):
        grasp_tsr = pb_robot.tsrs.panda_box.grasp(body)
        # Only use a top grasp (2, 4) and side grasps (7).
        for top_grasp_ix in range(len(grasp_tsr)):#[2, 4, 7, 6, 0, 1]: 
            sampled_tsr = grasp_tsr[top_grasp_ix]
            grasp_worldF = sampled_tsr.sample()

            grasp_objF = numpy.dot(numpy.linalg.inv(body.get_transform()), grasp_worldF)
            body_grasp = pb_robot.vobj.BodyGrasp(body, grasp_objF, robot.arm)
            yield (body_grasp,)
    return gen

def get_stable_gen_table(fixed=[]):
    def gen(body, surface, surface_pose):
        while True:
            pose = pb_robot.placements.sample_placement(body, surface) 

            if (pose is None) or any(pb_robot.collisions.pairwise_collision(body, b) for b in fixed):
                continue
            body_pose = pb_robot.vobj.BodyPose(body, pose)
            yield (body_pose,)
    return gen

def get_stable_gen_block(fixed=[]):
    def fn(body, surface, surface_pose, rel_pose):
        """
        @param rel_pose: A homogeneous transformation matrix.
        """
        surface_tform = pb_robot.geometry.tform_from_pose(surface_pose.pose)
        body_tform = surface_tform@rel_pose
        pose = pb_robot.geometry.pose_from_tform(body_tform)
        body_pose = pb_robot.vobj.BodyPose(body, pose)
        return (body_pose,)
    return fn

def get_ik_fn(robot, fixed=[], num_attempts=10):
    def fn(body, pose, grasp):
        obstacles = fixed + [body]
        obj_worldF = pb_robot.geometry.tform_from_pose(pose.pose)
        grasp_worldF = numpy.dot(obj_worldF, grasp.grasp_objF)
        approach_tform = misc.ComputePrePose(grasp_worldF, [0, 0, -0.05])

        for _ in range(num_attempts):
            q_approach = robot.arm.ComputeIK(approach_tform)
            if (q_approach is None) or not robot.arm.IsCollisionFree(q_approach, obstacles=obstacles):
                continue
            conf = pb_robot.vobj.BodyConf(robot, q_approach)
            q_grasp = robot.arm.ComputeIK(grasp_worldF, seed_q=q_approach)
            if (q_grasp is None) or not robot.arm.IsCollisionFree(q_grasp, obstacles=obstacles):
                continue

            path = robot.snap.PlanToConfiguration(robot.arm, q_approach, q_grasp, obstacles=obstacles)

            if path is None:
                if DEBUG_FAILURE: input('Approach motion failed')
                continue
            command = [pb_robot.vobj.JointSpacePath(robot.arm, path), grasp,
                       pb_robot.vobj.JointSpacePath(robot.arm, path[::-1])]
            return (conf, command)
        return None
    return fn

def assign_fluent_state(fluents):
    obstacles = []
    for fluent in fluents:
        name, args = fluent[0], fluent[1:]
        if name == 'atpose':
            o, p = args
            obstacles.append(o)
            o.set_base_link_pose(p.pose)
        else:
            raise ValueError(name)
    return obstacles

def get_free_motion_gen(robot, fixed=[]):
    def fn(conf1, conf2, fluents=[]):
        obstacles = fixed + assign_fluent_state(fluents)
        path = robot.birrt.PlanToConfiguration(robot.arm, conf1.configuration, conf2.configuration, obstacles=obstacles)

        if path is None:
            if DEBUG_FAILURE: input('Free motion failed')
            return None
        command = [pb_robot.vobj.JointSpacePath(robot.arm, path)]
        return (command,)
    return fn


def get_holding_motion_gen(robot, fixed=[]):
    def fn(conf1, conf2, body, grasp, fluents=[]):
        obstacles = fixed + assign_fluent_state(fluents)
        path = robot.birrt.PlanToConfiguration(robot.arm, conf1.configuration, conf2.configuration, obstacles=obstacles)

        if path is None:
            if DEBUG_FAILURE: input('Holding motion failed')
            return None
        command = [pb_robot.vobj.JointSpacePath(robot.arm, path)]
        return (command,)
    return fn

def get_movable_collision_test(robot):
    def test(command, body, pose):
        body.set_base_link_pose(pose.pose)
        obstacles = [body]
        print('Checking collisions!\n\n\n')
        print(command)
        for motion in command:
            if type(motion) != pb_robot.vobj.JointSpacePath: continue
            for q in motion.path:
                if not robot.arm.IsCollisionFree(q, obstacles=obstacles):
                    if DEBUG_FAILURE: input('Movable collision')
                    return False
                print('HERE')
        return True
    return test


