from mimetypes import init
import numpy as np
import os
import pb_robot

from pb_robot.planners.antipodalGraspPlanner import GraspableBody, GraspSampler


class GraspingAgent:

    def __init__(self, graspable_body, init_pose, use_gui=False):
        self.shapenet_root = '/home/mnosew/workspace/object_models/shapenet-sem/urdfs'

        self.client_id = pb_robot.utils.connect(use_gui=use_gui)
        pb_robot.utils.set_pbrobot_clientid(self.client_id)

        self.robot = pb_robot.panda.Panda()
        self.robot.hand.Open()

        floor_file = 'models/short_floor.urdf'
        self.floor = pb_robot.body.createBody(floor_file)

        self.graspable_body = graspable_body

        object_dataset, object_name = graspable_body.object_name.split('::')
        src_path = os.path.join(self.shapenet_root, f'{object_name}.urdf')
        self.pb_body = pb_robot.body.createBody(src_path)

        self.set_object_pose(init_pose, find_stable_z=True)

    def set_object_pose(self, pose, find_stable_z=True):
        pb_robot.utils.set_pbrobot_clientid(self.client_id)
        self.pb_body.set_base_link_pose(pose)
        if find_stable_z:
            z = pb_robot.placements.stable_z(self.pb_body, self.floor)
            pos, orn = pose
            self.pb_body.set_base_link_pose(((pos[0], pos[1], z), orn))        

    def execute_plan(self, plan):
        pass

    def sample_plan(self, horizon, wait=False):
        init_world = self._save_world()
        plan = []
        for hx in range(horizon):
            grasp = self._sample_grasp_action()
            plan.append(grasp)
            if wait:
                input('Next action?')
            placement = self._sample_place_action(grasp)
            plan.append(placement)
            if wait:
                input('Next action?')

        self._restore_world(init_world)
        return plan

    def _save_world(self):
        pb_robot.utils.set_pbrobot_clientid(self.client_id)
        robot_conf = self.robot.arm.GetJointValues()
        object_pose = self.pb_body.get_base_link_pose()
        return (robot_conf, object_pose)
    
    def _restore_world(self, world):
        pb_robot.utils.set_pbrobot_clientid(self.client_id)
        robot_conf, object_pose = world
        self.robot.arm.SetJointValues(robot_conf)
        self.pb_body.set_base_link_pose(object_pose)

    def _sample_grasp_action(self, max_attempts=100):
        
        for ax in range(0, max_attempts):
            # Step (1): Sample valid antipodal grasp for object.
            sampler = GraspSampler(graspable_body=self.graspable_body, 
                antipodal_tolerance=30,
                show_pybullet=False,
                urdf_directory='urdf_models')
            grasp = sampler.sample_grasp(force=20)
        
            # Step (2): Calculate world transform of gripper.
            pb_robot.utils.set_pbrobot_clientid(self.client_id)
            ee_obj = grasp.ee_relpose
            obj_pose = self.pb_body.get_base_link_pose()

            ee_world = pb_robot.geometry.multiply(obj_pose, ee_obj)
            #pb_robot.viz.draw_point(ee_world[0])
            
            # Step (3): Compute IK.
            ee_world_tform = pb_robot.geometry.tform_from_pose(ee_world)
            q_grasp = self.robot.arm.ComputeIK(ee_world_tform)
            if q_grasp is None:
                continue
            if not self.robot.arm.IsCollisionFree(q_grasp, obstacles=[self.floor]):
                continue
            self.robot.arm.SetJointValues(q_grasp)

            #sampler.sim_client.tm_show_grasp(grasp)
            sampler.disconnect()
            return grasp

    def _sample_place_action(self, grasp, max_attempts=100):

        placement_x_range = (0.3, 0.7)
        placement_y_range = (-0.4, 0.4)

        pb_robot.utils.set_pbrobot_clientid(self.client_id)
        for ax in range(0, max_attempts):

            # Step (1): Sample x, y location on table.
            x = np.random.uniform(*placement_x_range)
            y = np.random.uniform(*placement_y_range)

            # Step (2): Sample object rotation.
            quat = pb_robot.transformations.random_quaternion()
            
            self.pb_body.set_base_link_pose(((x, y, 1.0), quat))
            z = pb_robot.placements.stable_z(self.pb_body, self.floor)
            self.pb_body.set_base_link_pose(((x, y, z), quat))

            # Step (3): Compute world frame of gripper with given grasp.
            ee_obj = grasp.ee_relpose
            obj_pose = self.pb_body.get_base_link_pose()

            ee_world = pb_robot.geometry.multiply(obj_pose, ee_obj)
            # pb_robot.viz.draw_point(ee_world[0])
            
            # Step (4): Check IK.
            ee_world_tform = pb_robot.geometry.tform_from_pose(ee_world)
            q_grasp = self.robot.arm.ComputeIK(ee_world_tform)
            if q_grasp is None:
                continue
            if not self.robot.arm.IsCollisionFree(q_grasp, obstacles=[self.floor]):
                continue
            self.robot.arm.SetJointValues(q_grasp)
            return ((x, y, z), quat)
        


if __name__ == '__main__':
    object_name = 'ShapeNet::Desk_fe2a9f23035580ce239883c5795189ed'
    # object_name = 'ShapeNet::ComputerMouse_379e93edfd0cb9e4cc034c03c3eb69d'
    object_name = 'ShapeNet::WineGlass_2d89d2b3b6749a9d99fbba385cc0d41d'
    # object_name = 'ShapeNet::WallLamp_8be32f90153eb7281b30b67ce787d4d3'    
    # object_name = 'ShapeNet::PersonStanding_3dfe62d56a28f464c17f8c1c27c3df1'
    object_name = 'ShapeNet::DrinkingUtensil_c491d8bc77f4fb1ca4c322790a683350'
    graspable_body = GraspableBody(object_name=object_name, com=(0, 0, 0), mass=0.1, friction=1.0)

    agent = GraspingAgent(graspable_body=graspable_body, 
                          init_pose=((0.4, 0., 1.), (0, 0, 0, 1) ),
                          use_gui=True)
    plan = agent.sample_plan(horizon=10, wait=True)
    agent.execute_plan(plan)

    pb_robot.utils.wait_for_user()