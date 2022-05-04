"""
Make sure objects can be grasped.
"""
import os

from pb_robot.planners.antipodalGraspPlanner import GraspStabilityChecker, GraspableBody, GraspSampler
    
URDFS_PATH = '/home/mnosew/workspace/object_models/shapenet-sem/urdfs'
if __name__ == '__main__':
    # with open('learning/data/grasping/object_lists/non-watertight.txt', 'r') as handle:
    #     non_watertight = handle.read().split('\n')
    # with open('learning/data/grasping/object_lists/no-valid-grasps.txt', 'r') as handle:
    #     no_grasps = handle.read().split('\n')
    # print(non_watertight)

    non_watertight = []
    no_grasps = []

    object_names = []
    for urdf_path in os.listdir(URDFS_PATH):
        object_name = os.path.splitext(urdf_path)[0]
        model_name = f'ShapeNet::{object_name}'
        object_names.append(model_name)

     #caught_up = False
    for ox, object_name in enumerate(object_names):
        # if object_name == non_watertight[-1]:
        #     caught_up = True
        # print(object_name)
        # if not caught_up:
        #     print(f'Skipping: {ox}, {object_name}')
        #     continue
        
        graspable_body = GraspableBody(object_name=object_name, com=(0, 0, 0), mass=0.1, friction=1.0)
    
        grasp_sampler = GraspSampler(graspable_body=graspable_body, antipodal_tolerance=30, show_pybullet=False)

        if not grasp_sampler.sim_client.mesh.is_watertight:
            print(f'Mesh {ox} not watertight [{object_name}]')
            non_watertight.append(object_name)
            grasp_sampler.disconnect()
            continue
        
        for attempt in range(3):
            grasp = grasp_sampler.sample_grasp(force=20, show_trimesh=False)
            
            if grasp is None:
                print(f'No valid antipodal grasp found for mesh {ox} on attempt {attempt} [{object_name}]')
                no_grasps.append(object_name)
                break

        grasp_sampler.disconnect()

        with open('learning/data/grasping/object_lists/non-watertight.txt', 'w') as handle:
            handle.write('\n'.join(non_watertight))
        with open('learning/data/grasping/object_lists/no-valid-grasps.txt', 'w') as handle:
            handle.write('\n'.join(no_grasps))
        
    with open('learning/data/grasping/object_lists/non-watertight.txt', 'w') as handle:
            handle.write('\n'.join(non_watertight))
    with open('learning/data/grasping/object_lists/no-valid-grasps.txt', 'w') as handle:
        handle.write('\n'.join(no_grasps))

    print(non_watertight)