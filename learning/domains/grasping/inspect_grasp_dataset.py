import os
import random
from pybullet_object_models import ycb_objects
from PIL import Image

from pb_robot.planners.antipodalGraspPlanner import GraspableBodySampler, GraspSampler, GraspSimulationClient, GraspStabilityChecker


def generate_images_for_object(ycb_name, path, mass=None, friction=None, com=None, n_samples=50):
    labeler = GraspStabilityChecker(stability_direction='all', label_type='relpose')

    graspable_body = GraspableBodySampler.sample_random_object_properties(ycb_name, mass=mass, friction=friction, com=com)
    grasp_sampler = GraspSampler(graspable_body=graspable_body, antipodal_tolerance=30, show_pybullet=False)
    grasps = []
    for lx in range(0, n_samples):
        print('Sampling %d/%d...' % (lx, n_samples))
        grasp = grasp_sampler.sample_grasp(force=20, show_trimesh=False)
        grasps.append(grasp)
    grasp_sampler.disconnect()
    
    labels = []
    for lx, grasp in enumerate(grasps):
        print('Labeling %d/%d...' % (lx, n_samples))
        labels.append(labeler.get_label(grasp, show_pybullet=False))

    sim_client = GraspSimulationClient(graspable_body, show_pybullet=False, urdf_directory='object_models')
    sim_client.tm_show_grasps(grasps, labels, fname=path)
    sim_client.disconnect()


if __name__ == '__main__':
    ycb_name = 'YcbHammer'
    masses = [0.1, 0.5, 1.0, 1.5, 2.0]
    frictions = [0.1, 0.5, 1.0]
    n_com = 5

    for m in masses:
        for f in frictions:
            for cx in range(n_com):
                fname = 'learning/domains/grasping/images/%s_%.2fm_%.2ff_%d.png' % (ycb_name, m, f, cx)
                print('Working on %s...' % fname)
                generate_images_for_object(ycb_name, fname, m, f, n_samples=50)
