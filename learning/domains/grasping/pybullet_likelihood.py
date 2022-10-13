from learning.domains.grasping.active_utils import sample_unlabeled_data
import numpy as np
import pickle

from block_utils import ParticleDistribution
from learning.domains.grasping.active_utils import sample_unlabeled_data
from learning.domains.grasping.generate_grasp_datasets import vector_from_graspablebody
from pb_robot.planners.antipodalGraspPlanner import Grasp, GraspableBodySampler, ParallelGraspStabilityChecker



class PBLikelihood:

    def __init__(self, object_name, n_samples, batch_size):
        self.object_name = object_name
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.bodies_for_particles = None

    def eval(self):
        pass

    def init_particles(self, N):
        graspable_bodies = []
        graspable_vectors = []
        for px in range(N):
            graspable_bodies.append(GraspableBodySampler.sample_random_object_properties(self.object_name))
            graspable_vectors.append(vector_from_graspablebody(graspable_bodies[-1]))

        self.bodies_for_particles = graspable_bodies
        particle_dist = ParticleDistribution(np.array(graspable_vectors), np.ones(len(graspable_vectors)))
        return particle_dist

    def get_particle_likelihoods(self, particles, observation):
        if self.bodies_for_particles is None:
            print('[ERROR] Need to initialize particles first.')
            return

        tgrasp = observation['grasp_data']['raw_grasps'][0]
        
        labels = []
        n_batches =int(np.ceil(len(particles)/self.batch_size))
        for bx in range(n_batches):
            bodies = self.bodies_for_particles[bx*self.batch_size:(bx+1)*self.batch_size]
            grasps = []
            for body in bodies:
                g = Grasp(body, tgrasp.pb_point1, tgrasp.pb_point2, tgrasp.pitch, tgrasp.roll, tgrasp.ee_relpose, tgrasp.force)
                grasps.append(g)

            batch_labels = []
            labeler = ParallelGraspStabilityChecker(bodies, grasp_noise=0.0025)
            for sx in range(self.n_samples):
                batch_labels.append(labeler.get_labels(grasps))
            batch_labels = np.array(batch_labels).mean(axis=0).tolist()
            labels += batch_labels
            labeler.disconnect()

        return np.array(labels)


if __name__ == '__main__':
    with open('learning/data/grasping/train-sn100-test-sn10/objects/test_geo_test_props.pkl', 'rb') as handle:
        all_object_set = pickle.load(handle)['object_data']
    
    object_ix = 0
    object_set = {
        'object_names': [all_object_set['object_names'][object_ix]],
        'object_properties': [all_object_set['object_properties'][object_ix]],
    }

    object_name = object_set['object_names'][object_ix]
    likelihood = PBLikelihood(object_name=object_name, n_samples=5, batch_size=50)
    particles = likelihood.init_particles(100)

    grasp = sample_unlabeled_data(1, object_set)
    probs = likelihood.get_particle_likelihoods(particles.particles, grasp)
    print(probs)