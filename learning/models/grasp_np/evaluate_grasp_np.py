import argparse
import numpy as np
import os
import pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from pb_robot.planners.antipodalGraspPlanner import (
    GraspSimulationClient,
    graspablebody_from_vector
)

def get_object_metadata(dataset_name):
    """ Return interesting properties of the objects specified by the dataset. """

    dataset_path = os.path.join(
        'learning/data/grasping',
        dataset_name
    )

    objects_fname = os.path.join(dataset_path, 'objects', 'train_geo_train_props.pkl')
    with open(objects_fname, 'rb') as handle:
        objects = pickle.load(handle)

    ignore_path = os.path.join(dataset_path, 'ignore.txt')
    ignore_objects = []
    with open(ignore_path, 'r') as handle:
        for l in handle.readlines():
            if 'train' in l:
                ignore_objects.append(int(l.strip().split(',')[1]))

    object_names = objects['object_data']['object_names']
    object_properties = objects['object_data']['object_properties']
    object_metadata = []

    geom_props_cache = {}
    count = 0
    for ox, (name, props) in enumerate(zip(object_names, object_properties)):
        if ox in ignore_objects:
            continue
        print(name, props)
        graspable_body = graspablebody_from_vector(
            object_name=name,
            vector=props
        )
        if name in geom_props_cache:
            geom_props = geom_props_cache[name]
        else:
            client = GraspSimulationClient(
                graspable_body=graspable_body,
                show_pybullet=False,
            )

            geom_props = {
                'volume': client.mesh.volume
            }
            geom_props_cache[name] = geom_props
            # if client.mesh.volume < 0.0001:
            #     print(geom_props)
            #     input('Next?')

            client.disconnect()
        object_metadata.append(
            {
                'geometric': geom_props,
                'dynamic': {
                    'com': props[0:3],
                    'friction': props[4],
                    'mass': props[3]
                }
            }
        )

        count += 1

        # if count > 20: break

    return object_metadata


def compare_volume(metadata):
    with open('learning/experiments/metadata/grasp_np/results_val.pkl', 'rb') as handle:
        y_probs, target_ys = pickle.load(handle)
    per_obj_probs = y_probs.reshape(-1, 10)
    per_obj_target = target_ys.reshape(-1, 10)
    per_obj_acc = ((per_obj_probs > 0.5) == per_obj_target).mean(axis=1)

    target_rate = per_obj_target.mean(axis=1)
    import IPython; IPython.embed()
    print(per_obj_acc.mean())
    for ox in range(len(metadata)):
        # if target_rate[ox] > 0.1 and target_rate[ox] < 0.9:
        # plt.scatter(metadata[ox]['geometric']['volume'], per_obj_acc[ox], c='b', alpha=0.05)
        plt.scatter(
            metadata[ox]['dynamic']['friction'],
            per_obj_acc[ox],
            c='b', alpha=0.05
        )

    plt.show()
    # for ox in range(len(metadata)):

    #     print(ox)
    #     if ox % 10 == 0 and ox > 1:
    #         plt.ylim((0, 1))
    #         plt.show()
    #         plt.clf()
    #     plt.scatter(target_rate[ox], per_obj_acc[ox], c='b', alpha=0.1)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name')
    args = parser.parse_args()

    object_metadata = get_object_metadata(dataset_name=args.dataset_name)

    compare_volume(object_metadata)
    volumes = [o['geometric']['volume'] for o in object_metadata]
    v_hist, v_edges = np.histogram(volumes, bins=50)

    plt.hist(volumes, bins=25)
    plt.show()