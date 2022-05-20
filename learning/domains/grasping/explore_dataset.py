import argparse
import enum
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import trimesh
import time
import io
from matplotlib import cm
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid

from learning.domains.grasping.generate_grasp_datasets import graspablebody_from_vector
from pb_robot.planners.antipodalGraspPlanner import GraspSimulationClient

def get_success_per_test_object(dataset_root, n_bins=10):
    p_stables = []
    n = 0
    for ox in range(100):
        fname = os.path.join(dataset_root, 'grasps', 'fitting_phase', f'fit_grasps_train_geo_object{ox}.pkl' )
        with open(fname, 'rb') as handle:
            data = pickle.load(handle)
        p_stable = np.mean(data['grasp_data']['labels'])
        if p_stable >= 0.3 and p_stable <= 0.7:
            n+=1
        print(f'Object {ox}: {p_stable}')
        p_stables.append(p_stable)

    hist = np.histogram(p_stables, bins=n_bins)
    print(hist)
    print(n)
    plt.hist(p_stables, bins=n_bins)
    plt.show()

def inspect_train_dataset(dataset_root):

    train_fname = os.path.join(dataset_root, 'grasps', 'training_phase', 'train_grasps.pkl.tmp')
    with open(train_fname, 'rb') as handle:
        train_data = pickle.load(handle)

    object_names = train_data['object_data']['object_names']
    object_properties = train_data['object_data']['object_properties']

    grasps = train_data['grasp_data']['grasps']
    ids = train_data['grasp_data']['object_ids']
    labels = train_data['grasp_data']['labels']
    p_successes = []
    masses, frictions = [], []
    for ox in range(100):#(len(object_names)):
        ox_ids = np.array(ids)==ox
        ox_labels = np.array(labels)[ox_ids]
        p_success = np.mean(ox_labels)
        p_successes.append(p_success)

        m, f = object_properties[ox][-2:]
        masses.append(m)
        frictions.append(f)

    print(p_successes)
    hist = np.histogram(p_successes, bins=10)
    print(hist)
    plt.hist(p_successes, bins=10)
    plt.xlabel('Object-wise Grasp Success')
    plt.ylabel('Object Counts')
    plt.show()

    plt.scatter(masses, frictions, c=p_successes, cmap=plt.get_cmap('viridis'))
    plt.xlabel('mass')
    plt.ylabel('friction')
    plt.show()


def generate_object_grid(objects, dataset_figpath):
    names = objects['object_data']['object_names']
    properties = objects['object_data']['object_properties']
    
    n_objects = len(names)
    n_property_samples = objects['metadata'].n_property_samples
    n_geoms = n_objects // n_property_samples

    images = []
    for ox, (name, prop) in enumerate(zip(names, properties)):    
        graspable_body = graspablebody_from_vector(name, prop)
        sim_client = GraspSimulationClient(graspable_body, False, 'object_models')

        axis = sim_client._get_tm_com_axis()
        scene = trimesh.scene.Scene([sim_client.mesh, axis])
        scene.set_camera(angles=(np.pi/2, 0, np.pi/4), distance=0.5, center=sim_client.mesh.centroid)
        data = scene.save_image() 
        image = np.array(Image.open(io.BytesIO(data)))
        print(image.shape)
        images.append(image)
        sim_client.disconnect()
    

    for ix in range(0, n_geoms):
        images_for_geom = images[ix*n_property_samples:(ix+1)*n_property_samples]
        names_for_geom = names[ix*n_property_samples:(ix+1)*n_property_samples]
        properties_for_geom = properties[ix*n_property_samples:(ix+1)*n_property_samples]

        plt.clf()        
        fig = plt.figure(figsize=(20., 4.))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                        nrows_ncols=(1, 5),  # creates 2x2 grid of axes
                        axes_pad=0.1,  # pad between axes in inch.
                        )

        for ax, im, name, prop in zip(grid, images_for_geom, names_for_geom, properties_for_geom):
            mass = prop[-2]
            friction = prop[-1]
            n = name.split('::')[1].split('_')[0]
            
            # Iterating over the grid returns the Axes.
            ax.imshow(im)
            ax.text(50, 375, f'{n}\nm={mass: .2f}\nf={friction: .2f}', bbox=dict(fill=False, edgecolor='black', linewidth=1))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        
        fname = os.path.join(dataset_figpath, f'geom{ix}_{n}.png')
        print(fname)
        plt.savefig(fname)
    
def visualize_grasp_dataset(dataset_fname, labels=None, figpath=''):

    with open(dataset_fname, 'rb') as handle:
        val_data = pickle.load(handle)
    dataset_args = val_data['metadata']
    object_names = val_data['object_data']['object_names']
    object_properties = val_data['object_data']['object_properties']
    n_objects = len(object_names)

    grasp_vectors = val_data['grasp_data']['grasps']
    if labels is None:
        labels = val_data['grasp_data']['labels']
    
    for ix in range(n_objects):
        if ix*dataset_args.n_grasps_per_object > len(labels):
            break
        
        graspable_body = graspablebody_from_vector(object_names[ix], object_properties[ix])

        sim_client = GraspSimulationClient(graspable_body, False, 'object_models')
        grasps = []
        for gx in range(dataset_args.n_grasps_per_object):
            grasp_points = [
                grasp_vectors[ix*dataset_args.n_grasps_per_object+gx][0, 0:3],
                grasp_vectors[ix*dataset_args.n_grasps_per_object+gx][1, 0:3],
                grasp_vectors[ix*dataset_args.n_grasps_per_object+gx][2, 0:3]
            ]
            grasps.append(grasp_points)

        obj_labels = labels[ix*dataset_args.n_grasps_per_object:(ix+1)*dataset_args.n_grasps_per_object]

        fname = os.path.join(figpath, f'object{ix}.png')
        if len(figpath) > 0:
             sim_client.tm_show_grasps(grasps, obj_labels, fname=fname)
        else:
            sim_client.tm_show_grasps(grasps, obj_labels)
        sim_client.disconnect()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', type=str, required=True)
    args = parser.parse_args()
    print(args)

    dataset_root = os.path.join('learning', 'data', 'grasping', args.dataset_name)

    dataset_figpath = os.path.join(dataset_root, 'figures')
    if not os.path.exists(dataset_figpath):
        os.mkdir(dataset_figpath)
    object_meshes_path = os.path.join(dataset_figpath, 'object_meshes')
    if not os.path.exists(object_meshes_path):
        os.mkdir(object_meshes_path)


    train_objects_path = os.path.join(dataset_root, 'objects', 'train_geo_train_props.pkl')
    with open(train_objects_path, 'rb') as handle:
        train_objects = pickle.load(handle) 
    train_meshes_path = os.path.join(object_meshes_path, 'train_meshes')
    if not os.path.exists(train_meshes_path):
        os.mkdir(train_meshes_path)
    # generate_object_grid(train_objects, train_meshes_path)

    val_objects_path = os.path.join(dataset_root, 'objects', 'test_geo_test_props.pkl')
    with open(val_objects_path, 'rb') as handle:
        val_objects = pickle.load(handle) 
    val_meshes_path = os.path.join(object_meshes_path, 'val_meshes')
    if not os.path.exists(val_meshes_path):
        os.mkdir(val_meshes_path)
    # generate_object_grid(val_objects, val_meshes_path)

    inspect_train_dataset(dataset_root)

    train_grasps = os.path.join(dataset_root, 'grasps', 'training_phase', 'train_grasps.pkl.tmp')
    figpath = os.path.join(dataset_figpath, 'train_grasps')
    if not os.path.exists(figpath):
        os.mkdir(figpath)
    # visualize_grasp_dataset(train_grasps, figpath=figpath)

    # get_success_per_test_object(dataset_root)


