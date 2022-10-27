# Grasping Data

In this work, we develop a grasp feasibility model that applies to objects with arbitrary geometry and varying physical properties (e.g., CoM, friction, mass). For simulation we use PyBullet with geometries from ShapeNet-Sem or YCB. However these objects cannot be simulated in their default form by PyBullet and care must be taken to ensure reliable labels. This document describes what I have learned from making ShapeNet-Sem models compatible with the PyBullet simulator.

## Objects

### Primitive Objects
We can generate datasets with URDF primitives using [generate_primitive_data.py](../../learning/domains/grasping/generate_primitive_data.py)
as a way to debug or sanity check our results. Since these objects are URDF primitives
(spheres, cylinders, and cubes), we rely on Pybullet and Trimesh's internal primitives and do not
include any mesh representation.

### YCB Objects

### ShapeNet-Sem Objects

Each object is represented by two meshes:
- A collision mesh that is a collection of convex geometries used for simulation.
- A visual mesh that is watertight used for grasp sampling and point-cloud generation.
More details about the motivation and generation of each mesh can be found below.

#### Acronym

#### VHACD Meshes

For PyBullet to accurately simulate geometry, the mesh needs have a convex decomposition (otherwise, it seems like PyBullet uses the convex hull of the mesh for collision checking). To do this, we use the VHACD method provided in PyBullet.

#### Watertight Meshes

Many geometric algorithms used for grasp sampling rely on meshes only representing the surface of an object (e.g., ray tracing to find antipodal points). If this does not hold these algorithms may fail (e.g., by sampling internal structure that is present in a convex decomposition). A watertight mesh has no internal geometry or no holes and thus we also make each mesh watertight using `https://github.com/hjwdzh/Manifold`.

#### Origin Correction

Many of the meshes in ShapeNet are not centered around the origin. To simplify things, we compute the bounding box of the mesh and shift the point cloud such that the bounding box is centered at the origin.



## Grasping Simulation

### Antipodal Grasps

### Physical Property Sampling

### Grasp Simulation

#### Velocity Control

#### Stability Check