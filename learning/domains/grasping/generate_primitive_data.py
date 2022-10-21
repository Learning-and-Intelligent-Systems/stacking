import random
import argparse
import os

# WANT: generate URDF

# Use all primitives available in URDF: box (cube), cylinder, sphere
# need directory, number of objects to generate, maximum bounding box size?
# do we need to generate mass?

# probably a good idea to copy YCB/Shapenet structure so that our list gens
# should do less work

# directory structure is /SOURCE_ROOT/urdf (there are no visual models since we're using primitives, so that's it)
# urdf filename structure is ObjectName_SomethingThatLoocksHashlike.urdf

# sample file:

PRIMITIVES = ['cylinder', 'cube', 'sphere']
URDF_TEMPLATE = '''<?xml version="1.0"?>
<robot  name="UNNAMED_%i" >
 <link  name="UNNAMED_%i" >
  <inertial >
   <origin  rpy="0 0 0"  xyz="0 0 0" />
   <mass  value="1.0" />
   <inertia  ixx="0.001"  izz="0.001"  iyy="0.001" />
  </inertial>
  <visual >
   <geometry >
     <%s />
   </geometry>
  </visual>
  <collision >
   <geometry >
     <%s />
   </geometry>
  </collision>
 </link>
</robot>'''


def main(args):
    for p in args.primitives:
        if p not in PRIMITIVES:
            print('Please ensure all primitives specified are valid. Valid options: cylinder, cube, sphere.')
            return

    # make directories
    data_dir = (args.directory if args.directory[-1] == '/' else args.directory + '/') + 'urdfs/'
    try:
        os.mkdir(args.directory)  # do this first to ensure there isn't an existing directory with this name
        os.mkdir(data_dir)
    except FileExistsError as e:
        print('Directory already exists.')
        return
    except FileNotFoundError as e:
        print('Invalid directory.')
        return

    if 'cylinder' in args.primitives:
        for i in range(args.n_prim):
            side_len = random.uniform(args.min_len, args.max_len)
            urdf_text = URDF_TEMPLATE % (2 * i,
                                         2 * i + 1,
                                         'cylinder length=\"%f\" radius=\"%f\"' % (side_len, side_len / 2),
                                         'cylinder length=\"%f\" radius=\"%f\"' % (side_len, side_len / 2))
            f = open(data_dir + 'Cylinder_%i.urdf' % hash(random.uniform(0, 1)), 'w')
            f.write(urdf_text)
            f.close()

    if 'cube' in args.primitives:
        for i in range(args.n_prim):
            side_len = random.uniform(args.min_len, args.max_len)
            urdf_text = URDF_TEMPLATE % (2 * i,
                                         2 * i + 1,
                                         'box size=\"%f %f %f\"' % (side_len, side_len, side_len),
                                         'box size=\"%f %f %f\"' % (side_len, side_len, side_len))
            f = open(data_dir + 'Box_%i.urdf' % hash(random.uniform(0, 1)), 'w')
            f.write(urdf_text)
            f.close()

    if 'sphere' in args.primitives:
        for i in range(args.n_prim):
            side_len = random.uniform(args.min_len, args.max_len)
            urdf_text = URDF_TEMPLATE % (2 * i,
                                         2 * i + 1,
                                         'sphere radius=\"%f\"' % (side_len / 2),
                                         'sphere radius=\"%f\"' % (side_len / 2))
            f = open(data_dir + 'Sphere_%i.urdf' % hash(random.uniform(0, 1)), 'w')
            f.write(urdf_text)
            f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', '-d', type=str, required=True,
                        help='path and name of directory to store primitive data')
    parser.add_argument('--primitives', '-p', type=str, nargs='+', required=True,
                        help='primitives to generate in set. options are: cylinder, cube, and sphere.')
    parser.add_argument('--n_prim', '-n', type=int, required=True,
                        help='number of objects to generate (this is done per primitive)')
    parser.add_argument('--min_len', '-mil', type=float, default=0.01,
                        help='minimum side length of bounding volume on object')
    parser.add_argument('--max_len', '-mal', type=float, default=0.03,
                        help='maximum side length of bounding volume on object')

    args = parser.parse_args()
    main(args)
