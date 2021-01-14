import numpy as np
import pickle
from rotation_util import *

info = {
    'id': 4,
    'dimensions': np.array([0.06, 0.06 0.1]),
    24: {
        'name': 'top',
        'marker_size_cm': 5.4,
        'X_OT': np.eye(4)
    },
    25: {
        'name': 'top',
        'marker_size_cm': 5.4,
        'X_OT': np.eye(4)
    },
    26: {
        'name': 'top',
        'marker_size_cm': 5.4,
        'X_OT': np.eye(4)
    },
    27: {
        'name': 'top',
        'marker_size_cm': 5.4,
        'X_OT': np.eye(4)
    },
    28: {
        'name': 'front',
        'marker_size_cm': 5.4,
        'X_OT': Rt_to_pose_matrix(eul_to_rot([np.pi/2,0,np.pi/2]), [0.03, 0, 0])
    },
    29: {
        'name': 'top',
        'marker_size_cm': 5.4,
        'X_OT': np.eye(4)
    }
}

with open(f'tags/block_4_info.pkl', 'wb') as f:
        pickle.dump(info, f)