import numpy as np
from plyfile import PlyData

def load_gs_ply(ply_path):
    """
    Correctly loads Gaussian splat parameters from a PLY file.
    """
    plydata = PlyData.read(str(ply_path))
    vert = plydata['vertex']

    # 1. Extract Geometry
    means = np.stack([vert['x'], vert['y'], vert['z']], axis=-1).astype(np.float32)
    scales = np.stack([vert['scale_0'], vert['scale_1'], vert['scale_2']], axis=-1).astype(np.float32)
    quats = np.stack([vert['rot_0'], vert['rot_1'], vert['rot_2'], vert['rot_3']], axis=-1).astype(np.float32)
    opacities = vert['opacity'].astype(np.float32)

    # 2. Extract Spherical Harmonics (Colors)
    # DC components (usually f_dc_0, 1, 2)
    f_dc = np.zeros((means.shape[0], 3, 1))
    f_dc[:, 0, 0] = vert['f_dc_0']
    f_dc[:, 1, 0] = vert['f_dc_1']
    f_dc[:, 2, 0] = vert['f_dc_2']

    # Higher-order components (f_rest_0 to f_rest_44)
    extra_f_names = [f for f in vert.data.dtype.names if f.startswith('f_rest_')]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
    
    if len(extra_f_names) > 0:
        f_rest = np.zeros((means.shape[0], len(extra_f_names)))
        for idx, name in enumerate(extra_f_names):
            f_rest[:, idx] = vert[name]
        
        # Reshape to [N, 3, 15] (for degree 3 SH)
        f_rest = f_rest.reshape(means.shape[0], 3, -1)
        # Combine to [N, 3, 16]
        shs = np.concatenate([f_dc, f_rest], axis=2).transpose(0, 2, 1)
    else:
        # Only DC components exist
        shs = f_dc.transpose(0, 2, 1)

    return means, scales, quats, shs, opacities