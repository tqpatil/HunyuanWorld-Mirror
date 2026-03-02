import numpy as np
from plyfile import PlyData

def load_gs_ply(ply_path):
    """
    Load Gaussian splat parameters from a PLY file.
    Returns:
        means: [N, 3] float32
        scales: [N, 3] float32
        quats: [N, 4] float32
        colors: [N, 3] float32
        opacities: [N] float32
    """
    plydata = PlyData.read(str(ply_path))
    vert = plydata['vertex']
    means = np.stack([vert['x'], vert['y'], vert['z']], axis=-1).astype(np.float32)
    scales = np.stack([vert['scale_0'], vert['scale_1'], vert['scale_2']], axis=-1).astype(np.float32)
    quats = np.stack([vert['rot_0'], vert['rot_1'], vert['rot_2'], vert['rot_3']], axis=-1).astype(np.float32)
    colors = np.stack([vert['f_dc_0'], vert['f_dc_1'], vert['f_dc_2']], axis=-1).astype(np.float32)
    opacities = vert['opacity'].astype(np.float32)
    return means, scales, quats, colors, opacities
