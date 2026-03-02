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
    opacities = vert['opacity'].astype(np.float32)

    # Detect SH channels
    f_dc_fields = [f for f in vert.dtype.fields if f.startswith('f_dc_')]
    f_dc_fields_sorted = sorted(f_dc_fields, key=lambda x: int(x.split('_')[-1]))
    num_f_dc = len(f_dc_fields_sorted)
    if num_f_dc == 3:
        # RGB only
        colors = np.stack([vert['f_dc_0'], vert['f_dc_1'], vert['f_dc_2']], axis=-1).astype(np.float32)
    else:
        # SH: [N, num_sh_coeffs, 3]
        num_sh_coeffs = num_f_dc // 3
        colors = np.stack([vert[f_dc_fields_sorted[i]] for i in range(num_f_dc)], axis=-1).astype(np.float32)
        colors = colors.reshape(-1, num_sh_coeffs, 3)
    return means, scales, quats, colors, opacities
