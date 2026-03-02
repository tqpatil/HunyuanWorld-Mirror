import numpy as np
from plyfile import PlyData

def load_gs_ply(ply_path):
    """
    Load Gaussian splat parameters from a 3DGS-style PLY file.

    Returns:
        means:    [N, 3] float32      (x, y, z)
        scales:   [N, 3] float32      (scale_0..2; typically log-scales)
        quats:    [N, 4] float32      (rot_0..3; w,x,y,z or similar)
        colors:   [N, C, 3] float32   SH coefficients per color channel
                                       C = 1 if only f_dc_0..2 exist,
                                           >1 if f_rest_* present
        opacities:[N]   float32       (usually logits)
    """
    plydata = PlyData.read(str(ply_path))
    vert = plydata["vertex"]

    # Basic attributes
    means = np.stack(
        [vert["x"], vert["y"], vert["z"]],
        axis=-1,
    ).astype(np.float32)

    scales = np.stack(
        [vert["scale_0"], vert["scale_1"], vert["scale_2"]],
        axis=-1,
    ).astype(np.float32)

    quats = np.stack(
        [vert["rot_0"], vert["rot_1"], vert["rot_2"], vert["rot_3"]],
        axis=-1,
    ).astype(np.float32)

    opacities = np.asarray(vert["opacity"], dtype=np.float32)

    # --- Spherical harmonics coefficients ---
    # Standard 3DGS PLY has:
    #   f_dc_0, f_dc_1, f_dc_2          (DC RGB)
    #   f_rest_0 ... f_rest_44          (higher-order SH, flattened)
    # See e.g. PDAL and GraphDECO docs. [web:1][web:7][web:10]

    # DC component: always 3 fields (RGB)
    dc_fields = [f for f in vert.dtype.names if f.startswith("f_dc_")]
    dc_fields_sorted = sorted(dc_fields, key=lambda x: int(x.split("_")[-1]))
    if len(dc_fields_sorted) != 3:
        raise ValueError(f"Expected 3 f_dc_* fields, found {len(dc_fields_sorted)}")

    dc = np.stack(
        [vert[name] for name in dc_fields_sorted],
        axis=-1,
    ).astype(np.float32)  # [N, 3]

    # Higher-order SH (optional)
    rest_fields = [f for f in vert.dtype.names if f.startswith("f_rest_")]
    rest_fields_sorted = sorted(rest_fields, key=lambda x: int(x.split("_")[-1]))

    if len(rest_fields_sorted) == 0:
        # Only DC present → treat as 1 coefficient per channel
        # colors shape: [N, 1, 3]
        colors = dc[:, None, :]
    else:
        # Concatenate DC and higher-order coeffs into [N, K*3]
        # Order: (R,G,B) for DC, then (flattened) rest coefficients
        all_coeffs = np.concatenate(
            [
                dc,
                np.stack(
                    [vert[name] for name in rest_fields_sorted],
                    axis=-1,
                ).astype(np.float32),
            ],
            axis=-1,
        )  # [N, 3 + len(f_rest)]

        # Total scalar coeffs must be divisible by 3 (RGB)
        if all_coeffs.shape[1] % 3 != 0:
            raise ValueError(
                f"Total SH scalar count {all_coeffs.shape[1]} is not divisible by 3"
            )

        num_coeffs = all_coeffs.shape[1] // 3  # number of SH coeffs per channel
        # Reshape to [N, num_coeffs, 3] (coeff index, color channel)
        colors = all_coeffs.reshape(-1, num_coeffs, 3)

    return means, scales, quats, colors, opacities
