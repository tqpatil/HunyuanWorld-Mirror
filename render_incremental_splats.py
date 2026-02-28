import torch
import numpy as np
from pathlib import Path
from plyfile import PlyData
from PIL import Image

from src.models.models.rasterization import GaussianSplatRenderer


def load_gs_ply(ply_path):
    plydata = PlyData.read(str(ply_path))
    vert = plydata["vertex"]

    means = torch.tensor(
        np.stack([vert["x"], vert["y"], vert["z"]], axis=1),
        dtype=torch.float32
    )

    scales = torch.exp(torch.tensor(
        np.stack([vert["scale_0"], vert["scale_1"], vert["scale_2"]], axis=1),
        dtype=torch.float32
    ))

    quats = torch.tensor(
        np.stack([vert["rot_0"], vert["rot_1"], vert["rot_2"], vert["rot_3"]], axis=1),
        dtype=torch.float32
    )

    opacities = torch.tensor(vert["opacity"], dtype=torch.float32)

    sh = torch.tensor(
        np.stack([vert["f_dc_0"], vert["f_dc_1"], vert["f_dc_2"]], axis=1),
        dtype=torch.float32
    )

    return {
        "means": means,
        "scales": scales,
        "quats": quats,
        "opacities": opacities,
        "sh": sh,
    }


def render_incremental_from_deltas(output_dir, H, W):

    output_dir = Path(output_dir)
    incremental_dir = output_dir / "incremental_splats"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------------
    # Find delta files
    # --------------------------------------------------------
    delta_files = sorted(incremental_dir.glob("splats_delta_*to*.ply"))

    if len(delta_files) == 0:
        print("No delta splats found.")
        return

    # --------------------------------------------------------
    # Renderer
    # --------------------------------------------------------
    gs_renderer = GaussianSplatRenderer()
    if hasattr(gs_renderer, "to"):
        gs_renderer = gs_renderer.to(device)

    # --------------------------------------------------------
    # Cumulative storage
    # --------------------------------------------------------
    cumulative = {
        "means": [],
        "scales": [],
        "quats": [],
        "opacities": [],
        "sh": [],
    }

    # --------------------------------------------------------
    # Main reconstruction loop
    # --------------------------------------------------------
    for step, delta_file in enumerate(delta_files, start=1):

        print(f"\nReconstructing views 0..{step}")

        delta = load_gs_ply(delta_file)

        for k in cumulative:
            cumulative[k].append(delta[k])

        # Reconstruct cumulative splats
        means = torch.cat(cumulative["means"], dim=0).to(device)
        scales = torch.cat(cumulative["scales"], dim=0).to(device)
        quats = torch.cat(cumulative["quats"], dim=0).to(device)
        opacities = torch.cat(cumulative["opacities"], dim=0).to(device)
        sh = torch.cat(cumulative["sh"], dim=0).to(device)

        # --------------------------------------------------------
        # Load camera subset for this step
        # --------------------------------------------------------
        cam_pose_file = incremental_dir / f"camera_poses_0to{step}.npy"
        cam_intr_file = incremental_dir / f"camera_intrs_0to{step}.npy"

        if not cam_pose_file.exists():
            print(f"Missing camera file for step {step}")
            continue

        cam_poses = torch.from_numpy(np.load(cam_pose_file)).unsqueeze(0).to(device)
        cam_intrs = torch.from_numpy(np.load(cam_intr_file)).unsqueeze(0).to(device)

        # --------------------------------------------------------
        # Prepare splats with batch dimension
        # --------------------------------------------------------
        means = means.unsqueeze(0)
        scales = scales.unsqueeze(0)
        quats = quats.unsqueeze(0)
        opacities = opacities.unsqueeze(0)

        if gs_renderer.sh_degree > 0:
            colors_arg = sh.unsqueeze(0)
            sh_degree = gs_renderer.sh_degree
        else:
            colors_arg = sh.unsqueeze(0)
            sh_degree = None

        # --------------------------------------------------------
        # Render
        # --------------------------------------------------------
        try:
            render_colors, render_depths, _ = gs_renderer.rasterizer.rasterize_batches(
                means,
                quats,
                scales,
                opacities,
                colors_arg,
                cam_poses.float(),
                cam_intrs.float(),
                width=W,
                height=H,
                sh_degree=sh_degree,
            )

            renders_dir = incremental_dir / f"renders_views_0to{step}"
            renders_dir.mkdir(exist_ok=True)

            V_out = render_colors.shape[1]

            for v in range(V_out):

                rgb = render_colors[0, v].clamp(0, 1)
                rgb_img = (rgb * 255).to(torch.uint8).cpu().numpy()
                Image.fromarray(rgb_img).save(
                    renders_dir / f"render_view_{v:02d}_rgb.png"
                )

                depth = render_depths[0, v, :, :, 0].clamp(0, None)
                depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
                depth_img = (depth_norm * 255).to(torch.uint8).cpu().numpy()
                Image.fromarray(depth_img).save(
                    renders_dir / f"render_view_{v:02d}_depth.png"
                )

                print(f"   Rendered view {v}")

        except Exception as e:
            print(f"Failed rendering step {step}: {e}")
            import traceback
            traceback.print_exc()

    print("\nAll incremental renders complete.")