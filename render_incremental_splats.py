
import os
import numpy as np
import torch
from pathlib import Path
from src.models.models.rasterization import GaussianSplatRenderer
from plyfile import PlyData
from PIL import Image

def load_gs_ply(ply_path):
    plydata = PlyData.read(str(ply_path))
    vert = plydata['vertex']
    means = torch.tensor(np.stack([vert['x'], vert['y'], vert['z']], axis=1), dtype=torch.float32)
    scales = torch.exp(torch.tensor(np.stack([vert['scale_0'], vert['scale_1'], vert['scale_2']], axis=1), dtype=torch.float32))
    quats = torch.tensor(np.stack([vert['rot_0'], vert['rot_1'], vert['rot_2'], vert['rot_3']], axis=1), dtype=torch.float32)
    opacities = torch.tensor(vert['opacity'], dtype=torch.float32)
    sh = torch.tensor(np.stack([vert['f_dc_0'], vert['f_dc_1'], vert['f_dc_2']], axis=1), dtype=torch.float32)
    return {"means": means, "scales": scales, "quats": quats, "opacities": opacities, "sh": sh}

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Render incremental splats for each view in each set of splats.")
    parser.add_argument('--output_root', type=str, required=True, help='Root output directory containing incremental_splats folders.')
    parser.add_argument('--height', type=int, default=518, help='Render height (default: 518)')
    parser.add_argument('--width', type=int, default=518, help='Render width (default: 518)')
    args = parser.parse_args()

    output_root = Path(args.output_root)
    H, W = args.height, args.width

    # Find all incremental_splats directories recursively
    for inc_dir in output_root.rglob('incremental_splats'):
        print(f"Processing {inc_dir}")
        # Find all cumulative splat PLYs and their camera pose/intrinsics
        for ply_file in sorted(inc_dir.glob("splats_views_0to*.ply")):
            step = ply_file.stem.split("_0to")[-1]
            cam_poses_file = inc_dir / f"camera_poses_0to{step}.npy"
            cam_intrs_file = inc_dir / f"camera_intrs_0to{step}.npy"
            if not cam_poses_file.exists() or not cam_intrs_file.exists():
                print(f"Missing camera files for {ply_file}, skipping.")
                continue
            # Load splats
            splats = load_gs_ply(ply_file)
            # Load camera poses and intrinsics
            cam_poses = np.load(cam_poses_file)  # shape: (num_views, 4, 4)
            cam_intrs = np.load(cam_intrs_file)  # shape: (num_views, 3, 3)
            # Convert to torch
            cam_poses_torch = torch.from_numpy(cam_poses).unsqueeze(0)  # [1, V, 4, 4]
            cam_intrs_torch = torch.from_numpy(cam_intrs).unsqueeze(0)  # [1, V, 3, 3]
            # Set up renderer
            gs_renderer = GaussianSplatRenderer(voxel_size=0.002)
            renders_dir = inc_dir / f"renders_views_0to{step}"
            renders_dir.mkdir(exist_ok=True)
            # Prepare splat tensors
            means = splats["means"].unsqueeze(0) if splats["means"].ndim == 2 else splats["means"]
            quats = splats["quats"].unsqueeze(0) if splats["quats"].ndim == 2 else splats["quats"]
            scales = splats["scales"].unsqueeze(0) if splats["scales"].ndim == 2 else splats["scales"]
            opacities = splats["opacities"].unsqueeze(0) if splats["opacities"].ndim == 1 else splats["opacities"]
            sh = splats["sh"].unsqueeze(0) if splats["sh"].ndim == 2 else splats["sh"]
            # Render each view
            render_colors, render_depths, _ = gs_renderer.rasterizer.rasterize_batches(
                means, quats, scales, opacities,
                sh, cam_poses_torch, cam_intrs_torch,
                width=W, height=H, sh_degree=gs_renderer.sh_degree
            )
            V_out = render_colors.shape[1]
            for v in range(V_out):
                rgb = render_colors[0, v].clamp(0, 1)
                rgb_img = (rgb * 255).to(torch.uint8).cpu().numpy()
                Image.fromarray(rgb_img).save(str(renders_dir / f"render_view_{v:02d}_rgb.png"))
                depth = render_depths[0, v, :, :, 0].clamp(0, None)
                depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
                depth_img = (depth_normalized * 255).to(torch.uint8).cpu().numpy()
                Image.fromarray(depth_img).save(str(renders_dir / f"render_view_{v:02d}_depth.png"))

if __name__ == "__main__":
    main()
    print(f"Rendered view {v} for step 0to{step} saved in {renders_dir}")
