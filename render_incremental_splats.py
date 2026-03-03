import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from src.models.models.rasterization import GaussianSplatRenderer
from src.utils.load_gs_ply import load_gs_ply
from src.models.utils.sh_utils import SH2RGB, eval_sh

def render_incremental_splats(
    incremental_dir: Path,
    output_dir: Path,
    H: int,
    W: int,
    sh_degree: int = 0,
    device: str = 'cuda',
):
    incremental_dir = Path(incremental_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all splat PLY files and corresponding camera/intrinsics
    ply_files = sorted(incremental_dir.glob('splats_views_0to*.ply'))
    for ply_path in ply_files:
        # Extract view index
        view_str = ply_path.stem.split('_')[-1]
        end_view = int(view_str.replace('to', ''))
        cam_poses_path = incremental_dir / f"camera_poses_views_0to{end_view}.npz"
        cam_intrs_path = incremental_dir / f"camera_intrs_views_0to{end_view}.npz"
        renders_dir = output_dir / f"renders_views_0to{end_view}"
        renders_dir.mkdir(exist_ok=True)

        # Load splats
        means, scales, quats, colors, opacities = load_gs_ply(ply_path)
        means = means.unsqueeze(0) if means.ndim == 2 else means  # [1, N, 3/4]
        quats = quats.unsqueeze(0) if quats.ndim == 2 else quats  # [1, N, 4]
        scales = scales.unsqueeze(0) if scales.ndim == 2 else scales  # [1, N, 3]
        opacities = opacities.unsqueeze(0) if opacities.ndim == 1 else opacities  # [1, N]
        sh = colors.unsqueeze(0) if sh_degree == 3 else sh  # [1, N, num_sh_coeffs, 3]

        # Load cameras
        cam_poses = np.load(cam_poses_path)["camera_poses"]
        cam_intrs = np.load(cam_intrs_path)["camera_intrs"]
        cam_poses = torch.from_numpy(cam_poses).to(device)
        cam_intrs = torch.from_numpy(cam_intrs).to(device)

        # Create renderer
        gs_renderer = GaussianSplatRenderer(sh_degree=sh_degree).to(device)

        # Render each view one at a time and save
        V = cam_poses.shape[0]
        for v in range(V):
            # Prepare per-view camera pose/intrinsics
            cam_pose_v = cam_poses[v:v+1].to(torch.float32)  # [1, 4, 4]
            cam_intr_v = cam_intrs[v:v+1].to(torch.float32)  # [1, 3, 3]
            cams_c2w = cam_pose_v
            cams_K = cam_intr_v

            colors_arg = sh if sh is not None else colors

            print("DEBUG: means shape", means.shape)
            print("DEBUG: scales shape", scales.shape)
            print("DEBUG: quats shape", quats.shape)
            print("DEBUG: opacities shape", opacities.shape)
            if colors_arg is not None:
                print("DEBUG: colors shape", colors_arg.shape)
            if sh is not None:
                print("DEBUG: sh shape", sh.shape)
            print("DEBUG: cams_c2w shape", cams_c2w.shape)
            print("DEBUG: cams_K shape", cams_K.shape)
            print("DEBUG: width", W)
            print("DEBUG: height", H)
            print("DEBUG: sh_degree", gs_renderer.sh_degree if sh is not None else None)
            # Prepare splat batch
            # if sh_degree == 0:
            #     rgb = SH2RGB(colors.reshape(-1, 3)).reshape(-1, 3)
            #     splats = {
            #         "means": means.unsqueeze(0).to(torch.float32),
            #         "scales": scales.unsqueeze(0).to(torch.float32),
            #         "quats": quats.unsqueeze(0).to(torch.float32),
            #         "opacities": opacities.unsqueeze(0).to(torch.float32),
            #         "colors": rgb.unsqueeze(0).to(torch.float32),  # [1, N, 3]
            #     }
            # else:
            #     num_coeffs = (sh_degree + 1) ** 2
            #     if colors.shape[1] < num_coeffs:
            #         pad = torch.zeros((colors.shape[0], num_coeffs - colors.shape[1], 3), device=colors.device)
            #         sh = torch.cat([colors, pad], dim=1)
            #     else:
            #         sh = colors[:, :num_coeffs, :]
            #     # Evaluate SH for this view direction
            #     view_dir = cam_poses[v, :3, 2]
            #     view_dir = -view_dir / (view_dir.norm() + 1e-8)
            #     dirs = view_dir.expand(sh.shape[0], 3)
            #     rgb_v = eval_sh(sh_degree, sh, dirs)  # [N, 3]
            #     splats = {
            #         "means": means.unsqueeze(0).to(torch.float32),
            #         "scales": scales.unsqueeze(0).to(torch.float32),
            #         "quats": quats.unsqueeze(0).to(torch.float32),
            #         "opacities": opacities.unsqueeze(0).to(torch.float32),
            #         "colors": rgb_v.unsqueeze(0).to(torch.float32),  # [1, N, K, 3] (already [N, K, 3])
            #     }

            print(f"Rendering view {v} of {V} for splats_views_0to{end_view}")
            render_colors, render_depths, _ = gs_renderer.rasterizer.rasterize_batches(
                    means, quats, scales, opacities,
                    colors_arg,
                    cams_c2w, cams_K,
                    width=W, height=H,
                    sh_degree=gs_renderer.sh_degree if sh is not None else None,
            )

            # Save RGB
            rgb = render_colors[0, 0].clamp(0, 1)
            rgb_img = (rgb * 255).to(torch.uint8).cpu().numpy()
            Image.fromarray(rgb_img).save(str(renders_dir / f"render_view_{v:02d}_rgb.png"))

            # Save depth
            depth = render_depths[0, 0, :, :, 0].clamp(0, None)
            depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            depth_img = (depth_normalized * 255).to(torch.uint8).cpu().numpy()
            Image.fromarray(depth_img).save(str(renders_dir / f"render_view_{v:02d}_depth.png"))
            print(f"Saved render and depth for view {v} in splats_views_0to{end_view}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render incremental splats from saved PLY and camera files.")
    parser.add_argument("--incremental_dir", type=str, required=True, help="Path to incremental_splats folder")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for renders")
    parser.add_argument("--height", type=int, required=True, help="Image height")
    parser.add_argument("--width", type=int, required=True, help="Image width")
    parser.add_argument("--sh_degree", type=int, default=0, help="Spherical harmonics degree")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device")
    args = parser.parse_args()

    render_incremental_splats(
        incremental_dir=args.incremental_dir,
        output_dir=args.output_dir,
        H=args.height,
        W=args.width,
        sh_degree=args.sh_degree,
        device=args.device,
    )
