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
        means = torch.from_numpy(means).to(torch.float32).to(device)
        means = means.unsqueeze(0) if means.ndim == 2 else means  # [1, N, 3/4]
        quats = torch.from_numpy(quats).to(torch.float32).to(device)
        quats = quats.unsqueeze(0) if quats.ndim == 2 else quats  # [1, N, 4]
        scales = torch.from_numpy(scales).to(torch.float32).to(device)
        scales = scales.unsqueeze(0) if scales.ndim == 2 else scales  # [1, N, 3]
        opacities = torch.from_numpy(opacities).to(torch.float32).to(device)
        opacities = opacities.unsqueeze(0) if opacities.ndim == 1 else opacities  # [1, N]
        colors = torch.from_numpy(colors).to(torch.float32).to(device)
        sh = colors.unsqueeze(0) if colors.ndim == 3 else sh  # [1, N, num_sh_coeffs, 3]

        # Load cameras
        cam_poses = np.load(cam_poses_path)["camera_poses"]
        cam_intrs = np.load(cam_intrs_path)["camera_intrs"]
        print(f"Loaded from {cam_poses_path}: {cam_poses.shape}")
        print(f"Loaded from {cam_intrs_path}: {cam_intrs.shape}")
        cam_poses = torch.from_numpy(cam_poses).to(device)
        cam_intrs = torch.from_numpy(cam_intrs).to(device)
        print(f"Loaded cam_poses shape: {cam_poses.shape}")
        print(f"Loaded cam_intrs shape: {cam_intrs.shape}")

        # Create renderer
        gs_renderer = GaussianSplatRenderer(sh_degree=sh_degree).to(device)

        # Render each view one at a time and save
        V = cam_poses.shape[0]
        chunk_size = 2000  # Reduce chunk size for lower memory usage
        N_total = means.shape[1]
        for v in range(V):
            cam_pose_v = cam_poses[0, v:v+1].to(torch.float32)  # [1, 4, 4]
            cam_intr_v = cam_intrs[0, v:v+1].to(torch.float32)  # [1, 3, 3]
            cams_c2w = cam_pose_v
            cams_K = cam_intr_v

            colors_arg = sh if sh is not None else colors

            rgb_accum = None
            depth_accum = None
            alpha_accum = None

            for start in range(0, N_total, chunk_size):
                end = min(start + chunk_size, N_total)
                means_chunk = means[:, start:end].to(device)
                quats_chunk = quats[:, start:end].to(device)
                scales_chunk = scales[:, start:end].to(device)
                opacities_chunk = opacities[:, start:end].to(device)
                if sh is not None:
                    colors_chunk = colors_arg[:, start:end].to(device)
                else:
                    colors_chunk = colors_arg[:, start:end].to(device)

                with torch.no_grad():
                    render_colors, render_depths, render_alphas = gs_renderer.rasterizer.rasterize_batches(
                        means_chunk, quats_chunk, scales_chunk, opacities_chunk,
                        colors_chunk,
                        cams_c2w, cams_K,
                        width=W, height=H,
                        sh_degree=gs_renderer.sh_degree if sh is not None else None,
                    )

                alpha = render_alphas[0, 0, :, :, 0].unsqueeze(-1)  # [H, W, 1]
                if rgb_accum is None:
                    rgb_accum = render_colors[0, 0].cpu()
                    depth_accum = render_depths[0, 0, :, :, 0].cpu()
                    alpha_accum = render_alphas[0, 0, :, :, 0].cpu()
                else:
                    rgb_accum = rgb_accum * (1 - alpha.cpu()) + render_colors[0, 0].cpu() * alpha.cpu()
                    depth_accum = depth_accum * (1 - render_alphas[0, 0, :, :, 0].cpu()) + render_depths[0, 0, :, :, 0].cpu() * render_alphas[0, 0, :, :, 0].cpu()
                    alpha_accum = alpha_accum + render_alphas[0, 0, :, :, 0].cpu()
                torch.cuda.empty_cache()

            # Save RGB
            rgb_img = (rgb_accum.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
            Image.fromarray(rgb_img).save(str(renders_dir / f"render_view_{v:02d}_rgb.png"))

            # Save depth
            depth_normalized = (depth_accum.clamp(0, None) - depth_accum.min()) / (depth_accum.max() - depth_accum.min() + 1e-8)
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
