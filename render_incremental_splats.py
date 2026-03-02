import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from src.models.models.rasterization import GaussianSplatRenderer
from src.utils.load_gs_ply import load_gs_ply

def render_incremental_splats(
    incremental_dir: Path,
    output_dir: Path,
    H: int,
    W: int,
    sh_degree: int = 3,
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
        means = torch.from_numpy(means).to(device)
        scales = torch.from_numpy(scales).to(device)
        quats = torch.from_numpy(quats).to(device)
        opacities = torch.from_numpy(opacities).to(device)
        colors = torch.from_numpy(colors).to(device)
        # If colors are SH, reshape as needed
        if colors.ndim == 2 and colors.shape[1] == 3:
            sh = colors.unsqueeze(0)  # [1, N, 3]
        else:
            sh = colors  # fallback

        # Load cameras
        cam_poses = np.load(cam_poses_path)["camera_poses"]
        cam_intrs = np.load(cam_intrs_path)["camera_intrs"]
        cam_poses = torch.from_numpy(cam_poses).to(device)
        cam_intrs = torch.from_numpy(cam_intrs).to(device)

        # Prepare splats dict
        splats = {
            "means": means.unsqueeze(0),
            "scales": scales.unsqueeze(0),
            "quats": quats.unsqueeze(0),
            "opacities": opacities.unsqueeze(0),
            "sh": sh.unsqueeze(0) if sh.ndim == 2 else sh,
        }

        # Create renderer
        gs_renderer = GaussianSplatRenderer(sh_degree=sh_degree).to(device)

        # Render
        try:
            render_colors, render_depths, _ = gs_renderer.rasterizer.rasterize_batches(
                splats["means"], splats["quats"], splats["scales"], splats["opacities"],
                splats["sh"],
                cam_poses.to(torch.float32), cam_intrs.to(torch.float32),
                width=W, height=H, sh_degree=sh_degree,
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
                print(f"Rendered view {v} for splats_views_0to{end_view}")
        except Exception as e:
            print(f"Failed to render for splats_views_0to{end_view}: {e}")

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
