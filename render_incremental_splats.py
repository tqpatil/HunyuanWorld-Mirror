import torch
import numpy as np
from pathlib import Path
from PIL import Image
from src.renderers.gaussian_splat_renderer import GaussianSplatRenderer
from src.utils.save_utils import load_gs_ply
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
    # Create renderer once outside the loop (efficiency)
    gs_renderer = GaussianSplatRenderer(sh_degree=sh_degree).to(device)
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
        quats = torch.from_numpy(quats).to(torch.float32).to(device)
        scales = torch.from_numpy(scales).to(torch.float32).to(device)
        opacities = torch.from_numpy(opacities).to(torch.float32).to(device)
        colors = torch.from_numpy(colors).to(torch.float32).to(device)
        # Determine if colors represent SH coefficients or RGB based on sh_degree
        if sh_degree > 0:
            # Assume colors is [N, K, 3] where K is num_sh_coeffs
            sh = colors.unsqueeze(0)  # [1, N, K, 3]
            colors_arg = sh
            use_sh = True
        else:
            # Assume colors is [N, 3] RGB
            sh = None
            colors_arg = colors.unsqueeze(0)  # [1, N, 3]
            use_sh = False
        # Add batch dimension to other tensors
        means = means.unsqueeze(0)  # [1, N, 3]
        quats = quats.unsqueeze(0)  # [1, N, 4]
        scales = scales.unsqueeze(0)  # [1, N, 3]
        opacities = opacities.unsqueeze(0)  # [1, N]
        # Load cameras
        cam_poses_np = np.load(cam_poses_path)["camera_poses"]  # [B, V, 4, 4], assume B=1
        cam_intrs_np = np.load(cam_intrs_path)["camera_intrs"]  # [B, V, 3, 3], assume B=1
        cam_poses = torch.from_numpy(cam_poses_np).to(device).to(torch.float32)
        cam_intrs = torch.from_numpy(cam_intrs_np).to(device).to(torch.float32)
        # Prepare for rendering (match the batch rendering style from working code)
        cams_c2w = cam_poses  # [1, V, 4, 4]
        cams_K = cam_intrs    # [1, V, 3, 3]
        # Debug prints (optional, for matching working code style)
        print(f"Rendering for splats_views_0to{end_view}")
        print("DEBUG: means shape", means.shape)
        print("DEBUG: scales shape", scales.shape)
        print("DEBUG: quats shape", quats.shape)
        print("DEBUG: opacities shape", opacities.shape)
        if colors_arg is not None:
            print("DEBUG: colors_arg shape", colors_arg.shape)
        if sh is not None:
            print("DEBUG: sh shape", sh.shape)
        print("DEBUG: cams_c2w shape", cams_c2w.shape)
        print("DEBUG: cams_K shape", cams_K.shape)
        print("DEBUG: width", W)
        print("DEBUG: height", H)
        print("DEBUG: sh_degree", gs_renderer.sh_degree if use_sh else None)
        # Render all views at once (matching working code)
        render_colors, render_depths, _ = gs_renderer.rasterizer.rasterize_batches(
            means, quats, scales, opacities,
            colors_arg,
            cams_c2w, cams_K,
            width=W, height=H,
            sh_degree=gs_renderer.sh_degree if use_sh else None,
        )
        # render_colors: [B, V, H, W, 3], render_depths: [B, V, H, W, 1]
        V_out = render_colors.shape[1]
        for v in range(V_out):
            try:
                # Save RGB
                rgb = render_colors[0, v].clamp(0, 1)
                rgb_img = (rgb * 255).to(torch.uint8).cpu().numpy()
                Image.fromarray(rgb_img).save(str(renders_dir / f"render_view_{v:02d}_rgb.png"))
                # Save depth
                depth = render_depths[0, v, :, :, 0].clamp(0, None)
                depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
                depth_img = (depth_normalized * 255).to(torch.uint8).cpu().numpy()
                Image.fromarray(depth_img).save(str(renders_dir / f"render_view_{v:02d}_depth.png"))
                print(f"   Rendered and saved view {v}")
            except Exception as e:
                print(f"   Failed to save render for view {v}: {e}")
        print(f"Renders saved to {renders_dir}")
