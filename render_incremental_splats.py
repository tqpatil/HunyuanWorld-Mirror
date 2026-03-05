import os
import argparse
import re
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from src.models.models.rasterization import GaussianSplatRenderer

def load_gs_ply(ply_path, sh_degree, device):
    from plyfile import PlyData
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']
    # Attribute order: x, y, z, nx, ny, nz, f_dc_0, f_dc_1, f_dc_2, opacity, scale_0, scale_1, scale_2, rot_0, rot_1, rot_2, rot_3
    means = torch.tensor(np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1), device=device, dtype=torch.float32)
    scales = torch.tensor(np.stack([vertex['scale_0'], vertex['scale_1'], vertex['scale_2']], axis=-1), device=device, dtype=torch.float32)
    quats = torch.tensor(np.stack([vertex['rot_0'], vertex['rot_1'], vertex['rot_2'], vertex['rot_3']], axis=-1), device=device, dtype=torch.float32)
    rgbs = torch.tensor(np.stack([vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']], axis=-1), device=device, dtype=torch.float32)
    opacities = torch.tensor(vertex['opacity'], device=device, dtype=torch.float32).unsqueeze(-1)
    # SH: If sh_degree > 0, you may need to load more SH coefficients
    if sh_degree == 0:
        sh = rgbs.unsqueeze(1)  # [N, 1, 3]
    else:
        # If higher SH degree, extend this to load all SH bands
        sh = torch.zeros((means.shape[0], (sh_degree+1)**2, 3), device=device)
        sh[:, 0, :] = rgbs
    return {
        'means': means,
        'quats': quats,
        'scales': scales,
        'opacities': opacities,
        'sh': sh,
    }

def main():
    parser = argparse.ArgumentParser(description='Render incremental splats from PLY files.')
    parser.add_argument('--incremental_dir', type=str, required=True, help='Path to incremental_splats folder')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save renders')
    parser.add_argument('--height', type=int, required=True, help='Render image height')
    parser.add_argument('--width', type=int, required=True, help='Render image width')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--sh_degree', type=int, default=None, help='Spherical harmonics degree (default: inferred from data)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    pattern = re.compile(r".*0to(\d+)\.ply")

    # Original code with added regex filtering
    ply_files = sorted([f for f in os.listdir(args.incremental_dir) if pattern.match(f)])

    if not ply_files:
        print('No matching PLY files found in', args.incremental_dir)
        return  # Exit early if no files

    # Infer SH degree if not provided
    if args.sh_degree is None:
        # Try to infer from first PLY file
        first_ply_path = os.path.join(args.incremental_dir, ply_files[0])
        # Placeholder: replace with actual inference logic from load_gs_ply or similar
        args.sh_degree = 0  # Assuming default; adjust based on actual inference

    renderer = GaussianSplatRenderer(sh_degree=args.sh_degree).to(args.device)

    for ply_file in tqdm(ply_files, desc='Rendering incremental splats'):
        ply_path = os.path.join(args.incremental_dir, ply_file)
        splats = load_gs_ply(ply_path, args.sh_degree, args.device)

        # Extract end_view index from filename
        match = pattern.match(ply_file)
        end_view = int(match.group(1))

        # Construct camera file paths
        cam_poses_path = os.path.join(
            args.incremental_dir,
            f"camera_poses_views_0to{end_view}.npz"
        )

        cam_intrs_path = os.path.join(
            args.incremental_dir,
            f"camera_intrs_views_0to{end_view}.npz"
        )

        if not os.path.exists(cam_poses_path) or not os.path.exists(cam_intrs_path):
            print(f"Camera files missing for {ply_file}, skipping.")
            del splats  # Deallocate splats memory
            if args.device == 'cuda':
                torch.cuda.empty_cache()
            continue

        # Load data
        cam_poses = np.load(cam_poses_path)["camera_poses"]
        cam_intrs = np.load(cam_intrs_path)["camera_intrs"]
        cam_poses = torch.from_numpy(cam_poses).to(args.device)
        cam_intrs = torch.from_numpy(cam_intrs).to(args.device)

        # Prepare splat data (ensure batch dimension)
        means = splats["means"].unsqueeze(0) if splats["means"].ndim == 2 else splats["means"]  # [1, N, 3/4]
        quats = splats["quats"].unsqueeze(0) if splats["quats"].ndim == 2 else splats["quats"]  # [1, N, 4]
        scales = splats["scales"].unsqueeze(0) if splats["scales"].ndim == 2 else splats["scales"]  # [1, N, 3]
        opacities = splats["opacities"].squeeze(-1).unsqueeze(0)  # [1, N]
        sh = splats["sh"].unsqueeze(0) if splats["sh"].ndim == 3 else splats["sh"]  # [1, N, num_sh_coeffs, 3]

        colors_arg = sh if "sh" in splats else splats.get("colors") if "colors" in splats else None

        # Get number of views
        V = cam_poses.shape[0]
        print("Cam poses shape:", cam_poses.shape)
        # Render one view at a time to reduce memory usage
        for v in range(V):
            cam_c2w_single = cam_poses[v:v+1]  # [1, 4, 4]
            cam_K_single = cam_intrs[v:v+1]    # [1, 3, 3]

            rgb_images, depth_images, _ = renderer.rasterizer.rasterize_batches(
                means, quats, scales, opacities,
                colors_arg,
                cam_c2w_single, cam_K_single,
                width=args.width, height=args.height,
                sh_degree=renderer.sh_degree if "sh" in splats else None,
            )

            try:
                rgb = rgb_images[0, 0].clamp(0, 1)  # [H, W, 3]
                rgb_img = (rgb * 255).to(torch.uint8).cpu().numpy()
                Image.fromarray(rgb_img).save(os.path.join(args.output_dir, f"render_view_{v:02d}_rgb.png"))

                depth = depth_images[0, 0, :, :, 0].clamp(0, None)  # [H, W]
                depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
                depth_img = (depth_normalized * 255).to(torch.uint8).cpu().numpy()
                Image.fromarray(depth_img).save(os.path.join(args.output_dir, f"render_view_{v:02d}_depth.png"))

                print(f"   Rendered view {v}")
            except Exception as e:
                print(f"  Failed to save render for view {v}: {e}")

            # Deallocate per-view tensors to free memory
            del rgb_images, depth_images
            if args.device == 'cuda':
                torch.cuda.empty_cache()

        # Deallocate splats and camera data after processing all views for this ply_file
        del splats, cam_poses, cam_intrs, means, quats, scales, opacities, sh
        if colors_arg is not None:
            del colors_arg
        if args.device == 'cuda':
            torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
