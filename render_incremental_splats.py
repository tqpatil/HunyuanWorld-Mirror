import os
import argparse
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
    parser.add_argument('--incremental_dir', type=str,required=True, help='Path to incremental_splats folder')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save renders')
    parser.add_argument('--height', type=int, required=True, help='Render image height')
    parser.add_argument('--width', type=int, required=True, help='Render image width')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--sh_degree', type=int, default=None, help='Spherical harmonics degree (default: inferred from data)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ply_files = sorted([f for f in os.listdir(args.incremental_dir) if f.endswith('.ply')])
    if not ply_files:
        print('No PLY files found in', args.incremental_dir)
        return

    # Infer SH degree if not provided
    if args.sh_degree is None:
        # Try to infer from first PLY file
        # This is a placeholder: replace with actual logic
        args.sh_degree = 0

    renderer = GaussianSplatRenderer(sh_degree=args.sh_degree).to(args.device)

    for ply_file in tqdm(ply_files, desc='Rendering incremental splats'):
        ply_path = os.path.join(args.incremental_dir, ply_file)
        splats = load_gs_ply(ply_path, args.sh_degree, args.device)

        # Load camera poses/intrinsics for this step
        npz_path = ply_path.replace('.ply', '_cams.npz')
        if not os.path.exists(npz_path):
            print(f'Camera file {npz_path} not found, skipping {ply_file}')
            continue
        cam_data = np.load(npz_path)
        cam_poses = cam_data['poses']
        cam_intrs = cam_data['intrs']

        # Render
        rgb_images, depth_images = renderer.rasterize_batches(splats, cam_poses, cam_intrs, args.height, args.width)

        # Save images
        for i, (rgb, depth) in enumerate(zip(rgb_images, depth_images)):
            rgb_path = os.path.join(args.output_dir, f'{ply_file}_view{i}_rgb.png')
            depth_path = os.path.join(args.output_dir, f'{ply_file}_view{i}_depth.png')
            Image.fromarray(rgb).save(rgb_path)
            Image.fromarray((depth * 255).astype(np.uint8)).save(depth_path)

if __name__ == '__main__':
    main()
