import os
import argparse
import re
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from src.models.models.rasterization import GaussianSplatRenderer
from src.utils.load_gs_ply import load_gs_ply


def main():
    parser = argparse.ArgumentParser(description='Render incremental splats from PLY files.')
    parser.add_argument('--incremental_dir', type=str, required=True, help='Path to incremental_splats folder')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save renders')
    parser.add_argument('--height', type=int, required=True, help='Render image height')
    parser.add_argument('--width', type=int, required=True, help='Render image width')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--sh_degree', type=int, default=0, help='Spherical harmonics degree (default: inferred from data)')
    parser.add_argument('--chunk_size', type=int, default=2, help='Number of views to render at once (reduce if OOM)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    pattern = re.compile(r".*_(\d+)\.ply$")

    files = os.listdir(args.incremental_dir)

    matches = []
    for f in files:
        m = pattern.match(f)
        if m:
            matches.append((f, int(m.group(1))))

    print("Matched files:", matches)

    ply_files = [f for f, _ in sorted(matches, key=lambda x: x[1])]

    if not ply_files:
        print('No matching PLY files found in', args.incremental_dir)
        return    
    renderer = GaussianSplatRenderer(sh_degree=args.sh_degree).to(args.device)

    def preprocess_splats(splats, device, sh_degree):
        out = {}
        for k in ["means", "scales", "quats", "opacities"]:
            t = splats[k]
            t = t.to(device)
            if k == "scales":
                t = torch.exp(t)
            if torch.isnan(t).any() or torch.isinf(t).any():
                raise ValueError(f"NaN/Inf in {k}")
            out[k] = t
        out["opacities"] = out["opacities"].clamp(0, 1)
        if "sh" in splats:
            sh = splats["sh"].to(device)
            if sh.shape[-2] != (sh_degree + 1) ** 2:
                raise ValueError(f"SH degree mismatch: got {sh.shape[-2]}, expected {(sh_degree+1)**2}")
            if torch.isnan(sh).any() or torch.isinf(sh).any():
                raise ValueError("NaN/Inf in sh")
            out["sh"] = sh
        elif "colors" in splats:
            colors = splats["colors"].to(device)
            if torch.isnan(colors).any() or torch.isinf(colors).any():
                raise ValueError("NaN/Inf in colors")
            out["colors"] = colors
        else:
            N = out["means"].shape[0]
            out["colors"] = torch.ones((N, 3), device=device)
        return out

    def render_in_chunks(renderer, splats, cam_poses, cam_intrs, H, W, sh_degree, chunk_size=2):
        device = splats["means"].device
        N_views = cam_poses.shape[1]
        rgbs, depths = [], []
        for i in range(0, N_views, chunk_size):
            pose_chunk = cam_poses[:, i:i+chunk_size].to(device)
            intr_chunk = cam_intrs[:, i:i+chunk_size].to(device)
            with torch.no_grad():
                colors, depth, _ = renderer.rasterizer.rasterize_batches(
                    splats["means"].unsqueeze(0),
                    splats["quats"].unsqueeze(0),
                    splats["scales"].unsqueeze(0),
                    splats["opacities"].squeeze(-1).unsqueeze(0),
                    splats.get("sh", None).unsqueeze(0) if "sh" in splats else splats.get("colors", None).unsqueeze(0),
                    pose_chunk,
                    intr_chunk,
                    width=W,
                    height=H,
                    sh_degree=sh_degree,
                )
            rgbs.append(colors.cpu())
            depths.append(depth.cpu())
        rgbs = torch.cat(rgbs, dim=1)
        depths = torch.cat(depths, dim=1)
        return rgbs, depths


    # Incrementally load and concatenate splats for each view
    all_splats = None
    for ply_file in tqdm(ply_files, desc='Rendering incremental splats'):
        ply_path = os.path.join(args.incremental_dir, ply_file)
        # Use updated loader: returns means, scales, quats, shs, opacities, pixel_x, pixel_y, view_idx
        means, scales, quats, shs, opacities, pixel_x, pixel_y, view_idx = load_gs_ply(ply_path)
        means = torch.tensor(means, device=args.device, dtype=torch.float32)
        scales = torch.tensor(scales, device=args.device, dtype=torch.float32)
        quats = torch.tensor(quats, device=args.device, dtype=torch.float32)
        opacities = torch.tensor(opacities, device=args.device, dtype=torch.float32).unsqueeze(-1)
        sh = torch.tensor(shs, device=args.device, dtype=torch.float32)
        splats = {
            'means': means,
            'quats': quats,
            'scales': scales,
            'opacities': opacities,
            'sh': sh,
        }
        match = pattern.match(ply_file)
        view_idx = int(match.group(1))
        cam_poses_path = os.path.join(args.incremental_dir, f"camera_poses_view_{view_idx}.npy")
        cam_intrs_path = os.path.join(args.incremental_dir, f"camera_intrs_view_{view_idx}.npy")
        if not os.path.exists(cam_poses_path) or not os.path.exists(cam_intrs_path):
            print(f"Camera files missing for {ply_file}, skipping.")
            if args.device == 'cuda':
                torch.cuda.empty_cache()
            continue
        cam_poses = np.load(cam_poses_path)["camera_poses"]
        cam_intrs = np.load(cam_intrs_path)["camera_intrs"]
        cam_poses = torch.from_numpy(cam_poses).to(args.device)
        cam_intrs = torch.from_numpy(cam_intrs).to(args.device)
        splats = preprocess_splats(splats, args.device, args.sh_degree)

        # Concatenate splats from previous views
        if all_splats is None:
            all_splats = {k: v.clone() for k, v in splats.items()}
        else:
            for k in all_splats:
                all_splats[k] = torch.cat([all_splats[k], splats[k]], dim=0)

        # Debug: Print splat color, opacity, and camera pose stats
        print(f"[DEBUG] splats_view_0_to_{view_idx} stats:")
        print(f"  means: min {all_splats['means'].min().item():.4f}, max {all_splats['means'].max().item():.4f}")
        print(f"  opacities: min {all_splats['opacities'].min().item():.4f}, max {all_splats['opacities'].max().item():.4f}, mean {all_splats['opacities'].mean().item():.4f}, sum {all_splats['opacities'].sum().item():.4f}")
        if 'sh' in all_splats:
            print(f"  sh: min {all_splats['sh'].min().item():.4f}, max {all_splats['sh'].max().item():.4f}, mean {all_splats['sh'].mean().item():.4f}, abs sum {all_splats['sh'].abs().sum().item():.4f}")
            print("  sh sample (first 3 splats, first 5 bands):")
            print(all_splats['sh'][:3, :5, :])
        if 'colors' in all_splats:
            print(f"  colors: min {all_splats['colors'].min().item():.4f}, max {all_splats['colors'].max().item():.4f}, mean {all_splats['colors'].mean().item():.4f}, abs sum {all_splats['colors'].abs().sum().item():.4f}")
        print(f"  cam_poses: min {cam_poses.min().item():.4f}, max {cam_poses.max().item():.4f}, mean {cam_poses.mean().item():.4f}")
        print(f"  cam_intrs: min {cam_intrs.min().item():.4f}, max {cam_intrs.max().item():.4f}, mean {cam_intrs.mean().item():.4f}")
        if all_splats["opacities"].sum() == 0:
            print(f"Warning: all opacities zero in splats_view_0_to_{view_idx}")
        if "sh" in all_splats and all_splats["sh"].abs().sum() == 0:
            print(f"Warning: all SH coefficients zero in splats_view_0_to_{view_idx}")
        elif "colors" in all_splats and all_splats["colors"].abs().sum() == 0:
            print(f"Warning: all colors zero in splats_view_0_to_{view_idx}")

        # Render in chunks
        rgbs, depths = render_in_chunks(renderer, all_splats, cam_poses, cam_intrs, args.height, args.width, args.sh_degree, chunk_size=args.chunk_size)
        V_out = rgbs.shape[1]
        ply_base = f"splats_views_0_to_{view_idx}"
        render_dir = os.path.join(args.output_dir, ply_base)
        os.makedirs(render_dir, exist_ok=True)
        for vc in range(V_out):
            try:
                rgb = rgbs[0, vc]
                print(f"[DEBUG] rgb min: {rgb.min().item():.4f}, max: {rgb.max().item():.4f}, mean: {rgb.mean().item():.4f}")
                rgb = rgb.clamp(0, 1)
                if rgb.shape[0] == 3 and rgb.ndim == 3:
                    rgb = rgb.permute(1, 2, 0)  # [H, W, 3]
                rgb_img = (rgb * 255).to(torch.uint8).cpu().numpy()
                Image.fromarray(rgb_img).save(os.path.join(render_dir, f"render_view_{vc:02d}_rgb.png"))
                depth = depths[0, vc, :, :, 0].clamp(0, None)
                depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
                depth_img = (depth_normalized * 255).to(torch.uint8).cpu().numpy()
                Image.fromarray(depth_img).save(os.path.join(render_dir, f"render_view_{vc:02d}_depth.png"))
                print(f"   Rendered view {vc}")
            except Exception as e:
                print(f"  Failed to save render for view {vc}: {e}")
        del rgbs, depths
        if args.device == 'cuda':
            torch.cuda.empty_cache()
        del cam_poses, cam_intrs
        if args.device == 'cuda':
            torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
