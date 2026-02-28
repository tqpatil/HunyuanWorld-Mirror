
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
        ply_files = sorted(inc_dir.glob("splats_delta_*.ply"))
        if not ply_files:
            print(f"No delta splat files found in {inc_dir}, skipping.")
            continue
        cumulative_splats = {"means": [], "scales": [], "quats": [], "opacities": [], "sh": []}
        num_steps = len(ply_files)
        for step in range(num_steps):
            ply_file = ply_files[step]
            splats = load_gs_ply(ply_file)
            for k in cumulative_splats:
                cumulative_splats[k].append(splats[k])
            means = torch.cat(cumulative_splats["means"], dim=0)
            scales = torch.cat(cumulative_splats["scales"], dim=0)
            quats = torch.cat(cumulative_splats["quats"], dim=0)
            opacities = torch.cat(cumulative_splats["opacities"], dim=0)
            sh = torch.cat(cumulative_splats["sh"], dim=0)
            cam_poses_file = inc_dir / f"camera_poses_0to{step+1}.npy"
            cam_intrs_file = inc_dir / f"camera_intrs_0to{step+1}.npy"
            if not cam_poses_file.exists() or not cam_intrs_file.exists():
                print(f"Missing camera files for step {step+1}, skipping.")
                continue
            cam_poses = np.load(cam_poses_file)
            cam_intrs = np.load(cam_intrs_file)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            cam_poses_torch = torch.from_numpy(cam_poses).unsqueeze(0).to(device)
            cam_intrs_torch = torch.from_numpy(cam_intrs).unsqueeze(0).to(device)
            means = means.to(device)
            quats = quats.to(device)
            scales = scales.to(device)
            opacities = opacities.to(device)
            sh = sh.to(device)
            gs_renderer = GaussianSplatRenderer(voxel_size=0.002)
            if hasattr(gs_renderer, 'to'):
                gs_renderer = gs_renderer.to(device)
            # Directory structure and naming to match save_incremental_splats_and_render
            save_root = Path("/mnt/temp-data-volume/saved_renders")
            save_root.mkdir(parents=True, exist_ok=True)
            inc_dir_name = inc_dir.name
            renders_dir = save_root / inc_dir_name / f"renders_views_0to{step+1}"
            renders_dir.mkdir(parents=True, exist_ok=True)
            # --- Match reference rendering process exactly ---
            pruned_splats = {
                "means": means,
                "quats": quats,
                "scales": scales,
                "opacities": opacities,
                "sh": sh
            }
            means_r = pruned_splats["means"].unsqueeze(0) if pruned_splats["means"].ndim == 2 else pruned_splats["means"]  # [1, N, 3/4]
            quats_r = pruned_splats["quats"].unsqueeze(0) if pruned_splats["quats"].ndim == 2 else pruned_splats["quats"]  # [1, N, 4]
            scales_r = pruned_splats["scales"].unsqueeze(0) if pruned_splats["scales"].ndim == 2 else pruned_splats["scales"]  # [1, N, 3]
            opacities_r = pruned_splats["opacities"].unsqueeze(0) if pruned_splats["opacities"].ndim == 1 else pruned_splats["opacities"]  # [1, N]
            sh_r = pruned_splats["sh"].unsqueeze(0) if pruned_splats["sh"].ndim == 3 else pruned_splats["sh"]  # [1, N, num_sh_coeffs, 3]
            try:
                cams_c2w = cam_poses_torch.to(torch.float32)
                cams_K = cam_intrs_torch.to(torch.float32)
                colors_arg = sh_r if "sh" in pruned_splats else pruned_splats.get("colors")
                render_colors, render_depths, _ = gs_renderer.rasterizer.rasterize_batches(
                    means_r, quats_r, scales_r, opacities_r,
                    colors_arg,
                    cams_c2w, cams_K,
                    width=W, height=H,
                    sh_degree=gs_renderer.sh_degree if "sh" in pruned_splats else None,
                )
                V_out = render_colors.shape[1]
                for v in range(V_out):
                    try:
                        rgb = render_colors[0, v].clamp(0, 1)
                        rgb_img = (rgb * 255).to(torch.uint8).cpu().numpy()
                        Image.fromarray(rgb_img).save(str(renders_dir / f"render_view_{v:02d}_rgb.png"))
                        depth = render_depths[0, v, :, :, 0].clamp(0, None)
                        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
                        depth_img = (depth_normalized * 255).to(torch.uint8).cpu().numpy()
                        Image.fromarray(depth_img).save(str(renders_dir / f"render_view_{v:02d}_depth.png"))
                        print(f"   Rendered view {v}")
                    except Exception as e:
                        print(f"  Failed to save render for view {v}: {e}")
            except Exception as e:
                print(f" Failed to render views 0..{step+1}: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    main()
    
