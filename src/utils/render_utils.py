from pathlib import Path

import numpy as np
import torch
import moviepy.editor as mpy

from src.models.models.rasterization import GaussianSplatRenderer
from src.models.utils.sh_utils import RGB2SH, SH2RGB
from src.utils.gs_effects import GSEffects
from src.utils.color_map import apply_color_map_to_image
from tqdm import tqdm
from PIL import Image

def project_3d_to_2d(points_3d, c2w, K, H, W):
    """
    Project 3D world points to 2D pixel coordinates using camera extrinsics (c2w) and intrinsics (K).
    
    Args:
        points_3d: [N, 3] tensor of 3D points in world coordinates.
        c2w: [4, 4] camera-to-world matrix.
        K: [3, 3] intrinsics matrix.
        H, W: Image height/width.
    
    Returns:
        x, y: [N] tensors of pixel coordinates (0-based).
        valid: [N] boolean tensor indicating if projection is valid (in bounds, z > 0).
    """
    # Transform to camera coordinates: [N, 3] -> [3, N] -> [N, 3]
    points_cam = (c2w[:3, :3] @ points_3d.T + c2w[:3, 3:4]).T
    
    # Project to homogeneous 2D: [3, N]
    uv_hom = K @ points_cam.T
    
    # Normalize to pixel coords: [2, N]
    uv = uv_hom[:2] / uv_hom[2]
    
    x, y = uv[0], uv[1]
    
    # Validity: in image bounds and in front of camera
    valid = (x >= 0) & (x < W) & (y >= 0) & (y < H) & (points_cam[:, 2] > 0)
    
    return x, y, valid


def rotation_matrix_to_quaternion(R):
    """Convert rotation matrix to quaternion"""
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    
    q = torch.zeros(R.shape[:-2] + (4,), device=R.device, dtype=R.dtype)
    
    # Case where trace > 0
    mask1 = trace > 0
    s = torch.sqrt(trace[mask1] + 1.0) * 2  # s=4*qw 
    q[mask1, 0] = 0.25 * s  # qw
    q[mask1, 1] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s  # qx
    q[mask1, 2] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / s  # qy
    q[mask1, 3] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / s  # qz
    
    # Case where R[0,0] > R[1,1] and R[0,0] > R[2,2]
    mask2 = (~mask1) & (R[..., 0, 0] > R[..., 1, 1]) & (R[..., 0, 0] > R[..., 2, 2])
    s = torch.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2  # s=4*qx
    q[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s  # qw
    q[mask2, 1] = 0.25 * s  # qx
    q[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s  # qy
    q[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s  # qz
    
    # Case where R[1,1] > R[2,2]
    mask3 = (~mask1) & (~mask2) & (R[..., 1, 1] > R[..., 2, 2])
    s = torch.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2  # s=4*qy
    q[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s  # qw
    q[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s  # qx
    q[mask3, 2] = 0.25 * s  # qy
    q[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s  # qz
    
    # Remaining case
    mask4 = (~mask1) & (~mask2) & (~mask3)
    s = torch.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2  # s=4*qz
    q[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s  # qw
    q[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s  # qx
    q[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s  # qy
    q[mask4, 3] = 0.25 * s  # qz
    
    return q


def quaternion_to_rotation_matrix(q):
    """Convert quaternion to rotation matrix"""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    # Normalize quaternion
    norm = torch.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    
    R = torch.zeros(q.shape[:-1] + (3, 3), device=q.device, dtype=q.dtype)
    
    R[..., 0, 0] = 1 - 2*(y*y + z*z)
    R[..., 0, 1] = 2*(x*y - w*z)
    R[..., 0, 2] = 2*(x*z + w*y)
    R[..., 1, 0] = 2*(x*y + w*z)
    R[..., 1, 1] = 1 - 2*(x*x + z*z)
    R[..., 1, 2] = 2*(y*z - w*x)
    R[..., 2, 0] = 2*(x*z - w*y)
    R[..., 2, 1] = 2*(y*z + w*x)
    R[..., 2, 2] = 1 - 2*(x*x + y*y)
    
    return R

    
def slerp_quaternions(q1, q2, t):
    """Spherical linear interpolation between quaternions"""
    # Compute dot product
    dot = (q1 * q2).sum(dim=-1, keepdim=True)
    
    # If dot product is negative, slerp won't take the shorter path.
    # Note that q and -q represent the same rotation, so we can flip one.
    mask = dot < 0
    q2 = torch.where(mask, -q2, q2)
    dot = torch.where(mask, -dot, dot)
    
    # If the inputs are too close for comfort, linearly interpolate
    # and normalize the result.
    DOT_THRESHOLD = 0.9995
    mask_linear = dot > DOT_THRESHOLD
    
    result = torch.zeros_like(q1)
    
    # Linear interpolation for close quaternions
    if mask_linear.any():
        result_linear = q1 + t * (q2 - q1)
        norm = torch.norm(result_linear, dim=-1, keepdim=True)
        result_linear = result_linear / norm
        result = torch.where(mask_linear, result_linear, result)
    
    # Spherical interpolation for distant quaternions
    mask_slerp = ~mask_linear
    if mask_slerp.any():
        theta_0 = torch.acos(torch.abs(dot))
        sin_theta_0 = torch.sin(theta_0)
        
        theta = theta_0 * t
        sin_theta = torch.sin(theta)
        
        s0 = torch.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        
        result_slerp = (s0 * q1) + (s1 * q2)
        result = torch.where(mask_slerp, result_slerp, result)
    
    return result


def render_interpolated_video(gs_renderer: GaussianSplatRenderer,
                              splats: dict,
                              camtoworlds: torch.Tensor,
                              intrinsics: torch.Tensor,
                              hw: tuple[int, int],
                              out_path: Path,
                              interp_per_pair: int = 20,
                              loop_reverse: bool = True,
                              effects: GSEffects = None,
                              effect_type: int = 2,
                              save_mode: str = "split") -> None:
    # camtoworlds: [B, S, 4, 4], intrinsics: [B, S, 3, 3]
    b, s, _, _ = camtoworlds.shape
    h, w = hw

    # Build interpolated trajectory
    def build_interpolated_traj(index, nums):
        exts, ints = [], []
        tmp_camtoworlds = camtoworlds[:, index]
        tmp_intrinsics = intrinsics[:, index]
        for i in range(len(index)-1):
            exts.append(tmp_camtoworlds[:, i:i+1])
            ints.append(tmp_intrinsics[:, i:i+1])
            # Extract rotation and translation
            R0, t0 = tmp_camtoworlds[:, i, :3, :3], tmp_camtoworlds[:, i, :3, 3]
            R1, t1 = tmp_camtoworlds[:, i + 1, :3, :3], tmp_camtoworlds[:, i + 1, :3, 3]
            
            # Convert rotations to quaternions
            q0 = rotation_matrix_to_quaternion(R0)
            q1 = rotation_matrix_to_quaternion(R1)
            
            # Interpolate using smooth quaternion slerp
            for j in range(1, nums + 1):
                alpha = j / (nums + 1)
                
                # Linear interpolation for translation
                t_interp = (1 - alpha) * t0 + alpha * t1
                
                # Spherical interpolation for rotation
                q_interp = slerp_quaternions(q0, q1, alpha)
                R_interp = quaternion_to_rotation_matrix(q_interp)
                
                # Create interpolated extrinsic matrix
                ext = torch.eye(4, device=R_interp.device, dtype=R_interp.dtype)[None].repeat(b, 1, 1)
                ext[:, :3, :3] = R_interp
                ext[:, :3, 3] = t_interp
                
                # Linear interpolation for intrinsics
                K0 = tmp_intrinsics[:, i]
                K1 = tmp_intrinsics[:, i + 1]
                K = (1 - alpha) * K0 + alpha * K1
                
                exts.append(ext[:, None])
                ints.append(K[:, None])

        exts = torch.cat(exts, dim=1)[:1]
        ints = torch.cat(ints, dim=1)[:1]
        return exts, ints
    
    # Build wobble trajectory
    def build_wobble_traj(nums, delta):
        assert s==1
        t = torch.linspace(0, 1, nums, dtype=torch.float32, device=camtoworlds.device)
        t = (torch.cos(torch.pi * (t + 1)) + 1) / 2
        tf = torch.eye(4, dtype=torch.float32, device=camtoworlds.device)
        radius = delta * 0.15
        tf = tf.broadcast_to((*radius.shape, t.shape[0], 4, 4)).clone()
        radius = radius[..., None]
        radius = radius * t
        tf[..., 0, 3] = torch.sin(2 * torch.pi * t) * radius
        tf[..., 1, 3] = -torch.cos(2 * torch.pi * t) * radius
        exts = camtoworlds @ tf
        ints = intrinsics.repeat(1, exts.shape[1], 1, 1)
        return exts, ints
    
    if s > 1:
        all_ext, all_int = build_interpolated_traj([i for i in range(s)], interp_per_pair)
    else:
        all_ext, all_int = build_wobble_traj(interp_per_pair * 12, splats["means"][0].median(dim=0).values.norm(dim=-1)[None])

    rendered_rgbs, rendered_depths = [], []
    chunk = 40 
    t = 0
    t_skip = 0
    if False:
        try:
            pruned_splats = gs_renderer.prune_gs(splats, gs_renderer.voxel_size)
        except:
            pruned_splats = splats
        # indices = [x for x in range(0, all_ext.shape[1], 2)][:4]
        # add_ext, add_int = build_interpolated_traj(indices, 150)
        # add_ext = torch.flip(add_ext, dims=[1])
        # add_int = torch.flip(add_int, dims=[1])
        add_ext = all_ext[:, :1, :, :].repeat(1, 320, 1, 1)
        add_int = all_int[:, :1, :, :].repeat(1, 320, 1, 1)
        shift = pruned_splats["means"][0].median(dim=0).values
        scale_factor = (pruned_splats["means"][0] - shift).abs().quantile(0.95, dim=0).max()
        all_ext[0, :, :3, -1] = (all_ext[0, :, :3, -1] - shift) / scale_factor
        add_ext[0, :, :3, -1] = (add_ext[0, :, :3, -1] - shift) / scale_factor
        flag = None
        try:
            raw_splats = gs_renderer.rasterizer.runner.splats
        except:
            pass
        for st in range(0, add_ext.shape[1]):
            ed = min(st + 1, add_ext.shape[1])
            assert gs_renderer.sh_degree == 0
            if flag is not None and (flag < 0.99).any():
                break
            sample_gsplat = {"means": (pruned_splats["means"][0] - shift)/scale_factor, "quats": pruned_splats["quats"][0], "scales": pruned_splats["scales"][0]/scale_factor, 
                            "opacities": pruned_splats["opacities"][0],"colors": SH2RGB(pruned_splats["sh"][0].reshape(-1, 3))}
            effects_splats, flag = effects.apply_effect(sample_gsplat, t, effect_type=effect_type)
            t += 0.04
            effects_splats["sh"] = RGB2SH(effects_splats["colors"]).reshape(-1, 1, 3)
            try:
                gs_renderer.rasterizer.runner.splats
                effects_splats["sh0"] = effects_splats["sh"][:, :1, :]
                effects_splats["shN"] = effects_splats["sh"][:, 1:, :]
                effects_splats["scales"] = effects_splats["scales"].log()
                effects_splats["opacities"] = torch.logit(torch.clamp(effects_splats["opacities"], 1e-6, 1 - 1e-6))
                gs_renderer.rasterizer.runner.splats = effects_splats
                colors, depths, _ = gs_renderer.rasterizer.rasterize_batches(
                None, None, None, 
                None, None,
                add_ext[:, st:ed].to(torch.float32), add_int[:, st:ed].to(torch.float32),
                width=w, height=h, sh_degree=gs_renderer.sh_degree,
                )
            except:
                colors, depths, _ = gs_renderer.rasterizer.rasterize_batches(
                effects_splats["means"][None], effects_splats["quats"][None], effects_splats["scales"][None], 
                effects_splats["opacities"][None], effects_splats["sh"][None],
                add_ext[:, st:ed].to(torch.float32), add_int[:, st:ed].to(torch.float32),
                width=w, height=h, sh_degree=gs_renderer.sh_degree if "sh" in pruned_splats else None,
                )
            
            if st > add_ext.shape[1]*0.14:
                t_skip = t if t_skip == 0 else t_skip
                # break
                rendered_rgbs.append(colors)
                rendered_depths.append(depths)
            # if (flag == 0).all():
            #     break
    t_st = t
    t_ed = 0
    loop_dir = 1
    ignore_scale = False
    for st in tqdm(range(0, all_ext.shape[1], chunk)):
        ed = min(st + chunk, all_ext.shape[1])
        if False:
            try:
                sample_gsplat = {"means": (pruned_splats["means"][0] - shift)/scale_factor, "quats": pruned_splats["quats"][0], "scales": pruned_splats["scales"][0]/scale_factor, 
                                "opacities": pruned_splats["opacities"][0],"colors": SH2RGB(pruned_splats["sh"][0].reshape(-1, 3))}
            except:
                sample_gsplat = {"means": (pruned_splats["means"][0] - shift)/scale_factor, "quats": pruned_splats["quats"][0], "scales": pruned_splats["scales"][0]/scale_factor, 
                                "opacities": pruned_splats["opacities"][0],"colors": SH2RGB(pruned_splats["sh"][0].reshape(-1, 3))}
            effects_splats, flag = effects.apply_effect(sample_gsplat, t, effect_type=effect_type, ignore_scale=ignore_scale)
            if loop_dir < 0:
                t -= 0.04
            else:
                t += 0.04
            if flag.mean() < 0.01 and t_ed == 0:
                t_ed = t
            effects_splats["sh"] = RGB2SH(effects_splats["colors"]).reshape(-1, 1, 3)
            effects_splats["sh0"] = effects_splats["sh"][:, :1, :]
            effects_splats["shN"] = effects_splats["sh"][:, 1:, :]
            try:
                gs_renderer.rasterizer.runner.splats
                effects_splats["sh0"] = effects_splats["sh"][:, :1, :]
                effects_splats["shN"] = effects_splats["sh"][:, 1:, :]
                effects_splats["scales"] = effects_splats["scales"].log()
                effects_splats["opacities"] = torch.logit(torch.clamp(effects_splats["opacities"], 1e-6, 1 - 1e-6))
                gs_renderer.rasterizer.runner.splats = effects_splats
                colors, depths, _ = gs_renderer.rasterizer.rasterize_batches(
                None, None, None, 
                None, None,
                all_ext[:, st:ed].to(torch.float32), all_int[:, st:ed].to(torch.float32),
                width=w, height=h, sh_degree=gs_renderer.sh_degree,
                )
            except:
                colors, depths, _ = gs_renderer.rasterizer.rasterize_batches(
                effects_splats["means"][None], effects_splats["quats"][None], effects_splats["scales"][None], 
                effects_splats["opacities"][None], effects_splats["sh"][None],
                all_ext[:, st:ed].to(torch.float32), all_int[:, st:ed].to(torch.float32),
                width=w, height=h, sh_degree=gs_renderer.sh_degree if "sh" in pruned_splats else None,
                )
            
            if t > (all_ext.shape[1]) * 0.04 + t_st - (t_ed - t_st)*2 - 15*0.04 or t < t_st:
                # ignore_scale = True
                loop_dir *= -1
                t = t_ed if loop_dir == -1 else t
        else:
            colors, depths, _ = gs_renderer.rasterizer.rasterize_batches(
                splats["means"][:1], splats["quats"][:1], splats["scales"][:1], splats["opacities"][:1],
                splats["sh"][:1] if "sh" in splats else splats["colors"][:1],
                all_ext[:, st:ed].to(torch.float32), all_int[:, st:ed].to(torch.float32),
                width=w, height=h, sh_degree=gs_renderer.sh_degree if "sh" in splats else None,
            )
        rendered_rgbs.append(colors)
        rendered_depths.append(depths)


    rgbs = torch.cat(rendered_rgbs, dim=1)[0]          # [N, H, W, 3]
    depths = torch.cat(rendered_depths, dim=1)[0, ..., 0]     # [N, H, W]


    def depth_vis(d: torch.Tensor) -> torch.Tensor:
        valid = d > 0
        if valid.any():
            near = d[valid].float().quantile(0.01).log()
        else:
            near = torch.tensor(0.0, device=d.device)
        far = d.flatten().float().quantile(0.99).log()
        x = d.float().clamp(min=1e-9).log()
        x = 1.0 - (x - near) / (far - near + 1e-9)
        return apply_color_map_to_image(x, "turbo")
    
    frames = []
    rgb_frames = []
    depth_frames = []

    for rgb, dep in zip(rgbs, depths):
        rgb_img = rgb.permute(2, 0, 1)  # [3, H, W]
        depth_img = depth_vis(dep)      # [3, H, W]

        if save_mode == 'both':
            combined = torch.cat([rgb_img, depth_img], dim=1)  # [3, 2*H, W]
            frames.append(combined)
        elif save_mode == 'split':
            rgb_frames.append(rgb_img)
            depth_frames.append(depth_img)
        else:
            raise ValueError("save_mode must be 'both' or 'split'")

    def _make_video(frames, path):
        video = torch.stack(frames).clamp(0, 1)  # [N, 3, H, W]
        video = video.permute(0, 2, 3, 1)  # [N, H, W, 3] for moviepy
        video = (video * 255).to(torch.uint8).cpu().numpy()
        if loop_reverse and video.shape[0] > 1:
            video = np.concatenate([video, video[::-1][1:-1]], axis=0)
        clip = mpy.ImageSequenceClip(list(video), fps=30)
        clip.write_videofile(str(path), logger=None)

    # Save videos
    if save_mode == 'both':
        _make_video(frames, f"{out_path}.mp4")
    elif save_mode == 'split':
        _make_video(rgb_frames, f"{out_path}_rgb.mp4")
        _make_video(depth_frames, f"{out_path}_depth.mp4")

    print(f"Video saved to {out_path} (mode: {save_mode})")

    if False:
        try:
            gs_renderer.rasterizer.runner.splats = raw_splats
        except:
            pass
    torch.cuda.empty_cache()


def save_incremental_splats_and_render(
    splats,
    predictions,
    gs_renderer,
    output_dir,
    H,
    W,
    save_ply=True,
    save_renders=False,
    final_mask=None,
    cam_poses=None,
    cam_intrs=None,
):
    """
    Save splats incrementally as views accumulate, and render from those views.
    Multiprocessed: each end_view iteration runs in a separate worker process.
    No shared mutable state; data races avoided via process isolation + unique output paths.
    """
    import torch.multiprocessing as mp
    from src.utils.save_utils import save_gs_ply

    output_dir = Path(output_dir)
    incremental_dir = output_dir / "incremental_splats"
    incremental_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Resolve device
    # ------------------------------------------------------------------ #
    device = None
    for v in splats.values():
        if isinstance(v, torch.Tensor) and v.numel() > 0:
            device = v.device
            break
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
            device = v[0].device
            break

    if device is None:
        print("Could not determine device from splats; skipping incremental saving")
        return

    # ------------------------------------------------------------------ #
    # Resolve view_mapping
    # ------------------------------------------------------------------ #
    view_mapping = splats.get("view_mapping", None)
    if view_mapping is None:
        print("No view_mapping in splats; skipping incremental saving")
        return

    if isinstance(view_mapping, list):
        view_mapping_tensor = view_mapping[0]
    else:
        view_mapping_tensor = view_mapping[0] if view_mapping.ndim > 1 else view_mapping

    view_map_b = view_mapping_tensor
    num_views = int(view_map_b.max().item()) + 1

    # ------------------------------------------------------------------ #
    # Validate mask inputs
    # ------------------------------------------------------------------ #
    if final_mask is not None:
        if cam_poses is None or cam_intrs is None:
            raise ValueError("cam_poses and cam_intrs must be provided if final_mask is used")
        final_mask_tensor = torch.from_numpy(final_mask).to(device)
    else:
        final_mask_tensor = None

    print(f"\n Incremental splat saving for {num_views} views")

    # ------------------------------------------------------------------ #
    # Phase 1 (sequential): prune each subset so we can compute deltas.
    # Each iteration only depends on the previous one (prev_pruned_for_save),
    # so this cannot be parallelised without changing semantics.
    # ------------------------------------------------------------------ #
    pruned_results = {}          # end_view -> pruned_for_save dict (CPU tensors)
    prev_pruned_for_save = None

    for end_view in range(1, num_views):
        mask_ev = view_map_b <= end_view
        filtered_splats = {}
        for key in ["means", "quats", "scales", "opacities", "sh", "weights", "view_mapping"]:
            if key in splats:
                splat_entry = splats[key]
                if isinstance(splat_entry, list):
                    filtered_splats[key] = splat_entry[0][mask_ev].clone()
                elif isinstance(splat_entry, torch.Tensor) and splat_entry.ndim >= 2:
                    filtered_splats[key] = splat_entry[0][mask_ev].clone()
                else:
                    filtered_splats[key] = splat_entry

        filtered_splats_batched = {
            k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else v)
            for k, v in filtered_splats.items()
        }

        # Optional mask filtering
        if final_mask_tensor is not None:
            means_f = filtered_splats_batched["means"][0]
            vm_f = filtered_splats_batched["view_mapping"][0]
            cam_poses_0 = cam_poses[0]
            cam_intrs_0 = cam_intrs[0]
            keep_indices = []
            for i in range(means_f.shape[0]):
                v_idx_int = int(vm_f[i].item())
                if v_idx_int >= cam_poses_0.shape[0] or v_idx_int >= cam_intrs_0.shape[0]:
                    continue
                c2w = cam_poses_0[v_idx_int]
                K = cam_intrs_0[v_idx_int]
                pt = means_f[i].unsqueeze(0)
                x, y, valid = project_3d_to_2d(pt, c2w, K, H, W)
                if valid[0]:
                    x0, y0 = int(x[0]), int(y[0])
                    if (0 <= y0 < final_mask_tensor.shape[1] and
                            0 <= x0 < final_mask_tensor.shape[2] and
                            final_mask_tensor[v_idx_int, y0, x0]):
                        keep_indices.append(i)
            for key in ["means", "quats", "scales", "opacities", "sh", "weights", "view_mapping"]:
                if key in filtered_splats_batched:
                    tensor = filtered_splats_batched[key][0]
                    filtered_splats_batched[key] = tensor[keep_indices].unsqueeze(0)
            print(f"   Mask-filtered splats: {len(keep_indices)} retained")

        pruned = gs_renderer.prune_gs(filtered_splats_batched, voxel_size=gs_renderer.voxel_size)
        pruned = {k: (v[0] if isinstance(v, list) and len(v) > 0 else v)
                  for k, v in pruned.items()}

        # Move to CPU so workers can receive them without CUDA IPC issues
        pruned_cpu = {k: (v.cpu() if isinstance(v, torch.Tensor) else v)
                      for k, v in pruned.items()}

        pruned_results[end_view] = {
            "pruned": pruned_cpu,
            "prev_pruned": ({k: (v.cpu() if isinstance(v, torch.Tensor) else v)
                             for k, v in prev_pruned_for_save.items()}
                            if prev_pruned_for_save is not None else None),
        }

        prev_pruned_for_save = pruned   # stays on device for next iteration

    # ------------------------------------------------------------------ #
    # Phase 2 (parallel): save PLY files and render — all I/O, no shared
    # mutable state.  Each worker gets its own deep-copied data.
    # ------------------------------------------------------------------ #
    cam_poses_pred = predictions.get(
        "camera_poses", torch.eye(4, device=device).unsqueeze(0).unsqueeze(0))
    cam_intrs_pred = predictions.get(
        "camera_intrs", torch.eye(3, device=device).unsqueeze(0).unsqueeze(0))
    cam_poses_cpu = cam_poses_pred.detach().cpu()
    cam_intrs_cpu = cam_intrs_pred.detach().cpu()

    worker_args = []
    for end_view in range(1, num_views):
        worker_args.append((
            end_view,
            pruned_results[end_view],   # {"pruned": ..., "prev_pruned": ...}
            cam_poses_cpu,
            cam_intrs_cpu,
            str(incremental_dir),
            H, W,
            save_ply,
            save_renders,
            str(device),               # workers reconstruct device from string
            gs_renderer if save_renders else None,
        ))

    ctx = mp.get_context("spawn")
    num_workers = min(len(worker_args), mp.cpu_count())
    with ctx.Pool(processes=num_workers) as pool:
        pool.starmap(_incremental_worker, worker_args)


# --------------------------------------------------------------------------- #
# Worker function — runs in a child process; no shared state with main process #
# --------------------------------------------------------------------------- #
def _incremental_worker(
    end_view,
    pruned_bundle,       # {"pruned": cpu_dict, "prev_pruned": cpu_dict | None}
    cam_poses_cpu,       # [B, V, 4, 4] CPU tensor
    cam_intrs_cpu,       # [B, V, 3, 3] CPU tensor
    incremental_dir_str,
    H, W,
    save_ply,
    save_renders,
    device_str,
    gs_renderer,
):
    """
    Child-process worker.  All inputs are independent copies; writes only to
    paths uniquely determined by end_view, so no two workers share a file.
    """
    from src.utils.save_utils import save_gs_ply

    incremental_dir = Path(incremental_dir_str)
    device = torch.device(device_str)

    pruned_for_save = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in pruned_bundle["pruned"].items()
    }
    prev_pruned_for_save = (
        {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
         for k, v in pruned_bundle["prev_pruned"].items()}
        if pruned_bundle["prev_pruned"] is not None else None
    )

    curr_pruned_count = pruned_for_save["means"].shape[0] if "means" in pruned_for_save else 0
    pruned_view_multi = pruned_for_save.get("view_mapping_multi", None)

    print(f"\n  Processing views 0..{end_view} ({end_view + 1} views total)")
    print(f"   Views 0..{end_view}: pruned splats={curr_pruned_count}")

    # ---- Delta computation (identical logic to original) ---- #
    delta_indices = None
    if end_view > 0 and prev_pruned_for_save is not None and pruned_view_multi is not None:
        try:
            if end_view < pruned_view_multi.shape[1]:
                mask_involve_new_view = pruned_view_multi[:, end_view]
                curr_K = pruned_for_save["means"].shape[0]
                prev_K = prev_pruned_for_save["means"].shape[0] if prev_pruned_for_save is not None else 0
                num_new_splats = max(0, curr_K - prev_K)
                delta_mask = mask_involve_new_view.clone()
                if num_new_splats > 0:
                    delta_mask[-num_new_splats:] = True
                delta_indices = torch.where(delta_mask)[0]
        except Exception as e:
            print(f"Error computing delta from pruned contributor info: {e}")
            delta_indices = None

    added_pruned_count = len(delta_indices) if delta_indices is not None else 0
    print(f"      added (from delta): {added_pruned_count} splats")

    # ---- Save full PLY ---- #
    if save_ply:
        ply_path = incremental_dir / f"splats_views_0to{end_view}.ply"
        means    = pruned_for_save["means"]
        scales   = pruned_for_save["scales"]
        quats    = pruned_for_save["quats"]
        colors   = pruned_for_save.get("sh", torch.ones_like(means))
        opacities = pruned_for_save["opacities"]

        if means.ndim > 2:     means     = means.reshape(-1, 3)
        if scales.ndim > 2:    scales    = scales.reshape(-1, 3)
        if quats.ndim > 2:     quats     = quats.reshape(-1, 4)
        if colors.ndim > 2:    colors    = colors.reshape(-1, 3)
        if opacities.ndim > 1: opacities = opacities.reshape(-1)

        save_gs_ply(ply_path, means, scales, quats, colors, opacities)
        print(f"    Saved {len(means)} splats (pruned) to {ply_path.name}")

    # ---- Save delta PLY ---- #
    if save_ply and end_view > 0:
        ply_path_delta = incremental_dir / f"splats_delta_{end_view - 1}to{end_view}.ply"
        means_delta = scales_delta = quats_delta = colors_delta = opacities_delta = None

        if delta_indices is not None and len(delta_indices) > 0:
            means_delta    = pruned_for_save["means"][delta_indices]
            scales_delta   = pruned_for_save["scales"][delta_indices]
            quats_delta    = pruned_for_save["quats"][delta_indices]
            colors_delta   = pruned_for_save.get("sh", torch.ones_like(means_delta))[delta_indices]
            opacities_delta = pruned_for_save["opacities"][delta_indices]

        if means_delta is not None:
            if means_delta.ndim > 2:      means_delta     = means_delta.reshape(-1, 3)
            if scales_delta.ndim > 2:     scales_delta    = scales_delta.reshape(-1, 3)
            if quats_delta.ndim > 2:      quats_delta     = quats_delta.reshape(-1, 4)
            if colors_delta.ndim > 2:     colors_delta    = colors_delta.reshape(-1, 3)
            if opacities_delta.ndim > 1:  opacities_delta = opacities_delta.reshape(-1)

            save_gs_ply(ply_path_delta, means_delta, scales_delta, quats_delta,
                        colors_delta, opacities_delta)
            print(f"   Saved {len(means_delta)} delta splats to {ply_path_delta.name}")

    # ---- Camera subset ---- #
    if cam_poses_cpu.ndim == 4:
        cam_poses_subset = cam_poses_cpu[:, :end_view + 1]
        cam_intrs_subset = cam_intrs_cpu[:, :end_view + 1]
    else:
        cam_poses_subset = cam_poses_cpu
        cam_intrs_subset = cam_intrs_cpu

    # ---- Renders ---- #
    if save_renders and gs_renderer is not None:
        renders_dir = incremental_dir / f"renders_views_0to{end_view}"
        renders_dir.mkdir(exist_ok=True)

        means     = pruned_for_save["means"]
        quats     = pruned_for_save["quats"]
        scales    = pruned_for_save["scales"]
        opacities = pruned_for_save["opacities"]
        sh        = pruned_for_save.get("sh")

        def _ensure_batch(t, ndim_unbatched):
            return t.unsqueeze(0) if t.ndim == ndim_unbatched else t

        means     = _ensure_batch(means, 2)
        quats     = _ensure_batch(quats, 2)
        scales    = _ensure_batch(scales, 2)
        opacities = _ensure_batch(opacities, 1)
        if sh is not None:
            sh = _ensure_batch(sh, 3)

        print("DEBUG: means shape", means.shape)
        print("DEBUG: scales shape", scales.shape)
        print("DEBUG: quats shape", quats.shape)
        print("DEBUG: opacities shape", opacities.shape)
        if "colors" in pruned_for_save:
            print("DEBUG: colors shape", pruned_for_save["colors"].shape)
        if sh is not None:
            print("DEBUG: sh shape", sh.shape)

        try:
            cams_c2w    = cam_poses_subset.to(device, torch.float32)
            cams_K      = cam_intrs_subset.to(device, torch.float32)
            colors_arg  = sh if sh is not None else pruned_for_save.get("colors")

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
            print("DEBUG: width", W, "height", H)
            print("DEBUG: sh_degree", gs_renderer.sh_degree if sh is not None else None)

            render_colors, render_depths, _ = gs_renderer.rasterizer.rasterize_batches(
                means, quats, scales, opacities,
                colors_arg,
                cams_c2w, cams_K,
                width=W, height=H,
                sh_degree=gs_renderer.sh_degree if sh is not None else None,
            )

            for v in range(render_colors.shape[1]):
                try:
                    rgb = render_colors[0, v].clamp(0, 1)
                    Image.fromarray(
                        (rgb * 255).to(torch.uint8).cpu().numpy()
                    ).save(str(renders_dir / f"render_view_{v:02d}_rgb.png"))

                    depth = render_depths[0, v, :, :, 0].clamp(0)
                    depth_n = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
                    Image.fromarray(
                        (depth_n * 255).to(torch.uint8).cpu().numpy()
                    ).save(str(renders_dir / f"render_view_{v:02d}_depth.png"))

                    print(f"   Rendered view {v}")
                except Exception as e:
                    print(f"  Failed to save render for view {v}: {e}")

        except Exception as e:
            print(f" Failed to render views 0..{end_view}: {e}")
            import traceback
            traceback.print_exc()

        print(f"Renders saved to {renders_dir.name}")

    # ---- Camera pose/intrinsics npz ---- #
    cam_poses_np = cam_poses_subset.numpy()
    cam_intrs_np = cam_intrs_subset.numpy()
    np.savez(incremental_dir / f"camera_poses_views_0to{end_view}.npz",
             camera_poses=cam_poses_np)
    np.savez(incremental_dir / f"camera_intrs_views_0to{end_view}.npz",
             camera_intrs=cam_intrs_np)
    print(f"Saved camera poses/intrinsics for views 0..{end_view}")
        # prev_pruned_view_multi = pruned_view_multi.clone() if pruned_view_multi is not None else None