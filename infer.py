import argparse
import glob
from pathlib import Path
import os
import time

import cv2
import numpy as np
import torch
from PIL import Image
import onnxruntime

from src.models.models.worldmirror import WorldMirror
from src.utils.inference_utils import prepare_images_to_tensor
from src.utils.video_utils import select_frames_by_camera_poses, select_frames_from_dl3dv
from src.models.utils.geometry import depth_to_world_coords_points
from src.models.utils.geometry import create_pixel_coordinate_grid

from src.utils.save_utils import save_depth_png, save_depth_npy, save_normal_png
from src.utils.save_utils import save_scene_ply, save_gs_ply, save_points_ply
from src.utils.render_utils import render_interpolated_video

from src.utils.build_pycolmap_recon import build_pycolmap_reconstruction
from src.models.utils.camera_utils import vector_to_camera_matrices
from src.utils.render_utils import save_incremental_splats_and_render

# Import mask computation utilities
from src.utils.geometry import depth_edge, normals_edge
from src.utils.visual_util import segment_sky, download_file_from_url


def create_filter_mask(
    pts3d_conf: np.ndarray,
    depth_preds: np.ndarray, 
    normal_preds: np.ndarray,
    sky_mask: np.ndarray,
    confidence_percentile: float = 10.0,
    edge_normal_threshold: float = 5.0,
    edge_depth_threshold: float = 0.03,
    apply_confidence_mask: bool = True,
    apply_edge_mask: bool = True,
    apply_sky_mask: bool = False,
) -> np.ndarray:
    """
    Create comprehensive filter mask based on confidence, edges, and sky segmentation.
    This follows the same logic as app.py for consistent mask computation.
    
    Args:
        pts3d_conf: Point confidence scores [S, H, W]
        depth_preds: Depth predictions [S, H, W, 1]
        normal_preds: Normal predictions [S, H, W, 3]
        sky_mask: Sky segmentation mask [S, H, W]
        confidence_percentile: Percentile threshold for confidence filtering (0-100)
        edge_normal_threshold: Normal angle threshold in degrees for edge detection
        edge_depth_threshold: Relative depth threshold for edge detection
        apply_confidence_mask: Whether to apply confidence-based filtering
        apply_edge_mask: Whether to apply edge-based filtering
        apply_sky_mask: Whether to apply sky mask filtering
    
    Returns:
        final_mask: Boolean mask array [S, H, W] for filtering points
    """
    S, H, W = pts3d_conf.shape[:3]
    final_mask_list = []
    
    for i in range(S):
        final_mask = None
        
        if apply_confidence_mask:
            # Compute confidence mask based on the pointmap confidence
            confidences = pts3d_conf[i, :, :]  # [H, W]
            percentile_threshold = np.quantile(confidences, confidence_percentile / 100.0)
            conf_mask = confidences >= percentile_threshold
            if final_mask is None:
                final_mask = conf_mask
            else:
                final_mask = final_mask & conf_mask
        
        if apply_edge_mask:
            # Compute edge mask based on the normalmap
            normal_pred = normal_preds[i]  # [H, W, 3]
            normal_edges = normals_edge(
                normal_pred, tol=edge_normal_threshold, mask=final_mask
            )
            # Compute depth mask based on the depthmap
            depth_pred = depth_preds[i, :, :, 0]  # [H, W]
            depth_edges = depth_edge(
                depth_pred, rtol=edge_depth_threshold, mask=final_mask
            )
            edge_mask = ~(depth_edges & normal_edges)
            if final_mask is None:
                final_mask = edge_mask
            else:
                final_mask = final_mask & edge_mask
        
        if apply_sky_mask:
            # Apply sky mask filtering (sky_mask is already inverted: True = non-sky)
            sky_mask_frame = sky_mask[i]  # [H, W]
            if final_mask is None:
                final_mask = sky_mask_frame
            else:
                final_mask = final_mask & sky_mask_frame
        
        final_mask_list.append(final_mask)
    
    # Stack all frame masks
    if final_mask_list[0] is not None:
        final_mask = np.stack(final_mask_list, axis=0)  # [S, H, W]
    else:
        final_mask = np.ones(pts3d_conf.shape[:3], dtype=bool)  # [S, H, W]
    
    return final_mask


def main():
    parser = argparse.ArgumentParser(description="HunyuanWorld-Mirror inference")
    parser.add_argument("--input_path", type=str, default="examples/realistic", help="Input root directory containing 1K/2K/3K... subfolders, each with scene UUID folders.")
    parser.add_argument("--output_path", type=str, default="inference_output", help="Output root directory.")
    parser.add_argument("--fps", type=int, default=1, help="Frames per second for video extraction")
    parser.add_argument("--target_size", type=int, default=518, help="Target size for image resizing")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing scenes")
    parser.add_argument("--write_txt", action="store_true", help="Also write human-readable COLMAP txt (slow, huge)")
    # Mask filtering parameters
    parser.add_argument("--confidence_percentile", type=float, default=10.0, help="Confidence percentile threshold for filtering (0-100, filters bottom X percent)")
    parser.add_argument("--edge_normal_threshold", type=float, default=5.0, help="Normal angle threshold in degrees for edge detection")
    parser.add_argument("--edge_depth_threshold", type=float, default=0.03, help="Relative depth threshold for edge detection")
    parser.add_argument("--apply_confidence_mask", action="store_true", default=True, help="Apply confidence-based filtering")
    parser.add_argument("--apply_edge_mask", action="store_true", default=True, help="Apply edge-based filtering")
    parser.add_argument("--apply_sky_mask", action="store_true", default=False, help="Apply sky mask filtering")
    # Save flags
    parser.add_argument("--save_pointmap", action="store_true", default=True, help="Save points PLY")
    parser.add_argument("--save_depth", action="store_true", default=True, help="Save depth PNG")
    parser.add_argument("--save_normal", action="store_true", default=True, help="Save normal PNG")
    parser.add_argument("--save_gs", action="store_true", default=True, help="Save Gaussians PLY")
    parser.add_argument("--save_rendered", action="store_true", default=True, help="Save rendered video")
    parser.add_argument("--save_colmap", action="store_true", default=True, help="Save COLMAP sparse")
    # Conditioning flags
    parser.add_argument("--cond_pose", action="store_true", help="Use camera pose conditioning if available")
    parser.add_argument("--cond_intrinsics", action="store_true", help="Use intrinsics conditioning if available")
    parser.add_argument("--cond_depth", action="store_true", help="Use depth conditioning if available")
    args = parser.parse_args()

    # Print inference parameters
    print(f"🔧 Configuration:")
    print(f"  - FPS: {args.fps}")
    print(f"  - Target size: {args.target_size}px")
    print(f"  - Mask Filtering:")
    print(f"    - Confidence mask: {'✅' if args.apply_confidence_mask else '❌'} (percentile: {args.confidence_percentile}%)")
    print(f"    - Edge mask: {'✅' if args.apply_edge_mask else '❌'} (normal: {args.edge_normal_threshold}°, depth: {args.edge_depth_threshold})")
    print(f"    - Sky mask: {'✅' if args.apply_sky_mask else '❌'}")
    print(f"  - Conditioning:")
    print(f"    - Pose: {'✅' if args.cond_pose else '❌'}")
    print(f"    - Intrinsics: {'✅' if args.cond_intrinsics else '❌'}")
    print(f"    - Depth: {'✅' if args.cond_depth else '❌'}")


    # 1) Init model - This requires internet access or the huggingface hub cache to be pre-downloaded
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WorldMirror.from_pretrained("tencent/HunyuanWorld-Mirror").to(device)
    model.eval()

    input_root = Path(args.input_path)
    output_root = Path(args.output_path)

    # Find all scene UUID folders under 1K/2K/3K... structure
    scene_dirs = []
    for subdir in sorted(input_root.iterdir()):
        if not subdir.is_dir() or not subdir.name[0].isdigit():
            continue
        for uuid_dir in sorted(subdir.iterdir()):
            if uuid_dir.is_dir():
                scene_dirs.append(uuid_dir)

    print(f"[DEBUG] Found {len(scene_dirs)} scenes to process in {input_root}.")
    print(f"[DEBUG] Output root is {output_root} (cwd: {os.getcwd()})")

    # Batch processing
    batch_size = args.batch_size
    for batch_start in range(0, len(scene_dirs), batch_size):
        print(f"[DEBUG] Starting batch at index {batch_start}")
        batch_scene_dirs = scene_dirs[batch_start:batch_start+batch_size]
        batch_imgs = []
        batch_views = []
        batch_img_paths = []
        batch_scene_names = []
        batch_H, batch_W = None, None
        for scene_dir in batch_scene_dirs:
            print(f"[DEBUG] Processing scene_dir: {scene_dir}")
            # Find input images or video in uuid_dir
            img_paths = []
            for ext in ["*.jpeg", "*.jpg", "*.png", "*.webp"]:
                img_paths.extend(sorted(scene_dir.glob(ext)))
            video_exts = ['.mp4', '.avi', '.mov', '.webm', '.gif']
            if not img_paths:
                # Try video
                for file in scene_dir.iterdir():
                    if file.suffix.lower() in video_exts:
                        print(f"[DEBUG] Found video file: {file}")
                        # Extract frames
                        input_frames_dir = scene_dir / "input_frames"
                        input_frames_dir.mkdir(exist_ok=True)
                        img_paths = select_frames_by_camera_poses(str(file), n=10, output_dir=str(input_frames_dir))
                        img_paths = sorted(img_paths)
                        break
            print(f"[DEBUG] img_paths for scene_dir {scene_dir}: {img_paths}")
            if not img_paths:
                print(f"[DEBUG] No images or video found in {scene_dir}, skipping.")
                continue
            imgs = prepare_images_to_tensor(img_paths, target_size=args.target_size, resize_strategy="crop").to(device)
            batch_imgs.append(imgs)
            batch_views.append({"img": imgs})
            batch_img_paths.append(img_paths)
            batch_scene_names.append(scene_dir)
            if batch_H is None or batch_W is None:
                batch_H, batch_W = imgs.shape[-2], imgs.shape[-1]

        if not batch_imgs:
            print(f"[DEBUG] No valid images found in batch starting at {batch_start}, skipping batch.")
            continue

        # Stack for batch
        imgs_batch = torch.cat(batch_imgs, dim=0)  # [B, S, 3, H, W]
        views_batch = {"img": imgs_batch}
        B, S, C, H, W = imgs_batch.shape
        cond_flags = [0, 0, 0]

        print(f"[DEBUG] Processing batch of {B} scenes, {S} views each.")
        # Inference
        with torch.no_grad():
            use_amp = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            amp_dtype = torch.bfloat16 if use_amp else torch.float32
            with torch.amp.autocast('cuda', enabled=bool(use_amp), dtype=amp_dtype):
                predictions_batch = model(views=views_batch, cond_flags=cond_flags)

        # For each scene in batch, save outputs in organized structure
        for bidx, scene_dir in enumerate(batch_scene_names):
            scene_uuid = scene_dir.name
            parent_dir = scene_dir.parent.name
            outdir = output_root / parent_dir / scene_uuid
            print(f"[DEBUG] Creating output directory: {outdir}")
            outdir.mkdir(parents=True, exist_ok=True)

            # Save input frames
            input_frames_dir = outdir / "input_frames"
            print(f"[DEBUG] Creating input_frames directory: {input_frames_dir}")
            input_frames_dir.mkdir(exist_ok=True)
            for i, img_path in enumerate(batch_img_paths[bidx]):
                fname = f"image_{i+1:04d}.png"
                img = Image.open(img_path).convert("RGB")
                img.save(str(input_frames_dir / fname))

            # Save incremental splats and renders
            if "splats" in predictions_batch and args.save_gs:
                print(f"[DEBUG] Saving incremental splats for {scene_uuid}")
                model.gs_renderer.voxel_size = 0.002
                save_incremental_splats_and_render(
                    predictions_batch["splats"],
                    predictions_batch,
                    model.gs_renderer,
                    outdir,
                    H,
                    W,
                    save_ply=True,
                    save_renders=True,
                    final_mask=None,
                    cam_poses=predictions_batch.get('camera_poses'),
                    cam_intrs=predictions_batch.get('camera_intrs'),
                )
            else:
                print(f"[DEBUG] No splats found or save_gs not set for {scene_uuid}")

            # Save predicted camera poses
            if "camera_poses" in predictions_batch:
                cam_poses = predictions_batch["camera_poses"][bidx].detach().cpu().numpy()  # [S, 4, 4]
                np.save(outdir / "predicted_camera_poses.npy", cam_poses)
                print(f"[DEBUG] Saved predicted_camera_poses.npy for {scene_uuid}")
            if "camera_intrs" in predictions_batch:
                cam_intrs = predictions_batch["camera_intrs"][bidx].detach().cpu().numpy()  # [S, 3, 3]
                np.save(outdir / "predicted_camera_intrs.npy", cam_intrs)
                print(f"[DEBUG] Saved predicted_camera_intrs.npy for {scene_uuid}")

            # Optionally, save other outputs (depth, normals, etc.) as needed
if __name__ == "__main__":
    main()



