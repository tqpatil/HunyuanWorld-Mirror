import argparse
from pathlib import Path
import os
import numpy as np
import torch
import onnxruntime
import cv2

from src.models.models.worldmirror import WorldMirror
from src.utils.inference_utils import prepare_images_to_tensor
from src.utils.video_utils import select_frames_from_dl3dv
from src.utils.render_utils import save_incremental_splats_and_render
from src.utils.visual_util import segment_sky, download_file_from_url


def compute_sky_mask(img_paths, H, W):
    """
    Compute sky mask for all frames.
    Returns:
        sky_mask: [S, H, W] boolean
                  True = keep pixel
                  False = sky pixel (remove)
    """
    print("\n🌤️ Computing sky masks...")

    if not os.path.exists("skyseg.onnx"):
        print("Downloading skyseg.onnx...")
        download_file_from_url(
            "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx",
            "skyseg.onnx"
        )

    skyseg_session = onnxruntime.InferenceSession("skyseg.onnx")

    sky_mask_list = []
    for img_path in img_paths:
        sky_mask_frame = segment_sky(img_path, skyseg_session)

        # Resize mask if needed
        if sky_mask_frame.shape[0] != H or sky_mask_frame.shape[1] != W:
            sky_mask_frame = cv2.resize(
                sky_mask_frame, (W, H), interpolation=cv2.INTER_NEAREST
            )

        sky_mask_list.append(sky_mask_frame)

    sky_mask = np.stack(sky_mask_list, axis=0)  # [S, H, W]
    sky_mask = sky_mask > 0  # True = non-sky, False = sky

    print(f"✅ Sky masks computed for {len(img_paths)} frames")
    return sky_mask


def main():
    parser = argparse.ArgumentParser(description="Incremental splats and camera pose extraction")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--target_size", type=int, default=518)
    parser.add_argument("--apply_sky_mask", action="store_true", default=True,
                        help="Remove sky pixels from splats")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WorldMirror.from_pretrained("tencent/HunyuanWorld-Mirror").to(device)
    model.eval()

    input_root = Path(args.input_path)
    output_root = Path(args.output_path)

    scene_dirs = []
    for subset_dir in sorted(input_root.iterdir()):
        if not subset_dir.is_dir():
            continue
        for uuid_dir in sorted(subset_dir.iterdir()):
            if not uuid_dir.is_dir():
                continue
            if (uuid_dir / "transforms.json").exists():
                scene_dirs.append(uuid_dir)

    print(f"Found {len(scene_dirs)} scenes to process.")

    for batch_start in range(0, len(scene_dirs), args.batch_size):
        batch_scene_dirs = scene_dirs[batch_start:batch_start+args.batch_size]

        for scene_dir in batch_scene_dirs:
            subset_name = scene_dir.parent.name
            uuid_name = scene_dir.name
            print(f"\n=== Processing scene: {subset_name}/{uuid_name} ===")

            outdir = output_root / subset_name / uuid_name
            outdir.mkdir(parents=True, exist_ok=True)

            input_frames_dir = outdir / "input_frames"
            input_frames_dir.mkdir(exist_ok=True)

            img_paths = select_frames_from_dl3dv(
                str(scene_dir), n=10, output_dir=str(input_frames_dir)
            )
            if not img_paths:
                print(f"❌ Failed to select frames for {scene_dir}")
                continue

            img_paths = sorted(img_paths)
            print(f"✅ Selected {len(img_paths)} frames")

            # Load images
            views = {}
            imgs = prepare_images_to_tensor(
                img_paths,
                target_size=args.target_size,
                resize_strategy="crop"
            ).to(device)

            views["img"] = imgs
            B, S, C, H, W = imgs.shape
            cond_flags = [0, 0, 0]

            print(f"📸 Loaded {S} images of shape {imgs.shape}")

            # Inference
            print("🚀 Running inference...")
            with torch.no_grad():
                predictions = model(views=views, cond_flags=cond_flags)

            print("✅ Inference complete.")

            # -----------------------
            # 🌤 SKY SEGMENTATION
            # -----------------------
            if args.apply_sky_mask:
                sky_mask = compute_sky_mask(img_paths, H, W)
            else:
                sky_mask = np.ones((S, H, W), dtype=bool)

            # Convert to torch for renderer
            final_mask = sky_mask

            # -----------------------
            # Save incremental splats
            # -----------------------
            model.gs_renderer.voxel_size = 0.002

            save_incremental_splats_and_render(
                predictions["splats"],
                predictions,
                model.gs_renderer,
                outdir,
                H,
                W,
                save_ply=True,
                save_renders=False,
                final_mask=final_mask,   # 🔥 SKY FILTER APPLIED HERE
                cam_poses=predictions['camera_poses'],
                cam_intrs=predictions['camera_intrs'],
            )

            print("Done.")
if __name__ == "__main__":
    main()