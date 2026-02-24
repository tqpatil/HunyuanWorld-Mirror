
import argparse
from pathlib import Path
import torch
from src.models.models.worldmirror import WorldMirror
from src.utils.inference_utils import prepare_images_to_tensor
from src.utils.video_utils import select_frames_from_dl3dv
from src.utils.render_utils import save_incremental_splats_and_render





def main():
    parser = argparse.ArgumentParser(description="Incremental splats and camera pose extraction")
    parser.add_argument("--input_path", type=str, required=True, help="Input root directory containing 1K,2K,.../uuid/scene folders.")
    parser.add_argument("--output_path", type=str, required=True, help="Output root directory to mirror input structure.")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of scenes to process in parallel.")
    parser.add_argument("--target_size", type=int, default=518, help="Target size for image resizing")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WorldMirror.from_pretrained("tencent/HunyuanWorld-Mirror").to(device)
    model.eval()

    input_root = Path(args.input_path)
    output_root = Path(args.output_path)

    # Find all scene folders: input_root/1K/uuid, input_root/2K/uuid, ...
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
        print(f"\nProcessing batch: {batch_start} to {batch_start+len(batch_scene_dirs)-1}")
        for scene_dir in batch_scene_dirs:
            subset_name = scene_dir.parent.name
            uuid_name = scene_dir.name
            print(f"\n=== Processing scene: {subset_name}/{uuid_name} ===")
            outdir = output_root / subset_name / uuid_name
            outdir.mkdir(parents=True, exist_ok=True)

            # Select frames using DL3DV logic (saves selected frames in output)
            input_frames_dir = outdir / "input_frames"
            input_frames_dir.mkdir(exist_ok=True)
            img_paths = select_frames_from_dl3dv(str(scene_dir), n=10, output_dir=str(input_frames_dir))
            if not img_paths:
                print(f"❌ Failed to select frames for scene {scene_dir}")
                continue
            img_paths = sorted(img_paths)
            print(f"✅ Selected {len(img_paths)} frames for {scene_dir}")

            # Load and preprocess images
            views = {}
            imgs = prepare_images_to_tensor(img_paths, target_size=args.target_size, resize_strategy="crop").to(device)  # [1,S,3,H,W], in [0,1]
            views["img"] = imgs
            B, S, C, H, W = imgs.shape
            cond_flags = [0, 0, 0]
            print(f"📸 Loaded {S} images with shape {imgs.shape}")

            # Inference
            print("\n🚀 Starting inference pipeline...")
            with torch.no_grad():
                predictions = model(views=views, cond_flags=cond_flags)
            print(f"✅ Inference complete.")

            # Save incremental splats and camera poses/intrinsics
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
                final_mask=None,
                cam_poses=predictions['camera_poses'],
                cam_intrs=predictions['camera_intrs'],
            )
if __name__ == "__main__":
    main()