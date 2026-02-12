"""
Video utilities for visualization.

"""

import os
import cv2
import numpy as np
import subprocess
from PIL import Image
import tempfile
from pathlib import Path
import pycolmap
def video_to_image_frames(input_video_path, save_directory=None, fps=1):
    """
    Extracts image frames from a video file at the specified frame rate and saves them as JPEG format.
    Supports regular video files, webcam captures, WebM files, and GIF files, including incomplete files.
    
    Args:
        input_video_path: Path to the input video file
        save_directory: Directory to save extracted frames (default: None)
        fps: Number of frames to extract per second (default: 1)
    
    Returns: List of file paths to extracted frames
    """
    extracted_frame_paths = []
    
    # For GIF files, use PIL library for better handling
    if input_video_path.lower().endswith('.gif'):
        try:
            print(f"Processing GIF file using PIL: {input_video_path}")
            
            with Image.open(input_video_path) as gif_img:
                # Get GIF properties
                frame_duration_ms = gif_img.info.get('duration', 100)  # Duration per frame in milliseconds
                gif_frame_rate = 1000.0 / frame_duration_ms if frame_duration_ms > 0 else 10.0  # Convert to frame rate
                
                print(f"GIF properties: {gif_img.n_frames} frames, {gif_frame_rate:.2f} FPS, {frame_duration_ms}ms per frame")
                
                # Calculate sampling interval
                sampling_interval = max(1, int(gif_frame_rate / fps)) if fps < gif_frame_rate else 1
                
                saved_count = 0
                for current_frame_index in range(gif_img.n_frames):
                    gif_img.seek(current_frame_index)
                    
                    # Sample frames based on desired frame rate
                    if current_frame_index % sampling_interval == 0:
                        # Convert to RGB format if necessary
                        rgb_frame = gif_img.convert('RGB')
                        
                        # Convert PIL image to numpy array
                        frame_ndarray = np.array(rgb_frame)
                        
                        # Save frame as JPEG format
                        frame_output_path = os.path.join(save_directory, f"frame_{saved_count:06d}.jpg")
                        pil_image = Image.fromarray(frame_ndarray)
                        pil_image.save(frame_output_path, 'JPEG', quality=95)
                        extracted_frame_paths.append(frame_output_path)
                        saved_count += 1
                
                if extracted_frame_paths:
                    print(f"Successfully extracted {len(extracted_frame_paths)} frames from GIF using PIL")
                    return extracted_frame_paths
                    
        except Exception as error:
            print(f"PIL GIF extraction error: {str(error)}, falling back to OpenCV")
    
    # For WebM files, use FFmpeg directly for more stable processing
    if input_video_path.lower().endswith('.webm'):
        try:
            print(f"Processing WebM file using FFmpeg: {input_video_path}")
            
            # Create a unique output pattern for the frames
            output_frame_pattern = os.path.join(save_directory, "frame_%04d.jpg")
            
            # Use FFmpeg to extract frames at specified frame rate
            ffmpeg_command = [
                "ffmpeg", 
                "-i", input_video_path,
                "-vf", f"fps={fps}",  # Specified frames per second
                "-q:v", "2",     # High quality
                output_frame_pattern
            ]
            
            # Run FFmpeg process
            ffmpeg_process = subprocess.Popen(
                ffmpeg_command, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            process_stdout, process_stderr = ffmpeg_process.communicate()
            
            # Collect all extracted frames
            for filename in sorted(os.listdir(save_directory)):
                if filename.startswith("frame_") and filename.endswith(".jpg"):
                    full_frame_path = os.path.join(save_directory, filename)
                    extracted_frame_paths.append(full_frame_path)
            
            if extracted_frame_paths:
                print(f"Successfully extracted {len(extracted_frame_paths)} frames from WebM using FFmpeg")
                return extracted_frame_paths
            
            print("FFmpeg extraction failed, falling back to OpenCV")
        except Exception as error:
            print(f"FFmpeg extraction error: {str(error)}, falling back to OpenCV")
    
    # Standard OpenCV method for non-WebM files or as fallback
    try:
        video_capture = cv2.VideoCapture(input_video_path)
        
        # For WebM files, try setting more robust decoder options
        if input_video_path.lower().endswith('.webm'):
            video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'VP80'))
        
        source_fps = video_capture.get(cv2.CAP_PROP_FPS)
        extraction_interval = max(1, int(source_fps / fps))  # Extract at specified frame rate
        processed_frame_count = 0
        
        # Set error mode to suppress console warnings
        cv2.setLogLevel(0)
        
        while True:
            read_success, current_frame = video_capture.read()
            if not read_success:
                break
                
            if processed_frame_count % extraction_interval == 0:
                try:
                    # Additional check for valid frame data
                    if current_frame is not None and current_frame.size > 0:
                        rgb_converted_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
                        frame_output_path = os.path.join(save_directory, f"frame_{processed_frame_count:06d}.jpg")
                        cv2.imwrite(frame_output_path, cv2.cvtColor(rgb_converted_frame, cv2.COLOR_RGB2BGR))
                        extracted_frame_paths.append(frame_output_path)
                except Exception as error:
                    print(f"Warning: Failed to process frame {processed_frame_count}: {str(error)}")
                    
            processed_frame_count += 1
            
            # Safety limit to prevent infinite loops
            if processed_frame_count > 1000:
                break
                
        video_capture.release()
        print(f"Extracted {len(extracted_frame_paths)} frames from video using OpenCV")
        
    except Exception as error:
        print(f"Error extracting frames: {str(error)}")
            
    return extracted_frame_paths
def video_to_image_frames_custom(input_video_path, save_directory=None, n=10, return_indices=False):
    """
    Extracts exactly n image frames uniformly from a video file and saves them as JPEG format.
    Always includes the first and last frame.
    Supports regular video files, webcam captures, WebM files, and GIF files, including incomplete files.
    
    Args:
        input_video_path: Path to the input video file
        save_directory: Directory to save extracted frames (default: None)
        n: Total number of frames to extract (n >= 2)
        return_indices: If True, return (frame_paths, frame_indices) tuple; else just frame_paths
    
    Returns: List of file paths to extracted frames, or tuple (paths, indices) if return_indices=True
    """
    extracted_frame_paths = []
    selected_frame_indices = []
    
    # For GIF files, use PIL library for better handling
    if input_video_path.lower().endswith('.gif'):
        try:
            print(f"Processing GIF file using PIL: {input_video_path}")
            
            with Image.open(input_video_path) as gif_img:
                total_frames = gif_img.n_frames
                print(f"GIF properties: {gif_img.n_frames} frames")
                
                # Compute uniform frame indices (always include first and last)
                if n >= total_frames:
                    selected_indices = list(range(total_frames))
                else:
                    selected_indices = np.linspace(0, total_frames - 1, n, dtype=int).tolist()
                
                saved_count = 0
                for current_frame_index in selected_indices:
                    gif_img.seek(current_frame_index)
                    
                    # Convert to RGB format if necessary
                    rgb_frame = gif_img.convert('RGB')
                    
                    # Convert PIL image to numpy array
                    frame_ndarray = np.array(rgb_frame)
                    
                    # Save frame as JPEG format
                    frame_output_path = os.path.join(save_directory, f"frame_{saved_count:06d}.jpg")
                    pil_image = Image.fromarray(frame_ndarray)
                    pil_image.save(frame_output_path, 'JPEG', quality=95)
                    extracted_frame_paths.append(frame_output_path)
                    selected_frame_indices.append(current_frame_index)
                    saved_count += 1
                
                if extracted_frame_paths:
                    print(f"Successfully extracted {len(extracted_frame_paths)} frames from GIF using PIL")
                    return (extracted_frame_paths, selected_frame_indices) if return_indices else extracted_frame_paths
                    
        except Exception as error:
            print(f"PIL GIF extraction error: {str(error)}, falling back to OpenCV")
    
    # For WebM files, use FFmpeg directly for more stable processing
    if input_video_path.lower().endswith('.webm'):
        try:
            print(f"Processing WebM file using FFmpeg: {input_video_path}")
            
            # Create a unique output pattern for the frames
            output_frame_pattern = os.path.join(save_directory, "frame_%04d.jpg")
            
            # Use FFmpeg to extract all frames first
            ffmpeg_command = [
                "ffmpeg", 
                "-i", input_video_path,
                "-vsync", "0",
                "-q:v", "2",
                output_frame_pattern
            ]
            
            # Run FFmpeg process
            ffmpeg_process = subprocess.Popen(
                ffmpeg_command, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            process_stdout, process_stderr = ffmpeg_process.communicate()
            
            # Collect all extracted frames
            all_frames = []
            for filename in sorted(os.listdir(save_directory)):
                if filename.startswith("frame_") and filename.endswith(".jpg"):
                    all_frames.append(os.path.join(save_directory, filename))
            
            total_frames = len(all_frames)
            if total_frames == 0:
                print("FFmpeg extraction failed, falling back to OpenCV")
            else:
                # Compute uniform indices
                if n >= total_frames:
                    selected_indices = list(range(total_frames))
                else:
                    selected_indices = np.linspace(0, total_frames - 1, n, dtype=int).tolist()
                
                for idx in selected_indices:
                    extracted_frame_paths.append(all_frames[idx])
                    selected_frame_indices.append(idx)
                
                print(f"Successfully extracted {len(extracted_frame_paths)} frames from WebM using FFmpeg")
                return (extracted_frame_paths, selected_frame_indices) if return_indices else extracted_frame_paths
            
        except Exception as error:
            print(f"FFmpeg extraction error: {str(error)}, falling back to OpenCV")
    
    # Standard OpenCV method for non-WebM files or as fallback
    try:
        video_capture = cv2.VideoCapture(input_video_path)
        
        # For WebM files, try setting more robust decoder options
        if input_video_path.lower().endswith('.webm'):
            video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'VP80'))
        
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise RuntimeError("Could not determine total frame count")
        
        # Compute uniform frame indices
        if n >= total_frames:
            selected_indices = list(range(total_frames))
        else:
            selected_indices = np.linspace(0, total_frames - 1, n, dtype=int).tolist()
        
        selected_indices_set = set(selected_indices)
        processed_frame_count = 0
        
        # Set error mode to suppress console warnings
        cv2.setLogLevel(0)
        
        saved_count = 0
        while True:
            read_success, current_frame = video_capture.read()
            if not read_success:
                break
                
            if processed_frame_count in selected_indices_set:
                try:
                    if current_frame is not None and current_frame.size > 0:
                        rgb_converted_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
                        frame_output_path = os.path.join(save_directory, f"frame_{saved_count:06d}.jpg")
                        cv2.imwrite(frame_output_path, cv2.cvtColor(rgb_converted_frame, cv2.COLOR_RGB2BGR))
                        extracted_frame_paths.append(frame_output_path)
                        selected_frame_indices.append(processed_frame_count)
                        saved_count += 1
                except Exception as error:
                    print(f"Warning: Failed to process frame {processed_frame_count}: {str(error)}")
                    
            processed_frame_count += 1
            
            # Safety limit to prevent infinite loops
            if processed_frame_count > total_frames + 5:
                break
                
        video_capture.release()
        print(f"Extracted {len(extracted_frame_paths)} frames from video using OpenCV")
        
    except Exception as error:
        print(f"Error extracting frames: {str(error)}")
            
    return (extracted_frame_paths, selected_frame_indices) if return_indices else extracted_frame_paths


def select_frames_by_camera_poses(video_path, n=10, output_dir=None, colmap_temp_dir=None):
    """
    Select n frames from video using COLMAP camera pose estimation and pose-based constraints.
    
    Strategy:
    - Frame 0: first frame
    - Frame i (i > 0): furthest frame (in translation) with rotation <= (i+1)*(180/n) degrees from frame 0
               If no frame satisfies rotation constraint, take the furthest frame anyway
    - Repeat until n frames are selected
    
    Args:
        video_path: Path to input video file
        n: Number of frames to select
        output_dir: Directory to save selected frames (default: same as colmap_temp_dir)
        colmap_temp_dir: Temporary directory for COLMAP processing (default: creates temp dir)
    
    Returns:
        List of paths to selected frames
    """
    
    
    
    # Setup directories
    if colmap_temp_dir is None:
        colmap_temp_dir = tempfile.mkdtemp(prefix="colmap_frames_")
    colmap_temp_dir = Path(colmap_temp_dir)
    colmap_temp_dir.mkdir(parents=True, exist_ok=True)
    
    if output_dir is None:
        output_dir = colmap_temp_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    frames_dir = colmap_temp_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    
    print(f"\n📹 Selecting {n} frames from video using COLMAP poses...")
    print(f"   Temp dir: {colmap_temp_dir}")
    
    # Step 1: Extract all frames from video
    print(f"\n Extracting all frames from video...")
    all_frame_paths = _extract_all_frames(video_path, str(frames_dir))
    if not all_frame_paths:
        print(" Failed to extract frames")
        return None
    
    total_frames = len(all_frame_paths)
    print(f"  Extracted {total_frames} frames")
    
    if total_frames < n:
        print(f"Video has only {total_frames} frames but {n} requested. Returning all frames.")
        return all_frame_paths
    
    # Step 2: Run COLMAP on frames
    print(f"\n Running COLMAP to estimate camera poses...")
    reconstruction = _run_colmap_on_frames(str(frames_dir), colmap_temp_dir)
    if reconstruction is None:
        print("COLMAP failed. Falling back to uniform frame selection.")
        indices = np.linspace(0, total_frames - 1, n, dtype=int)
        return [all_frame_paths[i] for i in indices]
    
    # Step 3: Extract camera poses
    print(f"\n Extracting camera poses...")
    poses = _extract_camera_poses(reconstruction, total_frames)
    if poses is None or len(poses) < n:
        print(" Could not extract enough valid poses. Falling back to uniform selection.")
        indices = np.linspace(0, total_frames - 1, n, dtype=int)
        return [all_frame_paths[i] for i in indices]
    
    # Step 4: Select frames based on pose constraints
    print(f"\n Selecting frames by pose constraints...")
    selected_indices = _select_frames_by_pose_constraints(poses, n)
    
    # Step 5: Copy selected frames to output directory
    print(f"\n Saving selected frames...")
    selected_paths = []
    for out_idx, frame_idx in enumerate(selected_indices):
        src = Path(all_frame_paths[frame_idx])
        dst = output_dir / f"frame_{out_idx:06d}.jpg"
        import shutil
        shutil.copy2(src, dst)
        selected_paths.append(str(dst))
        print(f"   Frame {frame_idx} → {dst.name}")
    
    print(f"\n Selected {len(selected_paths)} frames")
    return selected_paths


def _extract_all_frames(video_path, save_dir, target_width=1280, target_height=720):
    """Extract all frames from video, downsample to target resolution, and save."""
    frame_paths = []
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f" Cannot open video: {video_path}")
            return None
        
        # Get original resolution
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"   Original resolution: {orig_width}x{orig_height}")
        print(f"   Downsampling to: {target_width}x{target_height}")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Downsample frame
            downsampled = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
            
            frame_path = os.path.join(save_dir, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_path, downsampled)
            frame_paths.append(frame_path)
            frame_count += 1
        
        cap.release()
        return frame_paths
    except Exception as e:
        print(f" Error extracting frames: {e}")
        return None


def _run_colmap_on_frames(frames_dir, colmap_work_dir):
    """Run COLMAP SfM pipeline on frames directory."""
    
    frames_dir = Path(frames_dir)
    work_dir = Path(colmap_work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Create database
    database_path = work_dir / "database.db"
    if database_path.exists():
        database_path.unlink()

    print("   Running feature extraction...")
    pycolmap.extract_features(
        str(database_path),
        str(frames_dir),
        camera_mode=pycolmap.CameraMode.SINGLE,
        sift_options=pycolmap.SiftExtractionOptions(num_threads=4),
    )

    print("   Running feature matching...")
    pycolmap.match_sequential(
        database_path=str(database_path),
        sift_options=pycolmap.SiftMatchingOptions(num_threads=4),
        matching_options=pycolmap.SequentialMatchingOptions(overlap=5),
    )

    print("   Running incremental mapping (SfM)...")
    reconstructions = pycolmap.incremental_mapping(
        database_path=str(database_path),
        image_path=str(frames_dir),
        output_path=str(work_dir / "sparse_tmp"),
        options=pycolmap.IncrementalPipelineOptions(num_threads=4),
    )

    if not reconstructions:
        print("COLMAP mapper produced no reconstructions")
        return None

    # Return largest reconstruction
    reconstruction = max(
        reconstructions,
        key=lambda r: r.num_registered_images()
    )

    return reconstruction

def _extract_camera_poses(reconstruction, total_frames):
    """
    Extract camera poses from COLMAP reconstruction.
    Returns list of 4x4 pose matrices (one per frame) or None if not enough valid poses.
    """
    try:
        poses = {}
        
        # Build mapping from image name to pose
        for image_id, image in reconstruction.images.items():
            # Image name is like "frame_000000.jpg"
            img_name = image.name
            frame_idx = int(img_name.split('_')[1].split('.')[0])
            
            if frame_idx < total_frames:
                # Convert quaternion and position to 4x4 matrix
                qvec = image.qvec  # [w, x, y, z]
                tvec = image.tvec  # [x, y, z]
                
                # pycolmap quaternion to rotation matrix
                q = qvec / np.linalg.norm(qvec)
                w, x, y, z = q
                
                R = np.array([
                    [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
                    [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
                    [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
                ])
                
                pose = np.eye(4)
                pose[:3, :3] = R
                pose[:3, 3] = tvec
                poses[frame_idx] = pose
        
        if not poses:
            print(f" No valid poses extracted from COLMAP")
            return None
        
        print(f"  Extracted {len(poses)} valid camera poses")
        return poses
    
    except Exception as e:
        print(f"Error extracting poses: {e}")
        import traceback
        traceback.print_exc()
        return None


def _select_frames_by_pose_constraints(poses, n):
    """
    Select n frames using pose-based constraints.
    
    Algorithm:
    - Start with frame 0
    - For each subsequent frame i (1 to n-1):
        - Rotation threshold: (i+1) * (180/n) degrees
        - Find frame with max translation from frame 0 that has rotation <= threshold
        - If no such frame exists, take the frame with max translation overall
        - Mark selected frame as used
    """
    frame_indices = sorted(poses.keys())
    selected_indices = []
    remaining_indices = set(frame_indices)
    
    # Always start with first frame
    selected_indices.append(0)
    remaining_indices.discard(0)
    
    ref_pose = poses[0]
    ref_position = ref_pose[:3, 3]
    
    for i in range(1, n):
        # Rotation threshold for this frame: (i+1) * (180/n) degrees
        rotation_threshold_deg = (i + 1) * (180.0 / n)
        rotation_threshold_rad = np.deg2rad(rotation_threshold_deg)
        
        # Find frame with max translation within rotation constraint
        best_idx = None
        best_dist = -1
        best_dist_unconstrained = -1
        best_idx_unconstrained = None
        
        for idx in remaining_indices:
            pose = poses[idx]
            position = pose[:3, 3]
            distance = np.linalg.norm(position - ref_position)
            
            # Compute rotation angle between ref_pose and pose
            R_rel = ref_pose[:3, :3].T @ pose[:3, :3]
            trace = np.trace(R_rel)
            # Clamp trace to [-1, 1] to avoid numerical issues
            trace = np.clip(trace, -1, 1)
            rotation_angle = np.arccos((trace - 1) / 2.0)
            
            # Track best unconstrained frame
            if distance > best_dist_unconstrained:
                best_dist_unconstrained = distance
                best_idx_unconstrained = idx
            
            # Check if within rotation constraint
            if rotation_angle <= rotation_threshold_rad:
                if distance > best_dist:
                    best_dist = distance
                    best_idx = idx
        
        # Select frame: prefer constrained frame, fall back to unconstrained
        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining_indices.discard(best_idx)
            print(f"   Frame {i}: selected idx={best_idx} (dist={best_dist:.3f}, rot<{rotation_threshold_deg:.1f}°)")
        else:
            selected_indices.append(best_idx_unconstrained)
            remaining_indices.discard(best_idx_unconstrained)
            print(f"   Frame {i}: selected idx={best_idx_unconstrained} (dist={best_dist_unconstrained:.3f}, no rot constraint satisfied, threshold was {rotation_threshold_deg:.1f}°)")
        
        if len(remaining_indices) == 0:
            print(f" Ran out of frames; selected {len(selected_indices)} out of {n}")
            break
    
    return selected_indices
