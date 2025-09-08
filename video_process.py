#!/usr/bin/env python3
"""
Video Processing Demo
This script processes a video using MOTIP model for multi-object tracking.
Converted from demo/video_process.ipynb
"""

import os
import sys
import torch
import cv2
from utils.nested_tensor import nested_tensor_from_tensor_list
from tqdm import tqdm
from demo.colormap import get_color


def setup_environment():
    """Setup the environment and check CUDA availability."""
    current_file_path = os.path.abspath(__file__)
    parent_dir = os.path.dirname(current_file_path)
    sys.path.append(parent_dir)
    os.chdir(parent_dir)
    print(f"Current root path is set to {parent_dir}")

    torch_version = torch.__version__
    cuda_available = torch.cuda.is_available()

    if not cuda_available:
        raise RuntimeError("CUDA is not available")

    print(f"Hello! Welcome to use the video process demo. Your torch version is {torch_version} and CUDA is available.")


def prepare_video_paths():
    """Prepare video input and output paths."""
    os.makedirs("outputs/", exist_ok=True)
    video_path = "videos/test.mp4"
    output_path = "outputs/test_tracking.mp4"
    return video_path, output_path


def build_model():
    """Build and load the MOTIP model."""
    from utils.misc import yaml_to_dict
    from configs.util import load_super_config
    from models.motip import build as build_model
    from models.misc import load_checkpoint
    from models.runtime_tracker import RuntimeTracker

    config_path = "./configs/r50_deformable_detr_motip_dancetrack.yaml"
    checkpoint_path = "./outputs/r50_deformable_detr_motip_dancetrack/r50_deformable_detr_motip_dancetrack.pth"
    config = yaml_to_dict(config_path)
    config = load_super_config(config, config["SUPER_CONFIG_PATH"])
    dtype = torch.float16  # torch.float32 or torch.float16, we select float16 for faster inference

    model, _ = build_model(config)
    # Load the model weights
    load_checkpoint(model, checkpoint_path)
    model.eval()
    model = model.cuda()
    if dtype == torch.float16:
        model.half()

    print("Model built successfully.")
    return model, dtype


def simple_transform(image, max_shorter, max_longer, image_dtype):
    """Apply simple transformation to input image."""
    from torchvision.transforms import functional as F

    image = F.to_tensor(image)
    image = F.resize(image, size=max_shorter, max_size=max_longer)
    image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if image_dtype != torch.float32:
        image = image.to(image_dtype)
    return image.cuda()


def process_video(video_path, output_path, model, dtype):
    """Process the video with MOTIP tracking."""
    from models.runtime_tracker import RuntimeTracker
    
    video_cap = cv2.VideoCapture(video_path)
    if not video_cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")
    
    # Get video properties
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    length = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"The video {video_path} seems OK. It has {fps} fps, {width} width and {height} height.")
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    runtime_tracker = RuntimeTracker(
        model=model,
        sequence_hw=(height, width),
        assignment_protocol="object-max",
        miss_tolerance=30,
        det_thresh=0.5,
        newborn_thresh=0.5,
        id_thresh=0.2,
        dtype=dtype,
    )

    for frame_idx in tqdm(range(length), desc="Processing video", unit="frame"):
        ret, frame = video_cap.read()
        if not ret:
            break

        # Convert the frame to a tensor
        frame_tensor = simple_transform(frame, max_shorter=800, max_longer=1440, image_dtype=dtype)
        frame_tensor = nested_tensor_from_tensor_list([frame_tensor])

        # Run the tracker on the frame
        runtime_tracker.update(frame_tensor)

        with torch.no_grad():
            track_results = runtime_tracker.get_track_results()

        for bbox, obj_id in zip(track_results["bbox"], track_results["id"]):
            x, y, w, h = map(int, bbox)
            cv2.rectangle(frame, (x, y), (x + w, y + h), get_color(obj_id, rgb=False, use_int=True), 2)
            cv2.putText(frame, f"ID: {obj_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, get_color(obj_id, rgb=False, use_int=True), 2)

        video_writer.write(frame)

    video_cap.release()
    video_writer.release()

    print(f"Video processing completed. The output video is saved to {output_path}.")


def main():
    """Main function to run the video processing demo."""
    # Setup environment
    setup_environment()
    
    # Prepare video paths
    video_path, output_path = prepare_video_paths()
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        print("Please place your video file at the specified path.")
        return
    
    # Build model
    model, dtype = build_model()
    
    # Process video
    process_video(video_path, output_path, model, dtype)


if __name__ == "__main__":
    main()