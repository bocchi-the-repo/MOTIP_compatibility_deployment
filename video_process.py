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
import ffmpeg
import numpy as np
from utils.nested_tensor import nested_tensor_from_tensor_list
from tqdm import tqdm
from colormap import get_color


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
    """Process the video with MOTIP tracking using GPU-accelerated decoding."""
    from models.runtime_tracker import RuntimeTracker
    
    # 获取视频信息
    probe = ffmpeg.probe(video_path)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    codec = video_info['codec_name']
    width, height = int(video_info['width']), int(video_info['height'])
    fps = float(video_info['r_frame_rate'].split('/')[0]) / float(video_info['r_frame_rate'].split('/')[1])
    length = int(video_info['nb_frames']) if 'nb_frames' in video_info else None
    
    print(f"输入视频: {width}x{height}, 编解码器: {codec}, FPS: {fps:.2f}")
    
    # 尝试GPU解码器，如果失败则回退到CPU
    codec_map = {'h264': 'h264_cuvid', 'hevc': 'hevc_cuvid', 'av1': 'av1_cuvid'}
    hw_decoder = codec_map.get(codec, 'h264_cuvid')
    
    process = None
    use_gpu = True
    
    try:
        # 尝试GPU加速解码
        print(f"尝试使用GPU解码器: {hw_decoder}")
        process = (
            ffmpeg.input(video_path, vcodec=hw_decoder)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run_async(pipe_stdout=True, pipe_stderr=True, quiet=True)
        )
        # 测试读取第一帧
        test_read = process.stdout.read(width * height * 3)
        if len(test_read) != width * height * 3:
            raise Exception("GPU解码器测试失败")
        # 重置进程，重新开始
        process.terminate()
        process.wait()
    except Exception as e:
        print(f"GPU解码器不可用({e})，回退到CPU解码")
        use_gpu = False
        if process:
            try:
                process.terminate()
                process.wait()
            except:
                pass
    
    # 根据是否支持GPU选择解码方式
    if use_gpu:
        print("使用GPU加速解码")
        process = (
            ffmpeg.input(video_path, vcodec=hw_decoder)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run_async(pipe_stdout=True, quiet=True)
        )
    else:
        print("使用CPU解码")
        process = (
            ffmpeg.input(video_path)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run_async(pipe_stdout=True, quiet=True)
        )
    
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

    frame_size = width * height * 3  # RGB24格式
    frame_idx = 0
    
    try:
        # 使用tqdm显示进度
        pbar = tqdm(desc="Processing video", unit="frame", total=length)
        
        while True:
            # 从ffmpeg管道读取原始RGB数据
            raw_frame = process.stdout.read(frame_size)
            if not raw_frame:
                print("视频流结束")
                break
            if len(raw_frame) != frame_size:
                print(f"帧大小不匹配: 期望{frame_size}, 实际{len(raw_frame)}")
                break
            
            # 转换为numpy数组并重塑为图像
            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(height, width, 3)
            # 转换RGB到BGR供OpenCV使用
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # 转换帧为张量
            frame_tensor = simple_transform(frame_bgr, max_shorter=800, max_longer=1440, image_dtype=dtype)
            frame_tensor = nested_tensor_from_tensor_list([frame_tensor])

            # 运行跟踪器
            runtime_tracker.update(frame_tensor)

            with torch.no_grad():
                track_results = runtime_tracker.get_track_results()

            # 绘制跟踪结果
            for bbox, obj_id in zip(track_results["bbox"], track_results["id"]):
                x, y, w, h = map(int, bbox)
                cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), get_color(obj_id, rgb=False, use_int=True), 2)
                cv2.putText(frame_bgr, f"ID: {obj_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, get_color(obj_id, rgb=False, use_int=True), 2)

            video_writer.write(frame_bgr)
            frame_idx += 1
            pbar.update(1)
            
    except Exception as e:
        print(f"处理视频时出错: {e}")
    finally:
        if process:
            process.stdout.close()
            process.wait()
        video_writer.release()
        pbar.close()

    print(f"视频处理完成。输出视频已保存到 {output_path}。")


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