from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import torch
import cv2
import subprocess
import os
from pathlib import Path
import gc
import shutil
import time

video_path = "30min_vid3.mp4"

def seconds_to_mmss(seconds):
    minutes = int(seconds) // 60
    remaining_seconds = int(seconds) % 60
    return f"{minutes:02}:{remaining_seconds:02}"

def initialize_models():
    """ Model Initialization """
    
    model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
    
    processor = LlavaNextProcessor.from_pretrained(
        model_id,
        use_fast=True
    )
    
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=None 
    ).to('cuda:0').eval()

    model.generation_config.pad_token_id = model.config.eos_token_id
    model.generation_config.eos_token_id = model.config.eos_token_id

    print(f"Model loaded on {next(model.parameters()).device}")
    print(f"VRAM usage: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    
    return processor, model

def extract_representative_frames(video_path, output_dir="segments"):
    """ Extract 3 frames per 1-minute segment"""
    
    os.makedirs(output_dir, exist_ok=True)
    project_root = Path(__file__).parent

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        ffmpeg_path = project_root / "ffmpeg" / "bin" / "ffmpeg.exe"
    
    if not ffmpeg_path or not ffmpeg_path.exists():
        raise FileNotFoundError(f"FFmpeg not found at {ffmpeg_path}")
    
    scene_frame_numbers = []
    
    result = subprocess.run([
        str(ffmpeg_path),
        "-i", video_path,
        "-vf", "select='(gt(scene,0.4))',showinfo",
        "-vsync", "0",
        "-frame_pts", "1",
        f"{output_dir}/frame_%04d.png"
    ], stderr=subprocess.PIPE, text=True)
    
    for line in result.stderr.splitlines():
        if "showinfo" in line and "n:" in line:
            parts = line.split("n:")
            if len(parts) > 1:
                frame_num = int(parts[1].split()[0])
                scene_frame_numbers.append(frame_num)
                
    frame_paths = sorted([str(p) for p in Path(output_dir).glob("frame_*.png")])
            
    with open(f"{output_dir}/frame_map.csv", "w") as f:
        f.write("filename,frame_number,timestamp,timestamp_mmss\n")
        for frame_path in frame_paths:
            fname = Path(frame_path).name
            frame_num = int(fname.split("_")[1].split(".")[0])
            timestamp = frame_num / fps if fps else 0
            timestamp_mmss = seconds_to_mmss(timestamp)
            f.write(f"{fname},{frame_num},{timestamp:.3f},{timestamp_mmss}\n")

    return frame_paths

def describe_frames_group(frames, processor, model):
    """ Process frames and summarize each segment """
    
    images = []
    for frame_path in frames:
        try:
            img = Image.open(frame_path).convert("RGB")
            img = img.resize((336, 336))
            images.append(img)
        except Exception as e:
            print(f"Could not open {frame_path}: {e}")
            return None

    texts = ["[INST] <image>\nDescribe this image in detail, focusing on the main objects, actions, and visual elements in one sentence. [/INST]"] * len(images)
    inputs = processor(images=images, text=texts, return_tensors="pt", padding=True).to("cuda:0")

    with torch.inference_mode():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=60,
            do_sample=False,
            num_beams=1,
            use_cache=True   
        )

    descriptions = [processor.decode(o, skip_special_tokens=True) for o in outputs]
    return descriptions

def save_results(results, output_file="scene_descriptions.txt"):
    """ Save the scene descriptions to a text file """
    
    with open(output_file, "w", encoding='utf-8') as f:
        for filename, desc in results:
            f.write(f"{filename} ||| {desc}\n")

def llava_main():
    torch.cuda.empty_cache()
    processor, model = initialize_models()

    frames_per_segment = extract_representative_frames(video_path)

    results = []
    total_scenes = len(frames_per_segment)
    start_time = time.time()
    cache_clear_interval = 5

    for i, frame_path in enumerate(frames_per_segment):
        scene_start = time.time()
        print(f"Processing scene {i+1}/{total_scenes}...")
        
        descriptions = describe_frames_group([frame_path], processor, model)
        
        if descriptions and descriptions[0]:
            results.append((os.path.basename(frame_path), descriptions[0]))
        
        scene_time = time.time() - scene_start
        elapsed_total = time.time() - start_time
        estimated_remaining = (elapsed_total / (i + 1)) * (total_scenes - i - 1)
        print(f"Scene {i+1} completed in {scene_time:.1f}s. ETA: {estimated_remaining/60:.1f}min")
        
        if i % cache_clear_interval == 0:
            torch.cuda.empty_cache()
            gc.collect()

    torch.cuda.empty_cache()
    save_results(results)
    print(f"Completed! Generated descriptions for {len(results)} scenes in {(time.time()-start_time)/60:.1f} minutes.\n")