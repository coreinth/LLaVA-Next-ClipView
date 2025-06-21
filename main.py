from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import torch
import subprocess
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Model Initialization
def initialize_models():
    """Initialize and load models with proper device placement"""
    processor = LlavaNextProcessor.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf",
        use_fast=True
    )
    
    model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf",
        torch_dtype=torch.float16,
        device_map=None 
    ).to('cuda:0').eval()
    
    assert next(model.parameters()).device.type == 'cuda'
    
    model.generation_config.pad_token_id = model.config.eos_token_id
    model.generation_config.eos_token_id = model.config.eos_token_id
    
    print(f"Model loaded on {next(model.parameters()).device}")
    print(f"VRAM usage: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    
    return processor, model

# Keyframe extraction
def extract_keyframes(video_path, output_dir="keyframes @ 0.35"):
    os.makedirs(output_dir, exist_ok=True)
    
    project_root = Path(__file__).parent
    ffmpeg_path = project_root / "ffmpeg" / "bin" / "ffmpeg.exe"
    
    if not ffmpeg_path.exists():
        raise FileNotFoundError(f"FFmpeg not found at {ffmpeg_path}")

    subprocess.run([
        str(ffmpeg_path),
        "-i", video_path,
        "-vf", "select='(gt(scene,0.35))'",
        "-vsync", "0",
        "-frame_pts", "1",
        f"{output_dir}/frame_%04d.png"
    ], check=True)
    
    return sorted([str(p) for p in Path(output_dir).glob("keyframe_*.png")])

# Optimized description generation

def process_frame(frame_path, processor, model):
    """Process a single frame with error handling"""
    try:
        image = Image.open(frame_path).convert('RGB')
        
        with torch.cuda.device('cuda:0'):
            inputs = processor(
                images=image,
                text="[INST] <image>\nDescribe this image. [/INST]",
                return_tensors="pt"
            ).to('cuda:0')
            
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False
                )
        
        return processor.decode(output[0], skip_special_tokens=True)
    
    except Exception as e:
        print(f"Error processing {frame_path.name}: {str(e)}")
        return None

def process_all_frames(frame_paths, processor, model):
    results = []
    for i, frame_path in enumerate(frame_paths):
        print(f"Processing {i+1}/{len(frame_paths)}")
        desc = process_frame(frame_path, processor, model)
        results.append((str(frame_path), desc))
        
        if (i+1) % 10 == 0:
            print(f"Completed {i+1}/{len(frame_paths)} frames")
            print(f"Current VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    
    return results

def save_results(results, output_file="descriptions.txt"):
    with open(output_file, "w", encoding='utf-8') as f:
        for path, desc in results:
            if desc:
                f.write(f"{path}\t{desc}\n")

# Main
def main():
    # processor, model = initialize_models()
    frame_paths = extract_keyframes("test2.mp4")
    #results = process_all_frames(frame_paths, processor, model)
    # save_results(results)
    print(f"Completed! Saved {len([r for r in results if r[1]])} descriptions")

if __name__ == "__main__":
    main()