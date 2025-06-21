from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import torch
import subprocess
import os
from pathlib import Path

# Model Initialization
def initialize_models():
    processor = LlavaNextProcessor.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf",
        use_fast=True
    )
    
    model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf",
        torch_dtype=torch.float16,
        device_map=None 
    ).to('cuda:0').eval()

    model.generation_config.pad_token_id = model.config.eos_token_id
    model.generation_config.eos_token_id = model.config.eos_token_id

    print(f"Model loaded on {next(model.parameters()).device}")
    print(f"VRAM usage: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    
    return processor, model

# Extract 3 frames per 1-minute segment
def extract_representative_frames(video_path, output_dir="segments"):
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
    
    frame_paths = sorted([str(p) for p in Path(output_dir).glob("frame_*.png")])
    return frame_paths

# Process frames and summarize each segment
def describe_frames_group(frames, processor, model):
    images = []
    for frame_path in frames:
        try:
            img = Image.open(frame_path).convert("RGB")
            images.append(img)
        except Exception as e:
            print(f"Could not open {frame_path}: {e}")
            return None

    texts = ["[INST] <image>\nDescribe this image. [/INST]"] * len(images)
    inputs = processor(images=images, text=texts, return_tensors="pt", padding=True).to("cuda:0")

    with torch.inference_mode():
        outputs = model.generate(
            **inputs, max_new_tokens=100, do_sample=False
        )

    descriptions = [processor.decode(o, skip_special_tokens=True) for o in outputs]
    combined_description = " ".join(descriptions)
    return combined_description.strip()

# Save results
def save_results(results, output_file="minute_descriptions.txt"):
    with open(output_file, "w", encoding='utf-8') as f:
        for minute, desc in results:
            f.write(f"[Minute {minute}] {desc}\n")

# Main
def llava_main():
    torch.cuda.empty_cache()
    processor, model = initialize_models()

    video_path = "30min_vid.mp4"
    frames_per_segment = extract_representative_frames(video_path)

    results = []
    for i, frame_path in enumerate(frames_per_segment):
        print(f"Processing minute {i}...")
        description = describe_frames_group([frame_path], processor, model)
        if description:
            results.append((i, description))

    save_results(results)
    print(f"Completed! Generated descriptions for {len(results)} minutes.")

if __name__ == "__main__":
    main()