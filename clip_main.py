import cv2
import csv
from PIL import Image
import torch
import clip
from torchvision import transforms
import torch.nn.functional as torch_func
import numpy as np
import time

def seconds_to_mmss(seconds):
    minutes = int(seconds) // 60
    remaining_seconds = int(seconds) % 60
    
    return f"{minutes:02}:{remaining_seconds:02}"

def analyze_scenes(image_features, segments, frame_info, diff_scores=None):
    print("Analyzing scores between scenes...")
    
    if diff_scores is None:
        diff_scores = []
        for i in range(len(image_features)):
            dists = []
            for j in range(len(image_features)):
                if i != j:
                    dists.append(torch.dist(image_features[i], image_features[j]).item())
            diff_scores.append(np.mean(dists))

    ranked_indices = np.argsort(diff_scores)[::-1]
    ranked_chapters = []
    for idx in ranked_indices:
        ranked_chapters.append({
            "filename": frame_info[idx]["filename"],
            "score": diff_scores[idx],
            "timestamp_mmss": seconds_to_mmss(segments[idx][0])
        })

    return ranked_chapters

def chapter_detection(frame_map_file="segments/frame_map.csv", desc_file="scene_descriptions.txt",
                               visual_change_threshold=0.2, similarity_threshold=0.15):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device)
    print("CLIP model loaded successfully.")
    
    # Add this debug code to see what CLIP tokenizes
    text = "The image is a simple black background with white text that reads \"Part 1: Why?\"."
    tokens = clip.tokenize([text])
    print(f"Tokenized text: {tokens}")
    print(f"Decoded tokens: {clip.decode(tokens[0])}")

    # Frame Information
    frame_info = []
    print("Loading frame information...")
    with open(frame_map_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            frame_info.append({
                "filename": row["filename"],
                "frame_number": int(row["frame_number"]),
                "timestamp": float(row["timestamp"])
            })
    
    # Llava Desc
    desc_map = {}
    print("Loading scene descriptions...")
    with open(desc_file, "r", encoding="utf-8") as f:
        content = f.read()
    entries = []
    lines = content.split('\n')
    current_entry = ""
    for line in lines:
        if line.strip() and '|||' in line:
            if current_entry:
                entries.append(current_entry)
            current_entry = line
        elif line.strip():
            current_entry += " " + line.strip()

    if current_entry:
        entries.append(current_entry)

    for entry in entries:
        if "|||" in entry:
            filename, desc = entry.split("|||", 1)
            desc = desc.strip()
            if "[/INST]" in desc:
                desc = desc.split("[/INST]", 1)[-1].strip()
            desc_map[filename.strip()] = desc
    
    # Process frames            
    descriptions = []
    frames = []
    segments = []
    print("Processing frames and extracting descriptions...")
    for i, info in enumerate(frame_info):
        frame_path = f"segments/{info['filename']}"
        frame = cv2.imread(frame_path)
        if frame is not None:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append((img, preprocess(img)))
            segments.append([info["timestamp"], info["timestamp"]])
            descriptions.append(desc_map.get(info["filename"], ""))
            print("Description for frame {}: {}".format(info["filename"], descriptions[-1]))

    if not frames:
        raise RuntimeError("No frames were loaded. Check that your segments/frame_*.png files exist and are readable.")

    image_inputs = torch.stack([p for _, p in frames]).to(device)

    with torch.no_grad():
        print("CLIP is Encoding images...")
        image_features = model.encode_image(image_inputs)
        text_tokens = clip.tokenize(descriptions).to(device)
        text_features = model.encode_text(text_tokens)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    visual_changes = [torch.dist(image_features[i], image_features[i+1]).item() for i in range(len(image_features)-1)]
    similarities = [(image_features[i] @ text_features[i]).item() for i in range(len(image_features))]

    diff_scores = []
    for i in range(len(image_features)):
        dists = []
        for j in range(len(image_features)):
            if i != j:
                dists.append(torch.dist(image_features[i], image_features[j]).item())
        diff_scores.append(np.mean(dists))

    chapters = []
    for i in range(len(image_features)):
        if diff_scores[i] > visual_change_threshold and similarities[i] > similarity_threshold:
            chapters.append({
                "filename": frame_info[i]["filename"],
                "timestamp_sec": segments[i][0],
                "timestamp_mmss": seconds_to_mmss(segments[i][0]),
                "visual_change": diff_scores[i],
                "similarity": similarities[i]
            })
    chapters = sorted(chapters, key=lambda x: x["timestamp_sec"])
    
    for chapter in chapters:
        chapter.pop("timestamp_sec", None)

    ranked_chapters = analyze_scenes(image_features, segments, frame_info, diff_scores)

    return chapters, ranked_chapters