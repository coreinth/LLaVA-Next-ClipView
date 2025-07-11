import cv2
import random
from PIL import Image
import torch
import clip
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as torch_func
from numpy.linalg import norm
import numpy as np

similar_segments = []
ground_truth = [
                  [[0, 47], [48, 237], [238, 330], [331, 898], [899, 1204], [1205, 1499], [1500, 1662]],
                  [[0, 31], [32, 404], [405, 623], [624, 1045], [1046, 1130], [1131, 1551], [1552, 1781]]
                 ]
text_labels = [["introduction", "recap: the surprise pi", "the game plan", "how to analyze the blocks", "the geometry puzzle", "small angle approximation", "the value of pure puzzles"],
               ["introduction", "twirling ties", "tarski plank problem", "monge's theorem", "3d volume, 4d answer", "the hypercube stack", "the sadness of higher dimensions"]]

cosine_accuracy = {}

def seconds_to_mmss(seconds):
    minutes = int(seconds) // 60
    remaining_seconds = int(seconds) % 60
    return f"{minutes:02}:{remaining_seconds:02}"

def calculate_iou(seg1, seg2):
    start1, end1 = seg1
    start2, end2 = seg2
    intersection = max(0, min(end1, end2) - max(start1, start2))
    union = max(end1, end2) - min(start1, start2)
    return intersection / union if union > 0 else 0

def compare_vectors(similarity_output):
  for i in range(len(similarity_output) - 1):
    vec1 = similarity_output[i]
    vec2 = similarity_output[i + 1]

    cos_sim = torch_func.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()
    euclid_dist = torch.dist(vec1, vec2).item()

    if (cos_sim > 0.85 and euclid_dist < 0.2):
      similar_segments.append([i, i+1])
    # print(f"Comparing row {i} & {i+1}:")
    # print(f"  Cosine Similarity: {cos_sim:.4f}")
    # print(f"  Euclidean Distance: {euclid_dist:.4f}")

def combine_segments(current_segments, combine_segments):
  # print(f"Segments to combine: {combine_segments}")
  from collections import defaultdict

  adjacency = defaultdict(set)
  for i, j in combine_segments:
      adjacency[i].add(j)
      adjacency[j].add(i)

  visited = set()
  groups = []

  for node in sorted(adjacency.keys()):
      if node not in visited:
          stack = [node]
          group = []

          while stack:
              current = stack.pop()
              if current not in visited:
                  visited.add(current)
                  group.append(current)
                  stack.extend(adjacency[current])

          if len(group) > 1:
              groups.append(sorted(group))

  segments_to_remove = set()
  for group in groups:
      start_idx = group[0]
      end_idx = group[-1]

      current_segments[start_idx] = [current_segments[start_idx][0], current_segments[end_idx][1]]

      for idx in group[1:]:
          segments_to_remove.add(idx)

  for idx in sorted(segments_to_remove, reverse=True):
      del current_segments[idx]

  return current_segments

def analyze_thresholds(image_features, ground_truth, segments,
                      cosine_thresholds=np.linspace(0.7, 0.95, 20), 
                      euclid_thresholds=np.linspace(0.1, 0.5, 5)):
    
    for euclid_thresh in euclid_thresholds:
        print(f"--- Euclidean Distance Threshold: {euclid_thresh:.2f} ---")
        segment_counts = []
        segment_frequency = np.zeros(len(image_features))
        
        for threshold in cosine_thresholds:
            temp_similar_segments = []
            
            for i in range(len(image_features) - 1):
                vec1 = image_features[i]
                vec2 = image_features[i + 1]
                cos_sim = torch_func.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()
                euclid_dist = torch.dist(vec1, vec2).item()
                
                if cos_sim > threshold and euclid_dist < euclid_thresh:
                    temp_similar_segments.append([i, i + 1])
                    segment_frequency[i] += 1
                    segment_frequency[i + 1] += 1

            temp_segments = segments.copy()
            merged = combine_segments(temp_segments, temp_similar_segments)
            print(f"\nCosine threshold: {threshold:.2f}")
            segment_counts.append(len(merged))

            matches = 0
            ious = []
            for gt in ground_truth:
                best_iou = 0
                for merged_seg in merged:
                    iou = calculate_iou(gt, merged_seg)
                    best_iou = max(best_iou, iou)
                ious.append(best_iou)
                if best_iou > 0.5:
                    matches += 1

            accuracy = round((matches / len(ground_truth)), 2)
            threshold_rounded = round(threshold, 2)
            cosine_accuracy.setdefault((threshold_rounded, round(euclid_thresh, 2)), []).append(accuracy)
            print(f"IoUs: {[f'{iou:.2f}' for iou in ious]}")
            print(f"Accuracy: {accuracy:.2f}")

    # # Plot: Threshold vs Final Segment Count
    # plt.figure(figsize=(10, 4))
    # plt.plot(thresholds, segment_counts, marker='o')
    # plt.xlabel("Cosine Similarity Threshold")
    # plt.ylabel("Number of Segments After Merging")
    # plt.title("Impact of Cosine Threshold on Segment Count")
    # plt.grid(True)
    # plt.show()

    # # Plot: Segment Similarity Frequency
    # plt.figure(figsize=(12, 4))
    # x_labels = [f"{seconds_to_mmss(i*60)}" for i in range(len(segment_frequency))]
    # plt.bar(range(len(segment_frequency)), segment_frequency, tick_label=x_labels)
    # plt.xticks(rotation=45, ha='right')
    # plt.xlabel("Segment Start Time")
    # plt.ylabel("Similarity Frequency")
    # plt.title("Segment Similarity Frequency Across Thresholds")
    # plt.tight_layout()
    # plt.show()

    # Optional: Return which segment(s) are most similar
    # top_segments = np.argsort(segment_frequency)[-3:][::-1]  # top 3 segments
    # print("Most clustered segments:")
    # for idx in top_segments:
    #     print(f"Segment {idx} ({seconds_to_mmss(idx*60)} - {seconds_to_mmss((idx+1)*60)}), Frequency: {segment_frequency[idx]}")

def load_clip_text_inputs_from_file(file_path="scene_descriptions.txt"):
    with open(file_path, "r", encoding="utf-8") as f:
        descriptions = []
        for line in f:
            if "]" in line:
                description = line.strip().split("]", 1)[-1].strip()
                if description:
                    descriptions.append(description)
        return descriptions
        
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
video_path = ["30min_vid.mp4"]

def load_scene_boundaries(scene_file="segments/scene_boundaries.txt"):
    try:
        with open(scene_file, "r") as f:
            boundaries = [int(line.strip()) for line in f if line.strip().isdigit()]
            if not boundaries:
                raise ValueError("No scene boundaries found")
            return boundaries
    except FileNotFoundError:
        raise FileNotFoundError(f"Scene boundaries file not found: {scene_file}")
    except ValueError as e:
        raise ValueError(f"Invalid scene boundaries file: {e}")

def run_evaluation_model(video_path):
    global similar_segments
    similar_segments = []
    
    for video_idx, video in enumerate(video_path):
        print(f"Video idx: {video_idx}; video_path: {video}")
        cap = cv2.VideoCapture(video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        scene_boundaries = load_scene_boundaries("segments/scene_boundaries.txt")
        scene_boundaries.append(total_frames)

        random_frames = []
        segments = []

        for i in range(len(scene_boundaries) - 1):
            start_frame = scene_boundaries[i]
            end_frame = scene_boundaries[i + 1] - 1
            
            if end_frame <= start_frame:
                end_frame = start_frame
            
            random_frame_index = random.randint(start_frame, end_frame)
            cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_index)
            ret, frame = cap.read()
            
            if ret:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                random_frames.append((img, preprocess(img)))
                
            segments.append([start_frame / fps, end_frame / fps])

        cap.release()

    # batch processing
    image_inputs = torch.stack([p for _, p in random_frames])
    image_inputs = image_inputs.to(device)

    descriptions = load_clip_text_inputs_from_file("scene_descriptions.txt")
    
    if len(descriptions) != len(random_frames):
        print(f"Warning: {len(descriptions)} descriptions but {len(random_frames)} scenes/frames!")
        min_len = min(len(descriptions), len(random_frames))
        descriptions = descriptions[:min_len]
        random_frames = random_frames[:min_len]
        print(f"Using {min_len} scenes for evaluation")
        
    text_inputs = clip.tokenize(descriptions).to(device)

    # cosine similarity of text query and image feature/video frame
    with torch.no_grad():
        image_features = model.encode_image(image_inputs)
        text_features = model.encode_text(text_inputs)
        similarity = (image_features @ text_features.T).softmax(dim=-1)
        print(similarity)
        
    cosine_changes = []
    euclid_changes = []
    for i in range(len(image_features) - 1):
        vec1 = image_features[i]
        vec2 = image_features[i + 1]
        cos_sim = torch_func.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()
        euclid_dist = torch.dist(vec1, vec2).item()
        cosine_changes.append(cos_sim)
        euclid_changes.append(euclid_dist)

    ranked_indices = np.argsort(euclid_changes)[::-1]

    print("All scene changes ranked by significance (Euclidean change):")
    for rank, idx in enumerate(ranked_indices, 1):
        print(f"Rank {rank}: Between scene {idx} and {idx+1} | Euclidean change={euclid_changes[idx]:.3f}, Cosine similarity={cosine_changes[idx]:.3f}")

    print(f"ground truth: {ground_truth[video_idx]}\n")
    analyze_thresholds(image_features, ground_truth[video_idx], segments)

    for cosine_threshold in cosine_accuracy:
            average_accuracy = np.mean(cosine_accuracy[cosine_threshold])
            print(f"Cosine threshold: {cosine_threshold}; Average Accuracy: {average_accuracy:.2f}")