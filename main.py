import json
from torch_check import torch_check
from llava_main import llava_main
from clip_main import chapter_detection
from score_fusion import create_final_chapters

def seconds_to_mmss(seconds):
    """Convert seconds to MM:SS format"""
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{minutes:02}:{seconds:02}"

def to_python_type(obj):
    import numpy as np
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_python_type(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python_type(v) for v in obj]
    return obj

def main():
    # torch_check()
    llava_main()
    chapters, ranked_chapters = chapter_detection()
    
    with open("clip_chapter_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "chapters": to_python_type(chapters),
            "ranked_chapters": to_python_type(ranked_chapters),
        }, f, indent=2)
        
    print("Chapter detection completed. Results saved to clip_chapter_results.json.")
    
    final_chapters = create_final_chapters()
    
    print("\n=== YOUTUBE CHAPTERS ===")
    for chapter in final_chapters:
        print(f'{{ "name": "{chapter["name"]}", "start": {seconds_to_mmss(chapter["start"])} }},')

if __name__ == "__main__":
    main()