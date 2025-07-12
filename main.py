import json
from torch_check import torch_check
from llava_main import llava_main
from clip_main import chapter_detection

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

if __name__ == "__main__":
    main()