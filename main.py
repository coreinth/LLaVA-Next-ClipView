from torch_check import torch_check
from llava_main import llava_main
# from clip_main import run_evaluation_model

def main():
    torch_check()
    llava_main()
    # run_evaluation_model(["30min_vid.mp4"])

if __name__ == "__main__":
    main()