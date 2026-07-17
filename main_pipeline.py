import sys
import os

# import modules here
from english_to_asl_gloss_llama import ask_llama as translate_to_gloss
from asl_llm_video_mapping import find_video_records
from numpy_merger import merge_numpy_arrays

def main():
    # 1. receive user input
    user_input = input("Please enter an English sentence: ").strip()
    if not user_input:
        return

    print(f"\n[System] Starting processing for input: '{user_input}'")

    try:
        # Stage 1: LLM Translation (Input: English string -> Output: ASL Gloss array)
        asl_gloss_list = translate_to_gloss(user_input)
        primary_glosses = [item.get("gloss", "") for item in asl_gloss_list if isinstance(item, dict)]
        print(f"[Stage 1 success] ASL Gloss (JSON): {asl_gloss_list}")
        print(f"[Stage 1 success] ASL Gloss (Primary): {primary_glosses}")

        # Stage 2: Video Mapping (Input: ASL Gloss array -> Output: Video path/ID array)
        video_records = find_video_records(asl_gloss_list, english_input=user_input)
        print(f"[Stage 2 success] Found corresponding videos/records")

        # Stage 3: Numpy Merging (Input: Video records -> Output: Final numpy file path)
        final_npy_path = merge_numpy_arrays(video_records)
        if final_npy_path:
            print(f"[Stage 3 success] Numpy merging complete, path: {final_npy_path}")
        else:
            print("[Stage 3 skipped] No numpy generated.")

    except Exception as e:
        print(f"❌ pipeline error: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()