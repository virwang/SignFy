import sys

# import modules here
from english_to_asl_gloss_llama import ask_llama as translate_to_gloss
from asl_llm_video_mapping import find_video_records
# from video_clipper import clip_and_merge_videos
# from web_renderer import update_web_ui

def main():
    # 1. receive user input
    user_input = input("Please enter an English sentence: ").strip()
    if not user_input:
        return

    print(f"\n[System] Starting processing for input: '{user_input}'")

    try:
        # Stage 1: LLM Translation (Input: English string -> Output: ASL Gloss array)
        asl_gloss_list = translate_to_gloss(user_input)
        gloss_array = asl_gloss_list.split(" ")  
        print(f"[Stage 1 success] ASL Gloss: {gloss_array}")

        # Stage 2: Video Mapping (Input: ASL Gloss array -> Output: Video path/ID array)
        video_records = find_video_records(gloss_array, english_input=user_input)
        print(f"[Stage 2 success] Found corresponding videos: {video_records}")

        #Stage 3: Video Clipping & Merging (Input: Video records -> Output: Final video file path)
        # final_video_path = clip_and_merge_videos(video_records)
        # print(f"[Stage 3 success] Video merging complete, path: {final_video_path}")

        # Stage 4: Web Rendering (Input: Final video file path -> Output: Updated web UI)
        # update_web_ui(final_video_path)
        # print("[Stage 4 success] website updated!")

    except Exception as e:
        print(f"❌ pipeline error: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()