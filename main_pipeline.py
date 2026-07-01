import sys

# import modules here
from english_to_asl_gloss_llama import ask_llama as translate_to_gloss
from asl_llm_video_mapping import find_video_records
from video_clipper import clip_and_merge_videos
# pyrefly: ignore [missing-import]
from media_pipline_converter import convert_video

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
        print(f"[Stage 2 success] Found corresponding videos: {video_records}")

        #Stage 3: Video Clipping & Merging (Input: Video records -> Output: Final video file path)
        final_video_path = clip_and_merge_videos(video_records)
        print(f"[Stage 3 success] Video merging complete, path: {final_video_path}")

        # Stage 3.5: Keypoint Extraction (Input: Final video file path -> Output: .npy file paths)
        if final_video_path:
            res = convert_video(final_video_path)
            if isinstance(res, tuple):
                npy_path, json_path = res
            else:
                npy_path, json_path = res, None
            print(f"[Stage 3.5 success] Keypoint extraction complete. NPY: {npy_path}, JSON: {json_path}")
            
            # Stage 3.6: Raw Frame Conversion (Input: Final video file path -> Output: .npy of raw frames)
            print("[Stage 3.6] Converting merged video to raw frames numpy array...")
            try:
                import subprocess
                import os
                result = subprocess.run(
                    [sys.executable, os.path.join("data_preprocessing", "video_npy_converter.py"), final_video_path],
                    capture_output=True,
                    text=True,
                    check=True
                )
                print(result.stdout.strip())
                print("[Stage 3.6 success] Raw frame conversion complete.")
            except Exception as e:
                print(f"❌ [Stage 3.6 error] Failed to convert video to raw frames: {e}", file=sys.stderr)
        else:
            print("[Stage 3.5 skipped] No merged video generated.")

        # Stage 4: Web Rendering (Input: Final video file path -> Output: Updated web UI)
        # update_web_ui(final_video_path)
        # print("[Stage 4 success] website updated!")

    except Exception as e:
        print(f"❌ pipeline error: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()