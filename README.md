# SignFy

A Python-based project designed to bridge communication barriers between hearing and deaf individuals through sign language recognition and translation.

## 🎯 Purpose

SignFy aims to facilitate seamless communication by recognizing and interpreting sign language, making it easier for people to connect and understand each other regardless of their hearing abilities.

## 🛠️ Technology Stack

- **Language**: Python (v3.10+ / runs inside the `asl_env` virtual environment)
- **Focus**: Sign Language Recognition & Translation
- **LLM Translation Agent**: Local Ollama service running `llama3.1` (or custom models)
- **Computer Vision & Processing**: OpenCV, MediaPipe, MoviePy

## 📁 Repository Directory Structure

- [asl_env/](file:///c:/Capstone/wlasl-complete/asl_env/): Python virtual environment for the workspace.
- `videos_raw/` & `Microsoft_Videos_raw/`: Raw sign language videos from WLASL and Microsoft datasets.
- `videos_best/` & `microsoft_best/`: Clarity- and stability-filtered best video clips.
- `videos_cut/` & `microsoft_cut/`: Trimmed video clips matching precise gloss frame boundaries.
- `videos_numpy/` & `microsoft_numpy/`: Pre-converted raw frame numpy array `.npy` files.
- `sign_out/`: Output directory for merged, end-to-end translated sign language video clips.
- `bone_sign_out/`: Output directory for extracted 225-dimensional MediaPipe Holistic landmark skeleton sequence `.npy` files.
- [data_preprocessing/](file:///c:/Capstone/wlasl-complete/data_preprocessing/): Directory containing scripts for video selection, clipping, and numpy conversions.
- [main_pipeline.py](file:///c:/Capstone/wlasl-complete/main_pipeline.py): Central entry point running the end-to-end English sentence to ASL video/skeleton pipeline.
- [npy_player.py](file:///c:/Capstone/wlasl-complete/npy_player.py): Skeleton-based CV2 player to visualize `.npy` landmark sequences.

## 🚀 Getting Started

### 1. Set Up the Python Environment

Activate the local python virtual environment:

#### Windows PowerShell:
```powershell
.\asl_env\Scripts\activate
```

#### Linux/macOS:
```bash
source asl_env/bin/activate
```

Install/verify dependencies if needed:
```bash
pip install -r requirements.txt
```

### 2. Set Up Local Ollama Service

1. Ensure Ollama is installed ([https://ollama.com/](https://ollama.com/))
2. Start the Ollama service:
   ```bash
   ollama serve
   ```
3. Pull the default `llama3.1` translation model:
   ```bash
   ollama pull llama3.1
   ```
4. The Ollama local API will be automatically queried by the translation script at `http://localhost:11434`.

---

## 📖 Translation Pipeline Flow ([main_pipeline.py](file:///c:/Capstone/wlasl-complete/main_pipeline.py))

Run the complete pipeline:
```bash
python main_pipeline.py
```

The pipeline operates in the following sequential stages:

1. **Stage 1: LLM Translation**: Uses [english_to_asl_gloss_llama.py](file:///c:/Capstone/wlasl-complete/english_to_asl_gloss_llama.py) to translate a user's English sentence into an ASL Gloss sequence using the local Ollama LLM.
2. **Stage 2: Video Mapping**: Uses [asl_llm_video_mapping.py](file:///c:/Capstone/wlasl-complete/asl_llm_video_mapping.py) to map the target ASL Glosses to the corresponding WLASL or Microsoft best candidate video files.
3. **Stage 3: Video Clipping & Merging**: Uses [video_clipper.py](file:///c:/Capstone/wlasl-complete/video_clipper.py) to clip and concatenate the mapped video files into a single merged translation video (saved under `sign_out/`).
4. **Stage 3.5: Keypoint Extraction**: Uses [media_pipline_converter.py](file:///c:/Capstone/wlasl-complete/media_pipline_converter.py) and MediaPipe Holistic to extract pose and hand landmark keypoints frame-by-frame, outputting a 225-dimensional `.npy` keypoint sequence (saved under `bone_sign_out/`).
5. **Stage 3.6: Raw Frame Conversion**: Invokes [video_npy_converter.py](file:///c:/Capstone/wlasl-complete/data_preprocessing/video_npy_converter.py) to compile the merged video back into a raw RGB frame numpy array.

---

## 🛠️ Preprocessing Tools ([data_preprocessing/](file:///c:/Capstone/wlasl-complete/data_preprocessing/))

The repository includes several preprocessing utilities to prepare the dataset:

1. **Evaluate and Select Best Videos**:
   ```bash
   python data_preprocessing/select_best_videos.py
   ```
   *Uses Laplacian clarity filtering and MediaPipe tracking stability to pick the best candidate video for each gloss. Results are saved in `data_preprocessing/best_asl_videos.json`.*

2. **Copy Best Candidate Videos**:
   ```bash
   python data_preprocessing/move_best_videos.py
   ```
   *Copies selected best videos from raw directories to `videos_best/` and `microsoft_best/`.*

3. **Frame Clipping**:
   ```bash
   python data_preprocessing/videos_frame_clipper.py
   ```
   *Clips candidate videos to their starting and ending frames, saving cropped outputs to `videos_cut/` and `microsoft_cut/`.*

4. **Numpy Conversion**:
   ```bash
   python data_preprocessing/video_npy_converter.py
   ```
   *Batch converts cropped videos to raw numpy frame arrays saved in `videos_numpy/` and `microsoft_numpy/`.*

---

## 👁️ Visualizing Keypoint Sequences

You can replay and inspect the skeleton coordinates extracted in `bone_sign_out/` using OpenCV-based visualizer [npy_player.py](file:///c:/Capstone/wlasl-complete/npy_player.py):

```bash
# Play all skeleton sequence files in the directory
python npy_player.py bone_sign_out/

# Play a specific skeleton file
python npy_player.py bone_sign_out/some_sequence.npy
```

### Keyboard Controls:
- **Space** - Pause / Resume
- **Q / Esc** - Quit playback
- **N** - Load next file
- **P** - Load previous file
- **Left / Right / A / D** - Step single frame back/forward (while paused)
- **+ / -** - Speed up / Slow down frame rate
- **R** - Replay current file

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues to help improve SignFy.

## 📄 License

[Add your license information here]

## 👥 Support

If you have questions or need assistance, please open an issue on this repository.
