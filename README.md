# SignFy

**Study / Practice Project for SignifyApp — Focus: text2sign**

SignFy is a research and practice implementation built to explore text-to-sign (text2sign) translation techniques and end-to-end pipelines for producing sign language video output from written English. This repository is NOT a production product; it is an experimental codebase used to prototype ideas and assemble components for the SignifyApp project.

## Goals

- Investigate and prototype text2sign: convert an English sentence into an ASL gloss sequence and map that sequence to recorded sign video clips or synthesized skeleton sequences.
- Provide tools for dataset preprocessing, mapping glosses to video snippets, extracting keypoints, and assembling merged sign-language outputs for inspection and evaluation.
- Serve as a practice playground for model and pipeline experiments used in SignifyApp.

## Key Features (text2sign focus)

1. LLM-based English → ASL Gloss translation (text2gloss)
   - Scripts use a local LLM (Ollama) to convert input English sentences into ASL gloss sequences.
   - File: `english_to_asl_gloss_llama.py`

2. Gloss → Video Mapping
   - Maps ASL gloss tokens to best-matching candidate clips from WLASL / Microsoft datasets and selects clips for each gloss.
   - File: `asl_llm_video_mapping.py`

3. Video Clipping & Merging (text2sign output generation)
   - Clips selected video segments and concatenates them into a single translation video that visualizes the ASL translation of the input text.
   - File: `video_clipper.py` (output saved under `sign_out/`)

4. Keypoint Extraction & Skeleton Outputs
   - Extracts MediaPipe Holistic landmarks (pose + hands) and saves 225-dim landmark sequences as `.npy` files.
   - Visualize skeleton sequences with `npy_player.py`.

5. Utilities for dataset preprocessing and frame-level conversions
   - Tools located in `data_preprocessing/` to select best videos, clip frames, and convert to numpy frame arrays.

## Quick Start (text2sign)

1. Activate the Python environment (`asl_env`) and install dependencies:

Windows PowerShell:

```powershell
.\asl_env\Scripts\activate
```

Linux/macOS:

```bash
source asl_env/bin/activate
```

```bash
pip install -r asl_env_requirements.txt
```

2. Start the local Ollama service (used for LLM-based translation):

```bash
ollama serve
ollama pull llama3.1
```

3. Run the end-to-end text2sign pipeline (example):

```bash
python main_pipeline.py
```

- main_pipeline.py orchestrates: English → ASL gloss (LLM) → gloss-to-video mapping → clip & merge → optional keypoint extraction and numpy conversion.

## Directory Overview

- `videos_raw/`, `Microsoft_Videos_raw/` — original dataset clips
- `videos_best/`, `microsoft_best/` — selected best candidate clips
- `videos_cut/`, `microsoft_cut/` — clipped videos per-gloss
- `videos_numpy/`, `microsoft_numpy/` — raw RGB frame arrays
- `sign_out/` — merged final sign video results (text2sign outputs)
- `bone_sign_out/` — extracted skeleton `.npy` sequences
- `data_preprocessing/` — helpers for selection, clipping, conversion
- `main_pipeline.py` — orchestrates the full text2sign pipeline
- `npy_player.py` — visualizer for skeleton sequences

## Intended Use & Limitations

- This repository is a study/practice project and intended for experimentation, research, and education. It is not production-ready and should not be used as a deployed accessibility product without further engineering, testing, and community consultation.
- Accuracy depends heavily on dataset coverage, gloss mapping heuristics, and the underlying LLM; results are demonstrative.

## Contributing

Contributions, experiment notes, and improvements are welcome. If you make changes that materially improve the text2sign pipeline, please open a pull request describing the change and reproducible steps.

## License

[Add your license information here]

## Support

For questions or issues, please open an issue on this repository.
