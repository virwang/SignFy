# SignFy Technology Analysis Report

## 📋 Executive Summary

**SignFy** is a comprehensive Python-based end-to-end sign language recognition and translation system designed to facilitate seamless communication between deaf and hearing individuals. The project integrates cutting-edge computer vision, natural language processing, and 3D human pose estimation technologies to automatically translate English sentences into American Sign Language (ASL) videos.

---

## 🛠️ Technology Stack Analysis

### Core Language and Runtime Environment

| Item | Details |
|------|---------|
| **Primary Language** | Python |
| **Python Versions** | 3.14 (main environment `asl_env`), 3.11 (auxiliary environment `hamer_env`) |
| **Language Distribution** | Python 99.8%, Batchfile 0.2% |
| **Repository Size** | ~1.28 MB |
| **Repository Type** | Private (personal project) |

### Dependency Frameworks and Core Libraries

#### **Main Environment (asl_env) Dependencies**

```
📦 Data Processing & Scientific Computing
  ├─ numpy==2.4.6
  ├─ pandas==3.0.3
  └─ scipy==1.17.1

📦 Computer Vision & Multimedia
  ├─ opencv-python==4.13.0.92
  ├─ opencv-contrib-python==4.13.0.92
  ├─ mediapipe==0.10.35
  ├─ moviepy==2.2.1
  ├─ imageio==2.37.3
  ├─ pillow==11.3.0
  └─ av==17.1.0 (PyAV - Video Decoding)

📦 LLM & Text Processing
  ├─ Local Ollama + llama3.1 (English → ASL Gloss Translation)
  └─ urllib (HTTP Connection to localhost:11434)

📦 Data Export & Report Generation
  ├─ openpyxl==3.1.5 (Excel Reports)
  └─ pandas==3.0.3 (Data Frames)

📦 UI & Service Framework
  ├─ streamlit==1.58.0 (Frontend - planned)
  ├─ starlette==1.3.0
  ├─ uvicorn==0.49.0
  └─ FastAPI-related dependencies

📦 Utilities & Helper Libraries
  ├─ tqdm==4.68.2 (Progress Bars)
  ├─ python-dotenv==1.2.2 (Environment Variable Management)
  └─ requests==2.34.2
```

#### **Auxiliary Environment (hamer_env) Dependencies — 3D Human Pose Estimation**

```
📦 Deep Learning Framework
  ├─ torch==2.5.1+cu121 (CUDA 12.1 GPU Acceleration)
  ├─ torchvision==0.20.1+cu121
  └─ pytorch-lightning==2.6.5

📦 3D Pose & Hand Estimation
  ├─ pytorch3d (3D Computer Vision)
  ├─ hamer (GitHub: geopavlakos/hamer)
  │   └─ Full-body 3D human mesh & hand pose estimation
  ├─ ViTPose (GitHub: ViTAE-Transformer/ViTPose)
  │   └─ Vision Transformer-based keypoint detector
  └─ detectron2 (Object Detection Framework)

📦 Vision & Graphics
  ├─ opencv-python==4.8.1.78
  ├─ pillow==9.5.0
  ├─ scikit-image==0.23.2
  ├─ pyrender==0.1.45 (3D Mesh Rendering)
  ├─ trimesh==4.12.2 (3D Mesh Processing)
  └─ PyOpenGL==3.1.0

📦 Data & Model Management
  ├─ mediapipe==0.10.35
  ├─ safetensors==0.8.0
  ├─ huggingface_hub==1.22.0
  └─ timm==1.0.27 (Transformer Image Models)

📦 SMPL / MANO Human Body Models
  ├─ chumpy==0.71
  ├─ smplx==0.1.28 (SMPL+X Full-body Model)
  └─ scipy, numpy (Matrix Computations)
```

---

## 🏗️ System Architecture

### System-Level Architecture Diagram

```
SignFy End-to-End Translation Pipeline
│
├─────────────────────────────────────────────────────────────────┐
│                    User Input: English Sentence                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: LLM Translation                                         │
│ ├─ Input: English sentence                                       │
│ ├─ Processing: english_to_asl_gloss_llama.py                     │
│ │       • Ollama + llama3.1 (local service :11434)              │
│ │       • System prompt loaded from english_asl_gloss_llama_    │
│ │         prompt.txt                                             │
│ │       • Temperature 0.2, top_p 0.9 (low randomness)           │
│ │       • TOKEN replacements standardization                    │
│ │         (e.g., "CAN NOT" → "CAN'T")                          │
│ ├─ Output: ASL gloss sequence (JSON array)                       │
│ │        [{"gloss":"HELLO","fallback":"HI"},...]                │
│ └─ Result: Primary + fallback synonym pairs                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2: Video Mapping                                           │
│ ├─ Input: ASL gloss sequence                                     │
│ ├─ Processing: asl_llm_video_mapping.py                          │
│ │       • Load best_asl_videos.json gloss→video database        │
│ │       • Priority search: primary gloss, fallback if failed    │
│ │       • In-memory video file caching                          │
│ │         (microsoft_cut/ + videos_cut/)                        │
│ │       • Generate Excel report (Found / Missing categories)    │
│ │       • Generate JSON tracking logs                            │
│ ├─ Output: Matched gloss → {video_id, video_path} mapping       │
│ └─ Edge Cases: Synonym degradation, missing video handling      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 3: Video Clipping & Merging                               │
│ ├─ Input: Video records (video_id + paths)                      │
│ ├─ Processing: video_clipper.py                                  │
│ │       • OpenCV cv2.VideoCapture / cv2.VideoWriter             │
│ │       • Clip individual sign language videos per frame        │
│ │       • Use moviepy for merging clips into single sequence    │
│ │       • Optional: PIL drawing for labels & progress bar       │
│ ├─ Output: Merged video file (MP4, saved to sign_out/)          │
│ └─ Result: {final_video_path}                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 3.5: Keypoint Extraction                                  │
│ ├─ Input: Merged video file                                     │
│ ├─ Processing: media_pipline_converter.py                        │
│ │       • MediaPipe Holistic frame-by-frame detection           │
│ │       • Extract coordinates: Pose(33) + Left Hand(21) +       │
│ │         Right Hand(21) = 225 dimensions                       │
│ │       • PyAV (av) video decoding to RGB24 format              │
│ │       • Output shape: (T, 225) float32 .npy file              │
│ ├─ Output: 225D skeleton sequence .npy                           │
│ │          (saved to bone_sign_out/)                            │
│ └─ Supplementary: JSON file (optional)                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 3.6: Raw Frame Conversion (Optional)                      │
│ ├─ Input: Merged video file                                     │
│ ├─ Processing: video_npy_converter.py                            │
│ │       • Read video frame-by-frame, convert to RGB array       │
│ └─ Output: Raw frame .npy (H×W×3, saved to videos_numpy/)       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Post-Processing & Visualization                                 │
│ ├─ npy_player.py: Play .npy skeleton sequences using OpenCV     │
│ │   └─ Keyboard Controls: Space (pause), Q (quit), N/P (next/  │
│ │       prev), A/D (frame-by-frame), +/- (speed adjust)         │
│ └─ Generated Reports: Excel + JSON                              │
│    (mapping statistics, missing item tracking)                   │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow and Directory Structure

```
SignFy/
├─ main_pipeline.py               ← Pipeline entry point
├─ english_to_asl_gloss_llama.py  ← Stage 1: LLM translation
│  ├─ english_asl_gloss_llama_prompt.txt (system prompt)
│  └─ Interface: Ollama @ http://localhost:11434/api/chat
├─ asl_llm_video_mapping.py       ← Stage 2: Gloss→Video mapping
│  └─ Dependency: data_preprocessing/best_asl_videos.json
├─ video_clipper.py               ← Stage 3: Video merging
│                                    (OpenCV + moviepy)
├─ media_pipline_converter.py      ← Stage 3.5: MediaPipe
│  │                                 keypoint extraction
│  └─ Model: holistic_landmarker.task
│     (auto-downloaded on first run)
├─ npy_player.py                  ← Visualization tool
│                                    (OpenCV player)
│
├─ data_preprocessing/            ← Preprocessing toolset
│  ├─ select_best_videos.py       (Laplacian clarity +
│  │                                MediaPipe stability)
│  ├─ move_best_videos.py         (Copy selected videos)
│  ├─ videos_frame_clipper.py     (Frame-by-frame clipping)
│  ├─ video_npy_converter.py      (Video→raw frames)
│  └─ best_asl_videos.json        (Gloss→{video_id,
│                                    source} mapping)
│
├─ hamer/                         ← Optional 3D pose
│  │                                estimation (PyTorch)
│  ├─ hamer_video_npy_converter.py (MediaPipe + HaMeR
│  │                                inference)
│  ├─ hamer_batch_npy.py          (Batch processing)
│  └─ holistic_landmarker.task    (MediaPipe model file)
│
├─ asl_env/                       ← Python 3.14 venv
├─ hamer_env/                     ← Python 3.11 venv (GPU)
│
├─ videos_raw/                    ← Raw WLASL videos
├─ microsoft_videos_raw/          ← Raw Microsoft videos
├─ videos_best/ & microsoft_best/ ← Selected best videos
├─ videos_cut/ & microsoft_cut/   ← Clipped videos
├─ videos_numpy/                  ← Raw frame .npy files
├─ bone_sign_out/                 ← 225D skeleton
│                                    sequence .npy files
└─ sign_out/                      ← Merged ASL
                                    translation videos
```

---

## 🔬 Technical Deep Dive

### 1️⃣ **Stage 1: English→ASL Gloss Translation (LLM)**

**Module**: `english_to_asl_gloss_llama.py`

| Aspect | Implementation |
|--------|-----------------|
| **Model** | Ollama + Llama 3.1 (local) |
| **API Endpoint** | `http://localhost:11434/api/chat` |
| **Protocol** | HTTP POST JSON (urllib) |
| **Prompt Management** | External file `english_asl_gloss_llama_prompt.txt` loaded at runtime |
| **Temperature Setting** | 0.2 (low randomness, deterministic) |
| **Top-P (Nucleus Sampling)** | 0.9 |
| **JSON Parsing** | Regex extraction of `[{...}]` format |
| **TOKEN Standardization** | Pre-defined replacement table (e.g., `CAN NOT` → `CAN'T`) |
| **Output Format** | Array: `[{"gloss": "WORD", "fallback": "SYNONYM"}, ...]` |
| **Error Handling** | Synonym degradation, filtering of helper verbs (AM/IS/ARE/BE/BEEN) |

**Code Logic**:
```python
# Call Ollama
payload = {
    "model": "llama3.1",
    "stream": False,
    "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Translate: {text}"}
    ]
}
response = urllib.request.urlopen(request)
gloss_list = clean_gloss(response_text)  # Parse JSON
```

---

### 2️⃣ **Stage 2: ASL Gloss→Video File Mapping**

**Module**: `asl_llm_video_mapping.py`

| Aspect | Implementation |
|--------|-----------------|
| **Database** | `best_asl_videos.json` (gloss→video ID + source) |
| **Caching Strategy** | In-memory dictionary (avoid repeated disk I/O) |
| **Video Sources** | Microsoft (microsoft_cut/) + WLASL (videos_cut/) |
| **Matching Logic** | Primary gloss → fallback gloss (2-tier degradation) |
| **Excel Report** | 4 sheets: Summary / Found Videos / Missing Videos / Source Statistics |
| **JSON Tracking** | UUID filename with gloss status, chosen_gloss, fallback_used flag |
| **Formatting** | openpyxl (dark blue headers, borders, auto-width) |
| **Performance Optimization** | Set-based video file existence check O(1) |

**Output Mapping Structure**:
```json
{
  "HELLO": {
    "video_id": "12345.mp4",
    "video_path": "./videos_cut/"
  },
  ...
}
```

---

### 3️⃣ **Stage 3: Video Clipping and Merging**

**Module**: `video_clipper.py`

| Aspect | Implementation |
|--------|-----------------|
| **Core Tools** | OpenCV (cv2) + MoviePy |
| **Video Reading** | `cv2.VideoCapture()` |
| **Frame Boundary Extraction** | WLASL JSON query (frame_start, frame_end, fps) |
| **Clipping Algorithm** | Read frames based on frame_start/frame_end positions |
| **Video Writing** | `cv2.VideoWriter()` with mp4v encoder |
| **Merging Method** | MoviePy `concatenate_videoclips()` |
| **Text Overlay** | PIL drawing (optional labels) |
| **Progress Bar** | OpenCV rectangle drawing |
| **Output** | Single MP4 (saved to sign_out/) |

**Core Code Snippet**:
```python
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
while reading_frames:
    ret, frame = cap.read()
    video_writer.write(frame)  # mp4v encoding
```

---

### 4️⃣ **Stage 3.5: Keypoint Extraction (MediaPipe + Optional HaMeR)**

#### **A. MediaPipe Holistic (Main Pipeline)**

**Module**: `media_pipline_converter.py`

| Aspect | Implementation |
|--------|-----------------|
| **Model** | MediaPipe Holistic Landmarker (float16) |
| **Model Download** | Auto-download from Google Storage on first run |
| **Detection Content** | Pose (33 points) + Left Hand (21 points) + Right Hand (21 points) |
| **Output Dimensions** | 225 = (33+21+21) × 3 (x, y, z coordinates) |
| **Video Decoding** | PyAV (av) library (RGB24 format) |
| **Frame Timestamp** | Calculated as `timestamp = frame_idx * 1000 / fps` |
| **Confidence Filtering** | min_confidence 0.5 (all detection types) |
| **Output Format** | NumPy .npy (float32, shape: (T, 225)) |
| **Storage Location** | bone_sign_out/ |

**Data Structure**:
```
.npy array structure (T, 225):
[
  [pose_x₁, pose_y₁, pose_z₁, ..., pose_x₃₃, ..., left_hand_x₁, ..., right_hand_z₂₁],
  [frame_2_data...],
  ...
]
```

#### **B. HaMeR (Optional High-End 3D)**

**Module**: `hamer/hamer_video_npy_converter.py` + `hamer_batch_npy.py`

| Aspect | Implementation |
|--------|-----------------|
| **Framework** | PyTorch + CUDA 12.1 |
| **Model** | HaMeR (Geopavlakos et al.) |
| **Input** | Raw RGB frames |
| **Keypoint Detection** | ViTPose (Vision Transformer) |
| **3D Estimation** | MANO hand model + SMPL+X full-body |
| **GPU Memory Management** | cuda.empty_cache() every 50 frames to reduce fragmentation |
| **Batch Size** | Default 48 (adjustable) |
| **Output** | 3D hand keypoints (HaMeR format) |
| **Dependencies** | PyTorch3D + detectron2 |

---

### 5️⃣ **Preprocessing Toolchain**

#### `data_preprocessing/select_best_videos.py`
- **Clarity Filtering**: Laplacian variance (edge detection)
- **Stability Scoring**: MediaPipe tracking confidence mean
- **Output**: `best_asl_videos.json`

#### `data_preprocessing/videos_frame_clipper.py`
- **Input**: Raw videos + WLASL frame boundaries
- **Logic**: 1-index to 0-index conversion, OpenCV frame range extraction
- **Output**: Precisely clipped videos to videos_cut/ / microsoft_cut/

#### `data_preprocessing/video_npy_converter.py`
- **Functionality**: Batch convert videos → raw RGB frame arrays
- **Format**: (T, H, W, 3) uint8 .npy

---

## 📊 Technical Highlights & Innovations

| # | Highlight | Details |
|----|-----------|---------|
| 1 | **Multi-level Fallback Mechanism** | Automatic synonym degradation (primary → fallback) |
| 2 | **Local LLM Integration** | Ollama low latency, no cloud dependency |
| 3 | **Dynamic Keypoint Extraction** | MediaPipe Holistic (standardized 225D output) |
| 4 | **Optional 3D Pose** | HaMeR high-fidelity hand/full-body 3D estimation |
| 5 | **Comprehensive Report Generation** | Styled Excel + JSON tracking logs |
| 6 | **In-Memory Caching** | Video file list cache avoids repeated disk scans |
| 7 | **Multi-Source Support** | WLASL + Microsoft videos with automatic selection |
| 8 | **GPU Optimization** | CUDA 12.1 + PyTorch + batch processing |
| 9 | **Cross-Platform** | Windows PowerShell + Linux/macOS bash support |
| 10 | **Parametric Configuration** | .env environment variables, CLI args, config files |

---

## 🔧 Runtime Environment Configuration

### Main Environment (asl_env)
```powershell
# Windows
.\asl_env\Scripts\activate
python -m venv asl_env
pip install -r asl_env_requirements.txt
```

**Dependency Characteristics**:
- NumPy 2.4.6 (latest version)
- MediaPipe 0.10.35 (stable)
- OpenCV 4.13.0.92 (contrib version)
- MoviePy 2.2.1 (video merging)
- Ollama HTTP client (built-in urllib)

### Auxiliary Environment (hamer_env)
```powershell
# Python 3.11 + CUDA 12.1
& "C:\path\to\Python311\python.exe" -m venv hamer_env
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install pytorch3d pytorch-lightning
git clone --recursive https://github.com/geopavlakos/hamer.git
pip install -e hamer/
```

**GPU Verification**:
```python
import torch
print(torch.cuda.is_available())  # True
print(torch.cuda.get_device_name(0))  # e.g., GeForce RTX 4060
```

### Ollama Local Service
```bash
# In another terminal
ollama serve
ollama pull llama3.1  # Or custom model
```

---

## 📈 Performance Metrics & Optimization

| Metric | Current / Optimization |
|--------|----------------------|
| **LLM Latency** | Local Ollama (no network latency), temperature 0.2 ensures fast convergence |
| **Video Processing** | OpenCV + moviepy, lossless quality, optional GPU decoding support |
| **Keypoint Extraction** | MediaPipe ~real-time (CPU), HaMeR GPU accelerated |
| **Caching** | Video file list caching O(1) lookup |
| **Parallelization** | ThreadPoolExecutor for batch video processing |
| **Report Generation** | Pandas + openpyxl styling (<10s for 1000+ glosses) |

---

## 🎯 Use Cases & Applications

```
┌────────────────────────────────────────────────────────────────┐
│ Use Case 1: Real-time Chat Translation                         │
│ ├─ User inputs English → ASL video generated → Sign user views │
│ └─ Latency: ~15-30 seconds (depends on sentence length & GPU) │
├────────────────────────────────────────────────────────────────┤
│ Use Case 2: Educational Training                              │
│ ├─ Instructor inputs course sentences → Large-scale ASL       │
│ │   video generation                                           │
│ └─ Utilize hamer_batch_npy.py for batch processing            │
├────────────────────────────────────────────────────────────────┤
│ Use Case 3: Motion Capture & Animation                        │
│ ├─ Extract 225D skeleton sequence → Animation engine          │
│ │   (Unity/Unreal)                                            │
│ └─ HaMeR 3D mesh → High-fidelity character animation          │
├────────────────────────────────────────────────────────────────┤
│ Use Case 4: Dataset Creation                                  │
│ ├─ Automated clipping of WLASL + Microsoft datasets           │
│ └─ Convert to .npy for machine learning training              │
└────────────────────────────────────────────────────────────────┘
```

---

## ⚠️ Technical Limitations & Future Directions

| Limitation | Explanation | Potential Solution |
|-----------|-------------|-------------------|
| **Vocabulary Coverage** | Dependent on WLASL/Microsoft dataset size | Expand datasets, use GAN video synthesis |
| **Context Processing** | LLM only does vocabulary-level translation, no global grammar | Train ASL-specific seq2seq models |
| **Real-time Performance** | Stages 1-3 combined ~15-30s | Model quantization, edge computing deployment |
| **Sign Language Dialects** | Currently ASL-specific | Multi-language extension (BSL, LSF, etc.) |
| **Fine Motion Details** | 225D may lose details | HaMeR 3D mesh as supplementary |
| **Facial Expressions** | MediaPipe 478 points underutilized | Fine-grained facial expression models |

---

## 📚 Technology Stack Summary Table

```
Layer                 Technology Choice         Version/Framework
────────────────────────────────────────────────────────────────
┌─ Language Generation ─┐
│ LLM              Ollama + Llama 3.1     Local service
│ Text Processing       urllib + json          Standard library
├─ Audio/Visual Processing ─┤
│ Video Decoding        PyAV (av)              17.1.0
│ Video Encoding        OpenCV cv2             4.13.0.92
│ Video Merging         MoviePy                2.2.1
│ Image Processing      Pillow + cv2           11.3.0 / 4.13.0.92
├─ Pose Estimation ─┤
│ Keypoint Detection    MediaPipe Holistic     0.10.35
│ 3D Pose             HaMeR (optional)         GitHub commit
│ 3D Models            PyTorch3D + SMPL+X     Latest
├─ Data & Computation ─┤
│ Scientific Computing  NumPy                  2.4.6
│ Data Frames          Pandas                 3.0.3
│ Deep Learning        PyTorch                2.5.1+cu121
│ GPU Acceleration     CUDA Toolkit            12.1
├─ Reporting & Visualization ─┤
│ Excel Generation     openpyxl + Pandas      3.1.5
│ Data Visualization   Matplotlib             3.11.0
│ Player               OpenCV cv2 namedwindow 4.13.0.92
├─ Development & Deployment ─┤
│ Environment Mgmt     Python venv            3.14 / 3.11
│ Package Mgmt         pip                    25.0+
│ Virtual Envs         asl_env / hamer_env    Standardized
├─ UI Framework ─┤
│ Frontend (planned)   Streamlit              1.58.0
│ Web Framework        Starlette + Uvicorn    1.3.0 / 0.49.0
└────────────────────────────────────────────────────────────────
```

---

## 🚀 Quick Start (Complete Pipeline)

```bash
# 1. Environment Setup
python -m venv asl_env
./asl_env/Scripts/activate  # Windows
source asl_env/bin/activate  # Linux/macOS

# 2. Install Dependencies
pip install -r asl_env_requirements.txt

# 3. Start Ollama (in another terminal)
ollama serve
ollama pull llama3.1

# 4. Run End-to-End Pipeline
python main_pipeline.py
# Input: "What is your name?"
# Output:
#   - Stage 1: [{"gloss":"WHAT","fallback":"HUH"},...]
#   - Stage 2: Video mapping report (.xlsx + .json)
#   - Stage 3: sign_out/merged_video.mp4
#   - Stage 3.5: bone_sign_out/merged_video.npy

# 5. Visualize Skeleton Sequence
python npy_player.py bone_sign_out/
```

---

## Conclusion

**SignFy** is a well-architected multi-modal AI system integrating:
1. **Natural Language Processing** (Ollama LLM)
2. **Computer Vision** (MediaPipe + OpenCV)
3. **3D Human Pose Estimation** (HaMeR + PyTorch)
4. **Multimedia Processing** (MoviePy + PyAV)

This project demonstrates a production-grade accessibility communication solution with scalability and high performance. Future enhancements could include fine-tuned language models, larger-scale datasets, and edge computing deployment to enable real-time communication for the global deaf community.

---

**Report Generated**: 2026-07-15  
**Repository**: [virwang/SignFy](https://github.com/virwang/SignFy)  
**Status**: Private Project  
**Language Composition**: Python 99.8%, Batchfile 0.2%
