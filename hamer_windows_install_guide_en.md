# Windows 11 Local Environment Setup and Testing Guide (HaMeR Windows 11 Setup Guide)

This guide provides the complete steps to deploy **HaMeR** and perform 3D hand reconstruction testing using `venv` under your Windows 11 local workspace (`c:\Capstone\wlasl-complete`). This will help you avoid runtime timeout issues commonly encountered in Google Colab.

---

## 🛠️ Environment and System Requirements

To run HaMeR smoothly on Windows, ensure your environment meets the following specifications:

1.  **Operating System**: Windows 10 or Windows 11 (64-bit).
2.  **GPU**: NVIDIA Graphics Card (Your local machine has an **RTX 4060 Laptop GPU**, which is perfect).
3.  **Python Version**: **Must be Python 3.10.x or 3.11.x**.
    > [!WARNING]
    > **Do not use Python 3.12 or higher** (e.g., 3.14), as there are no pre-compiled PyTorch3D wheels available for these versions on Windows. Compiling it manually will lead to MSVC build issues.

---

## 🚀 Setup Steps

Open **PowerShell** in the workspace root directory `c:\Capstone\wlasl-complete` and run the following commands in order:

### Step 1: Create and Activate Python Virtual Environment (venv)

Since the default `python` command in your system `PATH` points to Python 3.14, you must **specify the absolute path of Python 3.11** to create the virtual environment.

#### 1. Find Your Python 3.11 Installation Path
Run the following command in PowerShell to automatically search for the Python 3.11 executable path on your computer:
```powershell
# Detect common Windows installation paths
$userPath = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
$systemPath = "C:\Program Files\Python311\python.exe"
$storePath = "$env:LOCALAPPDATA\Microsoft\WindowsApps\python3.11.exe"

if (Test-Path $userPath) {
    Write-Host "Found Python 3.11 at: $userPath" -ForegroundColor Green
} elseif (Test-Path $systemPath) {
    Write-Host "Found Python 3.11 at: $systemPath" -ForegroundColor Green
} elseif (Test-Path $storePath) {
    Write-Host "Found Python 3.11 at: $storePath" -ForegroundColor Green
} else {
    Write-Host "Failed to find Python 3.11 in default paths. Please verify the installation path manually." -ForegroundColor Red
}
```

#### 2. Create and Activate the Virtual Environment Using Python 3.11
Once you find the path, use the ampersand `&` operator with the absolute path (replace with your actual path if different) to create `venv`:

```powershell
# 1. Specify the absolute path of Python 3.11 to create the virtual environment (using the default user installation path as an example)
& "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" -m venv hamer_env

# 2. Bypass Windows script execution restrictions (valid for the current PowerShell window only)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force

# 3. Activate the virtual environment
.\hamer_env\Scripts\Activate.ps1
```
> [!NOTE]
> **Good News**: You only need to specify the absolute path of Python 3.11 **once when creating the virtual environment**. Once the virtual environment is created and activated (indicated by `(hamer_env)` at the beginning of the PowerShell prompt), typing `python` or `pip` in this window will **automatically and strictly point to the 3.11 virtual environment**, so you do not need to enter the absolute path every time.


---

### Step 2: Install PyTorch (with CUDA 12.1 support)

For your RTX 4060 graphics card, it is recommended to install PyTorch with CUDA 12.1 support:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Verify if GPU support is successfully enabled**:
```powershell
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```
*(If the output displays `CUDA Available: True` and lists `GeForce RTX 4060...`, the PyTorch installation was successful.)*

---

### Step 3: Install PyTorch3D (Pre-compiled Windows Version)

Since official pre-compiled wheels for PyTorch3D on Windows are not provided, we use a stable, community-maintained pre-compiled index:

```powershell
# 1. Install core dependencies
pip install fvcore iopath

# 2. Download the Windows Wheel compatible with your Python/PyTorch version from the community Index
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d
```

---

### Step 4: Download HaMeR Source Code and Install

```powershell
# 1. Clone the source code and third-party submodules
git clone --recursive https://github.com/geopavlakos/hamer.git
cd hamer

# 2. Install HaMeR dependencies
pip install -e .

# 3. Install the third-party whole-body keypoint detector, ViTPose
pip install -v -e third-party/ViTPose
```

---

### Step 5: Download Model Weights and MANO Model

#### A. Download Pre-trained Weights
Run the following PowerShell commands in the `hamer/` directory to download the pre-trained data:
```powershell
# Install gdown
pip install gdown

# Download the weights archive using gdown
gdown https://drive.google.com/uc?id=1mv7CUAnm73oKsEEG1xE3xH2C_oqcFSzT

# Extract using Windows built-in tar (Windows 10/11 has built-in support for tar)
tar -xvf hamer_demo_data.tar.gz
```

#### B. Download and Place the MANO Model (3D Hand Articulated Model)
1.  Go to the [MANO Official Website](https://mano.is.tue.mpg.de/), register for free, and log in.
2.  Download **MANO v1.2** (`mano_v1_2.zip`), extract it, and locate `models/MANO_RIGHT.pkl` (you can also extract `MANO_LEFT.pkl` if you have left-hand data).
3.  Copy and place `MANO_RIGHT.pkl` into the following local directory:
    `c:\Capstone\wlasl-complete\hamer\_DATA\data\mano\MANO_RIGHT.pkl`

---

## 🎬 Testing and Evaluation Execution

Once the deployment is complete, you can perform the following tests in the `c:\Capstone\wlasl-complete\hamer` directory:

### 🏁 Test 1: Run the Official Demo Image Test
This is the demo image test provided by the repository, which can be used to verify that the environment works completely:

```powershell
python demo.py --img_folder example_data --out_folder official_demo_out --batch_size=48 --side_view --save_mesh --full_frame
```
*   **How to view results**: After execution, open the folder `c:\Capstone\wlasl-complete\hamer\official_demo_out`. It will contain rendered images with colored hand meshes. Confirm that the images are generated completely without errors.

---

### 🏁 Test 2: Test Any Sign Language Video under the `microsoft_cut` Directory
To make it easier for you to process videos locally, I have written a Python script that automatically handles "frame extraction $\rightarrow$ inference $\rightarrow$ video re-assembly".

1. In the `c:\Capstone\wlasl-complete\hamer\` directory, create a file named **`run_hamer_on_video.py`**.
2. Copy and paste the following code into the file:

```python
import os
import cv2
import shutil
import subprocess

# Select the Microsoft video you want to test (using 17361965357788933-W.H.A.T.mp4 as an example here)
VIDEO_PATH = r"..\microsoft_cut\17361965357788933-W.H.A.T.mp4" 
INPUT_FRAMES_DIR = "my_video_input_frames"
OUTPUT_FRAMES_DIR = "my_video_output_frames"

# 1. Clean up and create temporary folders
shutil.rmtree(INPUT_FRAMES_DIR, ignore_errors=True)
shutil.rmtree(OUTPUT_FRAMES_DIR, ignore_errors=True)
os.makedirs(INPUT_FRAMES_DIR, exist_ok=True)

# 2. Extract video frames using OpenCV
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or not cap.isOpened():
    print(f"❌ Error: Could not open video file {VIDEO_PATH}!")
    exit(1)

frame_idx = 0
print("Extracting video frames...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imwrite(os.path.join(INPUT_FRAMES_DIR, f"{frame_idx:05d}.jpg"), frame)
    frame_idx += 1
cap.release()
print(f"Frame extraction complete. Total frames: {frame_idx}, FPS: {fps:.2f}")

# 3. Call HaMeR for 3D reconstruction inference
print("Starting HaMeR 3D reconstruction inference...")
cmd = [
    "python", "demo.py",
    "--img_folder", INPUT_FRAMES_DIR,
    "--out_folder", OUTPUT_FRAMES_DIR,
    "--batch_size", "48",
    "--side_view",
    "--save_mesh",
    "--full_frame"
]
subprocess.run(cmd, check=True)

# 4. Re-assemble output frames back into a video
print("Re-assembling frames into video...")
output_video_name = "hamer_result_microsoft_video.mp4"
images = sorted([img for img in os.listdir(OUTPUT_FRAMES_DIR) if img.endswith(".jpg") or img.endswith(".png")])
if len(images) > 0:
    first_frame = cv2.imread(os.path.join(OUTPUT_FRAMES_DIR, images[0]))
    height, width, layers = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_name, fourcc, fps, (width, height))
    for image in images:
        video.write(cv2.imread(os.path.join(OUTPUT_FRAMES_DIR, image)))
    video.release()
    print(f"🎬 Reconstructed video has been successfully generated at: {os.path.abspath(output_video_name)}")
else:
    print("❌ Error: No inference result frames were generated.")
```

3. Run this script in PowerShell with `hamer_env` activated:
   ```powershell
   python run_hamer_on_video.py
   ```
4. **How to view results**: After execution, directly open and play `c:\Capstone\wlasl-complete\hamer\hamer_result_microsoft_video.mp4`. You will see the active hand covered by a colored hand mesh, allowing you to clearly evaluate HaMeR's performance on Microsoft sign language videos where the speaker is seated.
