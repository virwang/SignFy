# Windows 11 本地環境部署與測試指引 (HaMeR Windows 11 Setup Guide)

本手冊提供在 Windows 11 本地工作區（`c:\Capstone\wlasl-complete`）下，使用 `venv` 部署 **HaMeR** 並進行 3D 手部重建測試的完整步驟。這能避免您在 Google Colab 遇到運行階段（Runtime）超時的問題。

---

## 🛠️ 環境與系統要求

為了在 Windows 上順利執行 HaMeR，請確保您的環境符合以下配置：

1.  **作業系統**：Windows 10 或 Windows 11 (64-bit)。
2.  **GPU**：NVIDIA 顯示卡（您的本機為 **RTX 4060 Laptop GPU**，非常合適）。
3.  **Python 版本**：**必須是 Python 3.10.x 或 3.11.x**。
    > [!WARNING]
    > **請勿使用 Python 3.12 或更高版本**（例如 3.14），因為這些版本在 Windows 上沒有預編譯好的 PyTorch3D wheel，手動編譯將會陷入 MSVC 的編譯地獄。

---

## 🚀 部署步驟

請在工作區根目錄 `c:\Capstone\wlasl-complete` 開啟 **PowerShell**，並依序執行以下指令：

### 步驟 1：建立並啟用 Python 虛擬環境 (venv)

由於您的系統 `PATH` 預設的 `python` 指令指向了 Python 3.14，您必須**指定 Python 3.11 的絕對路徑**來建立虛擬環境。

#### 1. 尋找您的 Python 3.11 安裝路徑
請在 PowerShell 中執行以下命令，自動尋找您電腦中 Python 3.11 的可執行檔路徑：
```powershell
# 檢測常見的 Windows 安裝路徑
$userPath = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
$systemPath = "C:\Program Files\Python311\python.exe"
$storePath = "$env:LOCALAPPDATA\Microsoft\WindowsApps\python3.11.exe"

if (Test-Path $userPath) {
    Write-Host "找到 Python 3.11 位於: $userPath" -ForegroundColor Green
} elseif (Test-Path $systemPath) {
    Write-Host "找到 Python 3.11 位於: $systemPath" -ForegroundColor Green
} elseif (Test-Path $storePath) {
    Write-Host "找到 Python 3.11 位於: $storePath" -ForegroundColor Green
} else {
    Write-Host "未能在預設路徑中找到 Python 3.11，請手動確認安裝路徑。" -ForegroundColor Red
}
```

#### 2. 使用 Python 3.11 建立並啟用虛擬環境
找到路徑後，請使用 `&` 符號配合絕對路徑（請替換為您實際上查找到的路徑）來建立 `venv`：

```powershell
# 1. 指定 Python 3.11 絕對路徑來建立虛擬環境 (以預設個人安裝路徑為例)
& "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" -m venv hamer_env

# 2. 繞過 Windows 腳本執行限制（僅對當前 PowerShell 視窗有效）
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force

# 3. 啟用虛擬環境
.\hamer_env\Scripts\Activate.ps1
```
> [!NOTE]
> **好消息**：您只需要在**建立虛擬環境時**指定一次 Python 3.11 的絕對路徑。一旦虛擬環境建立並啟用後（PowerShell 最前端出現 `(hamer_env)`），在這個視窗中直接輸入 `python` 或 `pip` 就會**自動且強制指向 3.11 虛擬環境**，不需要每次都輸入絕對路徑。


---

### 步驟 2：安裝 PyTorch (CUDA 12.1 支援版)

針對您的 RTX 4060 顯示卡，建議安裝支援 CUDA 12.1 的 PyTorch：

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**驗證 GPU 支援是否成功啟用**：
```powershell
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```
*(如果輸出顯示 `CUDA Available: True` 且出現 `GeForce RTX 4060...` 代表 PyTorch 安裝成功。)*

---

### 步驟 3：安裝 PyTorch3D（Windows 預編譯免編譯版）

由於官方不提供 Windows 的 PyTorch3D wheel，我們使用社群維護的穩定預編譯索引：

```powershell
# 1. 安裝核心相依性
pip install fvcore iopath

# 2. 從社群 Index 下載相容您 Python/PyTorch 版本的 Windows Wheel
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d
```

---

### 步驟 4：下載 HaMeR 原始碼並安裝

```powershell
# 1. 複製原始碼與第三方子模組
git clone --recursive https://github.com/geopavlakos/hamer.git
cd hamer

# 2. 安裝 HaMeR 依賴套件
pip install -e .

# 3. 安裝第三方整體關鍵點定位套件 ViTPose
pip install -v -e third-party/ViTPose
```

---

### 步驟 5：下載模型權重與 MANO 模型

#### A. 下載預訓練權重
在 `hamer/` 目錄下執行以下 PowerShell 指令下載預訓練資料：
```powershell
# 安裝 gdown
pip install gdown

# 使用 gdown 下載權重壓縮包
gdown https://drive.google.com/uc?id=1mv7CUAnm73oKsEEG1xE3xH2C_oqcFSzT

# 使用 Windows 內建的 tar 解壓縮 (Windows 10/11 已內建 tar 支援)
tar -xvf hamer_demo_data.tar.gz
```

#### B. 下載並放置 MANO 模型 (手部 3D 約束物理模型)
1.  前往 [MANO 官網](https://mano.is.tue.mpg.de/) 免費註冊並登入。
2.  下載 **MANO v1.2**（`mano_v1_2.zip`），解壓並提取其中的 `models/MANO_RIGHT.pkl`（若有左手亦可提取 `MANO_LEFT.pkl`）。
3.  將 `MANO_RIGHT.pkl` 複製並放入您本機的以下目錄中：
    `c:\Capstone\wlasl-complete\hamer\_DATA\data\mano\MANO_RIGHT.pkl`

---

## 🎬 測試與成效評估執行

當上述部署完成後，您即可在 `c:\Capstone\wlasl-complete\hamer` 目錄下進行以下測試：

### 🏁 測試 1：執行官方提供的範例圖片測試
這是官方自帶的展示圖片測試，可用於確認環境是否已完全跑通：

```powershell
python demo.py --img_folder example_data --out_folder official_demo_out --batch_size=48 --side_view --save_mesh --full_frame
```
*   **如何檢視結果**：執行完畢後，打開資料夾 `c:\Capstone\wlasl-complete\hamer\official_demo_out`，裡面會生成帶有彩色手部網格渲染的圖片，確認圖片是否生成完整且無報錯。

---

### 🏁 測試 2：測試 `microsoft_cut` 目錄下的任意手語影片
為了方便您在本地直接處理影片，我寫好了一份自動「切影格 $\rightarrow$ 推論 $\rightarrow$ 組裝影片」的 Python 腳本。

1. 請在 `c:\Capstone\wlasl-complete\hamer\` 目錄下，建立一個名為 **`run_hamer_on_video.py`** 的檔案。
2. 將以下程式碼複製並貼入該檔案中：

```python
import os
import cv2
import shutil
import subprocess

# 選擇您要測試的微軟影片 (這裏以 17361965357788933-W.H.A.T.mp4 為例)
VIDEO_PATH = r"..\microsoft_cut\17361965357788933-W.H.A.T.mp4" 
INPUT_FRAMES_DIR = "my_video_input_frames"
OUTPUT_FRAMES_DIR = "my_video_output_frames"

# 1. 清理與建立暫存資料夾
shutil.rmtree(INPUT_FRAMES_DIR, ignore_errors=True)
shutil.rmtree(OUTPUT_FRAMES_DIR, ignore_errors=True)
os.makedirs(INPUT_FRAMES_DIR, exist_ok=True)

# 2. 使用 OpenCV 將影片拆成單張圖片
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or not cap.isOpened():
    print(f"❌ 錯誤：無法開啟影片檔案 {VIDEO_PATH}！")
    exit(1)

frame_idx = 0
print("正在拆解影片影格...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imwrite(os.path.join(INPUT_FRAMES_DIR, f"{frame_idx:05d}.jpg"), frame)
    frame_idx += 1
cap.release()
print(f"影格拆解完成，共 {frame_idx} 幀，FPS: {fps:.2f}")

# 3. 呼叫 HaMeR 進行 3D 重建推論
print("開始執行 HaMeR 3D 重建推論...")
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

# 4. 將輸出結果圖片合回影片
print("正在將影格組裝回影片...")
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
    print(f"🎬 重建影片已成功生成於: {os.path.abspath(output_video_name)}")
else:
    print("❌ 錯誤：未生成任何推論結果影格。")
```

3. 在啟用了 `hamer_env` 的 PowerShell 中執行該腳本：
   ```powershell
   python run_hamer_on_video.py
   ```
4. **如何檢視結果**：執行完畢後，直接開啟 `c:\Capstone\wlasl-complete\hamer\hamer_result_microsoft_video.mp4` 播放，即可看到主動手被彩色手部網格渲染包覆的成果，這將能讓您明確了解 HaMeR 在面對微軟坐姿影片時的抓取成效！
