"""
read_npy.py — Read and print NumPy .npy keypoint files in a human-readable format.

Usage:
    python read_npy.py <path_or_filename> [num_frames_to_print]
"""

import sys
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
WORKSPACE_DIR = SCRIPT_DIR.parent.resolve()

def resolve_npy_path(path_str: str) -> Path:
    """
    Resolves the numpy path flexibly:
    1. Check absolute or relative path directly.
    2. Try appending .npy suffix.
    3. Search in common folders (root, microsoft_numpy, videos_numpy, bone_sign_out).
    """
    p = Path(path_str)
    # 1. Direct match
    if p.exists() and p.is_file():
        return p.resolve()
        
    # 2. Try with .npy extension
    if p.suffix.lower() != ".npy":
        p_ext = p.with_suffix(".npy")
        if p_ext.exists():
            return p_ext.resolve()
            
    # 3. Search common directories
    filename = p.name
    if not filename.endswith(".npy"):
        filename += ".npy"
        
    search_dirs = [
        WORKSPACE_DIR,
        WORKSPACE_DIR / "microsoft_numpy",
        WORKSPACE_DIR / "videos_numpy",
        WORKSPACE_DIR / "bone_sign_out",
        SCRIPT_DIR
    ]
    
    for s_dir in search_dirs:
        candidate = s_dir / filename
        if candidate.exists():
            return candidate.resolve()
            
    # Fallback to original path
    return p.resolve()

def read_npy(npy_path: str, num_frames: int = 5):
    """
    Read a .npy file, print its content, and save a text version under bone_sign_out.
    """
    resolved_path = resolve_npy_path(npy_path)
    if not resolved_path.exists():
        print(f"Error: Could not find npy file: {npy_path}")
        print("Tried resolving to common directories, but file does not exist.")
        return

    try:
        data = np.load(str(resolved_path))
        print(f"Successfully loaded: {resolved_path.name}")
        print(f"Full path:         {resolved_path}")
        print(f"Array Shape:       {data.shape}")
        print(f"Data Type:         {data.dtype}")
        
        limit = min(num_frames, len(data))
        print(f"\nShowing first {limit} frames:")
        for i in range(limit):
            frame = data[i]
            # Use numpy default formatting (which truncates very long arrays with '...')
            print(f"Frame {i:02d}: {frame}")

        # Save to txt under bone_sign_out
        out_dir = WORKSPACE_DIR / "bone_sign_out"
        out_dir.mkdir(parents=True, exist_ok=True)
        txt_path = out_dir / resolved_path.with_suffix(".txt").name
        
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Numpy File: {resolved_path.name}\n")
            f.write(f"Shape: {data.shape}\n")
            f.write(f"Dtype: {data.dtype}\n\n")
            for i, frame in enumerate(data):
                frame_str = ", ".join(f"{val:.6f}" for val in frame)
                f.write(f"Frame {i:03d}: [{frame_str}]\n")
                
        print(f"\nSaved text representation to: {txt_path}")
    except Exception as e:
        print(f"Error loading or saving {resolved_path.name}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python read_npy.py <path_to_npy_or_filename> [num_frames_to_print]")
        sys.exit(1)
        
    npy_path = sys.argv[1]
    num_frames = 5
    if len(sys.argv) >= 3:
        try:
            num_frames = int(sys.argv[2])
        except ValueError:
            print(f"Warning: '{sys.argv[2]}' is not a valid integer. Using default value of 5.")
            
    read_npy(npy_path, num_frames)
