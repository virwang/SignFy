import sys
import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch

from hamer_video_npy_converter import initialize_hamer, process_video

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


def setup_logging(log_path: Path) -> None:
    """
    Log to both the console and a file under output_dir.

    For a batch job over a large dataset (potentially run unattended, e.g. via
    nohup/tmux on a remote box), a persistent log file matters more than
    console output alone — it lets you check progress or diagnose failures
    after the fact without having kept the terminal open.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def is_valid_npy(path: Path) -> bool:
    """
    Check that a previously-generated .npy is actually loadable, not just
    present. Guards against resuming a batch run over leftover files from an
    older/interrupted run and silently treating a corrupt file as "done".
    """
    try:
        arr = np.load(path, mmap_mode="r")
        return arr.size > 0
    except Exception:
        return False


def format_eta(seconds) -> str:
    if seconds is None or seconds < 0 or seconds != seconds:  # None/NaN guard
        return "unknown"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


def main():
    parser = argparse.ArgumentParser(description="Batch Convert ASL videos to .npy using MediaPipe + HaMeR")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing .mp4 videos")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output .npy files")
    parser.add_argument("--overwrite", action="store_true", help="Reprocess videos even if a valid .npy already exists")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N videos (useful for a quick test run)")
    parser.add_argument("--empty_cache_every", type=int, default=50,
                         help="Call torch.cuda.empty_cache() every N videos to reduce GPU memory fragmentation over a long run (0 to disable)")
    parser.add_argument("--batch_size", type=int, default=48, help="Batch size for HaMeR inference. Larger is faster but uses more VRAM.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: {input_dir} does not exist.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir / "batch_log.txt")
    log = logging.getLogger(__name__)

    failed_log_path = output_dir / "failed_videos.txt"

    mp4_files = sorted(input_dir.rglob("*.mp4"))  # deterministic order -> reproducible/resumable runs
    if len(mp4_files) == 0:
        log.warning(f"No .mp4 files found in {input_dir}")
        return

    if args.limit is not None:
        mp4_files = mp4_files[:args.limit]

    log.info(f"Found {len(mp4_files)} video(s). Initializing HaMeR and MediaPipe models...")
    hamer_models = initialize_hamer()

    ok = 0
    fail = 0
    skipped = 0
    # Running sum/count instead of a growing list of per-video times: avoids
    # re-summing the whole history on every single iteration (O(1) update
    # instead of O(n), which matters once you're batching thousands of videos).
    total_video_time = 0.0
    timed_count = 0
    start_time = time.time()
    interrupted = False

    progress_bar = tqdm(total=len(mp4_files), desc="Batch", unit="video") if _HAS_TQDM else None

    try:
        for i, mp4_file in enumerate(mp4_files):
            rel_path = mp4_file.relative_to(input_dir)
            npy_file = output_dir / rel_path.with_suffix(".npy")
            npy_file.parent.mkdir(parents=True, exist_ok=True)

            if not args.overwrite and npy_file.exists():
                if is_valid_npy(npy_file):
                    skipped += 1
                    ok += 1
                    if progress_bar is not None:
                        progress_bar.update(1)
                    continue
                else:
                    log.warning(f"{npy_file} exists but is not a valid/complete .npy — reprocessing {mp4_file.name}")

            avg = (total_video_time / timed_count) if timed_count else None
            eta = format_eta(avg * (len(mp4_files) - i)) if avg else "unknown"

            video_start = time.time()
            if progress_bar is None:
                log.info(f"--- Processing {mp4_file.name} ({i + 1}/{len(mp4_files)}, ETA {eta}) ---")

            try:
                success = process_video(str(mp4_file), str(npy_file), hamer_models, batch_size=args.batch_size)
                if success:
                    ok += 1
                else:
                    fail += 1
                    with open(failed_log_path, "a", encoding="utf-8") as f:
                        f.write(f"{mp4_file}\treturned_false\n")
                    log.error(f"process_video returned False for {mp4_file}")
            except Exception as e:
                fail += 1
                with open(failed_log_path, "a", encoding="utf-8") as f:
                    f.write(f"{mp4_file}\t{e.__class__.__name__}: {e}\n")
                log.exception(f"Failed to process {mp4_file}")

            total_video_time += time.time() - video_start
            timed_count += 1

            if args.empty_cache_every and torch.cuda.is_available() and (i + 1) % args.empty_cache_every == 0:
                torch.cuda.empty_cache()

            if progress_bar is not None:
                progress_bar.update(1)
                progress_bar.set_postfix(ok=ok, fail=fail, skipped=skipped, eta=eta)

    except KeyboardInterrupt:
        interrupted = True
        log.warning("Interrupted by user (Ctrl+C). Progress so far is preserved — rerun the same "
                    "command to resume (already-completed videos are skipped automatically).")
    finally:
        if progress_bar is not None:
            progress_bar.close()

    total_elapsed = time.time() - start_time
    processed = ok - skipped

    log.info("Batch processing stopped early." if interrupted else "Batch processing complete!")
    log.info(f"Total time elapsed: {total_elapsed / 60:.2f} minutes")
    log.info(f"Newly processed: {processed}, skipped (already done): {skipped}, failed: {fail}")
    log.info(f"Have {ok}/{len(mp4_files)} valid .npy files in total.")
    if fail > 0:
        log.info(f"See {failed_log_path} for the list of failed videos and error messages.")


if __name__ == "__main__":
    main()