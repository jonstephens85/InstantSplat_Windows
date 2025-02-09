import os
import subprocess
import time
from pathlib import Path
import torch
import argparse
from enum import IntEnum

class Views(IntEnum):
    TWO = 2
    THREE = 3
    SIX = 6
    TWELVE = 12

def run_process(cmd, log_file=None, show_output=False):
    print(f"Running command: {' '.join(cmd)}")
    try:
        if show_output:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            for line in process.stdout:
                print(line, end='')
                if log_file:
                    with open(log_file, 'a') as f:
                        f.write(line)
            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)
        else:
            if log_file:
                with open(log_file, 'w') as f:
                    process = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, text=True, check=True)
            else:
                process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nError: Process failed with return code {e.returncode}")
        print("Error output:", e.stderr)
        if log_file and os.path.exists(log_file):
            print(f"\nContents of {log_file}:")
            with open(log_file, 'r') as f:
                print(f.read())
        return False
    return True

def process_scene(input_path, output_dir, n_views=Views.TWO, iterations=1000):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing scene from: {input_path}")
    print(f"Output directory: {output_dir}")

    # 1. Co-visible Global Geometry Initialization
    print("\nStep 1: Running initialization...")
    init_cmd = [
        "python", "-W", "ignore", "init_geo.py",
        "-s", str(input_path),
        "-m", str(output_dir),
        "--n_views", str(n_views.value),
        "--focal_avg",
        "--co_vis_dsp",
        "--conf_aware_ranking",
        "--infer_video"
    ]
    if not run_process(init_cmd, output_dir / "01_init_geo.log"):
        return False

    # 2. Train
    print("\nStep 2: Running training...")
    train_cmd = [
        "python", "train.py",
        "-s", str(input_path),
        "-m", str(output_dir),
        "-r", "1",
        "--n_views", str(n_views.value),
        "--iterations", str(iterations),
        "--pp_optimizer",
        "--optim_pose"
    ]
    if not run_process(train_cmd, output_dir / "02_train.log"):
        return False

    # 3. Render Video
    print("\nStep 3: Running rendering...")
    render_cmd = [
        "python", "render.py",
        "-s", str(input_path),
        "-m", str(output_dir),
        "-r", "1",
        "--n_views", str(n_views.value),
        "--iterations", str(iterations),
        "--infer_video"
    ]
    if not run_process(render_cmd, output_dir / "03_render.log", show_output=True):
        return False

    # Check output video
    video_path = output_dir / "interp/ours_1000/interp_3_view.mp4"
    if video_path.exists():
        print(f"\nVideo generated successfully at: {video_path}")
    else:
        print("\nWarning: Video not found at expected location!")
        for mp4_file in output_dir.rglob("*.mp4"):
            print(f"Found video at: {mp4_file}")

    return True

def valid_views(value):
    ivalue = int(value)
    if ivalue not in [v.value for v in Views]:
        raise argparse.ArgumentTypeError(f"n_views must be one of {[v.value for v in Views]}")
    return ivalue

def main():
    parser = argparse.ArgumentParser(description='Run InstantSplat processing on a single scene')
    parser.add_argument('input_path', help='Path to input scene directory containing images')
    parser.add_argument('output_dir', help='Output directory for results')
    parser.add_argument('--n_views', type=valid_views, default=Views.TWO, 
                       help='Number of views (must be 2, 3, 6, or 12)')
    parser.add_argument('--iterations', type=int, default=1000, 
                       help='Number of training iterations (default: 1000)')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Error: CUDA is not available")
        return

    process_scene(args.input_path, args.output_dir, args.n_views, args.iterations)

if __name__ == "__main__":
    main()
