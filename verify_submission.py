#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--expected_count", type=int, default=400)
    parser.add_argument("--expected_height", type=int, default=256)
    parser.add_argument("--expected_width", type=int, default=256)
    args = parser.parse_args()

    files = sorted(Path(args.output_dir).glob("*.npy"))
    if len(files) != args.expected_count:
        raise RuntimeError(f"Expected {args.expected_count} files, found {len(files)}")

    for path in files:
        arr = np.load(path, allow_pickle=False)
        if arr.shape != (args.expected_height, args.expected_width):
            raise RuntimeError(f"{path.name}: bad shape {arr.shape}")
        if arr.dtype != np.float32:
            raise RuntimeError(f"{path.name}: bad dtype {arr.dtype}")
        if not np.isfinite(arr).all():
            raise RuntimeError(f"{path.name}: NaN/Inf detected")
        if arr.min() < 0.0 or arr.max() > 1.0:
            raise RuntimeError(f"{path.name}: values outside [0,1]")

    print("Submission output verification PASSED.")
    print("Files :", len(files))
    print("Shape :", (args.expected_height, args.expected_width))
    print("dtype : float32")
    print("range : [0,1]")


if __name__ == "__main__":
    main()
