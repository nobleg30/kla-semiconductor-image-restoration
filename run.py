#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from model import FastNAFSR

DEFAULT_MODEL_CFG = {"width": 64, "blocks": 12, "scale": 2}


def parse_args():
    parser = argparse.ArgumentParser(description="Restore degraded semiconductor .npy images.")
    parser.add_argument("input_dir_pos", nargs="?", type=str)
    parser.add_argument("output_dir_pos", nargs="?", type=str)
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    args = parser.parse_args()
    args.input_dir = args.input_dir or args.input_dir_pos
    args.output_dir = args.output_dir or args.output_dir_pos
    if args.input_dir is None or args.output_dir is None:
        parser.error("Both input and output directories are required.")
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive.")
    return args


def discover_npy_files(input_dir: Path):
    files = sorted(input_dir.glob("*.npy"))
    if not files:
        files = sorted(input_dir.rglob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files found under: {input_dir}")
    return files


def load_input_stack(files):
    arrays = []
    shape = None
    for path in files:
        arr = np.load(path, allow_pickle=False)
        if arr.ndim != 2:
            raise ValueError(f"{path} has shape {arr.shape}; expected 2-D grayscale data.")
        arr = np.asarray(arr, dtype=np.float32)
        if shape is None:
            shape = arr.shape
        elif arr.shape != shape:
            raise ValueError(f"Mixed image shapes found: expected {shape}, got {arr.shape} in {path}")
        arrays.append(np.ascontiguousarray(arr))
    return np.stack(arrays, axis=0)[:, None, :, :]


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_cfg = checkpoint.get("model_cfg", DEFAULT_MODEL_CFG) if isinstance(checkpoint, dict) else DEFAULT_MODEL_CFG
    model = FastNAFSR(**model_cfg)

    if isinstance(checkpoint, dict) and "ema" in checkpoint:
        state = checkpoint["ema"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state = checkpoint["model"]
    else:
        state = checkpoint

    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    return model, checkpoint if isinstance(checkpoint, dict) else {}


def run_inference(model, input_stack, device, batch_size, precision):
    n, _, h, w = input_stack.shape
    scale = int(model.scale)
    outputs = np.empty((n, h * scale, w * scale), dtype=np.float32)

    cpu_tensor = torch.from_numpy(input_stack)
    if device.type == "cuda":
        cpu_tensor = cpu_tensor.pin_memory()

    use_fp16 = device.type == "cuda" and precision == "fp16"

    with torch.inference_mode():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xb = cpu_tensor[start:end].to(
                device,
                dtype=torch.float32,
                non_blocking=(device.type == "cuda"),
            )
            if use_fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    yb = model(xb)
            else:
                yb = model(xb)

            yb = yb[:, 0].float().clamp_(0.0, 1.0).cpu().numpy()
            if not np.isfinite(yb).all():
                raise RuntimeError(f"Non-finite output detected for batch {start}:{end}")
            outputs[start:end] = yb

    return outputs


def save_outputs(files, outputs, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    names = [p.name for p in files]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate .npy basenames found in input tree.")
    for path, restored in zip(files, outputs):
        np.save(
            output_dir / path.name,
            restored.astype(np.float32, copy=False),
            allow_pickle=False,
        )


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    checkpoint_path = (
        Path(args.model_path).expanduser().resolve()
        if args.model_path is not None
        else script_dir / "best_model.pt"
    )

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    files = discover_npy_files(input_dir)
    input_stack = load_input_stack(files)
    model, checkpoint = load_model(checkpoint_path, device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    outputs = run_inference(
        model,
        input_stack,
        device,
        args.batch_size,
        args.precision,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - t0

    save_t0 = time.perf_counter()
    save_outputs(files, outputs, output_dir)
    save_seconds = time.perf_counter() - save_t0

    print(f"Device           : {device}")
    if device.type == "cuda":
        print(f"GPU              : {torch.cuda.get_device_name(0)}")
    print(f"Images processed : {len(files)}")
    print(f"Input shape      : {tuple(input_stack.shape[2:])}")
    print(f"Output shape     : {tuple(outputs.shape[1:])}")
    print(f"Batch size       : {args.batch_size}")
    print(f"Precision        : {args.precision if device.type == 'cuda' else 'fp32'}")
    print(f"Checkpoint epoch : {checkpoint.get('epoch', 'unknown')}")
    print(f"Validation PSNR  : {checkpoint.get('val_psnr', 'unknown')}")
    print(f"Validation SSIM  : {checkpoint.get('val_ssim', 'unknown')}")
    print(f"Inference time   : {inference_seconds:.4f} s")
    print(f"Save time        : {save_seconds:.4f} s")
    print(f"Throughput       : {len(files) / max(inference_seconds, 1e-12):.2f} images/s")
    print(f"Output directory : {output_dir}")


if __name__ == "__main__":
    main()
