"""GPU inference latency benchmark for PyTorch reference.

Usage:  .venv/bin/python bench/inference_gpu.py

Loads each .pt2 fixture on CUDA, runs the saved input on CUDA N times,
and reports min / median / mean latency. Requires a CUDA-capable PyTorch
build (check via `torch.cuda.is_available()`).

Paired with bench/inference_gpu.exs to measure ExTorch vs PyTorch on the
same GPU.
"""

import os
import statistics
import sys
import time

import torch

if not torch.cuda.is_available():
    print("ERROR: torch.cuda.is_available() == False")
    sys.exit(1)

# Match ExTorch's inference-mode setup: disable gradient tracking globally.
torch.set_grad_enabled(False)

DEVICE = torch.device("cuda")
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "test", "fixtures")
WARMUP = 5
ITERS = 30

MODELS = [
    ("alexnet",            (1, 3, 224, 224),  (1, 1000)),
    ("vgg11",              (1, 3, 224, 224),  (1, 1000)),
    ("squeezenet",         (1, 3, 224, 224),  (1, 1000)),
    ("mobilenetv2",        (1, 3, 224, 224),  (1, 1000)),
    ("resnet18",           (1, 3, 224, 224),  (1, 1000)),
    ("resnet50",           (1, 3, 224, 224),  (1, 1000)),
    ("simple_transformer", (1, 16, 32),       (1, 10)),
    ("mini_bert",          (1, 16, 32),       (1, 10)),
    ("autoencoder",        (1, 784),          (1, 784)),
    ("conv_autoencoder",   (1, 3, 32, 32),    (1, 3, 32, 32)),
    ("simple_lstm",        (1, 16, 32),       (1, 10)),
]


def time_one(runner, x):
    # Use CUDA events for accurate GPU timing: launch + synchronize.
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    runner(x)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)  # ms


def main():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"torch:  {torch.__version__}")
    print(f"CUDA:   {torch.version.cuda}")
    print(f"cuDNN:  {torch.backends.cudnn.version()}  enabled={torch.backends.cudnn.enabled}")
    print()
    print(f"== PyTorch torch.export.load GPU latency  ({ITERS} iters after {WARMUP} warmup) ==\n")
    print(f"{'model':<22} {'min(ms)':>10} {'median(ms)':>10} {'mean(ms)':>10}")
    print("-" * 56)

    for name, in_shape, _out_shape in MODELS:
        pt2 = os.path.join(FIXTURES_DIR, f"{name}.pt2")
        bin_path = os.path.join(FIXTURES_DIR, f"{name}_input.bin")
        if not os.path.exists(pt2) or not os.path.exists(bin_path):
            print(f"{name:<22}    SKIP (fixture missing)")
            continue

        with open(bin_path, "rb") as f:
            raw = f.read()
        cpu_input = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(in_shape)
        input_tensor = cpu_input.to(DEVICE)

        ep = torch.export.load(pt2)
        runner = ep.module().to(DEVICE)

        # Warmup: compile kernel selection / allocate CUDA caches
        for _ in range(WARMUP):
            runner(input_tensor)
        torch.cuda.synchronize()

        samples = [time_one(runner, input_tensor) for _ in range(ITERS)]
        samples_sorted = sorted(samples)
        print(
            f"{name:<22} "
            f"{samples_sorted[0]:>10.2f} "
            f"{samples_sorted[len(samples_sorted) // 2]:>10.2f} "
            f"{statistics.mean(samples):>10.2f}"
        )


if __name__ == "__main__":
    main()
