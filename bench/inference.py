"""Inference latency benchmark for PyTorch reference (torch.export.load).

Usage:  .venv/bin/python bench/inference.py

Loads each .pt2 fixture, runs the saved input through the loaded
ExportedProgram N times, and reports min / median / mean latency.

Used as the baseline against `bench/inference.exs` to measure how much
overhead the ExTorch.Export interpreter adds vs PyTorch's own dispatch.
"""

import os
import statistics
import time

import torch

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "test", "fixtures")
WARMUP = 3
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


def time_one(model_callable, input_tensor):
    t0 = time.perf_counter()
    with torch.no_grad():
        model_callable(input_tensor)
    return (time.perf_counter() - t0) * 1000.0  # ms


def main():
    print(f"== PyTorch torch.export.load latency  ({ITERS} iters after {WARMUP} warmup) ==\n")
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
        input_tensor = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(in_shape)

        ep = torch.export.load(pt2)
        runner = ep.module()

        for _ in range(WARMUP):
            with torch.no_grad():
                runner(input_tensor)

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
