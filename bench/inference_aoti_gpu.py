"""GPU inference latency benchmark for AOTI-compiled models (PyTorch reference).

Usage:  .venv/bin/python bench/inference_aoti_gpu.py

Paired with bench/inference_aoti_gpu.exs to measure ExTorch.AOTI vs PyTorch
AOTI on the same GPU.
"""

import os
import statistics
import sys

import torch
import torch._inductor

if not torch.cuda.is_available():
    print("ERROR: torch.cuda.is_available() == False")
    sys.exit(1)

torch.set_grad_enabled(False)

DEVICE = torch.device("cuda")
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "test", "fixtures")
WARMUP = 5
ITERS = 30

MODELS = [
    ("alexnet",            (1, 3, 224, 224)),
    ("vgg11",              (1, 3, 224, 224)),
    ("squeezenet",         (1, 3, 224, 224)),
    ("mobilenetv2",        (1, 3, 224, 224)),
    ("resnet18",           (1, 3, 224, 224)),
    ("resnet50",           (1, 3, 224, 224)),
    ("vit_b_16",           (1, 3, 224, 224)),
    ("simple_transformer", (1, 16, 32)),
    ("mini_bert",          (1, 16, 32)),
    ("autoencoder",        (1, 784)),
    ("conv_autoencoder",   (1, 3, 32, 32)),
]


def time_one(runner, x):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    runner(x)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def main():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"torch:  {torch.__version__}")
    print()
    print(f"== PyTorch AOTI GPU latency  ({ITERS} iters after {WARMUP} warmup) ==\n")
    print(f"{'model':<22} {'min(ms)':>10} {'median(ms)':>10} {'mean(ms)':>10}")
    print("-" * 56)

    for name, in_shape in MODELS:
        aoti_path = os.path.join(FIXTURES_DIR, f"{name}_aoti.pt2")
        bin_path = os.path.join(FIXTURES_DIR, f"{name}_input.bin")

        if not os.path.exists(aoti_path) or not os.path.exists(bin_path):
            print(f"{name:<22}    SKIP (fixture missing)")
            continue

        with open(bin_path, "rb") as f:
            raw = f.read()
        cpu_input = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(in_shape)
        input_tensor = cpu_input.to(DEVICE)

        runner = torch._inductor.aoti_load_package(aoti_path)

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
