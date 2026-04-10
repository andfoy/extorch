"""GPU version of bench/raw_op.py: times raw torch.nn.functional.conv2d
calls with CUDA tensors to measure per-op kernel launch overhead.

Paired with bench/raw_op_gpu.exs for ExTorch vs PyTorch comparison.

Usage:  .venv/bin/python bench/raw_op_gpu.py
"""

import sys
import time

import torch
import torch.nn.functional as F

if not torch.cuda.is_available():
    print("ERROR: CUDA not available")
    sys.exit(1)

torch.set_grad_enabled(False)

ITERS = 200
DEVICE = torch.device("cuda")


def cuda_tensor(shape):
    return torch.randn(shape, device=DEVICE)


def bench(name, input, weight, stride, padding, dilation, groups):
    # Warmup + sync so the first timed call is steady-state.
    for _ in range(5):
        F.conv2d(input, weight, bias=None, stride=stride,
                 padding=padding, dilation=dilation, groups=groups)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(ITERS):
        F.conv2d(input, weight, bias=None, stride=stride,
                 padding=padding, dilation=dilation, groups=groups)
    # Drain the stream before stopping the clock so we measure the
    # actual kernel time, not just queue latency.
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) * 1000.0 / ITERS

    print(f"{name:<48s} {elapsed:7.3f} ms/call")


def main():
    # ResNet50 first conv
    bench("conv2d 3->64 k=7 s=2 p=3 input=224x224",
          cuda_tensor((1, 3, 224, 224)), cuda_tensor((64, 3, 7, 7)),
          (2, 2), (3, 3), (1, 1), 1)

    # ResNet layer1 conv
    bench("conv2d 64->64 k=3 s=1 p=1 input=56x56",
          cuda_tensor((1, 64, 56, 56)), cuda_tensor((64, 64, 3, 3)),
          (1, 1), (1, 1), (1, 1), 1)

    # ResNet50 bottleneck 1x1
    bench("conv2d 256->64 k=1 s=1 p=0 input=56x56",
          cuda_tensor((1, 256, 56, 56)), cuda_tensor((64, 256, 1, 1)),
          (1, 1), (0, 0), (1, 1), 1)

    # MobileNetV2 depthwise
    bench("depthwise 96->96 k=3 groups=96 input=14x14",
          cuda_tensor((1, 96, 14, 14)), cuda_tensor((96, 1, 3, 3)),
          (1, 1), (1, 1), (1, 1), 96)

    # Deeper bottleneck
    bench("conv2d 512->512 k=3 s=1 p=1 input=7x7",
          cuda_tensor((1, 512, 7, 7)), cuda_tensor((512, 512, 3, 3)),
          (1, 1), (1, 1), (1, 1), 1)


if __name__ == "__main__":
    main()
