"""PyTorch-side equivalent of bench/raw_op.exs.

Times raw at::conv2d calls (via torch.nn.functional.conv2d) without any
exporter / interpreter overhead, to find out whether ExTorch's remaining
gap on CNNs is in NIF marshaling or in libtorch kernels themselves.
"""

import time
import torch
import torch.nn.functional as F

ITERS = 200


def bench(name, fn):
    # Warm up
    for _ in range(3):
        fn()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fn()
    elapsed = (time.perf_counter() - t0) * 1000.0 / ITERS  # ms per call
    print(f"{name:<48s} {elapsed:7.2f} ms/call")


def main():
    with torch.no_grad():
        # ResNet50's first conv
        x1 = torch.randn(1, 3, 224, 224)
        w1 = torch.randn(64, 3, 7, 7)
        bench("conv2d 3->64 k=7 s=2 p=3 input=224x224",
              lambda: F.conv2d(x1, w1, stride=2, padding=3))

        # ResNet18 layer1 conv
        x2 = torch.randn(1, 64, 56, 56)
        w2 = torch.randn(64, 64, 3, 3)
        bench("conv2d 64->64 k=3 s=1 p=1 input=56x56",
              lambda: F.conv2d(x2, w2, stride=1, padding=1))

        # ResNet50 bottleneck 1x1
        x3 = torch.randn(1, 256, 56, 56)
        w3 = torch.randn(64, 256, 1, 1)
        bench("conv2d 256->64 k=1 s=1 p=0 input=56x56",
              lambda: F.conv2d(x3, w3, stride=1, padding=0))

        # MobileNetV2 depthwise
        x4 = torch.randn(1, 96, 14, 14)
        w4 = torch.randn(96, 1, 3, 3)
        bench("depthwise 96->96 k=3 groups=96 input=14x14",
              lambda: F.conv2d(x4, w4, stride=1, padding=1, groups=96))


if __name__ == "__main__":
    main()
