"""Export small models with a dynamic batch dimension.

The point of these fixtures is to probe ExTorch.Export's handling of
torch.export's dynamic_shapes — not to benchmark. We want to know where
(if anywhere) the current interpreter breaks when `batch` is symbolic.

Three models with ascending complexity:
  * mlp_dynamic_batch   — pure Linear/ReLU, no shape arithmetic anywhere
  * convnet_dynamic_batch — conv + BN + pool + flatten; flatten is where
    graphs often bake in shape arithmetic at trace time
  * resnet18_dynamic_batch — a real architecture with many reshapes,
    batchnorms, and residual adds

For each, we write the .pt2 plus reference (input, output) pairs at three
different batch sizes so Elixir tests can verify numerical correctness.
"""

import os

import torch
import torch.nn as nn
from torch.export import Dim

FIXTURES_DIR = os.path.dirname(os.path.abspath(__file__))


class DynMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class DynConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 3)

    def forward(self, x):
        x = self.pool(self.relu(self.bn(self.conv(x))))
        x = x.flatten(1)
        return self.fc(x)


def save_tensor_bin(name, tensor):
    path = os.path.join(FIXTURES_DIR, f"{name}.bin")
    arr = tensor.detach().cpu().contiguous().float().numpy()
    with open(path, "wb") as f:
        f.write(arr.tobytes())
    print(f"  Saved {path} (shape={tuple(arr.shape)})")


def export_dynamic(name, model, example_input, dynamic_dim_spec, probe_inputs):
    """Export `model` with a dynamic batch dim, write reference IO at
    each of the `probe_inputs` batch sizes."""
    path = os.path.join(FIXTURES_DIR, f"{name}.pt2")
    model.eval()
    with torch.no_grad():
        exported = torch.export.export(
            model, (example_input,), dynamic_shapes=dynamic_dim_spec
        )
    torch.export.save(exported, path)
    print(f"Saved {path}")

    # Save the example input at its native batch size as the baseline,
    # plus outputs at each probe batch size.
    save_tensor_bin(f"{name}_input", example_input)
    with torch.no_grad():
        for bs, probe in probe_inputs.items():
            save_tensor_bin(f"{name}_input_bs{bs}", probe)
            save_tensor_bin(f"{name}_output_bs{bs}", model(probe))


def main():
    torch.manual_seed(42)

    batch = Dim("batch", min=1, max=64)

    # MLP: input (B, 10)
    export_dynamic(
        "mlp_dynamic_batch",
        DynMLP(),
        torch.randn(2, 10),
        {"x": {0: batch}},
        {1: torch.randn(1, 10), 4: torch.randn(4, 10), 8: torch.randn(8, 10)},
    )

    # ConvNet: input (B, 3, 8, 8)
    export_dynamic(
        "convnet_dynamic_batch",
        DynConvNet(),
        torch.randn(2, 3, 8, 8),
        {"x": {0: batch}},
        {
            1: torch.randn(1, 3, 8, 8),
            4: torch.randn(4, 3, 8, 8),
            8: torch.randn(8, 3, 8, 8),
        },
    )

    # ResNet18: a real architecture. Input (B, 3, 224, 224).
    try:
        import torchvision.models as tvm

        resnet = tvm.resnet18(weights=None).eval()
        export_dynamic(
            "resnet18_dynamic_batch",
            resnet,
            torch.randn(2, 3, 224, 224),
            {"x": {0: batch}},
            {
                1: torch.randn(1, 3, 224, 224),
                4: torch.randn(4, 3, 224, 224),
            },
        )
    except ImportError:
        print("torchvision not installed, skipping resnet18_dynamic_batch")

    print("\nDynamic-shape fixtures generated.")


if __name__ == "__main__":
    main()
