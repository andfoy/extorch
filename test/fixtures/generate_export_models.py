"""Generate torch.export.save .pt2 test models for ExTorch.Export.

Also writes reference (input, output) pairs as raw float32 bytes alongside
each archive so the Elixir test suite can verify numerical correctness against
PyTorch, not just shapes.
"""

import torch
import torch.nn as nn
import os

FIXTURES_DIR = os.path.dirname(os.path.abspath(__file__))


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, 3)
        self.bn = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 3)

    def forward(self, x):
        x = self.pool(self.relu(self.bn(self.conv(x))))
        x = x.flatten(1)
        return self.fc(x)


def save_tensor_bin(name, tensor):
    """Save a tensor as raw contiguous float32 bytes (native endian)."""
    path = os.path.join(FIXTURES_DIR, f"{name}.bin")
    arr = tensor.detach().cpu().contiguous().float().numpy()
    with open(path, "wb") as f:
        f.write(arr.tobytes())
    print(f"Saved {path} (shape={tuple(arr.shape)})")


def export_model(name, model, example_input):
    """Export a model via torch.export.save and write reference IO."""
    path = os.path.join(FIXTURES_DIR, f"{name}.pt2")
    model.eval()
    with torch.no_grad():
        exported = torch.export.export(model, (example_input,))
        reference_output = model(example_input)
    torch.export.save(exported, path)
    print(f"Saved {path}")

    save_tensor_bin(f"{name}_input", example_input)
    save_tensor_bin(f"{name}_output", reference_output)


if __name__ == "__main__":
    torch.manual_seed(42)

    export_model("simple_mlp_exported", SimpleMLP(), torch.randn(1, 10))
    export_model("convnet_exported", ConvNet(), torch.randn(1, 1, 8, 8))

    print("\nAll torch.export models generated successfully.")
