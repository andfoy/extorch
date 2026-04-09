"""Generate AOTI .pt2 test models for ExTorch."""

import torch
import torch.nn as nn
from torch._inductor import aoti_compile_and_package
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


if __name__ == "__main__":
    torch.manual_seed(42)

    model = SimpleMLP()
    model.eval()
    example = torch.randn(1, 10)

    exported = torch.export.export(model, (example,))
    path = os.path.join(FIXTURES_DIR, "simple_mlp.pt2")
    aoti_compile_and_package(exported, package_path=path)
    print(f"Saved {path}")

    # Verify
    loaded = torch._inductor.aoti_load_package(path)
    output = loaded(example)
    print(f"  Output shape: {output.shape}")
    print("\nAll AOTI models generated successfully.")
