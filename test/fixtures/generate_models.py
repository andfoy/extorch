"""Generate test TorchScript models for ExTorch JIT tests."""

import torch
import torch.nn as nn
import os

FIXTURES_DIR = os.path.dirname(os.path.abspath(__file__))


class SimpleMLP(nn.Module):
    """A simple MLP that returns a single tensor."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class TupleModel(nn.Module):
    """A model that returns a tuple of tensors."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(10, 3)

    def forward(self, x):
        return self.fc1(x), self.fc2(x)


class DictModel(nn.Module):
    """A model that returns a dict of tensors."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(10, 3)

    def forward(self, x):
        return {"logits": self.fc1(x), "features": self.fc2(x)}


class MultiInputModel(nn.Module):
    """A model that takes multiple input tensors."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(8, 5)

    def forward(self, x, y):
        return self.fc1(x) + self.fc2(y)


def save_scripted(model, name, example_inputs):
    """Script and save a model."""
    model.eval()
    scripted = torch.jit.script(model)
    path = os.path.join(FIXTURES_DIR, f"{name}.pt")
    scripted.save(path)
    print(f"Saved {path}")

    # Verify it loads and runs
    loaded = torch.jit.load(path)
    output = loaded(*example_inputs)
    print(f"  Output type: {type(output)}")
    if isinstance(output, torch.Tensor):
        print(f"  Output shape: {output.shape}")
    elif isinstance(output, tuple):
        print(f"  Output shapes: {[t.shape for t in output]}")
    elif isinstance(output, dict):
        print(f"  Output keys: {list(output.keys())}")
        print(f"  Output shapes: {[(k, v.shape) for k, v in output.items()]}")


if __name__ == "__main__":
    torch.manual_seed(42)

    # Simple MLP: single tensor input -> single tensor output
    mlp = SimpleMLP()
    save_scripted(mlp, "simple_mlp", [torch.randn(1, 10)])

    # Tuple model: single tensor input -> tuple of tensors
    tuple_model = TupleModel()
    save_scripted(tuple_model, "tuple_model", [torch.randn(1, 10)])

    # Dict model: single tensor input -> dict of tensors
    dict_model = DictModel()
    save_scripted(dict_model, "dict_model", [torch.randn(1, 10)])

    # Multi-input model: two tensor inputs -> single tensor output
    multi_model = MultiInputModel()
    save_scripted(multi_model, "multi_input_model", [torch.randn(1, 10), torch.randn(1, 8)])

    print("\nAll models generated successfully.")
