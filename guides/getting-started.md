# Getting Started

ExTorch is an Elixir library for production ML model serving, built on [libtorch](https://pytorch.org/cppdocs/) (the C++ backend of PyTorch). It lets you load TorchScript models, run inference with OTP fault tolerance, and monitor serving performance -- all from Elixir.

## Installation

Add `extorch` to your dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:extorch, "~> 0.2.0"}
  ]
end
```

Then fetch and compile:

```bash
mix deps.get
mix compile
```

On first compile, ExTorch downloads libtorch automatically. This takes a few minutes. If you already have PyTorch installed via Python, you can skip the download:

```elixir
# config/config.exs
config :extorch, libtorch: [
  version: :local,
  folder: :python
]
```

### Requirements

- Elixir >= 1.14
- Rust stable toolchain (`rustup` recommended)
- ~2 GB disk for libtorch

## Your first inference

The typical workflow is: train a model in Python, export it as TorchScript, then serve it from Elixir.

### Step 1: Export a model in Python

```python
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 3)

    def forward(self, x):
        return self.fc(x)

model = SimpleNet()
model.eval()
scripted = torch.jit.script(model)
scripted.save("simple_net.pt")
```

### Step 2: Load and run in Elixir

```elixir
# Load the model
model = ExTorch.JIT.load("simple_net.pt")
ExTorch.JIT.eval(model)

# Create an input tensor
input = ExTorch.randn({1, 10})

# Run inference
output = ExTorch.JIT.forward(model, [input])
# => %ExTorch.Tensor{size: {1, 3}, dtype: :float, ...}
```

That's it. The model runs entirely in the BEAM process via libtorch -- no Python, no HTTP, no serialization overhead.

## Inspecting a model

You can examine the structure of any loaded model:

```elixir
model = ExTorch.JIT.load("simple_net.pt")

# List methods
ExTorch.JIT.methods(model)
# => ["forward"]

# List submodules
ExTorch.JIT.modules(model)
# => ["fc"]

# Get parameters with their shapes
ExTorch.JIT.parameters(model)
# => [{"fc.weight", #Tensor<[3, 10]>}, {"fc.bias", #Tensor<[3]>}]

# View the computation graph
IO.puts(ExTorch.NN.Introspect.graph(model))
```

## Working with tensors

ExTorch provides 200+ tensor operations that mirror PyTorch's API:

```elixir
# Creation
a = ExTorch.ones({3, 3})
b = ExTorch.randn({3, 3})
c = ExTorch.tensor([[1.0, 2.0], [3.0, 4.0]])

# Math
d = ExTorch.add(a, b)
e = ExTorch.mul(c, 2.0)

# Inspection
ExTorch.Native.size(a)    # => {3, 3}
ExTorch.Native.dtype(a)   # => :float
```

## Next steps

- **[Serving Models](serving-models.md)** -- GenServer-based serving with telemetry and fault tolerance
- **[Neural Network DSL](neural-network-dsl.md)** -- Define architectures in Elixir and load pre-trained weights
- **[Observability](observability.md)** -- Metrics, monitoring, and LiveDashboard integration
- **[`ExTorch.JIT`](ExTorch.JIT.html)** -- Full JIT API reference
- **[`ExTorch.NN`](ExTorch.NN.html)** -- Full neural network API reference
