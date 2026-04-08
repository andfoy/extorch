# ExTorch

Elixir bindings for [libtorch](https://pytorch.org/cppdocs/) -- production ML model serving on the BEAM.

ExTorch lets you load TorchScript models, run inference with OTP fault tolerance, define neural network architectures with an Elixir DSL, and monitor serving performance through telemetry and LiveDashboard.

## Features

- **JIT Model Serving** -- Load `.pt` models, run inference with full IValue support (tensors, tuples, dicts, scalars), and serve behind GenServer with process isolation.
- **Neural Network DSL** -- Define PyTorch-compatible layers declaratively in Elixir with `deflayer`, backed by libtorch's C++ nn modules.
- **JIT IR Introspection** -- Extract model architecture, parameters, and computation graphs from any TorchScript model. Generate Elixir DSL source code from `.pt` files.
- **Zero-Copy Tensor Exchange** -- Share tensor memory with Nx/Torchx via raw pointer exchange (`data_ptr`/`from_blob`), no copies.
- **Telemetry & Observability** -- `:telemetry` events for load/inference, ETS-backed metrics (latency, throughput, errors), optional LiveDashboard page.
- **Tensor Operations** -- 200+ wrapped libtorch ops for tensor creation, manipulation, pointwise math, comparison, reduction, and indexing.

## Requirements

- Elixir >= 1.14
- Rust (stable toolchain)
- libtorch (automatically downloaded, or use a local PyTorch installation)

## Installation

Add `extorch` to your dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:extorch, "~> 0.1.0"}
  ]
end
```

ExTorch will download libtorch automatically on first compile. To use a local installation, configure in `config/config.exs`:

```elixir
config :extorch, libtorch: [
  version: :local,
  folder: :python  # or an absolute path to libtorch
]
```

## Quick Start

### Loading and Serving a TorchScript Model

```elixir
# Load a model exported from Python with torch.jit.script() or torch.jit.trace()
model = ExTorch.JIT.load("model.pt")
ExTorch.JIT.eval(model)

# Run inference
input = ExTorch.randn({1, 3, 224, 224})
output = ExTorch.JIT.forward(model, [input])

# Models returning tuples/dicts work naturally
{logits, features} = ExTorch.JIT.forward(multi_output_model, [input])
```

### GenServer-based Model Server

```elixir
# Start a supervised model server
{:ok, _pid} = ExTorch.JIT.Server.start_link(
  path: "model.pt",
  device: :cpu,
  name: MyModel
)

# Run inference (thread-safe, serialized through GenServer)
output = ExTorch.JIT.Server.predict(MyModel, [input])

# Check server stats
ExTorch.JIT.Server.info(MyModel)
# => %{path: "model.pt", device: :cpu, inference_count: 42, error_count: 0, uptime_ms: 15000}
```

### Neural Network DSL

```elixir
defmodule MyMLP do
  use ExTorch.NN.Module

  deflayer :fc1, ExTorch.NN.Linear, in_features: 784, out_features: 128
  deflayer :relu, ExTorch.NN.ReLU
  deflayer :dropout, ExTorch.NN.Dropout, p: 0.5
  deflayer :fc2, ExTorch.NN.Linear, in_features: 128, out_features: 10

  def forward(model, x) do
    x
    |> layer(model, :fc1)
    |> layer(model, :relu)
    |> layer(model, :dropout)
    |> layer(model, :fc2)
  end
end

model = MyMLP.new()
input = ExTorch.randn({32, 784})
output = MyMLP.forward(model, input)
# => %ExTorch.Tensor{size: {32, 10}, ...}

# Inspect parameters
MyMLP.parameters(model)
# => [{"fc1.weight", #Tensor<[128, 784]>}, {"fc1.bias", #Tensor<[128]>}, ...]
```

Available layers: `Linear`, `Conv1d`, `Conv2d`, `Conv3d`, `ConvTranspose1d`, `ConvTranspose2d`, `MaxPool1d`, `MaxPool2d`, `AvgPool1d`, `AvgPool2d`, `AdaptiveAvgPool1d`, `AdaptiveAvgPool2d`, `BatchNorm1d`, `BatchNorm2d`, `LayerNorm`, `GroupNorm`, `InstanceNorm1d`, `InstanceNorm2d`, `Dropout`, `Embedding`, `LSTM`, `GRU`, `MultiheadAttention`, `Flatten`, `Unflatten`, `ReLU`, `LeakyReLU`, `GELU`, `ELU`, `SiLU`, `Mish`, `PReLU`, `Sigmoid`, `Tanh`, `Softmax`, `LogSoftmax`.

### Loading Pre-trained Weights

There are two ways to use a trained model from Python with a DSL-defined module:

**Option A: `from_jit/1`** -- Use the JIT model's forward directly (simplest):

```elixir
# The JIT model's forward() runs the computation with pre-trained weights.
# The DSL definition is validated against the .pt file's submodules.
model = MyMLP.from_jit("trained_model.pt")
output = MyMLP.predict(model, [input])
```

**Option B: `load_weights/1`** -- Copy weights into Elixir-defined layers:

```elixir
# Creates DSL layers, then copies matching parameter tensors from the .pt file.
# The result is a regular DSL model that runs through your forward/2 function.
model = MyMLP.load_weights("trained_model.pt")
output = MyMLP.forward(model, input)
```

Both produce identical outputs. Use `from_jit` when you want the exact Python forward logic. Use `load_weights` when your Elixir `forward/2` differs (e.g., different dropout, custom post-processing) but you want the same trained parameters.

### JIT Model Introspection

```elixir
model = ExTorch.JIT.load("resnet18.pt")

# Extract structured schema
schema = ExTorch.NN.Introspect.schema(model)
schema.submodules
# => [%{name: "conv1", type_name: "...Conv2d", parameters: [%{name: "weight", shape: [64, 3, 7, 7], ...}]}, ...]

# View the computation graph
IO.puts(ExTorch.NN.Introspect.graph(model))

# Generate Elixir DSL source code from any .pt model
IO.puts(ExTorch.NN.Introspect.to_elixir(model, "ResNet18"))
```

### Zero-Copy Tensor Exchange with Nx

```elixir
# ExTorch tensor -> raw pointer (for passing to Torchx/Nx)
blob = ExTorch.Tensor.Blob.to_blob(tensor)
# => %Blob{ptr: 140234567890, shape: {3, 224, 224}, strides: [...], dtype: :float, ...}

# Foreign pointer -> ExTorch tensor (zero-copy, no data movement)
view = ExTorch.Tensor.Blob.from_blob(
  %{ptr: foreign_ptr, shape: {3, 224, 224}, dtype: :float32},
  owner: source_tensor  # prevents GC of source memory
)
view.tensor  # => %ExTorch.Tensor{...}
```

### Telemetry & Metrics

```elixir
# Enable metrics collection (call in your Application.start)
ExTorch.Metrics.setup()

# Metrics are automatically collected from JIT.Server telemetry events
ExTorch.Metrics.get("model.pt")
# => %{inference_count: 1500, error_count: 2, total_duration_ms: 4523.1,
#      min_duration_ms: 1.2, max_duration_ms: 89.2, load_duration_ms: 340.5, ...}

# Attach your own handlers to telemetry events
:telemetry.attach("my-handler", [:extorch, :jit, :forward, :stop], &handle_event/4, nil)
```

**LiveDashboard** (optional): Add `{:phoenix_live_dashboard, "~> 0.8"}` to your deps and configure:

```elixir
live_dashboard "/dashboard",
  additional_pages: [
    extorch: {ExTorch.Observer.Dashboard, []}
  ]
```

### CUDA Support

```elixir
ExTorch.Native.cuda_is_available()   # => true/false
ExTorch.Native.cuda_device_count()   # => 2

# Load model on GPU
model = ExTorch.JIT.load("model.pt", device: {:cuda, 0})

# Monitor GPU memory
ExTorch.Native.cuda_memory_allocated(0)   # bytes currently allocated
ExTorch.Native.cuda_memory_reserved(0)    # bytes reserved by caching allocator
```

## Architecture

ExTorch uses a three-layer architecture:

1. **C++** -- Wraps libtorch APIs (`torch::jit`, `torch::nn`, tensor ops) behind shared pointer types
2. **Rust** -- Bridges C++ to Erlang NIFs via [cxx](https://cxx.rs) and [Rustler](https://github.com/rusterlium/rustler), with type-safe encoding/decoding
3. **Elixir** -- Macro-generated API with `defbinding`/`nif_impl!` for tensor ops, hand-written modules for JIT/NN/telemetry

## License

MIT
