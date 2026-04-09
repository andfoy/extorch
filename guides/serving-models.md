# Serving Models

ExTorch provides `ExTorch.JIT.Server`, a GenServer that wraps a loaded model with OTP fault tolerance, serialized inference, and telemetry instrumentation.

## Basic serving

```elixir
# Start a model server
{:ok, pid} = ExTorch.JIT.Server.start_link(path: "model.pt")

# Run inference
input = ExTorch.randn({1, 10})
output = ExTorch.JIT.Server.predict(pid, [input])
```

The server loads the model on `init`, sets it to eval mode, and serializes all `predict` calls through the GenServer -- ensuring thread safety for models with mutable state (BatchNorm, Dropout).

## Named servers

```elixir
{:ok, _} = ExTorch.JIT.Server.start_link(
  path: "sentiment.pt",
  device: :cpu,
  name: SentimentModel
)

# Use from anywhere in the application
ExTorch.JIT.Server.predict(SentimentModel, [input])
```

## Supervision

Add model servers to your application's supervision tree:

```elixir
# application.ex
def start(_type, _args) do
  children = [
    # ExTorch starts its own DynamicSupervisor automatically
    {ExTorch.JIT.Server, path: "model_a.pt", name: ModelA},
    {ExTorch.JIT.Server, path: "model_b.pt", name: ModelB, device: {:cuda, 0}},
  ]

  Supervisor.start_link(children, strategy: :one_for_one)
end
```

If a model server crashes (e.g., due to a malformed input), the supervisor restarts it automatically, reloading the model from disk.

## Dynamic model loading

Use the built-in `ExTorch.ModelSupervisor` to start servers at runtime:

```elixir
DynamicSupervisor.start_child(ExTorch.ModelSupervisor, {
  ExTorch.JIT.Server,
  path: "new_model.pt", name: NewModel
})
```

## Server info

```elixir
ExTorch.JIT.Server.info(SentimentModel)
# => %{
#   path: "sentiment.pt",
#   device: :cpu,
#   inference_count: 1523,
#   error_count: 0,
#   uptime_ms: 3600000
# }
```

## Models with complex outputs

Models that return tuples, dicts, or nested structures work naturally:

```elixir
# Python: def forward(self, x): return {"logits": ..., "features": ...}
output = ExTorch.JIT.Server.predict(MyModel, [input])
# output is an Elixir map: %{"logits" => %Tensor{}, "features" => %Tensor{}}

# Python: def forward(self, x): return self.head1(x), self.head2(x)
{head1, head2} = ExTorch.JIT.Server.predict(MultiHead, [input])
```

## Using models from torch.export

PyTorch 2.x encourages `torch.export` over the deprecated `torch.jit.script`
for new models. ExTorch supports three paths from `torch.export`, depending on
your needs.

### Path 1: Load and run exported .pt2 directly (recommended)

`torch.export.save` produces a `.pt2` archive containing the model graph as
JSON and weight tensors as raw binaries. ExTorch reads these archives in pure
Elixir and interprets the ATen computation graph directly -- no Python, no JIT,
no C++ ExportedProgram support needed:

```python
# Python: export and save
exported = torch.export.export(model, (example_input,))
torch.export.save(exported, "model.pt2")
```

```elixir
# Elixir: load and run inference directly
model = ExTorch.Export.load("model.pt2")
output = ExTorch.Export.forward(model, [input])
```

The interpreter supports 60+ ATen operations and has been tested with AlexNet,
ResNet18, MobileNetV2, VGG11, SqueezeNet, transformers, and autoencoders.

You can also introspect the model, extract weights, or generate DSL code:

```elixir
# Read architecture
schema = ExTorch.Export.read_schema("model.pt2")

# Load weights into a DSL module
model = MyModel.load_weights_from_export("model.pt2")
output = MyModel.forward(model, input)

# Generate DSL source from the graph
IO.puts(ExTorch.Export.to_elixir("model.pt2", "MyModel"))
```

See `ExTorch.Export` for the full API.

### Path 2: AOTI compiled models (best throughput)

AOTInductor compiles the exported model into an optimized `.pt2` package with
fused kernels. ExTorch loads these via `AOTIModelPackageLoader` in libtorch:

```python
# Python: compile and package
from torch._inductor import aoti_compile_and_package
exported = torch.export.export(model, (example_input,))
aoti_compile_and_package(exported, package_path="model_compiled.pt2")
```

```elixir
# Elixir: load and run
model = ExTorch.AOTI.load("model_compiled.pt2")
[output] = ExTorch.AOTI.forward(model, [input])
```

AOTI models trade flexibility for throughput -- no introspection or weight
extraction, but benefit from kernel fusion optimizations. Use
`ExTorch.AOTI.Server` for production serving with telemetry.

Check availability: `ExTorch.AOTI.available?()`.

### Path 3: Convert to TorchScript (legacy)

For full JIT introspection, DSL generation, and `from_jit`/`load_weights`
support, you can convert an ExportedProgram to TorchScript:

```python
exported = torch.export.export(model, (example_input,))
jit_model = torch.jit.trace(exported.module(), (example_input,))
torch.jit.save(jit_model, "model.pt")
```

Note that `torch.jit` is in maintenance mode. Prefer Path 1 or 2 for new work.

## CUDA serving

```elixir
# Load on GPU
{:ok, _} = ExTorch.JIT.Server.start_link(
  path: "model.pt",
  device: {:cuda, 0},
  name: GPUModel
)

# Input tensors can be on CPU -- libtorch handles the transfer
# For best performance, pre-transfer inputs:
gpu_input = ExTorch.Tensor.to(input, device: {:cuda, 0})
output = ExTorch.JIT.Server.predict(GPUModel, [gpu_input])
```

## Telemetry events

Every `predict` call emits telemetry events that you can attach to:

```elixir
:telemetry.attach("my-logger", [:extorch, :jit, :forward, :stop], fn _event, measurements, metadata, _config ->
  duration_ms = System.convert_time_unit(measurements.duration, :native, :microsecond) / 1000
  IO.puts("#{metadata.path}: #{duration_ms}ms (#{metadata.input_count} inputs)")
end, nil)
```

See the [Observability guide](observability.md) for the full metrics and dashboard setup.
