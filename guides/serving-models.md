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
