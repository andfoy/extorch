# Observability

ExTorch integrates with the Elixir ecosystem's standard observability tools: `:telemetry` for events, ETS for metrics, and Phoenix LiveDashboard for visualization.

## Setup

Call `ExTorch.Metrics.setup/0` early in your application start:

```elixir
# application.ex
def start(_type, _args) do
  ExTorch.Metrics.setup()

  children = [
    {ExTorch.JIT.Server, path: "model.pt", name: MyModel}
  ]

  Supervisor.start_link(children, strategy: :one_for_one)
end
```

This creates an ETS table and attaches telemetry handlers. From this point, all `ExTorch.JIT.Server` predictions are automatically tracked.

## Telemetry events

`ExTorch.JIT.Server` emits these events:

| Event | When | Measurements | Metadata |
|---|---|---|---|
| `[:extorch, :jit, :load, :start]` | Model loading begins | `system_time` | `path`, `device` |
| `[:extorch, :jit, :load, :stop]` | Model loading completes | `duration` (native) | `path`, `device` |
| `[:extorch, :jit, :forward, :start]` | Inference begins | `system_time` | `path`, `device`, `input_count` |
| `[:extorch, :jit, :forward, :stop]` | Inference completes | `duration` (native) | `path`, `device`, `input_count` |
| `[:extorch, :jit, :forward, :exception]` | Inference fails | `system_time` | `path`, `device`, `input_count`, `kind`, `reason` |

### Attaching custom handlers

```elixir
# Log slow inferences
:telemetry.attach("slow-inference", [:extorch, :jit, :forward, :stop],
  fn _event, %{duration: duration}, metadata, _config ->
    ms = System.convert_time_unit(duration, :native, :microsecond) / 1000
    if ms > 100 do
      Logger.warning("Slow inference on #{metadata.path}: #{ms}ms")
    end
  end, nil)
```

## Querying metrics

```elixir
# Per-model metrics
ExTorch.Metrics.get("model.pt")
# => %{
#   inference_count: 1523,
#   error_count: 2,
#   total_duration_ms: 4523.1,
#   min_duration_ms: 1.2,
#   max_duration_ms: 89.2,
#   load_duration_ms: 340.5,
#   device: :cpu,
#   last_inference_at: 1712345678000
# }

# All tracked models
ExTorch.Metrics.all()
# => [{"model_a.pt", %{...}}, {"model_b.pt", %{...}}]

# Reset
ExTorch.Metrics.reset("model.pt")
ExTorch.Metrics.reset_all()
```

### Computing derived metrics

```elixir
metrics = ExTorch.Metrics.get("model.pt")

# Average latency
avg_ms = metrics.total_duration_ms / max(metrics.inference_count, 1)

# Throughput (inferences per second)
uptime_s = ExTorch.JIT.Server.info(MyModel).uptime_ms / 1000
throughput = metrics.inference_count / max(uptime_s, 1)

# Error rate
error_rate = metrics.error_count / max(metrics.inference_count, 1)
```

## CUDA memory monitoring

On GPU builds, ExTorch exposes memory allocator stats:

```elixir
ExTorch.Native.cuda_is_available()         # => true
ExTorch.Native.cuda_device_count()         # => 2
ExTorch.Native.cuda_memory_allocated(0)    # => 1073741824 (bytes)
ExTorch.Native.cuda_memory_reserved(0)     # => 2147483648 (bytes)
ExTorch.Native.cuda_max_memory_allocated(0) # => 1610612736 (bytes, peak)
```

On CPU-only builds, the memory functions return `-1`.

## Phoenix LiveDashboard

Add `phoenix_live_dashboard` to your deps (it's optional):

```elixir
{:phoenix_live_dashboard, "~> 0.8"}
```

Then register the ExTorch page:

```elixir
# router.ex
live_dashboard "/dashboard",
  additional_pages: [
    extorch: ExTorch.Observer.Dashboard
  ]
```

The dashboard page shows a table of all loaded models with their inference count, error count, latency stats (avg/min/max), and load time.

## OTP Observer integration

`ExTorch.JIT.Server` implements `format_status/2`, so model servers are introspectable in the standard Erlang `:observer`:

1. Start observer: `:observer.start()`
2. Navigate to the Processes tab
3. Find your model server process
4. The State tab shows: path, device, inference count, error count, and uptime

This works out of the box with no additional configuration.
