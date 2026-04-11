# ExTorch

Elixir bindings for [libtorch](https://pytorch.org/cppdocs/) -- production ML model serving on the BEAM.

Train in Python, serve from Elixir. ExTorch runs PyTorch models with OTP fault tolerance, beating Python's own inference performance by 1.35x on average.

## Why ExTorch?

**Faster than Python.** The pre-compiled graph executor beats Python's FX interpreter on every tested model -- 1.35x faster on average, bit-for-bit identical outputs.

| Model | Python Export | ExTorch Compiled | Speedup |
|---|---:|---:|---:|
| ResNet50 | 7.21ms | 4.96ms | **1.45x** |
| MobileNetV2 | 6.56ms | 4.07ms | **1.61x** |
| ViT-B/16 | 9.53ms | 9.46ms | **1.01x** |
| SqueezeNet | 2.77ms | 1.98ms | **1.40x** |
| DistilBERT | 0.78ms | 0.59ms | **1.32x** |

*RTX 3060, median latency, 30 iterations. Full results for 12 models in [examples/models](examples/models).*

**Four inference paths** for every use case:

| Path | Use case | ViT-B/16 latency |
|---|---|---:|
| `forward/2` | Debug, profile, op-by-op introspection | 54.9ms |
| `forward_native/2` | Production, single NIF call | 11.9ms |
| `forward_compiled/2` | Pre-compiled, fastest Export path | 9.5ms |
| `ExTorch.AOTI` | Compiled kernels, maximum throughput | 8.8ms |

**Production-ready serving.** GenServer model pools, telemetry events, ETS-backed metrics, zero-downtime hot model reload -- not bolted on, designed in.

**Extensible op ecosystem.** The generic `c10::Dispatcher` bridge lets pure-Elixir packages register new ops without C++ code. [ExTorch.Vision](https://github.com/andfoy/extorch_vision) adds torchvision ops (NMS, ROI Align, deformable conv, image I/O) this way.

**Zero-copy with Nx.** Share tensor memory between ExTorch and Nx/Torchx via raw pointer exchange -- no data copying.

**Bit-for-bit accurate.** All inference paths produce identical outputs to Python (verified across 11 models, 3 paths each, max absolute error = 0.0).

## Features

- **torch.export Inference** -- Load `.pt2` files from `torch.export.save` and run inference through a compiled C++ graph executor (89+ ATen ops). Tested with AlexNet, ResNet, VGG, MobileNet, ViT, EfficientNet, DeepLab, DistilBERT, Whisper, LSTM, and more.
- **AOTI Compiled Models** -- Load AOTInductor `.pt2` packages for optimized inference with fused kernels.
- **JIT Model Serving** -- Load `.pt` TorchScript models with full IValue support (tensors, tuples, dicts, scalars).
- **Generic c10 Dispatcher** -- Call any PyTorch op by name through `dispatch_op/3`. Load external op libraries (torchvision, torchaudio) via `load_torch_library/1`.
- **Op Extension System** -- `ExTorch.Export.OpHandler` behaviour + `OpRegistry` for registering custom ops from external packages.
- **Neural Network DSL** -- Define PyTorch-compatible layers in Elixir with `deflayer`, backed by libtorch's C++ nn modules (35 layer types).
- **Zero-Copy Tensor Exchange** -- Share tensor memory with Nx/Torchx via `data_ptr`/`from_blob`.
- **Telemetry & Observability** -- `:telemetry` events for load/inference, ETS-backed metrics, optional LiveDashboard page.
- **Tensor Operations** -- 200+ wrapped libtorch ops for creation, manipulation, math, comparison, reduction, and indexing.

## Requirements

- Elixir >= 1.16
- Rust (stable toolchain)
- libtorch (automatically downloaded, or use a local PyTorch installation)
- CMake (for ExTorch.Vision)
- CUDA toolkit (optional, for GPU support)

## Installation

Add `extorch` to your dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:extorch, "~> 0.3.0"}
  ]
end
```

ExTorch downloads libtorch automatically on first compile. To use a local installation:

```elixir
config :extorch, libtorch: [
  version: :local,
  folder: :python  # or an absolute path to libtorch
]
```

## Quick Start

### Train in Python, Serve from Elixir

```python
# Python: export your model
import torch
model = torchvision.models.resnet50(pretrained=True).eval()
exported = torch.export.export(model, (torch.randn(1, 3, 224, 224),))
torch.export.save(exported, "resnet50.pt2")
```

```elixir
# Elixir: load and serve
model = ExTorch.Export.load("resnet50.pt2", device: :cuda)
input = ExTorch.Tensor.to(ExTorch.randn({1, 3, 224, 224}), device: :cuda)

# Fastest path — pre-compiled graph, zero per-op overhead
output = ExTorch.Export.forward_compiled(model, [input])

# Or use AOTI for maximum throughput (requires pre-compilation in Python)
aoti_model = ExTorch.AOTI.load("resnet50_aoti.pt2", device_index: 0)
[output] = ExTorch.AOTI.forward(aoti_model, [input])
```

### Production Serving with GenServer

```elixir
# Supervised model server with telemetry
{:ok, _} = ExTorch.Export.Server.start_link(
  path: "resnet50.pt2",
  device: :cuda,
  name: :resnet
)

# Thread-safe inference
{:ok, output} = ExTorch.Export.Server.predict(:resnet, [input])

# Monitor performance
ExTorch.Metrics.setup()
ExTorch.Metrics.get("resnet50.pt2")
# => %{inference_count: 1500, min_duration_ms: 4.9, max_duration_ms: 12.1, ...}
```

### Hot Model Reload

```elixir
# Swap models without dropping requests
# See examples/serving/hot_reload.exs for the full pattern
GenServer.cast(:resnet, {:reload, "resnet50_v2.pt2"})
# In-flight requests complete on old model, new requests use new model
```

### Extending with Custom Ops

```elixir
# Load torchvision ops (NMS, ROI Align, etc.)
ExTorch.Native.load_torch_library("/path/to/libtorchvision.so")

# Call any registered op by name
keep = ExTorch.Native.dispatch_op("torchvision::nms", "", [
  {:tensor, boxes}, {:tensor, scores}, {:float, 0.5}
])

# Or use ExTorch.Vision for a clean API
ExTorch.Vision.nms(boxes, scores, 0.5)
ExTorch.Vision.roi_align(features, rois, 1.0, 7, 7)
```

### Zero-Copy Tensor Exchange with Nx

```elixir
# ExTorch → Nx (via Torchx): share memory, no copy
blob = ExTorch.Tensor.Blob.to_blob(tensor)
# => %Blob{ptr: 140234567890, shape: {3, 224, 224}, dtype: :float, ...}

# Nx → ExTorch: wrap foreign memory
view = ExTorch.Tensor.Blob.from_blob(
  %{ptr: torchx_ptr, shape: {3, 224, 224}, dtype: :float32},
  owner: nx_tensor
)
```

### CUDA Support

```elixir
ExTorch.Native.cuda_is_available()    # => true
ExTorch.Native.cuda_device_count()    # => 2

model = ExTorch.Export.load("model.pt2", device: :cuda)
ExTorch.Native.cuda_memory_allocated(0)  # bytes on GPU 0
```

## Deployment Examples

See [examples/serving/](examples/serving/) for production patterns:

- **basic_inference.exs** -- Three inference paths side-by-side with benchmarks
- **genserver_pool.exs** -- Supervised model pool with concurrent inference and p50/p95/p99 reporting
- **hot_reload.exs** -- Zero-downtime model swapping
- **telemetry_dashboard.exs** -- Live metrics and monitoring

See [examples/models/](examples/models/) for real-world model deployment:

- 8 production models: CLIP, DistilBERT, MobileNetV3, EfficientNet, ResNet50, ViT-B/16, DeepLabV3, Whisper
- Export script, multi-model benchmark, full image classification pipeline

## Architecture

Three-layer design: C++ (libtorch wrapper) → Rust (cxx bridge + Rustler NIFs) → Elixir (macro-generated API).

- **C++ sources**: `native/extorch/src/csrc/*.cc` + `native/extorch/include/*.h`
- **Rust bridge**: `native/extorch/src/native/*.rs.in` (Tera templates rendered by `build.rs`)
- **Rust NIFs**: `native/extorch/src/nifs/*.rs`
- **Elixir API**: `lib/extorch/`

The generic `c10::Dispatcher` NIF bridge (`dispatch_op`, `execute_graph`, `compile_graph`) enables calling any PyTorch op without per-op C++ wrappers, and the `OpHandler` behaviour allows external packages to extend the Export interpreter.

## License

MIT
