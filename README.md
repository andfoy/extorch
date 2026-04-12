# ExTorch

**PyTorch bindings for Elixir — Python-free.** Load and execute serialized PyTorch models (`.pt2` AOTInductor, `.pt2` torch.export) directly from the BEAM. No Python process, no ports, no subprocess pool, no RPC queue, no ONNX conversion, no model rewrite. Just a library you call.

Bit-for-bit identical outputs to Python (verified across 11 tested models). Inference performance comparable to Python on the equivalent path — see [Benchmarks](#benchmarks) for details.

## What ExTorch is

A library you embed in your BEAM application to run PyTorch models you've already serialized elsewhere. It gives you the *execution* primitive (`load` + `forward`) and a customization surface built on normal Elixir/OTP tools — not a packaged serving product.

## What ExTorch is not

- Not a serving platform. There is no built-in dynamic batching, request coalescing, multi-model scheduler, or HTTP layer. Bring Phoenix, Bandit, Broadway, or roll your own around `forward/2`.
- Not a replacement for Triton or TorchServe. If you want a closed-box high-throughput inference server with batching as a config knob, those are the right tools.
- Not faster than PyTorch for apples-to-apples compiled paths. We match Python AOTI (1.03x); we don't beat it.

## Why use it

**Run PyTorch models from Elixir, directly.** If you're already on BEAM and want to add inference, you no longer need a second runtime, an RPC hop to Python, or an ONNX conversion step. Load the `.pt2` your ML team produced and call it.

**Customization surface.** Because it's a library, you wrap `forward/2` in whatever your app already does — Phoenix controllers, Broadway pipelines, GenServers with custom supervision, telemetry handlers, feature flags, fallback to a second model on error, ensembling, preprocessing in Nx. This is the flipside of not being Triton: batching is your problem, but *everything around the model* is trivially customizable.

**Distribution is whatever you already use.** `ExTorch.Export.Server` and `ExTorch.AOTI.Server` are plain GenServers. If you want multiple replicas, start several under a Supervisor. Multi-node replicas? `Node.connect` + `:global`, `Horde`, or `Swarm`. Pooling? `poolboy`, `NimblePool`, or a `DynamicSupervisor`. Rolling model updates with drain? Custom `terminate/2`. There is no ExTorch-specific clustering, routing, or pool primitive — the integration point is a process, so the BEAM tools you already use work as-is.

**Two supported formats**, both from modern PyTorch export paths:

| Format | Loaded via | Notes |
|---|---|---|
| AOTInductor `.pt2` | `ExTorch.AOTI.load/2` | Fused compiled kernels; matches Python AOTI |
| torch.export `.pt2` | `ExTorch.Export.load/2` | Four execution paths — interpreter, native, compiled graph, or extract as an Elixir DSL |

**Zero-copy with Nx.** Share tensor memory with Nx/Torchx via raw pointer exchange. Preprocessing in Nx composes with inference in ExTorch without copying.

**Extensible op set.** The `c10::Dispatcher` bridge lets Elixir packages register new ops without C++ code. [ExTorch.Vision](https://github.com/andfoy/extorch_vision) adds torchvision ops (NMS, ROI Align, deformable conv, image I/O) this way.

## Features

- **torch.export Inference** -- Load `.pt2` files from `torch.export.save` and run inference through a compiled C++ graph executor (89+ ATen ops). Tested with AlexNet, ResNet, VGG, MobileNet, ViT, EfficientNet, DeepLab, DistilBERT, Whisper, LSTM, and more.
- **AOTI Compiled Models** -- Load AOTInductor `.pt2` packages for optimized inference with fused kernels.
- **Generic c10 Dispatcher** -- Call any PyTorch op by name through `dispatch_op/3`. Load external op libraries (torchvision, torchaudio) via `load_torch_library/1`.
- **Op Extension System** -- `ExTorch.Export.OpHandler` behaviour + `OpRegistry` for registering custom ops from external packages.
- **Neural Network DSL** -- Define PyTorch-compatible layers in Elixir with `deflayer`, backed by libtorch's C++ nn modules (35 layer types).
- **Zero-Copy Tensor Exchange** -- Share tensor memory with Nx/Torchx via `data_ptr`/`from_blob`.
- **Telemetry hooks** -- `:telemetry` events for load/inference, optional ETS-backed counters, optional LiveDashboard page.
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
    {:extorch, "~> 0.4.0"}
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

### Export in Python, load from Elixir

```python
# Python: export your model
import torch
import torchvision

model = torchvision.models.resnet50(pretrained=True).eval()
exported = torch.export.export(model, (torch.randn(1, 3, 224, 224),))
torch.export.save(exported, "resnet50.pt2")
```

```elixir
# Elixir: load and call
model = ExTorch.Export.load("resnet50.pt2", device: :cuda)
input = ExTorch.Tensor.to(ExTorch.randn({1, 3, 224, 224}), device: :cuda)

# Fastest path — pre-compiled graph, zero per-op overhead
output = ExTorch.Export.forward_compiled(model, [input])

# Or use AOTI for maximum throughput (requires pre-compilation in Python)
aoti_model = ExTorch.AOTI.load("resnet50_aoti.pt2", device_index: 0)
[output] = ExTorch.AOTI.forward(aoti_model, [input])
```

### Dynamic batch size (and other dynamic dims)

Export with a symbolic dim and run at any batch size that fits the
constraint — no re-export needed:

```python
# Python: export once with dynamic batch
from torch.export import Dim
batch = Dim("batch", min=1, max=64)
exported = torch.export.export(
    model,
    (torch.randn(2, 3, 224, 224),),     # example input
    dynamic_shapes={"x": {0: batch}},    # 0th dim is dynamic
)
torch.export.save(exported, "resnet.pt2")
```

```elixir
# Elixir: load once, call at any batch
model = ExTorch.Export.load("resnet.pt2")

ExTorch.Export.forward(model, [ExTorch.randn({1, 3, 224, 224})])   # bs=1
ExTorch.Export.forward(model, [ExTorch.randn({4, 3, 224, 224})])   # bs=4
ExTorch.Export.forward(model, [ExTorch.randn({16, 3, 224, 224})])  # bs=16
```

Works across all three Export inference paths (`forward/2`,
`forward_native/2`, `forward_compiled/2`) and with any dimension that
`torch.export`'s tracer can express symbolically — batch size,
variable H/W on convolutional models, variable sequence length on
transformer classifiers. See
[`test/export/dynamic_batch_test.exs`](test/export/dynamic_batch_test.exs)
for verified coverage (MLP, ConvNet, ResNet18 at multiple batch sizes).

**Limitation — data-dependent shapes.** Ops whose output shape depends
on input *values* (e.g. `nonzero`, `torchvision::nms`) are not yet
supported when the graph performs downstream arithmetic on the
variable-length output. This affects detection models like Mask R-CNN.
Non-data-dependent dynamic dims (batch, H/W) work today.

### Customize around the model

Because there's no built-in server, you compose inference with whatever your app already uses. A minimal supervised model behind a mailbox:

```elixir
{:ok, _} = ExTorch.Export.Server.start_link(
  path: "resnet50.pt2",
  device: :cuda,
  name: :resnet
)

{:ok, output} = ExTorch.Export.Server.predict(:resnet, [input])
```

…but the point is you don't have to use that wrapper. A Phoenix controller that does preprocessing in Nx, zero-copies to ExTorch, runs inference, and falls back to a smaller model on OOM is all ordinary Elixir code. See [examples/serving/](examples/serving/).

### Optional telemetry

```elixir
ExTorch.Metrics.setup()
ExTorch.Metrics.get("resnet50.pt2")
# => %{inference_count: 1500, min_duration_ms: 4.9, max_duration_ms: 12.1, ...}
```

### Hot model reload

```elixir
# Swap models without dropping in-flight requests
# See examples/serving/hot_reload.exs for the full pattern
GenServer.cast(:resnet, {:reload, "resnet50_v2.pt2"})
```

### Extending with custom ops

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

### Zero-copy tensor exchange with Nx

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

## Benchmarks

Four inference paths on `torch.export` models, all producing bit-for-bit identical outputs to Python. RTX 3060, ViT-B/16, median of 30 runs:

| Path | When to use | Latency |
|---|---|---:|
| `ExTorch.Export.forward/2` | Op-by-op interpreter, useful for debugging | 54.9ms |
| `ExTorch.Export.forward_native/2` | Native graph execution, one NIF call | 11.9ms |
| `ExTorch.Export.forward_compiled/2` | Pre-compiled graph, no per-op overhead | 9.5ms |
| `ExTorch.AOTI.forward/2` | AOTInductor kernels | 8.8ms |

Against Python on the same hardware:

- **AOTI**: 1.03x of Python AOTI — effectively tied. Both sides are compiled shared objects, so neither has interpreter overhead in the hot path.
- **Compiled Export**: 1.37x of Python's `torch.export.export().module()` (geomean across 12 models). The gap is Python's FX interpreter walking the graph per-call; ExTorch pre-resolves the graph to c10 dispatcher handles at load time.

The 1.37x number is the symptom, not the point. The architectural fact is that there is no Python interpreter in ExTorch's hot path — on paths where Python also skips its interpreter (AOTI), we match.

Full results for 12 models in [examples/models](examples/models).

## Examples

See [examples/serving/](examples/serving/) for integration patterns:

- **basic_inference.exs** -- Three inference paths side-by-side with benchmarks
- **genserver_pool.exs** -- Model pool with concurrent inference and p50/p95/p99 reporting
- **hot_reload.exs** -- Swapping models without dropping in-flight requests
- **telemetry_dashboard.exs** -- Metrics via LiveDashboard

See [examples/models/](examples/models/) for end-to-end model usage:

- 8 models: CLIP, DistilBERT, MobileNetV3, EfficientNet, ResNet50, ViT-B/16, DeepLabV3, Whisper
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
