# ExTorch: Production ML Model Serving on the BEAM

[ExTorch](https://github.com/andfoy/extorch), the Elixir/Erlang bindings for
[libtorch](https://pytorch.org/cppdocs/), is getting a major update. In this
post, we'll take a deep dive into the new capabilities that make ExTorch a
production-ready framework for serving machine learning models from Elixir,
including direct `torch.export` inference, a neural network DSL, and
telemetry-based observability.

For those unfamiliar with the project, ExTorch bridges Elixir and PyTorch's C++
backend through a three-layer architecture: C++ wraps the libtorch API directly,
Rust provides the NIF bridge using [cxx](https://cxx.rs) and
[Rustler](https://github.com/rusterlium/rustler), and Elixir offers the
user-facing API with macros that generate type-safe bindings. The project
originally focused on tensor operations -- providing over 200 wrapped libtorch
ops for creation, manipulation, pointwise math, comparison, reduction, and
indexing. While this foundation remains, the v0.3 release shifts the focus
toward what the BEAM VM does uniquely well: reliable, concurrent, observable
service infrastructure.

The motivation behind this shift is straightforward. Machine learning models are
typically trained in Python using PyTorch, TensorFlow, or JAX, and there is no
compelling reason to change that -- the Python ecosystem for training is mature
and well-tooled. However, when it comes to deploying those models behind a
production backend, the story is different. Python's GIL limits true
concurrency, dedicated serving solutions like TorchServe or Triton add
operational complexity, and general-purpose languages like Go or Rust lack
native ML framework integration. The BEAM VM, originally designed for telecom
systems requiring millions of concurrent connections with fault tolerance and
hot code reloading, maps naturally to the demands of production ML serving:
process isolation prevents one bad input from taking down the system, OTP
supervision trees restart crashed model servers automatically, and the BEAM's
scheduler handles thousands of concurrent inference requests without manual
thread pool tuning.

## Serving Models from torch.export

PyTorch 2.x has introduced `torch.export` as the recommended path for model
deployment, replacing the now-deprecated `torch.jit.script` and
`torch.jit.trace`. ExTorch supports `torch.export` through two complementary
paths -- a pure Elixir graph interpreter for maximum flexibility, and an
AOTInductor loader for maximum throughput.

### Pure Elixir Inference

The recommended path for most use cases is `ExTorch.Export`, which reads `.pt2`
archives produced by `torch.export.save` and interprets the ATen computation
graph directly in Elixir. No C++ `ExportedProgram` support is needed, no Python
at runtime, and no dependency on the deprecated `torch.jit` API:

```python
# Python: train and export
model.eval()
exported = torch.export.export(model, (example_input,))
torch.export.save(exported, "model.pt2")
```

```elixir
# Elixir: load and serve
model = ExTorch.Export.load("model.pt2")
output = ExTorch.Export.forward(model, [input])
```

The `.pt2` archive contains the model's computation graph as JSON and its weight
tensors as raw binary files. ExTorch reads both in pure Elixir -- unzipping the
archive with the Erlang `:zip` module, parsing the graph with `Jason`, and
loading weights into libtorch tensors via a `from_binary` NIF that copies the
data directly from Erlang binaries.

The interpreter dispatches 60+ ATen operations (`aten.linear.default`,
`aten.conv2d.default`, `aten.relu.default`, `aten.batch_norm.default`, etc.)
to the corresponding ExTorch tensor functions and NN layer modules. We tested
it against seven popular architectures -- AlexNet, ResNet18, MobileNetV2,
VGG11, SqueezeNet, a multi-head attention transformer, and an autoencoder --
all producing correct output shapes and values matching their Python
counterparts.

### AOTInductor Compiled Models

For workloads where inference latency is critical, ExTorch also supports
AOTInductor-compiled `.pt2` packages via the `AOTIModelPackageLoader` C++ API
in libtorch:

```python
from torch._inductor import aoti_compile_and_package
exported = torch.export.export(model, (example_input,))
aoti_compile_and_package(exported, package_path="model_compiled.pt2")
```

```elixir
model = ExTorch.AOTI.load("model_compiled.pt2")
[output] = ExTorch.AOTI.forward(model, [input])
```

AOTI models benefit from kernel fusion and compiler optimizations that the
interpreter cannot perform, but they don't support introspection or weight
extraction. Both paths provide GenServer wrappers (`ExTorch.Export.Server`
and `ExTorch.AOTI.Server`) with telemetry instrumentation and OTP fault
tolerance.

### OTP Model Serving

The real value of ExTorch's serving infrastructure comes from wrapping models in
OTP processes. All three serving paths -- Export, AOTI, and JIT -- provide
GenServers that load a model on `init`, serialize inference calls for thread
safety, and emit telemetry events for every prediction:

```elixir
children = [
  {ExTorch.Export.Server, path: "sentiment.pt2", name: SentimentModel},
  {ExTorch.AOTI.Server, path: "fast_model.pt2", name: FastModel}
]

Supervisor.start_link(children, strategy: :one_for_one)
```

If a model server crashes due to a malformed input or an out-of-memory error,
the supervisor restarts it -- the model is reloaded from disk and the server is
back online in milliseconds. This is the kind of reliability that is difficult
to retrofit into Python serving solutions.

### TorchScript (Legacy)

ExTorch also supports loading TorchScript `.pt` files via `ExTorch.JIT.load`,
with full `IValue` marshalling for complex return types (tuples, dicts, nested
structures), model introspection (`parameters`, `modules`, `methods`), and
graph IR extraction. This path remains useful for existing `.pt` deployments
and for models that require JIT-specific features like `torch.jit.export` for
multiple entry points. However, since `torch.jit` is in maintenance mode, we
recommend `torch.export` for new work.

## Neural Network DSL

Beyond serving pre-trained models, ExTorch provides an Elixir DSL for defining
neural network architectures. This is not meant to replace PyTorch for training
-- rather, it gives Elixir developers a way to express model architectures in
their own language and load pre-trained weights into them.

```elixir
defmodule TextClassifier do
  use ExTorch.NN.Module

  deflayer :embedding, ExTorch.NN.Embedding,
    num_embeddings: 30000, embedding_dim: 128
  deflayer :dropout, ExTorch.NN.Dropout, p: 0.3
  deflayer :fc1, ExTorch.NN.Linear, in_features: 128, out_features: 64
  deflayer :relu, ExTorch.NN.ReLU
  deflayer :fc2, ExTorch.NN.Linear, in_features: 64, out_features: 3

  def forward(model, x) do
    x
    |> layer(model, :embedding)
    |> layer(model, :dropout)
    |> layer(model, :fc1)
    |> layer(model, :relu)
    |> layer(model, :fc2)
  end
end
```

The `deflayer` macro registers a named layer at compile time, while `layer/3`
is a runtime function that looks up the layer in the model map and applies it.
Under the hood, each layer wraps a `torch::nn::AnyModule` -- the same C++
module types used by PyTorch's C++ frontend -- which means that layer creation,
forward passes, and parameter access all go through libtorch natively.

The release includes 35 layer types covering convolutions (1D, 2D, 3D,
and transposed variants), pooling (max, average, and adaptive), normalization
(batch, layer, group, and instance), recurrent layers (LSTM and GRU),
multi-head attention, and 11 activation functions including ReLU, GELU, SiLU,
Mish, and PReLU. Each function's documentation follows PyTorch's format with
mathematical formulas, shape specifications, and examples.

### Loading Pre-trained Weights

The recommended path for loading weights into a DSL module is directly from
a `torch.export.save` archive -- no JIT dependency needed:

```elixir
model = TextClassifier.load_weights_from_export("trained.pt2")
output = TextClassifier.forward(model, input)
```

This creates the DSL layers, reads the weight tensors from the `.pt2` archive,
and copies matching parameters into each layer. Loading from TorchScript
(`.pt`) files via `load_weights` and `from_jit` is also supported for existing
deployments.

### Generating DSL from Existing Models

For models where no DSL definition exists yet, `ExTorch.Export.to_elixir/2`
generates the Elixir source code from a `.pt2` archive by parsing its ATen
graph. The code generator performs data flow analysis on the computation graph
to correctly handle branching architectures. For a ResNet, the generated code
includes explicit variable bindings for skip connections:

```elixir
# Generated from ResNet18 -- residual connections are automatic
conv2d_1 = max_pool2d |> layer(model, :layer1_0_conv1)
batch_norm_1 = conv2d_1 |> layer(model, :layer1_0_bn1)
relu__1 = batch_norm_1 |> layer(model, :relu__1)
conv2d_2 = relu__1 |> layer(model, :layer1_0_conv2)
batch_norm_2 = conv2d_2 |> layer(model, :layer1_0_bn2)
add_ = ExTorch.add(batch_norm_2, max_pool2d)    # skip connection
```

The generator tracks which values are consumed by multiple downstream nodes
and emits proper variable bindings at branch points, rather than producing a
flat pipe chain that loses the branching structure.

## Observability

Production serving requires monitoring, and ExTorch integrates with the standard
Elixir observability stack. All model servers (`Export.Server`, `AOTI.Server`,
`JIT.Server`) emit `:telemetry` events for every model load and inference call,
including duration, input count, and device metadata. The built-in
`ExTorch.Metrics` module attaches to these events and maintains per-model
statistics in an ETS table:

```elixir
ExTorch.Metrics.setup()

# After some predictions...
ExTorch.Metrics.get("sentiment.pt2")
# => %{inference_count: 15230, error_count: 2,
#      total_duration_ms: 4523.1, min_duration_ms: 0.2,
#      max_duration_ms: 89.1, load_duration_ms: 340.5, ...}
```

For teams using Phoenix, an optional LiveDashboard page provides a visual
overview of all loaded models with their inference counts, latency stats, and
error rates. Model servers also implement `format_status/2`, making them
introspectable in the standard Erlang `:observer`.

On GPU builds, ExTorch exposes CUDA memory allocator statistics through
`cuda_memory_allocated/1`, `cuda_memory_reserved/1`, and
`cuda_max_memory_allocated/1`, which can be used to monitor device memory
pressure during serving.

## Zero-Copy Tensor Exchange with Nx

ExTorch does not try to replace the [Nx](https://github.com/elixir-nx/nx)
ecosystem -- it complements it. For teams that use Nx for data preprocessing and
ExTorch for model serving, the `ExTorch.Tensor.Blob` module provides zero-copy
tensor exchange by sharing raw memory pointers between libraries that use
libtorch in the same BEAM process:

```elixir
blob = ExTorch.Tensor.Blob.to_blob(tensor)
# => %Blob{ptr: 140234567890, shape: {3, 224, 224},
#          strides: [150528, 224, 1], dtype: :float, ...}

view = ExTorch.Tensor.Blob.from_blob(
  %{ptr: foreign_ptr, shape: {3, 224, 224}, dtype: :float32},
  owner: source_tensor
)
```

Since both ExTorch and Torchx use libtorch under the hood, the raw pointer
exchange works for both CPU and CUDA memory without any data movement. The
`%BlobView{}` wrapper holds a reference to the source tensor, ensuring it is
not garbage collected while the view is alive.

## Infrastructure

The v0.3 release updates all dependencies to current versions: Rustler 0.37,
cxx 1.0.194, and libtorch 2.11. The minimum Elixir version is 1.16, with CI
testing across OTP 26-27, Elixir 1.16-1.18, and both stable and nightly
PyTorch builds.

A memory safety audit was conducted across all layers of the FFI bridge. The
refcount chain from Elixir's garbage collector through Rustler's `ResourceArc`
to cxx's `SharedPtr` to libtorch's tensor storage is correct for tensors, JIT
modules, and nn modules. The `from_blob` path, which creates tensors backed by
foreign memory, uses `%BlobView{}` to hold lifetime references preventing
premature deallocation. A build-time sync check in `build.rs` validates that
every function declared in the cxx bridge has a matching C++ header declaration,
catching common "forgot to update the header" mistakes at compile time rather
than at link time.

## What's Next

The current release covers the serving use case well. Areas under consideration
for future releases include an explicit `torch::NoGradGuard` context for
reducing GPU memory usage during inference, binary tensor serialization for
efficient tensor I/O, model pooling for high-throughput serving across multiple
replicas, and expanding the ATen interpreter's op coverage for more exotic
architectures.

## Getting Started

ExTorch is available on [Hex](https://hex.pm/packages/extorch):

```elixir
{:extorch, "~> 0.3.0"}
```

The [documentation](https://hexdocs.pm/extorch) includes four guides covering
installation, model serving, the neural network DSL, and observability. The
[GitHub repository](https://github.com/andfoy/extorch) has the full source and
a test suite of 593 tests.

ExTorch is MIT licensed. Contributions, issues, and feedback are welcome at
https://github.com/andfoy/extorch.
