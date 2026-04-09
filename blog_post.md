# ExTorch: Production ML Model Serving on the BEAM

[ExTorch](https://github.com/andfoy/extorch), the Elixir/Erlang bindings for
[libtorch](https://pytorch.org/cppdocs/), is getting a major update. In this
post, we'll take a deep dive into the new capabilities that make ExTorch a
production-ready framework for serving machine learning models from Elixir,
including JIT model deployment, a neural network DSL, and telemetry-based
observability.

For those unfamiliar with the project, ExTorch bridges Elixir and PyTorch's C++
backend through a three-layer architecture: C++ wraps the libtorch API directly,
Rust provides the NIF bridge using [cxx](https://cxx.rs) and
[Rustler](https://github.com/rusterlium/rustler), and Elixir offers the
user-facing API with macros that generate type-safe bindings. The project
originally focused on tensor operations -- providing over 200 wrapped libtorch
ops for creation, manipulation, pointwise math, comparison, reduction, and
indexing. While this foundation remains, the v0.2 release shifts the focus
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

## JIT Model Serving

The core workflow is simple: train a model in Python, export it as TorchScript,
and serve it from Elixir. On the Python side, this is a matter of calling
`torch.jit.script()` or `torch.jit.trace()` and saving the result:

```python
model.eval()
scripted = torch.jit.script(model)
scripted.save("sentiment_model.pt")
```

On the Elixir side, `ExTorch.JIT.load/2` loads the model and
`ExTorch.JIT.forward/2` runs inference, with no Python runtime in the loop and
no HTTP serialization overhead:

```elixir
model = ExTorch.JIT.load("sentiment_model.pt")
ExTorch.JIT.eval(model)

input = ExTorch.randn({1, 512})
output = ExTorch.JIT.forward(model, [input])
```

One of the more interesting technical challenges was handling PyTorch's `IValue`
type across the FFI boundary. A model's forward method can return not just
tensors, but also tuples, lists, dicts, strings, scalars, and nested
combinations thereof. The cxx bridge doesn't support recursive types, so we
flatten the IValue tree into a pre-order traversal of tagged nodes with
parent-child index references, pass that flat vector across the bridge, and
reconstruct the nested Elixir terms on the Rust side using `enif_*` functions.
The result is that models returning complex types work naturally with Elixir's
pattern matching:

```elixir
# A model that returns {"logits": tensor, "features": tensor}
%{"logits" => logits, "features" => features} =
  ExTorch.JIT.forward(model, [input])
```

The real value, however, comes from wrapping models in OTP processes.
`ExTorch.JIT.Server` is a GenServer that loads a model on `init`, sets it to
eval mode, serializes inference calls for thread safety, and emits telemetry
events for every prediction:

```elixir
children = [
  {ExTorch.JIT.Server, path: "sentiment.pt", name: SentimentModel},
  {ExTorch.JIT.Server, path: "embedding.pt", name: EmbeddingModel,
   device: {:cuda, 0}}
]

Supervisor.start_link(children, strategy: :one_for_one)
```

If a model server crashes due to a malformed input or an out-of-memory error,
the supervisor restarts it -- the model is reloaded from disk and the server is
back online in milliseconds. This is the kind of reliability that is difficult
to retrofit into Python serving solutions.

### A note on torch.export

PyTorch 2.x has introduced `torch.export` as the new recommended path for
model export, alongside `torch.compile` for training-time optimization. It is
worth clarifying how these relate to ExTorch.

`torch.compile` is a training-time JIT compiler -- it optimizes Python execution
but does not produce a serializable artifact. It is not relevant to the
deployment use case that ExTorch addresses. `torch.export`, on the other hand,
produces an `ExportedProgram` with a cleaner IR and better support for dynamic
shapes. It is where the PyTorch team is directing new deployment work.

However, as of early 2026, there is no C++ API for loading an `ExportedProgram`
directly in libtorch. A [feature request](https://github.com/pytorch/pytorch/issues/144663)
for this has been open since January 2025, with the PyTorch team indicating
intent to deliver it but not yet shipping a solution. The only C++ path for
`ExportedProgram` currently goes through AOTInductor, which compiles the model
into a platform-specific `.so` file -- losing the portability that TorchScript
provides.

ExTorch supports three paths for `torch.export` users, covering different
points on the flexibility-vs-throughput spectrum.

The first -- and recommended -- path is a pure Elixir reader and interpreter
for `.pt2` archives produced by `torch.export.save`. These archives contain the
model graph as JSON and weight tensors as raw binary files. ExTorch reads the
archive, loads the weights, and interprets the ATen computation graph directly
-- no C++ ExportedProgram support, no Python, and no `torch.jit` dependency:

```python
exported = torch.export.export(model, (example_input,))
torch.export.save(exported, "model.pt2")
```

```elixir
# Load and run inference directly
model = ExTorch.Export.load("model.pt2")
output = ExTorch.Export.forward(model, [input])
```

The interpreter dispatches 60+ ATen operations (`aten.linear.default`,
`aten.conv2d.default`, `aten.relu.default`, `aten.batch_norm.default`, etc.)
to the corresponding ExTorch tensor functions and NN layer modules. We have
tested it against seven popular architectures -- AlexNet, ResNet18,
MobileNetV2, VGG11, SqueezeNet, a multi-head attention transformer, and an
autoencoder -- all producing correct output shapes and values matching
their Python counterparts.

Beyond inference, `ExTorch.Export` also provides introspection: `read_schema`
extracts the graph IR and weight metadata, `read_weights` loads the raw tensors,
and `to_elixir` generates a complete `ExTorch.NN.Module` DSL definition from the
ATen graph. The extracted weights can be loaded into DSL-defined modules via
`load_weights_from_export`, giving a fully JIT-free workflow from Python
training to Elixir serving.

The second path uses AOTInductor to compile the exported model into an
optimized `.pt2` package with fused kernels, which ExTorch loads via the
`AOTIModelPackageLoader` API in libtorch:

```python
from torch._inductor import aoti_compile_and_package
exported = torch.export.export(model, (example_input,))
aoti_compile_and_package(exported, package_path="model_compiled.pt2")
```

```elixir
model = ExTorch.AOTI.load("model_compiled.pt2")
[output] = ExTorch.AOTI.forward(model, [input])
```

AOTI models trade flexibility for throughput -- they don't support
introspection or weight extraction, but benefit from kernel fusion and
compiler optimizations. `ExTorch.AOTI.Server` provides the same GenServer
serving pattern as `ExTorch.JIT.Server`, with telemetry instrumentation and
OTP fault tolerance.

The third path converts the exported model to TorchScript via
`torch.jit.trace(exported.module(), ...)` for full JIT introspection and DSL
compatibility. This works today but depends on the deprecated `torch.jit` API,
so we recommend the first two paths for new work.

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

The v0.2 release includes 35 layer types covering convolutions (1D, 2D, 3D,
and transposed variants), pooling (max, average, and adaptive), normalization
(batch, layer, group, and instance), recurrent layers (LSTM and GRU),
multi-head attention, and 11 activation functions including ReLU, GELU, SiLU,
Mish, and PReLU. Each function's documentation follows PyTorch's format with
mathematical formulas, shape specifications, and examples.

The DSL supports two ways to use pre-trained weights from Python. The first,
`TextClassifier.from_jit("model.pt")`, loads a TorchScript model and validates
its submodules against the DSL definition at load time, then delegates to the
JIT model's forward method directly. The second,
`TextClassifier.load_weights("model.pt")`, copies parameter tensors from the
`.pt` file into the Elixir-defined layers, which is useful when the Elixir
`forward/2` differs from Python's (for instance, different dropout behavior or
custom post-processing). Both produce identical outputs given the same weights.

For models where no DSL definition exists yet,
`ExTorch.NN.Introspect.to_elixir/2` can generate the Elixir source code from
any `.pt` file by inspecting its JIT IR, mapping qualified PyTorch type names
to ExTorch layer modules, and inferring layer parameters from weight shapes:

```elixir
model = ExTorch.JIT.load("model.pt")
IO.puts(ExTorch.NN.Introspect.to_elixir(model, "MyModel"))
# defmodule MyModel do
#   use ExTorch.NN.Module
#
#   deflayer :fc1, ExTorch.NN.Linear, in_features: 10, out_features: 20
#   deflayer :relu, ExTorch.NN.ReLU
#   deflayer :fc2, ExTorch.NN.Linear, in_features: 20, out_features: 5
#   ...
# end
```

This outputs a complete `defmodule` with `deflayer` declarations that mirrors
the model's architecture -- a starting point that can be pasted into a project
and customized. For complex models like ResNets, the introspection correctly
identifies all layer types (Conv2d, BatchNorm2d, MaxPool2d,
AdaptiveAvgPool2d, etc.) though nested Sequential blocks need to be
expanded manually.

## Observability

Production serving requires monitoring, and ExTorch integrates with the standard
Elixir observability stack. `ExTorch.JIT.Server` emits `:telemetry` events for
every model load and inference call, including duration, input count, and device
metadata. The built-in `ExTorch.Metrics` module attaches to these events and
maintains per-model statistics in an ETS table:

```elixir
ExTorch.Metrics.setup()

# After some predictions...
ExTorch.Metrics.get("sentiment.pt")
# => %{inference_count: 15230, error_count: 2,
#      total_duration_ms: 4523.1, min_duration_ms: 0.2,
#      max_duration_ms: 89.1, load_duration_ms: 340.5, ...}
```

For teams using Phoenix, an optional LiveDashboard page provides a visual
overview of all loaded models with their inference counts, latency stats, and
error rates. And since `ExTorch.JIT.Server` implements `format_status/2`, model
servers are introspectable in the standard Erlang `:observer` as well -- showing
their path, device, inference count, and uptime without any additional
configuration.

On GPU builds, ExTorch also exposes CUDA memory allocator statistics through
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

The v0.2 release updates all dependencies to current versions: Rustler 0.37,
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
reducing GPU memory usage during inference,
`torch::jit::optimize_for_inference` for JIT optimization passes that fuse
operations, binary tensor serialization for efficient tensor I/O without going
through lists or external frameworks, and model pooling for high-throughput
serving across multiple replicas.

## Getting Started

ExTorch is available on [Hex](https://hex.pm/packages/extorch):

```elixir
{:extorch, "~> 0.2.0"}
```

The [documentation](https://hexdocs.pm/extorch) includes four guides covering
installation, model serving, the neural network DSL, and observability. The
[GitHub repository](https://github.com/andfoy/extorch) has the full source and
a test suite of 559 tests.

ExTorch is MIT licensed. Contributions, issues, and feedback are welcome at
https://github.com/andfoy/extorch.
