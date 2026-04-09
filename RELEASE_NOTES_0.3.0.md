# ExTorch 0.3.0

**Production ML model serving on the BEAM, now with `torch.export` as a first-class citizen.**

This release adds a pure Elixir reader and interpreter for `torch.export.save` `.pt2` archives, AOTInductor compiled model support, and significantly expands the tensor operation coverage. The serving story now spans three complementary paths (Export, AOTI, JIT), all with OTP supervision and telemetry built in.

## Highlights

- **Pure Elixir `torch.export` interpreter** -- load and run `.pt2` files directly without C++ ExportedProgram support, Python, or the deprecated `torch.jit`. Tested against AlexNet, ResNet18, MobileNetV2, VGG11, SqueezeNet, a transformer, and an autoencoder.
- **AOTInductor compiled models** -- load `.pt2` packages from `aoti_compile_and_package` via the `AOTIModelPackageLoader` C++ API for maximum inference throughput.
- **Data flow analysis in DSL generation** -- `ExTorch.Export.to_elixir/2` emits proper variable bindings for skip connections and branching architectures (e.g., ResNet residual blocks).
- **27 new tensor ops** including `matmul`, `mm`, `bmm`, `where`, `masked_fill`, `clamp`, `einsum`, `view`, `expand`, `clone`, `detach`, and the core arithmetic ops `add`/`sub`/`mul`/`div`.
- **OTP serving** -- `ExTorch.Export.Server` and `ExTorch.AOTI.Server` join `ExTorch.JIT.Server` with the same GenServer pattern: fault tolerance, telemetry events, and `:observer` integration.

## New Features

### torch.export Support (`ExTorch.Export`)

A new pure Elixir module for reading and interpreting `torch.export.save` archives:

- `ExTorch.Export.load/1` -- loads a `.pt2` archive, returns a `Model` struct
- `ExTorch.Export.forward/2` -- runs inference via the ATen graph interpreter (60+ ops)
- `ExTorch.Export.read_schema/1` -- extracts graph IR, inputs, outputs, and weight metadata
- `ExTorch.Export.read_weights/1` -- loads weight tensors into an FQN-keyed map
- `ExTorch.Export.to_elixir/2` -- generates `ExTorch.NN.Module` DSL source with flow analysis
- `ExTorch.Export.Server` -- GenServer wrapper with telemetry events (`[:extorch, :export, :load/:forward, ...]`)

The interpreter dispatches the following ATen op categories: linear algebra (`linear`, `mm`, `bmm`, `addmm`), activations (`relu`, `gelu`, `sigmoid`, `tanh`, `silu`, `elu`, `leaky_relu`, `hardtanh`, with in-place variants), softmax, arithmetic, unary math, clamping, comparisons, reductions, conditional/masking, shape ops, convolutions (including generic `aten.convolution`), normalization (batch, layer, group), pooling, dropout, embedding, gather/scatter, and more.

### AOTInductor Support (`ExTorch.AOTI`)

A new module for loading AOTInductor-compiled `.pt2` packages:

- `ExTorch.AOTI.load/2` -- loads an AOTI package via `torch::inductor::AOTIModelPackageLoader`
- `ExTorch.AOTI.forward/2` -- runs inference on compiled kernels
- `ExTorch.AOTI.metadata/1` -- returns package metadata (device, ISA, platform)
- `ExTorch.AOTI.constant_names/1` -- returns fully-qualified parameter names
- `ExTorch.AOTI.available?/0` -- checks if AOTI support is compiled in
- `ExTorch.AOTI.Server` -- GenServer wrapper with telemetry events (`[:extorch, :aoti, :load/:forward, ...]`)

### Tensor Operations

27 new pointwise, math, and linear algebra operations:

- **Arithmetic**: `add`, `sub`, `mul`, `tensor_div`, `neg`, `tensor_abs`, `pow_tensor`, `clamp`
- **Math**: `tensor_exp`, `tensor_log`, `tensor_sqrt`, `tensor_sin`, `tensor_cos`
- **Linear algebra**: `matmul`, `mm`, `bmm`
- **Conditional**: `tensor_where`, `masked_fill`
- **Manipulation**: `contiguous`, `clone`, `detach`, `view`, `expand`
- **Functional activations**: `functional_softmax`, `functional_log_softmax`, `functional_relu`
- **Einsum**: `einsum`

### Neural Network DSL

- `MyModule.load_weights_from_export/2` -- JIT-free weight loading from `.pt2` archives
- `ExTorch.Export.to_elixir/2` now performs data flow analysis to emit proper variable bindings for skip connections and merge points, supporting ResNet-style architectures

### Zero-Copy Tensor Exchange

- New `from_binary` NIF -- creates tensors from Erlang binaries (copies data into libtorch memory)

## Bug Fixes

- `ExTorch.JIT.Introspect.to_elixir/2` now correctly expands Sequential containers into their leaf layers with dotted path names
- Layer name mapping in introspection handles `MaxPool2d`, `AdaptiveAvgPool2d`, and other pooling types
- Memory safety audit: the refcount chain from Elixir GC through Rustler `ResourceArc` to cxx `SharedPtr` to libtorch is verified correct for tensors, JIT modules, and nn modules

## Infrastructure

- Dependencies updated: Rustler 0.29 → 0.37, cxx 1.0.97 → 1.0.194, libtorch 2.1 → 2.11
- Minimum Elixir version bumped to 1.16
- CI matrix: OTP 26-27, Elixir 1.16-1.18, PyTorch 2.11 stable + nightly
- 593 tests (up from 576 in 0.2.x), 0 failures, 0 warnings
- Build-time sync check validates cxx bridge functions against C++ headers
- Removed the Credo lint workflow to simplify CI

## Breaking Changes

None. All 0.2.x APIs remain compatible.

## Deprecation Notices

While `ExTorch.JIT` remains fully supported, we recommend using `ExTorch.Export` or `ExTorch.AOTI` for new work. `torch.jit` is in maintenance mode upstream, and the torch.export-based paths don't depend on it.

## Installation

```elixir
def deps do
  [
    {:extorch, "~> 0.3.0"}
  ]
end
```

## Getting Started

```elixir
# Recommended path: torch.export
model = ExTorch.Export.load("model.pt2")
output = ExTorch.Export.forward(model, [input])

# Throughput path: AOTI compiled
model = ExTorch.AOTI.load("model_compiled.pt2")
[output] = ExTorch.AOTI.forward(model, [input])

# Production serving with OTP
children = [
  {ExTorch.Export.Server, path: "sentiment.pt2", name: SentimentModel}
]
Supervisor.start_link(children, strategy: :one_for_one)
```

See the [documentation](https://hexdocs.pm/extorch) for the complete getting started guide, serving guide, neural network DSL guide, and observability guide.

## Acknowledgments

Thanks to the PyTorch team for the `AOTIModelPackageLoader` API that made pure C++ AOTI loading possible, and to everyone who has filed issues and provided feedback on the project.
