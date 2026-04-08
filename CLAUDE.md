# ExTorch

Elixir bindings for libtorch -- production ML model serving on the BEAM.

## Build

```bash
mix deps.get
mix compile
```

Requires Rust (stable) and libtorch. On first compile, libtorch is downloaded automatically to `priv/native/libtorch/`. Alternatively, set `config :extorch, libtorch: [version: :local, folder: :python]` to use a local PyTorch installation.

## Test

```bash
# Generate test model fixtures (requires Python + torch in .venv)
.venv/bin/python test/fixtures/generate_models.py

mix test
```

CUDA tests are auto-excluded on CPU-only builds via tags in `test/test_helper.exs`.

## Architecture

Three-layer design: C++ (libtorch wrapper) → Rust (cxx bridge + Rustler NIFs) → Elixir (macro-generated API).

- **C++ sources**: `native/extorch/src/csrc/*.cc` + `native/extorch/include/*.h`
- **Rust bridge**: `native/extorch/src/native/*.rs.in` (Tera templates rendered by `build.rs`)
- **Rust NIFs**: `native/extorch/src/nifs/*.rs`
- **Elixir API**: `lib/extorch/`

Adding a new function touches: `.h` header → `.cc` implementation → `.rs.in` bridge → NIF (or `nif_impl!`) → Elixir binding (or `defbinding`). The build.rs sync check warns if bridge functions are missing from headers.

## Key modules

- `ExTorch.JIT` -- Load/serve TorchScript models
- `ExTorch.JIT.Server` -- GenServer model serving with telemetry
- `ExTorch.NN` -- Neural network layer creation (35 layer types)
- `ExTorch.NN.Module` -- DSL for defining models (`deflayer`, `from_jit`, `load_weights`)
- `ExTorch.NN.Introspect` -- Extract model architecture from .pt files
- `ExTorch.Tensor.Blob` -- Zero-copy tensor exchange via data_ptr/from_blob
- `ExTorch.Metrics` -- ETS-backed serving metrics from telemetry events
