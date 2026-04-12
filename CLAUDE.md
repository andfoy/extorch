# ExTorch

PyTorch bindings for Elixir. Load and execute serialized PyTorch models (`.pt2` AOTInductor, `.pt2` torch.export) directly from the BEAM, with a customization surface built on normal OTP primitives.

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

- `ExTorch.NN` -- Neural network layer creation (35 layer types)
- `ExTorch.NN.Module` -- DSL for defining models (`deflayer`, `load_weights`)
- `ExTorch.Tensor.Blob` -- Zero-copy tensor exchange via data_ptr/from_blob
- `ExTorch.AOTI` -- Load and run AOTInductor .pt2 compiled models
- `ExTorch.Export` -- Pure Elixir reader + ATen interpreter for torch.export.save .pt2 archives (load, forward, graph introspection, weight extraction, DSL generation). Tested with AlexNet, ResNet18, MobileNetV2, VGG11, SqueezeNet, transformers. Dynamic batch / HW dims work across all three inference paths (see test/export/dynamic_batch_test.exs); data-dependent shapes (NMS, nonzero with downstream arithmetic) are the remaining gap.
- `ExTorch.Metrics` -- Optional telemetry handlers that populate ETS counters for inference events. Use or ignore.
