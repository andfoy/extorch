# Changelog

## 0.4.0 (2026-04-11)

### Highlights

ExTorch now **beats Python's inference performance by 1.35x** on average across 12 models via a new pre-compiled graph executor. All inference paths produce bit-for-bit identical results to Python.

### New Features

- **Pre-compiled graph executor** (`forward_compiled/2`) — Resolves all op schemas and converts string refs to integer slot indices at load time. Zero per-op overhead at inference time. 1.35x faster than Python Export on average across 12 models on RTX 3060.
- **Native graph executor** (`forward_native/2`) — Runs entire Export graphs in a single NIF call via C++ loop, eliminating per-node NIF boundary crossings. ViT-B/16 goes from 55ms to 12ms.
- **Generic c10::Dispatcher NIF** — `dispatch_op/3` calls any PyTorch op by name through the c10 dispatcher. `load_torch_library/1` loads external `.so` files (torchvision, torchaudio). `list_registered_ops/1` discovers available ops.
- **Op extension system** — `ExTorch.Export.OpHandler` behaviour + `ExTorch.Export.OpRegistry` for registering custom ops from external packages (e.g., ExTorch.Vision).
- **Schema-aware arg reordering** — The graph executor matches export graph args to schema positions by name, inserting defaults for omitted optional params (e.g., `layout` in `zeros.default`).
- **Device threading for native/compiled paths** — Tensor creation ops respect the target device from `load/2`, fixing GPU inference for models with hardcoded `device: "cpu"` in the export graph.
- **AOTI libtorch preloading** — `ExTorch.AOTI.load/2` pre-loads libtorch shared libraries via `dlopen` so AOTI `.pt2` packages work without `LD_LIBRARY_PATH`.
- **ccache integration** — `build.rs` auto-detects `ccache`/`sccache` for faster C++ rebuilds (~35s instead of ~3min).
- **Deployment examples** — Four production serving patterns: basic inference, GenServer pool, hot reload, telemetry dashboard.
- **Real-world model benchmarks** — Export + benchmark scripts for 8 production models (CLIP, DistilBERT, MobileNetV3, EfficientNet, ResNet50, ViT-B/16, DeepLabV3, Whisper).
- **ViT-B/16 in benchmarks** — Added to popular model fixture generator and all benchmark scripts.
- **AOTI GPU benchmarks** — Side-by-side AOTI comparison scripts (Python + Elixir).

### Bug Fixes

- **Build OOM fix** — Cap C++ compile parallelism by removing Cargo-inherited jobserver vars so `NUM_JOBS` takes effect. Lowered default to 2 jobs.
- **Export GPU device mismatch** — Thread target device through `compile_node` / `build_runner` / `execute_node` so tensor creation ops (zeros, full, ones) land on the correct device.
- **ScalarType mapping** — torch.export JSON uses 7 for float32, but C++ enum maps 7 to float64. Added export→C++ conversion table.
- **Scalar→tensor coercion** — Wrap float/int/bool IValues as scalar tensors when the schema expects Tensor (fixes `aten::mul.Tensor` with scalar args).
- **IntList typing** — Use `c10::impl::GenericList(c10::IntType::get())` for typed int lists in the dispatcher.
- **ivalue_utils extraction** — Moved `flatten_ivalue` from JIT-specific code to a shared module using `c10::IValue` directly, decoupling from the deprecated `torch::jit` namespace.

### Breaking Changes

- None. All existing APIs preserved.

## 0.3.0

Initial public release with JIT serving, AOTI, Export interpreter, NN DSL, telemetry, and tensor operations.
