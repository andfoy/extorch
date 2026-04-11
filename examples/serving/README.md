# ExTorch Model Serving Examples

Production deployment patterns for serving PyTorch models from Elixir.

## Quick Start

```bash
# Generate model fixtures
cd ../..
.venv/bin/python test/fixtures/generate_popular_models.py
.venv/bin/python test/fixtures/generate_aoti_popular_models.py --device cuda

# Run any example
mix run examples/serving/basic_inference.exs
mix run examples/serving/genserver_pool.exs
mix run examples/serving/telemetry_dashboard.exs
mix run examples/serving/hot_reload.exs
```

## Examples

### 1. `basic_inference.exs` — Three Inference Paths

Demonstrates the three ways to run a model, when to use each, and
relative performance:

- **Export** (`forward/2`) — Elixir interpreter, good for debugging
- **Export Native** (`forward_native/2`) — C++ graph executor, production-ready
- **AOTI** — Compiled kernels, fastest

### 2. `genserver_pool.exs` — Supervised Model Pool

Production-grade serving with:

- Multiple model replicas behind a pool
- Request routing with backpressure
- Graceful degradation on failure
- Per-model telemetry

### 3. `telemetry_dashboard.exs` — Metrics and Monitoring

Live inference metrics using ExTorch.Metrics:

- Latency percentiles (p50/p95/p99)
- Throughput (inferences/sec)
- Error rate tracking
- Integration with Phoenix LiveDashboard

### 4. `hot_reload.exs` — Zero-Downtime Model Updates

Swap models without dropping requests:

- Load new model in background
- Atomic swap via GenServer state
- Drain in-flight requests to old model
- Rollback on load failure
