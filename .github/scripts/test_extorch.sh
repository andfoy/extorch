set -exou

export RUSTFLAGS="-C link-args=-Wl,-rpath,$(pwd)/priv/native/libtorch/lib"

# Generate test fixture models
python -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet
.venv/bin/python test/fixtures/generate_models.py

mix test --trace
