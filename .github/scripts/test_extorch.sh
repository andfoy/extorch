set -exou

export RUSTFLAGS="-C link-args=-Wl,-rpath,$(pwd)/priv/native/libtorch/lib"
mix test --trace
