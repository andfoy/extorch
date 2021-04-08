set -exou

export RUSTFLAGS="-C link-args=-Wl,-rpath,$(pwd)/priv/native/libtorch/lib"
MIX_ENV=test mix compile
