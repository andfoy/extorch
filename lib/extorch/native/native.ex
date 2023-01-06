defmodule ExTorch.Native do
  @moduledoc """
  The `ExTorch.Native` module contains all NIF declarations to call libtorch in C++.

  All the declarations contained here are placeholder to native calls generated with `Rustler` and
  implemented via Rust.

  Argument optional function declarations with default values are provided on the `ExTorch` module.
  """
  # use ExTorch.Native.Tensor.Creation
  # use ExTorch.Native.Tensor.Info
  # use ExTorch.Native.Tensor.Ops

  use Rustler, otp_app: :extorch, crate: "extorch", env: [{"CARGO_TERM_VERBOSE", "true"}]

  # When your NIF is loaded, it will override this function.
  def add(_a, _b), do: :erlang.nif_error(:nif_not_loaded)

  def empty(_size, _opts), do: :erlang.nif_error(:nif_not_loaded)

  def repr(_t), do: :erlang.nif_error(:nif_not_loaded)

end
