defmodule ExTorch.Native do
  @moduledoc """
  The `ExTorch.Native` module contains all NIF declarations to call libtorch in C++.

  All the declarations contained here are placeholder to native calls generated with `Rustler` and
  implemented via Rust.

  Argument optional function declarations with default values are provided on the `ExTorch` module.
  """

  use ExTorch.Native.Tensor.Creation
  use ExTorch.Native.Tensor.Info
  use ExTorch.Native.Tensor.Ops.Indexing
  use ExTorch.Native.Tensor.Ops.Manipulation
  use ExTorch.Native.Tensor.Ops.PointWise
  use ExTorch.Native.Tensor.Ops.Comparison
  use ExTorch.Native.Tensor.Ops.Reduction
  use ExTorch.Native.Tensor.Ops.Other

  use ExTorch.Utils.DownloadTorch
  use Rustler, otp_app: :extorch, crate: "extorch", env: [{"CARGO_TERM_VERBOSE", "true"}]

  # When your NIF is loaded, it will override this function.
  def add(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
end
