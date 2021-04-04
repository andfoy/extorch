defmodule ExTorch do
  @moduledoc """
  The ``ExTorch`` namespace contains data structures for multi-dimensional tensors and mathematical operations over these are defined.
  Additionally, it provides many utilities for efficient serializing of Tensors and arbitrary types, and other useful utilities.

  It has a CUDA counterpart, that enables you to run your tensor computations on an NVIDIA GPU with compute capability >= 3.0
  """

  use ExTorch.DelegateWithDocs
  import ExTorch.ModuleMixin
  extends(ExTorch.Native.Tensor.Creation)
  extends(ExTorch.Native.Tensor.Info)
  extends(ExTorch.Native.Tensor.Ops)

  # native_calls do
  #   # set_num_threads(num_threads)
  #   size(tensor)
  #   unsqueeze(tensor, dim)
  # end
end
