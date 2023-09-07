defmodule ExTorch do
  @moduledoc """
  The ``ExTorch`` namespace contains data structures for multi-dimensional tensors and mathematical operations over these are defined.
  Additionally, it provides many utilities for efficient serializing of Tensors and arbitrary types, and other useful utilities.

  It has a CUDA counterpart, that enables you to run your tensor computations on an NVIDIA GPU with compute capability >= 3.0
  """
  use ExTorch.DelegateWithDocs
  import ExTorch.ModuleMixin

  # Native operations
  extends(ExTorch.Native.Tensor.Creation)
  extends(ExTorch.Native.Tensor.Ops.Manipulation)
  extends(ExTorch.Native.Tensor.Ops.Indexing)
  extends(ExTorch.Native.Tensor.Ops.PointWise)
  extends(ExTorch.Native.Tensor.Ops.Other)
  extends(ExTorch.Registry.DType)
  extends(ExTorch.Registry.Device)

  # extends(ExTorch.Native.Tensor.Info)
end
