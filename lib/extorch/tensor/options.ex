defmodule ExTorch.Tensor.Options do
  @moduledoc """
  The ``ExTorch.Tensor.Options`` struct defines the creation parameters of a tensor.
  """
  # @behaviour Access

  @typedoc """
  An ``ExTorch.Tensor.Options`` defines the creation parameters of a tensor.
  """
  @type t :: %__MODULE__{
    dtype: ExTorch.DType.dtype(),
    layout: ExTorch.Layout.layout(),
    device: ExTorch.Device.device(),
    requires_grad: boolean(),
    pin_memory: boolean(),
    memory_format: ExTorch.MemoryFormat.memory_format()
  }

  defstruct [
    # Type of the tensor reference
    dtype: :int,

    # The layout format of the tensor
    layout: :strided,

    # Device where the tensor lives on
    device: :cpu,

    # true if the tensor will accumulate gradients, false otherwise
    requires_grad: false,

    # true if the tensor will pin memory on a GPU device
    pin_memory: false,

    # The memory format which the tensor should have
    memory_format: :contiguous

  ]
end
