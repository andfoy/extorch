defmodule ExTorch.MemoryFormat do
  @moduledoc """
  A ``torch.memory_format`` is an object representing the memory format on which
  a ``torch.Tensor`` is or will be allocated.

  Possible values are:
  * ``:contiguous``: Tensor is or will be allocated in dense non-overlapping memory. Strides represented by values in decreasing order.
  * ``:channels_last``: Tensor is or will be allocated in dense non-overlapping memory. Strides represented by values in ``strides[0] > strides[2] > strides[3] > strides[1] == 1`` aka NHWC order.
  * ``:preserve_format``: Used in functions like clone to preserve the memory format of the input tensor. If input tensor is allocated in dense non-overlapping memory, the output tensor strides will be copied from the input. Otherwise output strides will follow ``contiguous``
  """

  @typedoc """
  A ``torch.memory_format`` is an object representing the memory format on which
  a ``torch.Tensor`` is or will be allocated.
  """
  @type memory_format :: :contiguous | :channels_last | :preserve_format | :channels_last_3d

  @memory_format [:contiguous, :channels_last, :preserve_format, :channels_last_3d]

  defguard is_memory_format(memory_format) when memory_format in @memory_format
end
