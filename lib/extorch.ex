defmodule ExTorch do
  import ExTorch.Macros

  @moduledoc """
  The ``ExTorch`` namespace contains data structures for multi-dimensional tensors and mathematical operations over these are defined.
  Additionally, it provides many utilities for efficient serializing of Tensors and arbitrary types, and other useful utilities.

  It has a CUDA counterpart, that enables you to run your tensor computations on an NVIDIA GPU with compute capability >= 3.0
  """

  native_calls do
    # set_num_threads(num_threads :: integer())
    size(tensor)

    empty(
      size,
      dtype \\ :float,
      layout \\ :strided,
      device \\ :cpu,
      requires_grad \\ false,
      pin_memory \\ false,
      memory_format \\ :contiguous
    )

    zeros(
      size,
      dtype \\ :float,
      layout \\ :strided,
      device \\ :cpu,
      requires_grad \\ false,
      pin_memory \\ false,
      memory_format \\ :contiguous
    )

    ones(
      size,
      dtype \\ :float,
      layout \\ :strided,
      device \\ :cpu,
      requires_grad \\ false,
      pin_memory \\ false,
      memory_format \\ :contiguous
    )

    full(
      size,
      scalar,
      dtype \\ :float,
      layout \\ :strided,
      device \\ :cpu,
      requires_grad \\ false,
      pin_memory \\ false,
      memory_format \\ :contiguous
    )

  end

  # _dtype, _layout, _device, _requires_grad, _pin_memory
  # deftensor empty(
  #             sizes,
  #             dtype \\ :float,
  #             layout \\ :strided,
  #             device \\ :cpu,
  #             requires_grad \\ false,
  #             pin_memory \\ false,
  #             memory_format \\ :contiguous
  #           ) do
  #   ExTorch.Native.empty(sizes, dtype, layout, device, requires_grad, pin_memory, memory_format)
  # end

  # deftensor zeros(
  #             sizes,
  #             dtype \\ :float,
  #             layout \\ :strided,
  #             device \\ :cpu,
  #             requires_grad \\ false,
  #             pin_memory \\ false,
  #             memory_format \\ :contiguous
  #           ) do
  #   ExTorch.Native.zeros(sizes, dtype, layout, device, requires_grad, pin_memory, memory_format)
  # end

  # deftensor ones(
  #             sizes,
  #             dtype \\ :float,
  #             layout \\ :strided,
  #             device \\ :cpu,
  #             requires_grad \\ false,
  #             pin_memory \\ false,
  #             memory_format \\ :contiguous
  #           ) do
  #   ExTorch.Native.ones(sizes, dtype, layout, device, requires_grad, pin_memory, memory_format)
  # end

  # deftensor full(
  #             sizes,
  #             scalar,
  #             dtype \\ :float,
  #             layout \\ :strided,
  #             device \\ :cpu,
  #             requires_grad \\ false,
  #             pin_memory \\ false,
  #             memory_format \\ :contiguous
  #           ) do
  #   ExTorch.Native.full(sizes, scalar, dtype, layout, device, requires_grad, pin_memory, memory_format)
  # end
end
