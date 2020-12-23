defmodule ExTorch do
  import ExTorch.Macros

  @moduledoc """
  Documentation for `ExTorch`.

  The ``ExTorch`` namespace contains data structures for multi-dimensional tensors and mathematical operations over these are defined.
  Additionally, it provides many utilities for efficient serializing of Tensors and arbitrary types, and other useful utilities.

  It has a CUDA counterpart, that enables you to run your tensor computations on an NVIDIA GPU with compute capability >= 3.0
  """

  @doc """
  Hello world.

  ## Examples

      iex> ExTorch.hello()
      :world

  """
  def hello do
    :world
  end

  native_calls do
    empty(sizes :: tuple() | [integer()],
      dtype: :float :: ExTorch.DType.dtype(),
      layout: :strided :: ExTorch.Layout.layout(),
      device: :cpu :: ExTorch.Device.device(),
      requires_grad: false :: bool(),
      pin_memory: false :: bool(),
      memory_format: :contiguous :: ExTorch.MemoryFormat.memory_format()
    )

    zeros(sizes :: tuple() | [integer()],
      dtype: :float :: ExTorch.DType.dtype(),
      layout: :strided :: ExTorch.Layout.layout(),
      device: :cpu :: ExTorch.Device.device(),
      requires_grad: false :: bool(),
      pin_memory: false :: bool(),
      memory_format: :contiguous :: ExTorch.MemoryFormat.memory_format()
    )
  end

  # _dtype, _layout, _device, _requires_grad, _pin_memory
  deftensor empty(
              sizes,
              dtype \\ :float,
              layout \\ :strided,
              device \\ :cpu,
              requires_grad \\ false,
              pin_memory \\ false,
              memory_format \\ :contiguous
            ) do
    ExTorch.Native.empty(sizes, dtype, layout, device, requires_grad, pin_memory, memory_format)
  end

  deftensor zeros(
              sizes,
              dtype \\ :float,
              layout \\ :strided,
              device \\ :cpu,
              requires_grad \\ false,
              pin_memory \\ false,
              memory_format \\ :contiguous
            ) do
    ExTorch.Native.zeros(sizes, dtype, layout, device, requires_grad, pin_memory, memory_format)
  end

  deftensor ones(
              sizes,
              dtype \\ :float,
              layout \\ :strided,
              device \\ :cpu,
              requires_grad \\ false,
              pin_memory \\ false,
              memory_format \\ :contiguous
            ) do
    ExTorch.Native.ones(sizes, dtype, layout, device, requires_grad, pin_memory, memory_format)
  end

  deftensor full(
              sizes,
              scalar,
              dtype \\ :float,
              layout \\ :strided,
              device \\ :cpu,
              requires_grad \\ false,
              pin_memory \\ false,
              memory_format \\ :contiguous
            ) do
    ExTorch.Native.full(sizes, scalar, dtype, layout, device, requires_grad, pin_memory, memory_format)
  end
end
