defmodule ExTorch do
  import ExTorch.Macros

  @moduledoc """
  Documentation for `ExTorch`.
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

  # native_calls do
  #   empty(sizes,
  #     dtype: :float,
  #     layout: :strided,
  #     device: :cpu,
  #     requires_grad: false,
  #     pin_memory: false,
  #     memory_format: :contiguous
  #   )

  #   zeros(sizes,
  #     dtype: :float,
  #     layout: :strided,
  #     device: :cpu,
  #     requires_grad: false,
  #     pin_memory: false,
  #     memory_format: :contiguous
  #   )
  # end

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
end
