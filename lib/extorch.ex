defmodule ExTorch do
  # require ExTorch.Native
  import ExTorch.Macros

  @moduledoc """
  The ``ExTorch`` namespace contains data structures for multi-dimensional tensors and mathematical operations over these are defined.
  Additionally, it provides many utilities for efficient serializing of Tensors and arbitrary types, and other useful utilities.

  It has a CUDA counterpart, that enables you to run your tensor computations on an NVIDIA GPU with compute capability >= 3.0
  """

  native_calls do
    # set_num_threads(num_threads)
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

    rand(
      size,
      dtype \\ :float,
      layout \\ :strided,
      device \\ :cpu,
      requires_grad \\ false,
      pin_memory \\ false,
      memory_format \\ :contiguous
    )

    randn(
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

    deftensor eye(
      n,
      dtype \\ :float,
      layout \\ :strided,
      device \\ :cpu,
      requires_grad \\ false,
      pin_memory \\ false,
      memory_format \\ :contiguous
    ) do
      ExTorch.Native.eye(
        n,
        n,
        dtype,
        layout,
        device,
        requires_grad,
        pin_memory,
        memory_format
      )
    end

    eye(
      n,
      m,
      dtype \\ :float,
      layout \\ :strided,
      device \\ :cpu,
      requires_grad \\ false,
      pin_memory \\ false,
      memory_format \\ :contiguous
    ) when is_integer(m)

    deftensor arange(
      end_bound,
      step \\ 1,
      dtype \\ :float,
      layout \\ :strided,
      device \\ :cpu,
      requires_grad \\ false,
      pin_memory \\ false,
      memory_format \\ :contiguous
    )
    when is_number(end_bound) do
      ExTorch.Native.arange(
        0,
        end_bound,
        step,
        dtype,
        layout,
        device,
        requires_grad,
        pin_memory,
        memory_format
      )
    end

    arange(
      start,
      end_bound,
      step \\ 1,
      dtype \\ :float,
      layout \\ :strided,
      device \\ :cpu,
      requires_grad \\ false,
      pin_memory \\ false,
      memory_format \\ :contiguous
    )

    arange(
      start,
      end_bound,
      step,
      dtype \\ :float,
      layout \\ :strided,
      device \\ :cpu,
      requires_grad \\ false,
      pin_memory \\ false,
      memory_format \\ :contiguous
    )
    when is_number(step)

    linspace(
      start,
      end_bound,
      steps,
      dtype \\ :float,
      layout \\ :strided,
      device \\ :cpu,
      requires_grad \\ false,
      pin_memory \\ false,
      memory_format \\ :contiguous
    )

    logspace(
      start,
      end_bound,
      steps,
      base \\ 10,
      dtype \\ :float,
      layout \\ :strided,
      device \\ :cpu,
      requires_grad \\ false,
      pin_memory \\ false,
      memory_format \\ :contiguous
    )
  end
end
