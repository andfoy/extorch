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
  extends(ExTorch.Native.Tensor.Ops.Comparison)
  extends(ExTorch.Native.Tensor.Ops.Reduction)
  extends(ExTorch.Native.Tensor.Ops.Other)
  extends(ExTorch.Registry.DType)
  extends(ExTorch.Registry.Device)

  # extends(ExTorch.Native.Tensor.Info)

  @doc """
  Enable or disable autograd tracking process-wide.

  Equivalent to `torch.set_grad_enabled(enabled)` in Python. When disabled,
  tensor operations skip building the autograd graph and dispatch to faster
  inference-mode kernels in the underlying libtorch backends (notably
  oneDNN/MKLDNN). For pure inference workloads, disable grad once at startup.

  Returns `:ok`.
  """
  @spec set_grad_enabled(boolean()) :: :ok
  def set_grad_enabled(enabled) when is_boolean(enabled) do
    ExTorch.Native.aten_set_grad_enabled(enabled)
  end

  @doc """
  Returns whether autograd tracking is currently enabled process-wide.
  """
  @spec grad_enabled?() :: boolean()
  def grad_enabled?() do
    ExTorch.Native.aten_is_grad_enabled()
  end

  @doc """
  Temporarily disable autograd tracking around the given zero-arity function
  and restore the previous setting afterward. Mirrors `torch.no_grad()` as a
  Python context manager.

  ## Example

      ExTorch.no_grad(fn ->
        ExTorch.Export.forward(model, [input])
      end)
  """
  @spec no_grad((-> result)) :: result when result: any()
  def no_grad(fun) when is_function(fun, 0) do
    prev = grad_enabled?()
    :ok = set_grad_enabled(false)
    try do
      fun.()
    after
      :ok = set_grad_enabled(prev)
    end
  end

  @doc """
  Clear the current OS thread's CPU affinity mask so libtorch's OpenMP
  worker threads can run on all available cores.

  NIF calls run on BEAM scheduler threads which are typically bound to
  individual CPUs. OpenMP workers spawned by libtorch inherit that mask,
  so all intra-op parallelism happens on a single core. This resets the
  mask to include every online CPU, letting libtorch parallelize freely.

  Call once per BEAM process that will dispatch tensor ops. Returns `true`
  on Linux if the syscall succeeded, `false` on other platforms.
  """
  @spec clear_cpu_affinity() :: boolean()
  def clear_cpu_affinity() do
    ExTorch.Native.aten_clear_cpu_affinity()
  end
end
