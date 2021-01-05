defmodule ExTorch.Native do
  @moduledoc """
  The `ExTorch.Native` module contains all NIF declarations to call libtorch in C++.

  All the declarations contained here are placeholder to native calls generated with `Rustler` and
  implemented via Rust.

  Argument optional function declarations with default values are provided on the `ExTorch` module.
  """

  use Rustler, otp_app: :extorch, crate: "extorch_native"

  # When your NIF is loaded, it will override this function.
  def add(_a, _b), do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Get the size of a tensor.

  ## Arguments
    - `tensor`: Input tensor
  """
  @spec size(ExTorch.Tensor.t()) :: tuple()
  def size(_a), do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Get the dtype of a tensor.

  ## Arguments
    - tensor (`ExTorch.Tensor`): Input tensor
  """
  @spec dtype(ExTorch.Tensor.t()) :: ExTorch.DType.dtype()
  def dtype(_a), do: :erlang.nif_error(:nif_not_loaded)

  def device(_a), do: :erlang.nif_error(:nif_not_loaded)

  def repr(_a), do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Returns a tensor filled with uninitialized data. The shape of the tensor is
  defined by the tuple argument `size`.

  ## Arguments
    - `size`: a tuple/list of integers defining the shape of the output tensor.

  ## Keyword args
    - dtype (`ExTorch.DType`, optional): the desired data type of returned tensor.
      **Default**: if `nil`, uses a global default (see `ExTorch.set_default_tensor_type`).

    - layout (`ExTorch.Layout`, optional): the desired layout of returned Tensor.
      **Default**: `:strided`.

    - device (`ExTorch.Device`, optional): the desired device of returned tensor.
        Default: if `nil`, uses the current device for the default tensor type
        (see `ExTorch.set_default_tensor_type`). `device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.

    - requires_grad (`boolean()`, optional): If autograd should record operations on the
        returned tensor. **Default**: `false`.

    - pin_memory (`bool`, optional): If set, returned tensor would be allocated in
        the pinned memory. Works only for CPU tensors. Default: `false`.

    - memory_format (`ExTorch.MemoryFormat`, optional): the desired memory format of
        returned Tensor. **Default**: `:contiguous`

  ## Examples
      iex> ExTorch.empty({2, 3})
      #Tensor<-6.2093e+29  4.5611e-41  0.0000e+00
      0.0000e+00  1.1673e-42  0.0000e+00
      [ CPUFloatType{2,3} ]>

      iex> ExTorch.empty({2, 3}, dtype: :int64, device: :cpu)
      #Tensor< 1.4023e+14  0.0000e+00  0.0000e+00
      1.0000e+00  7.0000e+00  1.4023e+14
      [ CPULongType{2,3} ]>
  """
  @spec empty(
          tuple() | [integer()],
          ExTorch.DType.dtype(),
          ExTorch.Layout.layout(),
          ExTorch.Device.device(),
          boolean(),
          boolean(),
          ExTorch.MemoryFormat.memory_format()
        ) :: ExTorch.Tensor.t()
  def empty(_size, _dtype, _layout, _device, _requires_grad, _pin_memory, _mem_fmt) do
    :erlang.nif_error(:nif_not_loaded)
  end

  @doc """
  Returns a tensor filled with the scalar value `0`, with the shape defined
  by the variable argument `size`.

  ## Arguments
    - `size`: a tuple/list of integers defining the shape of the output tensor.

  ## Keyword args
    - dtype (`ExTorch.DType`, optional): the desired data type of returned tensor.
      **Default**: if `nil`, uses a global default (see `ExTorch.set_default_tensor_type`).

    - layout (`ExTorch.Layout`, optional): the desired layout of returned Tensor.
      **Default**: `:strided`.

    - device (`ExTorch.Device`, optional): the desired device of returned tensor.
        Default: if `nil`, uses the current device for the default tensor type
        (see `ExTorch.set_default_tensor_type`). `device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.

    - requires_grad (`boolean()`, optional): If autograd should record operations on the
        returned tensor. **Default**: `false`.

    - pin_memory (`bool`, optional): If set, returned tensor would be allocated in
        the pinned memory. Works only for CPU tensors. Default: `false`.

    - memory_format (`ExTorch.MemoryFormat`, optional): the desired memory format of
        returned Tensor. **Default**: `:contiguous`

  ## Examples
      iex> ExTorch.zeros({2, 3})
      #Tensor< 0  0  0
      0  0  0
      [ CPUFloatType{2,3} ]>

      iex> ExTorch.zeros({2, 3}, dtype: :uint8, device: :cpu)
      #Tensor< 0  0  0
      0  0  0
      [ CPUByteType{2,3} ]>

      iex> ExTorch.zeros({5})
      #Tensor< 0
      0
      0
      0
      0
      [ CPUFloatType{5} ]>
  """
  @spec zeros(
          tuple() | [integer()],
          ExTorch.DType.dtype(),
          ExTorch.Layout.layout(),
          ExTorch.Device.device(),
          boolean(),
          boolean(),
          ExTorch.MemoryFormat.memory_format()
        ) :: ExTorch.Tensor.t()
  def zeros(_sizes, _dtype, _layout, _device, _requires_grad, _pin_memory, _mem_fmt) do
    :erlang.nif_error(:nif_not_loaded)
  end

  @doc """
  Returns a tensor filled with the scalar value `1`, with the shape defined
  by the variable argument `size`.

  ## Arguments
    - `size`: a tuple/list of integers defining the shape of the output tensor.

  ## Keyword args
    - dtype (`ExTorch.DType`, optional): the desired data type of returned tensor.
      **Default**: if `nil`, uses a global default (see `ExTorch.set_default_tensor_type`).

    - layout (`ExTorch.Layout`, optional): the desired layout of returned Tensor.
      **Default**: `:strided`.

    - device (`ExTorch.Device`, optional): the desired device of returned tensor.
        Default: if `nil`, uses the current device for the default tensor type
        (see `ExTorch.set_default_tensor_type`). `device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.

    - requires_grad (`boolean()`, optional): If autograd should record operations on the
        returned tensor. **Default**: `false`.

    - pin_memory (`bool`, optional): If set, returned tensor would be allocated in
        the pinned memory. Works only for CPU tensors. Default: `false`.

    - memory_format (`ExTorch.MemoryFormat`, optional): the desired memory format of
        returned Tensor. **Default**: `:contiguous`

  ## Examples
      iex> ExTorch.ones({2, 3})
      #Tensor< 1  1  1
      1  1  1
      [ CPUFloatType{2,3} ]>

      iex> ExTorch.ones({2, 3}, dtype: :uint8, device: :cpu)
      #Tensor< 1  1  1
      1  1  1
      [ CPUByteType{2,3} ]>

      iex> ExTorch.ones({5})
      #Tensor< 1
      1
      1
      1
      1
      [ CPUFloatType{5} ]>
  """
  @spec ones(
          tuple() | [integer()],
          ExTorch.DType.dtype(),
          ExTorch.Layout.layout(),
          ExTorch.Device.device(),
          boolean(),
          boolean(),
          ExTorch.MemoryFormat.memory_format()
        ) :: ExTorch.Tensor.t()
  def ones(_size, _dtype, _layout, _device, _requires_grad, _pin_memory, _mem_fmt) do
    :erlang.nif_error(:nif_not_loaded)
  end

  @doc """
  Returns a tensor filled with the scalar value `scalar`, with the shape defined
  by the variable argument `size`.

  ## Arguments
    - `size`: a tuple/list of integers defining the shape of the output tensor.
    - `scalar`: the value to fill the output tensor with.

  ## Keyword args
    - dtype (`ExTorch.DType`, optional): the desired data type of returned tensor.
      **Default**: if `nil`, uses a global default (see `ExTorch.set_default_tensor_type`).

    - layout (`ExTorch.Layout`, optional): the desired layout of returned Tensor.
      **Default**: `:strided`.

    - device (`ExTorch.Device`, optional): the desired device of returned tensor.
        Default: if `nil`, uses the current device for the default tensor type
        (see `ExTorch.set_default_tensor_type`). `device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.

    - requires_grad (`boolean()`, optional): If autograd should record operations on the
        returned tensor. **Default**: `false`.

    - pin_memory (`bool`, optional): If set, returned tensor would be allocated in
        the pinned memory. Works only for CPU tensors. Default: `false`.

    - memory_format (`ExTorch.MemoryFormat`, optional): the desired memory format of
        returned Tensor. **Default**: `:contiguous`

  ## Examples
      iex> ExTorch.full({2, 3}, 2)
      #Tensor< 2  2  2
      2  2  2
      [ CPUFloatType{2,3} ]>

      iex> ExTorch.full({2, 3}, 23, dtype: :uint8, device: :cpu)
      #Tensor< 23  23  23
      23  23  23
      [ CPUByteType{2,3} ]>

      iex> ExTorch.full({2, 3}, 3.1416)
      #Tensor< 3.1416  3.1416  3.1416
      3.1416  3.1416  3.1416
      [ CPUFloatType{5} ]>
  """
  @spec full(
          tuple() | [integer()],
          number(),
          ExTorch.DType.dtype(),
          ExTorch.Layout.layout(),
          ExTorch.Device.device(),
          boolean(),
          boolean(),
          ExTorch.MemoryFormat.memory_format()
        ) :: ExTorch.Tensor.t()
  def full(_sizes, _scalar, _dtype, _layout, _device, _requires_grad, _pin_memory, _mem_fmt) do
    :erlang.nif_error(:nif_not_loaded)
  end

  @spec eye(
    integer(),
    integer(),
    ExTorch.DType.dtype(),
    ExTorch.Layout.layout(),
    ExTorch.Device.device(),
    boolean(),
    boolean(),
    ExTorch.MemoryFormat.memory_format()
  ) :: ExTorch.Tensor.t()
  def eye(_n, _m, _dtype, _layout, _device, _requires_grad, _pin_memory, _mem_fmt) do
    :erlang.nif_error(:nif_not_loaded)
  end
end
