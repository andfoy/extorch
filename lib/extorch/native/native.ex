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

  @doc """
  Returns a tensor filled with random numbers from a uniform distribution
  on the interval $[0, 1)$

  The shape of the tensor is defined by the variable argument `size`.

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
      iex> ExTorch.rand({3, 3, 3})
      #Tensor<
      (1,.,.) =
        0.5997  0.3569  0.7639
        0.1939  0.0923  0.0942
        0.3355  0.3534  0.6490

      (2,.,.) =
        0.7250  0.5877  0.9215
        0.1583  0.7270  0.3289
        0.7083  0.1259  0.0050

      (3,.,.) =
        0.1731  0.9534  0.6758
        0.8523  0.0659  0.3623
        0.0747  0.6079  0.7227
      [ CPUFloatType{3,3,3} ]
      >

      iex> ExTorch.rand({2, 3}, dtype: :float64)
      #Tensor<
      0.6012  0.6164  0.2413
      0.9720  0.7804  0.4863
      [ CPUDoubleType{2,3} ]
      >
  """
  @spec rand(
    tuple() | [integer()],
    ExTorch.DType.dtype(),
    ExTorch.Layout.layout(),
    ExTorch.Device.device(),
    boolean(),
    boolean(),
    ExTorch.MemoryFormat.memory_format()
  ) :: ExTorch.Tensor.t()
  def rand(_size, _dtype, _layout, _device, _requires_grad, _pin_memory, _mem_fmt) do
    :erlang.nif_error(:nif_not_loaded)
  end

  @doc ~S"""
  Returns a tensor filled with random numbers from a normal distribution
  with mean `0` and variance `1` (also called the standard normal
  distribution).

  $$\text{{out}}_{{i}} \sim \mathcal{{N}}(0, 1)$$

  The shape of the tensor is defined by the variable argument :attr:`size`.

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
      iex> ExTorch.randn({3, 3, 5})
      #Tensor<
      (1,.,.) =
        0.0784 -0.3355 -0.0159 -0.0606 -1.2691
      -0.6146  0.2346  0.8563  0.8795  0.0645
      -1.9992  0.6692  0.2269  1.9263  0.1033

      (2,.,.) =
        0.2647  0.7078  0.0270 -1.1330 -0.4143
        1.2061 -1.1191  0.7465  0.2140  0.7406
        0.3587 -0.6102  0.3359 -0.4517 -0.5276

      (3,.,.) =
        1.7122  0.3814 -0.6218  0.8047 -0.6067
        0.1693  0.4957 -0.6139  0.7341  1.4272
        0.1630 -0.1142  0.8823  0.8026  1.3355
      [ CPUFloatType{3,3,5} ]

      iex> ExTorch.randn({3, 3, 5}, device: :cpu)
      #Tensor<
      (1,.,.) =
      -0.8990 -0.3449 -1.2916 -0.0318  0.7116
        0.9068 -0.3159 -0.6416 -1.8414 -0.1421
      -0.9251 -0.8209  0.0830 -2.5484  0.3731

      (2,.,.) =
        0.5975  0.0690 -0.2972 -0.0328 -0.2672
        1.3053  0.7803 -0.1992 -2.1078 -0.7520
        1.3048  0.6391  0.1137  2.0412  0.2380

      (3,.,.) =
      -1.1820 -1.9329 -0.3965 -0.0618 -1.1190
        0.7926 -1.8551  1.1356 -0.7451 -0.6003
        1.0266  0.5791  0.2724  0.6952 -3.1296
      [ CPUFloatType{3,3,5} ]
      >
  """
  @spec randn(
    tuple() | [integer()],
    ExTorch.DType.dtype(),
    ExTorch.Layout.layout(),
    ExTorch.Device.device(),
    boolean(),
    boolean(),
    ExTorch.MemoryFormat.memory_format()
  ) :: ExTorch.Tensor.t()
  def randn(_size, _dtype, _layout, _device, _requires_grad, _pin_memory, _mem_fmt) do
    :erlang.nif_error(:nif_not_loaded)
  end

  @doc """
  Returns a 2-D tensor with ones on the diagonal and zeros elsewhere.

  ## Arguments
    - `n`: the number of rows
    - `m`: the number of columns

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
      iex> ExTorch.eye(3, 3)
      #Tensor<
      1  0  0
      0  1  0
      0  0  1
      [ CPUFloatType{3,3} ]
      >

      iex> ExTorch.eye(4, 6, dtype: :uint8, device: :cpu)
      #Tensor<
      1  0  0  0  0  0
      0  1  0  0  0  0
      0  0  1  0  0  0
      0  0  0  1  0  0
      [ CPUByteType{4,6} ]
      >
  """
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

  @doc ~S"""
  Returns a 1-D tensor of size $\left\lceil \frac{\text{end} - \text{start}}{\text{step}} \right\rceil$
  with values from the interval ``[start, end)`` taken with common difference
  `step` beginning from `start`.

  Note that non-integer `step` is subject to floating point rounding errors when
  comparing against `end`; to avoid inconsistency, we advise adding a small epsilon
  to `end` in such cases.

  $$out_{i + 1} = out_i + step$$

  ## Arguments
    - `start`: the starting value for the set of points. Default: ``0``.
    - `end`: the ending value for the set of points.
    - `step`: the gap between each pair of adjacent points. Default: ``1``.

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
      # Single argument, end only
      iex> ExTorch.arange(5)
      #Tensor<
      0
      1
      2
      3
      4
      [ CPUFloatType{5} ]
      >

      # End only with options
      iex> ExTorch.arange(5, dtype: :uint8)
      #Tensor<
      0
      1
      2
      3
      4
      [ CPUByteType{5} ]

      # Start to end
      iex> ExTorch.arange(1, 7)
      #Tensor<
      1
      2
      3
      4
      5
      6
      [ CPUFloatType{6} ]
      >

      # Start to end with options
      iex> ExTorch.arange(1, 7, device: :cpu, dtype: :float64)
      #Tensor<
      1
      2
      3
      4
      5
      6
      [ CPUDoubleType{6} ]
      >

      # Start to end with step
      iex> ExTorch.arange(-1.3, 2.4, 0.5)
      #Tensor<
      -1.3000
      -0.8000
      -0.3000
      0.2000
      0.7000
      1.2000
      1.7000
      2.2000
      [ CPUFloatType{8} ]
      >

      # Start to end with step and options
      iex> ExTorch.arange(-1.3, 2.4, 0.5, dtype: :float64)
      #Tensor<
      -1.3000
      -0.8000
      -0.3000
      0.2000
      0.7000
      1.2000
      1.7000
      2.2000
      [ CPUDoubleType{8} ]
      >
  """
  @spec arange(
    number(),
    number(),
    number(),
    ExTorch.DType.dtype(),
    ExTorch.Layout.layout(),
    ExTorch.Device.device(),
    boolean(),
    boolean(),
    ExTorch.MemoryFormat.memory_format()
  ) :: ExTorch.Tensor.t()
  def arange(_start, _end, _step, _dtype, _layout, _device, _requires_grad, _pin_memory, _mem_fmt) do
    :erlang.nif_error(:nif_not_loaded)
  end

  @doc ~S"""
  Creates a one-dimensional tensor of size `steps` whose values are evenly
  spaced from `start` to `end`, inclusive. That is, the value are:

  $$(\text{start},
    \text{start} + \frac{\text{end} - \text{start}}{\text{steps} - 1},
    \ldots,
    \text{start} + (\text{steps} - 2) * \frac{\text{end} - \text{start}}{\text{steps} - 1},
    \text{end})$$

  ## Arguments
    - `start`: the starting value for the set of points.
    - `end`: the ending value for the set of points.
    - `steps`: size of the constructed tensor.

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
      # Returns a tensor with 10 evenly-spaced values between -2 and 10
      iex> ExTorch.linspace(-2, 10, 10)
      #Tensor<
      -2.0000
      -0.6667
       0.6667
       2.0000
       3.3333
       4.6667
       6.0000
       7.3333
       8.6667
      10.0000
      [ CPUFloatType{10} ]
      >

      # Returns a tensor with 10 evenly-spaced int32 values between -2 and 10
      iex> ExTorch.linspace(-2, 10, 10, dtype: :int32)
      #Tensor<
      -2
       0
       0
       1
       3
       4
       6
       7
       8
      10
      [ CPUIntType{10} ]
      >
  """
  @spec linspace(
    number(),
    number(),
    integer(),
    ExTorch.DType.dtype(),
    ExTorch.Layout.layout(),
    ExTorch.Device.device(),
    boolean(),
    boolean(),
    ExTorch.MemoryFormat.memory_format()
  ) :: ExTorch.Tensor.t()
  def linspace(_start, _end, _steps, _dtype, _layout, _device, _requires_grad, _pin_memory, _mem_fmt) do
    :erlang.nif_error(:nif_not_loaded)
  end

  @doc ~S"""
  Creates a one-dimensional tensor of size `steps` whose values are evenly
  spaced from ${{\text{{base}}}}^{{\text{{start}}}}$ to
  ${{\text{{base}}}}^{{\text{{end}}}}$, inclusive, on a logarithmic scale
  with base `base`. That is, the values are:

  $$(\text{base}^{\text{start}},
    \text{base}^{(\text{start} + \frac{\text{end} - \text{start}}{ \text{steps} - 1})},
    \ldots,
    \text{base}^{(\text{start} + (\text{steps} - 2) * \frac{\text{end} - \text{start}}{ \text{steps} - 1})},
    \text{base}^{\text{end}})$$

  ## Arguments
    - `start`: the starting value for the set of points.
    - `end`: the ending value for the set of points.
    - `steps`: size of the constructed tensor.
    - `base`: base of the logarithm function. Default: ``10.0``.

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
      iex> ExTorch.logspace(-10, 10, 5)
      #Tensor<
      1.0000e-10
      1.0000e-05
      1.0000e+00
      1.0000e+05
      1.0000e+10
      [ CPUFloatType{5} ]
      >

      iex> ExTorch.logspace(0.1, 1.0, 5)
      #Tensor<
        1.2589
        2.1135
        3.5481
        5.9566
       10.0000
      [ CPUFloatType{5} ]
      >

      iex> ExTorch.logspace(0.1, 1.0, 3, base: 2)
      #Tensor<
      1.0718
      1.4641
      2.0000
      [ CPUFloatType{3} ]
      >

      iex> ExTorch.logspace(0.1, 1.0, 3, base: 2, dtype: :float64)
      #Tensor<
      1.0718
      1.4641
      2.0000
      [ CPUDoubleType{3} ]
      >
  """
  @spec logspace(
    number(),
    number(),
    integer(),
    number(),
    ExTorch.DType.dtype(),
    ExTorch.Layout.layout(),
    ExTorch.Device.device(),
    boolean(),
    boolean(),
    ExTorch.MemoryFormat.memory_format()
  ) :: ExTorch.Tensor.t()
  def logspace(_start, _end, _steps, _base, _dtype, _layout, _device, _requires_grad, _pin_memory, _mem_fmt) do
    :erlang.nif_error(:nif_not_loaded)
  end
end
