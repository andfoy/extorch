defmodule ExTorch.Native.Tensor.Creation do
  @moduledoc false

  use ExTorch.Native.BindingDeclaration
  @doc_section :creation

  defbindings(:tensor_creation) do
    @doc """
    Returns a tensor filled with uninitialized data. The shape of the tensor is
    defined by the tuple argument `size`.

    ## Arguments
      - `size`: a tuple/list of integers defining the shape of the output tensor.

    ## Keyword args
      - dtype (`ExTorch.DType`, optional): the desired data type of returned tensor.
        **Default**: if `nil`, uses a global default (see `ExTorch.set_default_dtype`).

      - layout (`ExTorch.Layout`, optional): the desired layout of returned Tensor.
        **Default**: `:strided`.

      - device (`ExTorch.Device`, optional): the desired device of returned tensor.
          Default: if `nil`, uses the current device for the default tensor type
          (see `ExTorch.set_default_device`). `device` will be the CPU
          for CPU tensor types and the current CUDA device for CUDA tensor types.

      - requires_grad (`boolean()`, optional): If autograd should record operations on the
          returned tensor. **Default**: `false`.

      - pin_memory (`bool`, optional): If set, returned tensor would be allocated in
          the pinned memory. Works only for CPU tensors. Default: `false`.

      - memory_format (`ExTorch.MemoryFormat`, optional): the desired memory format of
          returned Tensor. **Default**: `:contiguous`

    ## Examples
        iex> ExTorch.empty({2, 3})
        #Tensor<
        [[ 6.7262e-44,  0.0000e+00,  7.2868e-44],
         [ 0.0000e+00, -2.7524e+24,  4.5880e-41]]
        [
          size: {2, 3},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

        iex> ExTorch.empty({2, 3}, dtype: :int64, device: :cuda)
        #Tensor<
        [[0, 0, 0],
         [0, 0, 0]]
        [
          size: {2, 3},
          dtype: :long,
          device: {:cuda, 0},
          requires_grad: false
        ]>

    """
    @spec empty(
            tuple() | [integer()],
            ExTorch.Tensor.Options.t()
          ) :: ExTorch.Tensor.t()
    defbinding(
      empty(
        size,
        opts \\ %ExTorch.Tensor.Options{}
      )
    )

    @doc """
    Returns a tensor filled with the scalar value `0`, with the shape defined
    by the variable argument `size`.

    ## Arguments
      - `size`: a tuple/list of integers defining the shape of the output tensor.

    ## Keyword args
      - dtype (`ExTorch.DType`, optional): the desired data type of returned tensor.
        **Default**: if `nil`, uses a global default (see `ExTorch.set_default_dtype`).

      - layout (`ExTorch.Layout`, optional): the desired layout of returned Tensor.
        **Default**: `:strided`.

      - device (`ExTorch.Device`, optional): the desired device of returned tensor.
          Default: if `nil`, uses the current device for the default tensor type
          (see `ExTorch.set_default_device`). `device` will be the CPU
          for CPU tensor types and the current CUDA device for CUDA tensor types.

      - requires_grad (`boolean()`, optional): If autograd should record operations on the
          returned tensor. **Default**: `false`.

      - pin_memory (`bool`, optional): If set, returned tensor would be allocated in
          the pinned memory. Works only for CPU tensors. Default: `false`.

      - memory_format (`ExTorch.MemoryFormat`, optional): the desired memory format of
          returned Tensor. **Default**: `:contiguous`

    ## Examples
        iex> ExTorch.zeros({2, 3})
        #Tensor<
        [[0., 0., 0.],
         [0., 0., 0.]]
        [
          size: {2, 3},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>


        iex> ExTorch.zeros({2, 3}, dtype: :uint8, device: :cuda)
        #Tensor<
        [[0, 0, 0],
         [0, 0, 0]]
        [
          size: {2, 3},
          dtype: :byte,
          device: {:cuda, 0},
          requires_grad: false
        ]>

    """
    @spec zeros(
            tuple() | [integer()],
            ExTorch.Tensor.Options.t()
          ) :: ExTorch.Tensor.t()
    defbinding(
      zeros(
        size,
        opts \\ %ExTorch.Tensor.Options{}
      )
    )

    @doc """
    Returns a tensor filled with the scalar value `1`, with the shape defined
    by the variable argument `size`.

    ## Arguments
      - `size`: a tuple/list of integers defining the shape of the output tensor.

    ## Keyword args
      - dtype (`ExTorch.DType`, optional): the desired data type of returned tensor.
        **Default**: if `nil`, uses a global default (see `ExTorch.set_default_dtype`).

      - layout (`ExTorch.Layout`, optional): the desired layout of returned Tensor.
        **Default**: `:strided`.

      - device (`ExTorch.Device`, optional): the desired device of returned tensor.
          Default: if `nil`, uses the current device for the default tensor type
          (see `ExTorch.set_default_device`). `device` will be the CPU
          for CPU tensor types and the current CUDA device for CUDA tensor types.

      - requires_grad (`boolean()`, optional): If autograd should record operations on the
          returned tensor. **Default**: `false`.

      - pin_memory (`bool`, optional): If set, returned tensor would be allocated in
          the pinned memory. Works only for CPU tensors. Default: `false`.

      - memory_format (`ExTorch.MemoryFormat`, optional): the desired memory format of
          returned Tensor. **Default**: `:contiguous`

    ## Examples
        iex> ExTorch.ones({2, 3})
        #Tensor<
        [[1., 1., 1.],
         [1., 1., 1.]]
        [
          size: {2, 3},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

        iex> ExTorch.ones({2, 3}, dtype: :uint8, device: :cuda)
        #Tensor<
        [[1, 1, 1],
         [1, 1, 1]]
        [
          size: {2, 3},
          dtype: :byte,
          device: {:cuda, 0},
          requires_grad: false
        ]>
    """
    @spec ones(
            tuple() | [integer()],
            ExTorch.Tensor.Options.t()
          ) :: ExTorch.Tensor.t()
    defbinding(
      ones(
        size,
        opts \\ %ExTorch.Tensor.Options{}
      )
    )

    @doc """
    Returns a tensor filled with random numbers from a uniform distribution
    on the interval $[0, 1)$

    The shape of the tensor is defined by the variable argument `size`.

    ## Arguments
      - `size`: a tuple/list of integers defining the shape of the output tensor.

    ## Keyword args
      - dtype (`ExTorch.DType`, optional): the desired data type of returned tensor.
        **Default**: if `nil`, uses a global default (see `ExTorch.set_default_dtype`).

      - layout (`ExTorch.Layout`, optional): the desired layout of returned Tensor.
        **Default**: `:strided`.

      - device (`ExTorch.Device`, optional): the desired device of returned tensor.
          Default: if `nil`, uses the current device for the default tensor type
          (see `ExTorch.set_default_device`). `device` will be the CPU
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
        [[[0.4099, 0.8473, 0.6221],
          [0.9906, 0.3174, 0.9849],
          [0.6988, 0.1157, 0.9424]],

         [[0.0550, 0.9723, 0.4380],
          [0.9304, 0.2973, 0.4920],
          [0.1860, 0.9460, 0.2602]],

         [[0.9208, 0.9713, 0.8194],
          [0.8109, 0.1395, 0.1245],
          [0.5742, 0.5222, 0.0937]]]
        [
          size: {3, 3, 3},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

        iex> ExTorch.rand({2, 3}, dtype: :float32, device: :cuda)
        #Tensor<
        [[0.1583, 0.5184, 0.6711],
         [0.3829, 0.3248, 0.3524]]
        [
          size: {2, 3},
          dtype: :float,
          device: {:cuda, 0},
          requires_grad: false
        ]>
    """
    @spec rand(
            tuple() | [integer()],
            ExTorch.Tensor.Options.t()
          ) :: ExTorch.Tensor.t()
    defbinding(
      rand(
        size,
        opts \\ %ExTorch.Tensor.Options{}
      )
    )

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
        **Default**: if `nil`, uses a global default (see `ExTorch.set_default_dtype`).

      - layout (`ExTorch.Layout`, optional): the desired layout of returned Tensor.
        **Default**: `:strided`.

      - device (`ExTorch.Device`, optional): the desired device of returned tensor.
          Default: if `nil`, uses the current device for the default tensor type
          (see `ExTorch.set_default_device`). `device` will be the CPU
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
        [[[ 0.6246, -0.4914,  1.1007, -0.0740, -1.6833],
          [-0.3883,  1.2653, -0.7250,  0.4994, -0.0219],
          [-1.3880,  1.8336, -1.7369, -0.2781, -0.0703]],

         [[ 0.2841,  0.7564, -0.3294,  0.1375,  2.0717],
          [-0.6085, -0.8361,  0.5009,  1.5529,  0.5856],
          [-0.3905, -0.3704,  1.1392,  0.3159, -0.5587]],

         [[ 0.8050, -0.0064, -0.6925, -0.0121, -1.2824],
          [-1.7309, -1.4089, -1.0207,  0.2222, -0.5027],
          [-0.4363, -0.1095,  1.3950, -0.4580,  0.2475]]]
        [
          size: {3, 3, 5},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

        iex> ExTorch.randn({3, 3, 5}, device: :cuda)
        #Tensor<
        [[[ 3.5948e-01,  1.9308e-01, -1.0206e-01, -8.1509e-01, -1.6322e+00],
          [ 5.3390e-02, -1.2340e-01, -4.0909e-01,  3.5126e-01, -1.4023e-01],
          [ 6.5496e-01,  1.4283e+00, -1.2375e+00,  1.3729e+00,  4.2116e-01]],

         [[ 1.4638e+00,  6.9129e-03, -1.4147e+00, -1.8253e+00, -1.9235e+00],
          [-1.3941e-01, -7.3455e-01,  3.7658e-01, -1.0569e-01,  6.8978e-01],
          [ 3.7640e-01, -3.5241e-01, -1.1376e-01, -5.2477e-01, -1.6157e-01]],

         [[-2.8951e-01, -1.5665e+00,  3.4778e-01, -2.1329e+00, -1.0400e+00],
          [ 4.7831e-04,  1.2714e+00,  1.6693e+00, -2.1787e+00,  4.4486e-01],
          [-3.2052e-01,  2.3278e+00,  6.2929e-01,  2.5321e-01, -1.4433e+00]]]
        [
          size: {3, 3, 5},
          dtype: :float,
          device: {:cuda, 0},
          requires_grad: false
        ]>
    """
    @spec randn(
            tuple() | [integer()],
            ExTorch.Tensor.Options.t()
          ) :: ExTorch.Tensor.t()
    defbinding(
      randn(
        size,
        opts \\ %ExTorch.Tensor.Options{}
      )
    )

    @doc """
    Returns a tensor filled with random integers generated uniformly
    between `low` (inclusive) and `high` (exclusive).

    The shape of the tensor is defined by the variable argument `size`.

    ## Arguments
      - `low`: Lowest integer to be drawn from the distribution. Default: `0`.
      - `high`: One above the highest integer to be drawn from the distribution.
      - `size`: a tuple/list of integers defining the shape of the output tensor.

    ## Keyword args
      - dtype (`ExTorch.DType`, optional): the desired data type of returned tensor.
        **Default**: if `nil`, uses a global default (see `ExTorch.set_default_dtype`).

      - layout (`ExTorch.Layout`, optional): the desired layout of returned Tensor.
        **Default**: `:strided`.

      - device (`ExTorch.Device`, optional): the desired device of returned tensor.
          Default: if `nil`, uses the current device for the default tensor type
          (see `ExTorch.set_default_device`). `device` will be the CPU
          for CPU tensor types and the current CUDA device for CUDA tensor types.

      - requires_grad (`boolean()`, optional): If autograd should record operations on the
          returned tensor. **Default**: `false`.

      - pin_memory (`bool`, optional): If set, returned tensor would be allocated in
          the pinned memory. Works only for CPU tensors. Default: `false`.

      - memory_format (`ExTorch.MemoryFormat`, optional): the desired memory format of
          returned Tensor. **Default**: `:contiguous`

    ## Examples
        # Sample numbers between 0 and 3
        iex> ExTorch.randint(3, {3, 3, 4})
        #Tensor<
        [[[0., 2., 0., 0.],
          [1., 2., 0., 2.],
          [2., 2., 1., 2.]],

         [[2., 1., 1., 1.],
          [2., 1., 0., 1.],
          [1., 1., 0., 0.]],

         [[1., 1., 1., 1.],
          [2., 2., 0., 1.],
          [1., 2., 1., 1.]]]
        [
          size: {3, 3, 4},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

        # Sample numbers between 0 and 3 of type int64
        iex> ExTorch.randint(3, {3, 3, 4}, dtype: :int64)
        #Tensor<
        [[[0, 0, 0, 2],
          [1, 1, 0, 0],
          [1, 0, 1, 1]],

         [[0, 1, 1, 1],
          [2, 0, 2, 0],
          [2, 1, 0, 2]],

         [[1, 0, 2, 1],
          [2, 2, 1, 0],
          [0, 0, 0, 2]]]
        [
          size: {3, 3, 4},
          dtype: :long,
          device: :cpu,
          requires_grad: false
        ]>

        # Sample numbers between -2 and 4
        iex> ExTorch.randint(-2, 3, {2, 2, 4})
        #Tensor<
        [[[ 2.,  1.,  0., -1.],
          [ 2.,  2., -2.,  2.]],

         [[-1., -1.,  1., -1.],
          [ 2., -1.,  1., -1.]]]
        [
          size: {2, 2, 4},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

        # Sample numbers between -2 and 4 on gpu
        iex> ExTorch.randint(-2, 3, {2, 2, 4}, device: :cuda)
        #Tensor<
        [[[-2.,  2.,  0., -2.],
          [ 0.,  0.,  0., -2.]],

         [[ 0.,  0., -2.,  0.],
          [ 0.,  1., -2.,  1.]]]
        [
          size: {2, 2, 4},
          dtype: :float,
          device: {:cuda, 0},
          requires_grad: false
        ]>
    """
    @spec randint(
            integer(),
            integer(),
            tuple() | [integer()],
            ExTorch.Tensor.Options.t()
          ) :: ExTorch.Tensor.t()
    defbinding(
      randint(
        low \\ 0,
        high,
        size,
        opts \\ %ExTorch.Tensor.Options{}
      )
    )

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
        **Default**: if `nil`, uses a global default (see `ExTorch.set_default_dtype`).

      - layout (`ExTorch.Layout`, optional): the desired layout of returned Tensor.
        **Default**: `:strided`.

      - device (`ExTorch.Device`, optional): the desired device of returned tensor.
          Default: if `nil`, uses the current device for the default tensor type
          (see `ExTorch.set_default_device`). `device` will be the CPU
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
        [0., 1., 2., 3., 4.]
        [
          size: {5},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

        # End only with options
        iex> ExTorch.arange(5, dtype: :uint8)
        #Tensor<
        [0, 1, 2, 3, 4]
        [
          size: {5},
          dtype: :byte,
          device: :cpu,
          requires_grad: false
        ]>

        # Start to end
        iex> ExTorch.arange(1, 7)
        #Tensor<
        [1., 2., 3., 4., 5., 6.]
        [
          size: {6},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

        # Start to end with options
        iex> ExTorch.arange(1, 7, device: :cuda, dtype: :float16)
        #Tensor<
        [1., 2., 3., 4., 5., 6.]
        [
          size: {6},
          dtype: :half,
          device: {:cuda, 0},
          requires_grad: false
        ]>

        # Start to end with step
        iex> ExTorch.arange(-1.3, 2.4, 0.5)
        #Tensor<
        [-1.3000, -0.8000, -0.3000,  0.2000,  0.7000,  1.2000,  1.7000,  2.2000]
        [
          size: {8},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>


        # Start to end with step and options
        iex> ExTorch.arange(-1.3, 2.4, 0.5, dtype: :float64)
        #Tensor<
        [-1.3000, -0.8000, -0.3000,  0.2000,  0.7000,  1.2000,  1.7000,  2.2000]
        [
          size: {8},
          dtype: :double,
          device: :cpu,
          requires_grad: false
        ]>
    """
    @spec arange(
            number(),
            number(),
            number(),
            ExTorch.Tensor.Options.t()
          ) :: ExTorch.Tensor.t()
    defbinding(
      arange(
        start \\ 0,
        end_bound,
        step \\ 1,
        opts \\ %ExTorch.Tensor.Options{}
      )
    )

    @doc """
    Returns a 2-D tensor with ones on the diagonal and zeros elsewhere.

    ## Arguments
      - `n`: the number of rows
      - `m`: the number of columns

    ## Keyword args
      - dtype (`ExTorch.DType`, optional): the desired data type of returned tensor.
        **Default**: if `nil`, uses a global default (see `ExTorch.set_default_dtype`).

      - layout (`ExTorch.Layout`, optional): the desired layout of returned Tensor.
        **Default**: `:strided`.

      - device (`ExTorch.Device`, optional): the desired device of returned tensor.
          Default: if `nil`, uses the current device for the default tensor type
          (see `ExTorch.set_default_device`). `device` will be the CPU
          for CPU tensor types and the current CUDA device for CUDA tensor types.

      - requires_grad (`boolean()`, optional): If autograd should record operations on the
          returned tensor. **Default**: `false`.

      - pin_memory (`bool`, optional): If set, returned tensor would be allocated in
          the pinned memory. Works only for CPU tensors. Default: `false`.

      - memory_format (`ExTorch.MemoryFormat`, optional): the desired memory format of
          returned Tensor. **Default**: `:contiguous`

    ## Examples
        iex> ExTorch.eye(3)
        #Tensor<
        [[1., 0., 0.],
         [0., 1., 0.],
         [0., 0., 1.]]
        [
          size: {3, 3},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

        iex> ExTorch.eye(3, 3)
        #Tensor<
        [[1., 0., 0.],
         [0., 1., 0.],
         [0., 0., 1.]]
        [
          size: {3, 3},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

        iex> ExTorch.eye(4, 6, dtype: :uint8, device: :cuda)
        #Tensor<
        [[1, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0],
         [0, 0, 0, 1, 0, 0]]
        [
          size: {4, 6},
          dtype: :byte,
          device: {:cuda, 0},
          requires_grad: false
        ]>

    """
    @spec eye(
            integer(),
            integer(),
            ExTorch.Tensor.Options.t()
          ) :: ExTorch.Tensor.t()
    defbinding(
      eye(
        n,
        m \\ n,
        opts \\ %ExTorch.Tensor.Options{}
      )
    )

    @doc """
    Returns a tensor filled with the scalar value `scalar`, with the shape defined
    by the variable argument `size`.

    ## Arguments
      - `size`: a tuple/list of integers defining the shape of the output tensor.
      - `scalar`: the value to fill the output tensor with.

    ## Keyword args
      - dtype (`ExTorch.DType`, optional): the desired data type of returned tensor.
        **Default**: if `nil`, uses a global default (see `ExTorch.set_default_dtype`).

      - layout (`ExTorch.Layout`, optional): the desired layout of returned Tensor.
        **Default**: `:strided`.

      - device (`ExTorch.Device`, optional): the desired device of returned tensor.
          Default: if `nil`, uses the current device for the default tensor type
          (see `ExTorch.set_default_device`). `device` will be the CPU
          for CPU tensor types and the current CUDA device for CUDA tensor types.

      - requires_grad (`boolean()`, optional): If autograd should record operations on the
          returned tensor. **Default**: `false`.

      - pin_memory (`bool`, optional): If set, returned tensor would be allocated in
          the pinned memory. Works only for CPU tensors. Default: `false`.

      - memory_format (`ExTorch.MemoryFormat`, optional): the desired memory format of
          returned Tensor. **Default**: `:contiguous`

    ## Examples
        iex> ExTorch.full({2, 3}, 2)
        #Tensor<
        [[2., 2., 2.],
         [2., 2., 2.]]
        [
          size: {2, 3},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

        iex> ExTorch.full({2, 3}, 23, dtype: :uint8, device: :cuda)
        #Tensor<
        [[2, 2, 2],
         [2, 2, 2]]
        [
          size: {2, 3},
          dtype: :byte,
          device: {:cuda, 0},
          requires_grad: false
        ]>

    """
    @spec full(
            tuple() | [integer()],
            number() | ExTorch.Complex.t() | :nan | :inf | :ninf,
            ExTorch.Tensor.Options.t()
          ) :: ExTorch.Tensor.t()
    defbinding(
      full(
        size,
        scalar,
        opts \\ %ExTorch.Tensor.Options{}
      )
    )

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
        **Default**: if `nil`, uses a global default (see `ExTorch.set_default_dtype`).

      - layout (`ExTorch.Layout`, optional): the desired layout of returned Tensor.
        **Default**: `:strided`.

      - device (`ExTorch.Device`, optional): the desired device of returned tensor.
          Default: if `nil`, uses the current device for the default tensor type
          (see `ExTorch.set_default_device`). `device` will be the CPU
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
        [-2.0000, -0.6667,  0.6667,  2.0000,  3.3333,  4.6667,  6.0000,  7.3333,
          8.6667, 10.0000]
        [
          size: {10},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

        # Returns a tensor with 10 evenly-spaced int32 values between -2 and 10
        iex> ExTorch.linspace(-2, 10, 10, dtype: :int32)
        #Tensor<
        [-2,  0,  0,  1,  3,  4,  6,  7,  8, 10]
        [
          size: {10},
          dtype: :int,
          device: :cpu,
          requires_grad: false
        ]>
    """
    @spec linspace(
            number(),
            number(),
            integer(),
            ExTorch.Tensor.Options.t()
          ) :: ExTorch.Tensor.t()
    defbinding(
      linspace(
        start,
        end_bound,
        steps,
        opts \\ %ExTorch.Tensor.Options{}
      )
    )

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
        **Default**: if `nil`, uses a global default (see `ExTorch.set_default_dtype`).

      - layout (`ExTorch.Layout`, optional): the desired layout of returned Tensor.
        **Default**: `:strided`.

      - device (`ExTorch.Device`, optional): the desired device of returned tensor.
          Default: if `nil`, uses the current device for the default tensor type
          (see `ExTorch.set_default_device`). `device` will be the CPU
          for CPU tensor types and the current CUDA device for CUDA tensor types.

      - requires_grad (`boolean()`, optional): If autograd should record operations on the
          returned tensor. **Default**: `false`.

      - pin_memory (`bool`, optional): If set, returned tensor would be allocated in
          the pinned memory. Works only for CPU tensors. Default: `false`.

      - memory_format (`ExTorch.MemoryFormat`, optional): the desired memory format of
          returned Tensor. **Default**: `:contiguous`

    ## Examples
        # Returns a tensor containing five logarithmic-spaced values between -10 and 10
        iex> ExTorch.logspace(-10, 10, 5)
        #Tensor<
        [1.0000e-10, 1.0000e-05, 1.0000e+00, 1.0000e+05, 1.0000e+10]
        [
          size: {5},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

        # Returns a tensor containing five logarithmic-spaced values between 0.1 and 1.0
        iex> ExTorch.logspace(0.1, 1.0, 5)
        #Tensor<
        [ 1.2589,  2.1135,  3.5481,  5.9566, 10.0000]
        [
          size: {5},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

        # Returns a tensor containing three logarithmic-spaced (base 2) values between 0.1 and 1.0
        iex> ExTorch.logspace(0.1, 1.0, 3, base: 2)
        #Tensor<
        [1.0718, 1.4641, 2.0000]
        [
          size: {3},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

        # Returns a float64 tensor containing three logarithmic-spaced (base 2) values between 0.1 and 1.0
        iex> ExTorch.logspace(0.1, 1.0, 3, base: 2, dtype: :float64)
        #Tensor<
        [1.0718, 1.4641, 2.0000]
        [
          size: {3},
          dtype: :double,
          device: :cpu,
          requires_grad: false
        ]>
    """
    @spec logspace(
            number(),
            number(),
            integer(),
            number(),
            ExTorch.Tensor.Options.t()
          ) :: ExTorch.Tensor.t()
    defbinding(
      logspace(
        start,
        end_bound,
        steps,
        base \\ 10,
        opts \\ %ExTorch.Tensor.Options{}
      )
    )

    @doc """
    Constructs a tensor with data.

    ## Arguments
      - `list`: Initial data for the tensor. Can be a list, tuple or number.

    ## Keyword args
      - dtype (`ExTorch.DType`, optional): the desired data type of returned tensor.
        **Default**: if `nil`, uses a global default (see `ExTorch.set_default_dtype`).

      - layout (`ExTorch.Layout`, optional): the desired layout of returned Tensor.
        **Default**: `:strided`.

      - device (`ExTorch.Device`, optional): the desired device of returned tensor.
          Default: if `nil`, uses the current device for the default tensor type
          (see `ExTorch.set_default_device`). `device` will be the CPU
          for CPU tensor types and the current CUDA device for CUDA tensor types.

      - requires_grad (`boolean()`, optional): If autograd should record operations on the
          returned tensor. **Default**: `false`.

      - pin_memory (`bool`, optional): If set, returned tensor would be allocated in
          the pinned memory. Works only for CPU tensors. Default: `false`.

      - memory_format (`ExTorch.MemoryFormat`, optional): the desired memory format of
          returned Tensor. **Default**: `:contiguous`

    ## Examples
        iex> ExTorch.tensor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]])
        #Tensor<
        [[0.1000, 1.2000],
         [2.2000, 3.1000],
         [4.9000, 5.2000]]
        [
          size: {3, 2},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

        # Type inference
        iex> ExTorch.tensor([0, 1])
        #Tensor<
        [0, 1]
        [size: {2}, dtype: :byte, device: :cpu, requires_grad: false]>

        iex> ExTorch.tensor([[0.11111, 0.222222, 0.3333333]], dtype: :float64)
        #Tensor<
        [[0.1111, 0.2222, 0.3333]]
        [
          size: {1, 3},
          dtype: :double,
          device: :cpu,
          requires_grad: false
        ]>
    """
    @spec tensor(
            list() | tuple() | number() | boolean() | ExTorch.Complex.t() | :nan | :inf | :ninf,
            ExTorch.Tensor.Options.t()
          ) :: ExTorch.Tensor.t()
    defbinding(
      tensor(
        list,
        opts \\ %ExTorch.Tensor.Options{dtype: :auto}
      ),
      list: ExTorch.Utils.to_list_wrapper(list),
      opts:
        case opts.dtype do
          :auto ->
            struct(opts, dtype: list.dtype)

          _ ->
            opts
        end
    )

    @doc """
    Returns an uninitialized tensor, with the same size as `input`.

    `ExTorch.empty_like(input)` is equivalent to
    `ExTorch.empty(input.size, dtype: input.dtype, layout: input.layout, device: input.device)`

    ## Arguments
      - `input`: The input tensor (`ExTorch.Tensor`)

    ## Keyword args
      - dtype (`ExTorch.DType`, optional): the desired data type of returned tensor.
        **Default**: `auto`. If `auto`, it will use the same data type as the input.
        If `nil`, it will use a global default (see `ExTorch.set_default_dtype`).

      - layout (`ExTorch.Layout`, optional): the desired layout of returned Tensor.
        **Default**: `nil`. If `nil`, it will use the same layout as the input.

      - device (`ExTorch.Device`, optional): the desired device of returned tensor.
        Default: `auto`. If `auto`, it will use the same device as the input.
        If `nil`, it will use the current device for the default tensor type
        (see `ExTorch.set_default_device`). `device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.

      - requires_grad (`boolean()`, optional): If autograd should record operations on the
          returned tensor. **Default**: `false`.

      - pin_memory (`bool`, optional): If set, returned tensor would be allocated in
          the pinned memory. Works only for CPU tensors. Default: `false`.

      - memory_format (`ExTorch.MemoryFormat`, optional): the desired memory format of
          returned Tensor. **Default**: `:preserve`. If `preserve`, it will use the same
          memory format as the input.

    ## Examples
        # Create an empty tensor from another
        iex> a = ExTorch.empty({4, 5})
        iex> ExTorch.empty_like(a)
        #Tensor<
        [[ 8.3624e+06,  4.5880e-41, -2.8874e+24,  4.5880e-41,  2.5223e-44],
         [ 0.0000e+00,  2.5223e-44,  0.0000e+00,  5.1482e+22,  1.6816e-43],
         [ 9.8511e-43,  0.0000e+00,  8.3624e+06,  4.5880e-41, -3.1780e+24],
         [ 4.5880e-41,  2.5223e-44,  0.0000e+00,  2.5223e-44,  0.0000e+00]]
        [
          size: {4, 5},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

        # Create an empty tensor in GPU from a CPU one
        iex> a = ExTorch.empty({3, 3})
        iex> ExTorch.empty_like(a, device: :cuda)
        #Tensor<
        [[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]]
        [
          size: {3, 3},
          dtype: :float,
          device: {:cuda, 0},
          requires_grad: false
        ]>

    """
    @spec empty_like(ExTorch.Tensor.t(), ExTorch.Tensor.Options.t()) :: ExTorch.Tensor.t()
    defbinding(
      empty_like(
        input,
        opts \\ %ExTorch.Tensor.Options{
          dtype: :auto,
          device: :auto,
          layout: nil,
          memory_format: :preserve
        }
      ),
      opts: ExTorch.Tensor.Options.merge_input(input, opts)
    )

    @doc """
    Returns a tensor filled with random numbers from a uniform distribution
    on the interval $[0, 1)$, with the same size as `input`.

    `ExTorch.rand_like(input)` is equivalent to
    `ExTorch.rand(input.size, dtype: input.dtype, layout: input.layout, device: input.device)`

    ## Arguments
      - `input`: The input tensor (`ExTorch.Tensor`)

    ## Keyword args
      - dtype (`ExTorch.DType`, optional): the desired data type of returned tensor.
        **Default**: `auto`. If `auto`, it will use the same data type as the input.
        If `nil`, it will use a global default (see `ExTorch.set_default_dtype`).

      - layout (`ExTorch.Layout`, optional): the desired layout of returned Tensor.
        **Default**: `nil`. If `nil`, it will use the same layout as the input.

      - device (`ExTorch.Device`, optional): the desired device of returned tensor.
        Default: `auto`. If `auto`, it will use the same device as the input.
        If `nil`, it will use the current device for the default tensor type
        (see `ExTorch.set_default_device`). `device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.

      - requires_grad (`boolean()`, optional): If autograd should record operations on the
          returned tensor. **Default**: `false`.

      - pin_memory (`bool`, optional): If set, returned tensor would be allocated in
          the pinned memory. Works only for CPU tensors. Default: `false`.

      - memory_format (`ExTorch.MemoryFormat`, optional): the desired memory format of
          returned Tensor. **Default**: `:preserve`. If `preserve`, it will use the same
          memory format as the input.

    ## Examples
        # Derive a new float64 tensor from another one
        iex> a = ExTorch.empty({3, 2, 2}, dtype: :float64)
        iex> ExTorch.rand_like(a)
        #Tensor<
        [[[0.6495, 0.9480],
          [0.3083, 0.7135]],

         [[0.5482, 0.3676],
          [0.2825, 0.1806]],

         [[0.4742, 0.8673],
          [0.4542, 0.4239]]]
        [
          size: {3, 2, 2},
          dtype: :double,
          device: :cpu,
          requires_grad: false
        ]>

        # Derive a GPU tensor from a CPU one
        iex> b = ExTorch.ones({2, 3}, dtype: :complex64)
        iex> ExTorch.rand_like(b, device: :cuda)
        #Tensor<
        [[0.1554+0.6794j, 0.5356+0.2049j, 0.7555+0.3877j],
         [0.0148+0.0772j, 0.8368+0.3802j, 0.6820+0.1727j]]
        [
          size: {2, 3},
          dtype: :complex_float,
          device: {:cuda, 0},
          requires_grad: false
        ]>

    """
    @spec rand_like(ExTorch.Tensor.t(), ExTorch.Tensor.Options.t()) :: ExTorch.Tensor.t()
    defbinding(
      rand_like(
        input,
        opts \\ %ExTorch.Tensor.Options{
          dtype: :auto,
          device: :auto,
          layout: nil,
          memory_format: :preserve
        }
      ),
      opts: ExTorch.Tensor.Options.merge_input(input, opts)
    )

    @doc """
    Returns a tensor filled with random numbers from a normal distribution
    with mean `0` and variance `1` (also called the standard normal
    distribution), with the same size as `input`.

    `ExTorch.randn_like(input)` is equivalent to
    `ExTorch.randn(input.size, dtype: input.dtype, layout: input.layout, device: input.device)`

    ## Arguments
      - `input`: The input tensor (`ExTorch.Tensor`)

    ## Keyword args
      - dtype (`ExTorch.DType`, optional): the desired data type of returned tensor.
        **Default**: `auto`. If `auto`, it will use the same data type as the input.
        If `nil`, it will use a global default (see `ExTorch.set_default_dtype`).

      - layout (`ExTorch.Layout`, optional): the desired layout of returned Tensor.
        **Default**: `nil`. If `nil`, it will use the same layout as the input.

      - device (`ExTorch.Device`, optional): the desired device of returned tensor.
        Default: `auto`. If `auto`, it will use the same device as the input.
        If `nil`, it will use the current device for the default tensor type
        (see `ExTorch.set_default_device`). `device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.

      - requires_grad (`boolean()`, optional): If autograd should record operations on the
          returned tensor. **Default**: `false`.

      - pin_memory (`bool`, optional): If set, returned tensor would be allocated in
          the pinned memory. Works only for CPU tensors. Default: `false`.

      - memory_format (`ExTorch.MemoryFormat`, optional): the desired memory format of
          returned Tensor. **Default**: `:preserve`. If `preserve`, it will use the same
          memory format as the input.

    ## Examples
        # Derive a new float64 tensor from another one
        iex> a = ExTorch.empty({3, 2, 2}, dtype: :float64)
        iex> ExTorch.rand_like(a)
        #Tensor<
        [[[0.6394, 0.0540],
          [0.8050, 0.6426]],

         [[0.7196, 0.6789],
          [0.2813, 0.4029]],

         [[0.0898, 0.4235],
          [0.3301, 0.2744]]]
        [
          size: {3, 2, 2},
          dtype: :double,
          device: :cpu,
          requires_grad: false
        ]>

        # Derive a new cuda float64 tensor from another one
        iex> b = ExTorch.empty({3, 2}, device: :cuda)
        iex> ExTorch.rand_like(b, dtype: :float64)
        #Tensor<
        [[0.2639, 0.7628],
         [0.5935, 0.4772],
         [0.0176, 0.2496]]
        [
          size: {3, 2},
          dtype: :double,
          device: {:cuda, 0},
          requires_grad: false
        ]>

    """
    @spec randn_like(ExTorch.Tensor.t(), ExTorch.Tensor.Options.t()) :: ExTorch.Tensor.t()
    defbinding(
      randn_like(
        input,
        opts \\ %ExTorch.Tensor.Options{
          dtype: :auto,
          device: :auto,
          layout: nil,
          memory_format: :preserve
        }
      ),
      opts: ExTorch.Tensor.Options.merge_input(input, opts)
    )

    @doc """
    Returns a tensor filled with random integers generated uniformly
    between `low` (inclusive) and `high` (exclusive),
    with the same size as `input`.

    `ExTorch.randint_like(input, low, high)` is equivalent to
    `ExTorch.randint(low, high, input.size, dtype: input.dtype, layout: input.layout, device: input.device)`

    ## Arguments
      - `input`: The input tensor (`ExTorch.Tensor`)
      - `low`: Lowest integer to be drawn from the distribution. Default: `0`.
      - `high`: One above the highest integer to be drawn from the distribution.

    ## Keyword args
      - dtype (`ExTorch.DType`, optional): the desired data type of returned tensor.
        **Default**: `auto`. If `auto`, it will use the same data type as the input.
        If `nil`, it will use a global default (see `ExTorch.set_default_dtype`).

      - layout (`ExTorch.Layout`, optional): the desired layout of returned Tensor.
        **Default**: `nil`. If `nil`, it will use the same layout as the input.

      - device (`ExTorch.Device`, optional): the desired device of returned tensor.
        Default: `auto`. If `auto`, it will use the same device as the input.
        If `nil`, it will use the current device for the default tensor type
        (see `ExTorch.set_default_device`). `device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.

      - requires_grad (`boolean()`, optional): If autograd should record operations on the
          returned tensor. **Default**: `false`.

      - pin_memory (`bool`, optional): If set, returned tensor would be allocated in
          the pinned memory. Works only for CPU tensors. Default: `false`.

      - memory_format (`ExTorch.MemoryFormat`, optional): the desired memory format of
          returned Tensor. **Default**: `:preserve`. If `preserve`, it will use the same
          memory format as the input.

    ## Examples
        # Create a random tensor with values between 0 and 10 from a float32 one.
        iex> a = ExTorch.zeros({3, 4, 5}, dtype: :float32)
        iex> ExTorch.randint_like(a, 10)
        #Tensor<
        [[[2., 5., 0., 7., 5.],
          [9., 0., 9., 1., 4.],
          [6., 3., 6., 0., 2.],
          [2., 6., 5., 9., 0.]],

         [[9., 4., 7., 9., 8.],
          [2., 8., 0., 8., 3.],
          [6., 6., 1., 9., 0.],
          [5., 2., 1., 7., 8.]],

         [[8., 3., 6., 8., 9.],
          [5., 7., 0., 7., 6.],
          [5., 4., 0., 3., 3.],
          [4., 3., 7., 3., 5.]]]
        [
          size: {3, 4, 5},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

        # Create a CUDA random tensor with values between 0 and 5 from a CPU one
        iex> b = ExTorch.rand({3, 3})
        iex> ExTorch.randint_like(b, 5, device: :cuda)
        #Tensor<
        [[4., 1., 4.],
         [1., 1., 1.],
         [2., 4., 3.]]
        [
          size: {3, 3},
          dtype: :float,
          device: {:cuda, 0},
          requires_grad: false
        ]>


        # Create a random tensor with values between -1 and 5 from a int32 one.
        iex> c = ExTorch.ones({3, 3}, dtype: :int32)
        iex> ExTorch.randint_like(c, -1, 5)
        #Tensor<
        [[ 2,  2,  4],
         [-1,  4, -1],
         [ 0,  3,  1]]
        [
          size: {3, 3},
          dtype: :int,
          device: :cpu,
          requires_grad: false
        ]>

        # Create a float32 CUDA random tensor with values between -1 and 5 from a int32 one.
        iex> ExTorch.randint_like(c, -1, 5, dtype: :float32, device: :cuda)
        #Tensor<
        [[4., 0., 4.],
         [0., 1., 2.],
         [1., 2., 1.]]
        [
          size: {3, 3},
          dtype: :float,
          device: {:cuda, 0},
          requires_grad: false
        ]>
    """
    @spec randint_like(ExTorch.Tensor.t(), integer(), integer(), ExTorch.Tensor.Options.t()) ::
            ExTorch.Tensor.t()
    defbinding(
      randint_like(
        input,
        low \\ 0,
        high,
        opts \\ %ExTorch.Tensor.Options{
          dtype: :auto,
          device: :auto,
          layout: nil,
          memory_format: :preserve
        }
      ),
      opts: ExTorch.Tensor.Options.merge_input(input, opts)
    )

    @doc """
    Returns a tensor filled with the scalar value `fill_value`, with the same size as `input`.

    `ExTorch.full_like(input, fill_value)` is equivalent to
    `ExTorch.full(input.size, fill_value, dtype: input.dtype, layout: input.layout, device: input.device)`

    ## Arguments
      - `input`: The input tensor (`ExTorch.Tensor`)
      - `fill_value`: the value to fill the output tensor with.

    ## Keyword args
      - dtype (`ExTorch.DType`, optional): the desired data type of returned tensor.
        **Default**: `auto`. If `auto`, it will use the same data type as the input.
        If `nil`, it will use a global default (see `ExTorch.set_default_dtype`).

      - layout (`ExTorch.Layout`, optional): the desired layout of returned Tensor.
        **Default**: `nil`. If `nil`, it will use the same layout as the input.

      - device (`ExTorch.Device`, optional): the desired device of returned tensor.
        Default: `auto`. If `auto`, it will use the same device as the input.
        If `nil`, it will use the current device for the default tensor type
        (see `ExTorch.set_default_device`). `device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.

      - requires_grad (`boolean()`, optional): If autograd should record operations on the
          returned tensor. **Default**: `false`.

      - pin_memory (`bool`, optional): If set, returned tensor would be allocated in
          the pinned memory. Works only for CPU tensors. Default: `false`.

      - memory_format (`ExTorch.MemoryFormat`, optional): the desired memory format of
          returned Tensor. **Default**: `:preserve`. If `preserve`, it will use the same
          memory format as the input.

    ## Examples
        # Create a tensor filled with -1 from an int64 input.
        iex> a = ExTorch.empty({1, 2, 2}, dtype: :int64)
        iex> ExTorch.full_like(a, -1)
        #Tensor<
        [[[-1, -1],
          [-1, -1]]]
        [
          size: {1, 2, 2},
          dtype: :long,
          device: :cpu,
          requires_grad: false
        ]>

        # Create a CUDA complex tensor filled with a given value from a CPU input.
        iex> b = ExTorch.ones({3, 3}, dtype: :complex128)
        iex> ExTorch.full_like(b, ExTorch.Complex.complex(0.8, -0.5), device: :cuda)
        #Tensor<
        [[0.8000-0.5000j, 0.8000-0.5000j, 0.8000-0.5000j],
         [0.8000-0.5000j, 0.8000-0.5000j, 0.8000-0.5000j],
         [0.8000-0.5000j, 0.8000-0.5000j, 0.8000-0.5000j]]
        [
          size: {3, 3},
          dtype: :complex_double,
          device: {:cuda, 0},
          requires_grad: false
        ]>
    """
    @spec full_like(
            ExTorch.Tensor.t(),
            number() | ExTorch.Complex.t(),
            ExTorch.Tensor.Options.t()
          ) :: ExTorch.Tensor.t()
    defbinding(
      full_like(
        input,
        fill_value,
        opts \\ %ExTorch.Tensor.Options{
          dtype: :auto,
          device: :auto,
          layout: nil,
          memory_format: :preserve
        }
      ),
      opts: ExTorch.Tensor.Options.merge_input(input, opts)
    )

    @doc """
    Returns a tensor filled with the scalar value 0, with the same size as `input`.

    `ExTorch.zeros_like(input)` is equivalent to
    `ExTorch.zeros(input.size, dtype: input.dtype, layout: input.layout, device: input.device)`

    ## Arguments
      - `input`: The input tensor (`ExTorch.Tensor`)

    ## Keyword args
      - dtype (`ExTorch.DType`, optional): the desired data type of returned tensor.
        **Default**: `auto`. If `auto`, it will use the same data type as the input.
        If `nil`, it will use a global default (see `ExTorch.set_default_dtype`).

      - layout (`ExTorch.Layout`, optional): the desired layout of returned Tensor.
        **Default**: `nil`. If `nil`, it will use the same layout as the input.

      - device (`ExTorch.Device`, optional): the desired device of returned tensor.
        Default: `auto`. If `auto`, it will use the same device as the input.
        If `nil`, it will use the current device for the default tensor type
        (see `ExTorch.set_default_device`). `device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.

      - requires_grad (`boolean()`, optional): If autograd should record operations on the
          returned tensor. **Default**: `false`.

      - pin_memory (`bool`, optional): If set, returned tensor would be allocated in
          the pinned memory. Works only for CPU tensors. Default: `false`.

      - memory_format (`ExTorch.MemoryFormat`, optional): the desired memory format of
          returned Tensor. **Default**: `:preserve`. If `preserve`, it will use the same
          memory format as the input.

    ## Examples
        # Create a tensor filled with ones from another float64 tensor.
        iex> a = ExTorch.rand({3, 4})
        iex> ExTorch.zeros_like(a)
        #Tensor<
        [[0., 0., 0., 0.],
         [0., 0., 0., 0.],
         [0., 0., 0., 0.]]
        [
          size: {3, 4},
          dtype: :double,
          device: :cpu,
          requires_grad: false
        ]>

        # Create a complex tensor with real part equal to one in GPU from another CPU tensor.
        iex> a = ExTorch.rand({3, 4}, dtype: :complex64)
        iex> ExTorch.zeros_like(a, device: :cuda)
        #Tensor<
        [[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
         [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
         [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]]
        [
          size: {3, 4},
          dtype: :complex_float,
          device: {:cuda, 0},
          requires_grad: false
        ]>

    """
    @spec zeros_like(ExTorch.Tensor.t(), ExTorch.Tensor.Options.t()) :: ExTorch.Tensor.t()
    defbinding(
      zeros_like(
        input,
        opts \\ %ExTorch.Tensor.Options{
          dtype: :auto,
          device: :auto,
          layout: nil,
          memory_format: :preserve
        }
      ),
      opts: ExTorch.Tensor.Options.merge_input(input, opts)
    )

    @doc """
    Returns a tensor filled with the scalar value 1, with the same size as `input`.

    `ExTorch.ones_like(input)` is equivalent to
    `ExTorch.ones(input.size, dtype: input.dtype, layout: input.layout, device: input.device)`

    ## Arguments
      - `input`: The input tensor (`ExTorch.Tensor`)

    ## Keyword args
      - dtype (`ExTorch.DType`, optional): the desired data type of returned tensor.
        **Default**: `auto`. If `auto`, it will use the same data type as the input.
        If `nil`, it will use a global default (see `ExTorch.set_default_dtype`).

      - layout (`ExTorch.Layout`, optional): the desired layout of returned Tensor.
        **Default**: `nil`. If `nil`, it will use the same layout as the input.

      - device (`ExTorch.Device`, optional): the desired device of returned tensor.
        Default: `auto`. If `auto`, it will use the same device as the input.
        If `nil`, it will use the current device for the default tensor type
        (see `ExTorch.set_default_device`). `device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.

      - requires_grad (`boolean()`, optional): If autograd should record operations on the
          returned tensor. **Default**: `false`.

      - pin_memory (`bool`, optional): If set, returned tensor would be allocated in
          the pinned memory. Works only for CPU tensors. Default: `false`.

      - memory_format (`ExTorch.MemoryFormat`, optional): the desired memory format of
          returned Tensor. **Default**: `:preserve`. If `preserve`, it will use the same
          memory format as the input.

    ## Examples
        # Create a tensor filled with ones from another float64 tensor.
        iex> a = ExTorch.rand({3, 4})
        iex> ExTorch.ones_like(a)
        #Tensor<
        [[1., 1., 1., 1.],
         [1., 1., 1., 1.],
         [1., 1., 1., 1.]]
        [
          size: {3, 4},
          dtype: :double,
          device: :cpu,
          requires_grad: false
        ]>

        # Create a complex tensor with real part equal to one in GPU from another CPU tensor.
        iex> a = ExTorch.rand({3, 4}, dtype: :complex64)
        iex> ExTorch.ones_like(a, device: :cuda)
        #Tensor<
        [[1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
         [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
         [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j]]
        [
          size: {3, 4},
          dtype: :complex_float,
          device: {:cuda, 0},
          requires_grad: false
        ]>

    """
    @spec ones_like(ExTorch.Tensor.t(), ExTorch.Tensor.Options.t()) :: ExTorch.Tensor.t()
    defbinding(
      ones_like(
        input,
        opts \\ %ExTorch.Tensor.Options{
          dtype: :auto,
          device: :auto,
          layout: nil,
          memory_format: :preserve
        }
      ),
      opts: ExTorch.Tensor.Options.merge_input(input, opts)
    )

    @doc """
    Constructs a complex tensor with its real part equal to `real` and its
    imaginary part equal to `imag`.

    ## Arguments
      - `real`: An `ExTorch.Tensor` containing the real parts.
      - `imag`: An `ExTorch.Tensor` containing the imag parts.

    ## Notes
    If both the inputs are `:float32`, the output will be `:complex64`.
    Comparatively, if both the inputs are `:float64`, the output will be
    `:complex128`.

    ## Examples
        iex> real = ExTorch.arange(5)
        iex> imag = ExTorch.arange(-5, 0)
        iex> ExTorch.complex(real, imag)
        #Tensor<
        [0.-5.j, 1.-4.j, 2.-3.j, 3.-2.j, 4.-1.j]
        [
          size: {5},
          dtype: :complex_float,
          device: :cpu,
          requires_grad: false
        ]>

    """
    @spec complex(ExTorch.Tensor.t(), ExTorch.Tensor.t()) :: ExTorch.Tensor.t()
    defbinding(complex(real, imag))

    @doc ~S"""
    Constructs a complex tensor whose elements are Cartesian coordinates
    corresponding to the polar coordinates with absolute value `abs` and
    angle `angle`.

    $$(\text{out} = \text{abs} \cdot \cos(\text{angle}) + \text{abs} \cdot \sin(\text{angle}) \cdot j)$$

    ## Arguments
      - `real`: An `ExTorch.Tensor` containing the real parts.
      - `imag`: An `ExTorch.Tensor` containing the imag parts.

    ## Notes
    If both the inputs are `:float32`, the output will be `:complex64`.
    Comparatively, if both the inputs are `:float64`, the output will be
    `:complex128`.

    ## Examples
        iex> real = ExTorch.arange(5)
        iex> imag = ExTorch.arange(-5, 0)
        iex> ExTorch.polar(real, imag)
        #Tensor<
        [ 0.0000+0.0000j, -0.6536+0.7568j, -1.9800-0.2822j, -1.2484-2.7279j,
          2.1612-3.3659j]
        [
          size: {5},
          dtype: :complex_float,
          device: :cpu,
          requires_grad: false
        ]>
    """
    @spec polar(ExTorch.Tensor.t(), ExTorch.Tensor.t()) :: ExTorch.Tensor.t()
    defbinding(polar(real, imag))
  end
end
