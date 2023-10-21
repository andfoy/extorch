defmodule ExTorch.Native.Tensor.Ops.Reduction do
  @moduledoc false
  use ExTorch.Native.BindingDeclaration

  defbindings(:tensor_reduction) do
    @doc """
    Check if all elements (or in a dimension) in `input` evaluate to `true`.

    If `dim` is `nil`, tests if all elements in `input` evaluate to `true`.
    Else, for each row of `input` in the given dimension `dim`, returns `true`
    if all elements in the row evaluate to `true` and `false` otherwise.

    * The `keepdim` and `out` options only work if `dim` is not `nil`.
    * If keepdim is `true`, the output tensor is of the same size as `input`,
    except in the dimension `dim` where it is of size 1. Otherwise, `dim` is
    squeezed (see `ExTorch.squeeze`), resulting in the output tensor having
    1 fewer dimension than `input`.

    ## Arguments
    - `input` - the input tensor. (`ExTorch.Tensor`)

    ## Optional arguments
      - `dim` - the dimension to reduce. (`nil | integer()`). Default: `nil`
      - `keepdim` - whether the output tensor has dim retained or not. (`boolean()`). Default: `false`
      - `out` - the optional output pre-allocated tensor. (`ExTorch.Tensor | nil`). Default: `nil`

    ## Examples
        # Find if all elements in a tensor are true
        iex> a = ExTorch.tensor([[true, true, true], [true, true, true]])
        iex> ExTorch.all(a)
        #Tensor<
        true
        [size: {}, dtype: :bool, device: :cpu, requires_grad: false]>

        # Find if all elements (per dimension) are true
        iex> b = ExTorch.empty({3, 3}, dtype: :bool)
        #Tensor<
        [[false,  true, false],
         [ true,  true,  true],
         [false, false, false]]
        [
          size: {3, 3},
          dtype: :bool,
          device: :cpu,
          requires_grad: false
        ]>
        iex> ExTorch.all(b, -1)
        #Tensor<
        [false,  true, false]
        [
          size: {3},
          dtype: :bool,
          device: :cpu,
          requires_grad: false
        ]>

        # Preserve tensor dimensions
        iex> ExTorch.all(b, -1, keepdim: true)
        #Tensor<
        [[false],
         [ true],
         [false]]
        [
          size: {3, 1},
          dtype: :bool,
          device: :cpu,
          requires_grad: false
        ]>

    """
    @spec all(ExTorch.Tensor.t(), nil | integer(), boolean(), nil | ExTorch.Tensor.t()) ::
            ExTorch.Tensor.t()
    defbinding(all(input, dim \\ nil, keepdim \\ false, out \\ nil))

    @doc """
    Check if at least one element (or element in a dimension) in `input` evaluates to `true`.

    If `dim` is `nil`, tests if at least one element in `input` evaluate to `true`.
    Else, for each row of `input` in the given dimension `dim`, returns `true`
    if at least one element in the row evaluates to `true` and `false` otherwise.

    * The `keepdim` and `out` options only work if `dim` is not `nil`.
    * If keepdim is `true`, the output tensor is of the same size as `input`,
    except in the dimension `dim` where it is of size 1. Otherwise, `dim` is
    squeezed (see `ExTorch.squeeze`), resulting in the output tensor having
    1 fewer dimension than `input`.

    ## Arguments
    - `input` - the input tensor. (`ExTorch.Tensor`)

    ## Optional arguments
      - `dim` - the dimension to reduce. (`nil | integer()`). Default: `nil`
      - `keepdim` - whether the output tensor has dim retained or not. (`boolean()`). Default: `false`
      - `out` - the optional output pre-allocated tensor. (`ExTorch.Tensor | nil`). Default: `nil`

    ## Examples
        # Find if any element in a tensor is true
        iex> a = ExTorch.tensor([[true, false, true], [false, true, true]])
        iex> ExTorch.any(a)
        #Tensor<
        true
        [size: {}, dtype: :bool, device: :cpu, requires_grad: false]>

        # Find if any elements (per dimension) is true
        iex> b = ExTorch.empty({3, 3}, dtype: :bool)
        #Tensor<
        [[false,  true, false],
         [ true,  true,  true],
         [false, false, false]]
        [
          size: {3, 3},
          dtype: :bool,
          device: :cpu,
          requires_grad: false
        ]>
        iex> ExTorch.any(b, -1)
        #Tensor<
        [ true,  true, false]
        [
          size: {3},
          dtype: :bool,
          device: :cpu,
          requires_grad: false
        ]>

        # Preserve tensor dimensions
        iex> ExTorch.any(b, -1, keepdim: true)
        #Tensor<
        [[ true],
         [ true],
         [false]]
        [
          size: {3, 1},
          dtype: :bool,
          device: :cpu,
          requires_grad: false
        ]>
    """
    @spec any(ExTorch.Tensor.t(), nil | integer(), boolean(), nil | ExTorch.Tensor.t()) ::
            ExTorch.Tensor.t()
    defbinding(any(input, dim \\ nil, keepdim \\ false, out \\ nil))

    @doc """
    Returns the indices of the maximum value of all elements (or elements in a dimension)
    in the input tensor.

    * If `dim` is `nil`, it will return the index of the maximum element on the overall input
    tensor, else it will return the maximum values alongside the specified dimension.

    ## Arguments
    - `input` (`ExTorch.Tensor`) - the input tensor.

    ## Optional arguments
    - `dim` (`nil | integer()`) - the dimension to reduce. Default: `nil`
    - `keepdim` (`boolean()`) - whether the output tensor has dim retained or not. Default: `false`

    ## Notes
    If there are multiple maximal values then the indices of the first maximal value are returned.

    ## Examples
        iex> a = ExTorch.randn({4, 4})
        #Tensor<
        [[ 1.2023, -0.1142,  0.5077,  1.2127],
         [ 0.5873, -0.7416, -0.0758,  0.0578],
         [-0.8066,  1.7030,  0.2894,  0.0539],
         [ 0.2353,  0.4396, -0.1846, -0.7395]]
        [
          size: {4, 4},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

        # Get the overall maximum index.
        iex> ExTorch.argmax(a)
        #Tensor<
        9
        [size: {}, dtype: :long, device: :cpu, requires_grad: false]>

        # Get the maximum index on the last dimension.
        iex> ExTorch.argmax(a, -1)
        #Tensor<
        [3, 0, 1, 1]
        [
          size: {4},
          dtype: :long,
          device: :cpu,
          requires_grad: false
        ]>

        # Keep the reduced dimension on the output
        iex> ExTorch.argmax(a, 0, keepdim: true)
        #Tensor<
        [[0, 2, 0, 0]]
        [
          size: {1, 4},
          dtype: :long,
          device: :cpu,
          requires_grad: false
        ]>

    """
    @spec argmax(ExTorch.Tensor.t(), integer() | nil, boolean()) :: ExTorch.Tensor.t()
    defbinding(argmax(input, dim \\ nil, keepdim \\ false))

    @doc """
    Returns the indices of the minimum value of all elements (or elements in a dimension)
    in the input tensor.

    * If `dim` is `nil`, it will return the index of the minimum element on the overall input
    tensor, else it will return the minimum values alongside the specified dimension.

    ## Arguments
    - `input` (`ExTorch.Tensor`) - the input tensor.

    ## Optional arguments
    - `dim` (`nil | integer()`) - the dimension to reduce. Default: `nil`
    - `keepdim` (`boolean()`) - whether the output tensor has dim retained or not. Default: `false`

    ## Notes
    If there are multiple minimal values then the indices of the first minimal value are returned.

    ## Examples
        iex> a = ExTorch.randn({4, 4})
        #Tensor<
        [[-0.6192, -0.4204,  0.1524, -0.1544],
         [ 1.4040,  1.0165,  1.6355,  0.6480],
         [-0.6566,  1.0730, -0.1548, -0.2488],
         [-1.0406,  0.0883,  1.0485, -0.3025]]
        [
          size: {4, 4},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

        # Get the overall minimum index.
        iex> ExTorch.argmin(a)
        #Tensor<
        12
        [size: {}, dtype: :long, device: :cpu, requires_grad: false]>

        # Get the minimum index on the last dimension.
        iex> ExTorch.argmin(a, -1)
        #Tensor<
        [0, 3, 0, 0]
        [
          size: {4},
          dtype: :long,
          device: :cpu,
          requires_grad: false
        ]>

        # Keep the reduced dimension on the output
        iex> ExTorch.argmin(a, 0, keepdim: true)
        #Tensor<
        [[3, 0, 2, 3]]
        [
          size: {1, 4},
          dtype: :long,
          device: :cpu,
          requires_grad: false
        ]>

    """
    @spec argmin(ExTorch.Tensor.t(), integer() | nil, boolean()) :: ExTorch.Tensor.t()
    defbinding(argmin(input, dim \\ nil, keepdim \\ false))

    @doc """
    Returns the maximum value of all elements (or elements in a dimension) in the
    `input` tensor.

    If `dim` is `nil`, `max` returns the maximum element in the tensor.
    Else, it returns a namedtuple `{max, argmax}` where `max` are the maximum values
    of each row of the `input` tensor in the given dimension `dim`. And `argmax`
    is the index location of each maximum value found (See `ExTorch.argmax/3`).

    ## Arguments
    - `input` (`ExTorch.Tensor`) - the input tensor.

    ## Optional arguments
    - `dim` (`nil | integer()`) - the dimension to reduce. Default: `nil`
    - `keepdim` (`boolean()`) - whether the output tensor has dim retained or not, this . Default: `false`
    - `out` (`ExTorch.Tensor | {ExTorch.Tensor, ExTorch.Tensor} | nil`) - the optional output pre-allocated
    tensor if computing the overall maximum of a tensor, else it must be a pre-allocated
    tensor tuple `{max, argmax}`. Default: `nil`

    ## Notes
    * `keepdim` and `out` won't take any effect if `dim` is `nil`.
    * Unlike PyTorch, `max` will not take two tensors as input as an alias to `ExTorch.maximum/3`,
    please use that function explicitly instead.

    ## Examples
        iex> a = ExTorch.randn({4, 4})
        #Tensor<
        [[-0.6730,  0.9223, -0.3803,  0.2369],
         [ 0.5956, -0.2750,  1.3838, -2.1479],
         [ 0.4648, -1.8987,  0.8329, -0.5854],
         [ 1.1679, -0.4866, -0.5227, -0.4399]]
        [
          size: {4, 4},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

        # Find the overall maximum in the tensor
        iex> ExTorch.max(a)
        #Tensor<
        1.3838
        [size: {}, dtype: :float, device: :cpu, requires_grad: false]>

        # Find the maximum in the last dimension
        iex> {max, argmax} = ExTorch.max(a, -1)
        iex> max
        #Tensor<
        [0.9223, 1.3838, 0.8329, 1.1679]
        [
          size: {4},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>
        iex> argmax
        #Tensor<
        [1, 2, 2, 0]
        [
          size: {4},
          dtype: :long,
          device: :cpu,
          requires_grad: false
        ]>

        # Preserve the original number of dimensions
        iex> {max, argmax} = ExTorch.max(a, -1, keepdim: true)
        iex> max
        #Tensor<
        [[0.9223],
         [1.3838],
         [0.8329],
         [1.1679]]
        [
          size: {4, 1},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>
        iex> argmax
        #Tensor<
        [[1],
         [2],
         [2],
         [0]]
        [
          size: {4, 1},
          dtype: :long,
          device: :cpu,
          requires_grad: false
        ]>
    """
    @spec max(
            ExTorch.Tensor.t(),
            integer() | nil,
            boolean(),
            {ExTorch.Tensor.t(), ExTorch.Tensor.t()} | nil
          ) ::
            ExTorch.Tensor.t() | {ExTorch.Tensor.t(), ExTorch.Tensor.t()}
    defbinding(max(input, dim \\ nil, keepdim \\ false, out \\ nil))

    @doc """
    Returns the minimum value of all elements (or elements in a dimension) in the
    `input` tensor.

    If `dim` is `nil`, `min` returns the minimum element in the tensor.
    Else, it returns a namedtuple `{min, argmin}` where `min` are the minimum values
    of each row of the `input` tensor in the given dimension `dim`. And `argmin`
    is the index location of each minimum value found (See `ExTorch.argmin/3`).

    ## Arguments
    - `input` (`ExTorch.Tensor`) - the input tensor.

    ## Optional arguments
    - `dim` (`nil | integer()`) - the dimension to reduce. Default: `nil`
    - `keepdim` (`boolean()`) - whether the output tensor has dim retained or not, this . Default: `false`
    - `out` (`ExTorch.Tensor | {ExTorch.Tensor, ExTorch.Tensor} | nil`) - the optional output pre-allocated
    tensor if computing the overall minimum of a tensor, else it must be a pre-allocated
    tensor tuple `{min, argmin}`. Default: `nil`

    ## Notes
    * `keepdim` and `out` won't take any effect if `dim` is `nil`.
    * Unlike PyTorch, `min` will not take two tensors as input as an alias to `ExTorch.minimum/3`,
    please use that function explicitly instead.

    ## Examples
        iex> a = ExTorch.randn({4, 4})
        #Tensor<
        [[-0.6730,  0.9223, -0.3803,  0.2369],
         [ 0.5956, -0.2750,  1.3838, -2.1479],
         [ 0.4648, -1.8987,  0.8329, -0.5854],
         [ 1.1679, -0.4866, -0.5227, -0.4399]]
        [
          size: {4, 4},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

        # Find the overall minimum in the tensor
        iex> ExTorch.min(a)
        #Tensor<
        -2.1479
        [size: {}, dtype: :float, device: :cpu, requires_grad: false]>

        # Find the minimum in the last dimension
        iex> {min, argmin} = ExTorch.min(a, -1)
        iex> min
        #Tensor<
        [-0.6730, -2.1479, -1.8987, -0.5227]
        [
          size: {4},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>
        iex> argmin
        #Tensor<
        [0, 3, 1, 2]
        [
          size: {4},
          dtype: :long,
          device: :cpu,
          requires_grad: false
        ]>

        # Preserve the original number of dimensions
        iex> {min, argmin} = ExTorch.min(a, -1, keepdim: true)
        iex> min
        #Tensor<
        [[-0.6730],
         [-2.1479],
         [-1.8987],
         [-0.5227]]
        [
          size: {4, 1},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>
        iex(16)> argmin
        #Tensor<
        [[0],
         [3],
         [1],
         [2]]
        [
          size: {4, 1},
          dtype: :long,
          device: :cpu,
          requires_grad: false
        ]>

    """
    @spec min(
            ExTorch.Tensor.t(),
            integer() | nil,
            boolean(),
            {ExTorch.Tensor.t(), ExTorch.Tensor.t()} | nil
          ) ::
            ExTorch.Tensor.t() | {ExTorch.Tensor.t(), ExTorch.Tensor.t()}
    defbinding(min(input, dim \\ nil, keepdim \\ false, out \\ nil))

    @doc """
    Returns the maximum value of each slice of the `input` tensor in the given dimension(s) dim.

    If `keepdim` is `true`, the output tensor is of the same size as `input` except in the dimension(s)
    `dim` where it is of size 1. Otherwise, `dim` is squeezed (see `Extorch.squeeze`), resulting in
    the output tensor having 1 (or `length(dim)`) fewer dimension(s).

    ## Arguments
    - `input` (`ExTorch.Tensor`) - the input tensor.
    - `dim` (`nil | integer() | tuple()`) - the dimension(s) to reduce. If `nil`, then it reduces all the dimensions.

    ## Optional arguments
    - `keepdim` (`boolean()`) - whether the output tensor has dim retained or not, this . Default: `false`
    - `out` (`ExTorch.Tensor | nil`) - the optional output pre-allocated tensor. Default: `nil`

    ## Notes
    The difference between `ExTorch.max`/`ExTorch.min` and `ExTorch.amax`/`ExTorch.amin` is:
    * `ExTorch.amax`/`ExTorch.amin` supports reducing on multiple dimensions
    * `ExTorch.amax`/`ExTorch.amin` does not return indices
    * `ExTorch.amax`/`ExTorch.amin` evenly distributes gradient between equal values,
    while `max(dim)`/`min(dim)` propagates gradient only to a single index in the source tensor.

    ## Examples
        iex> a = ExTorch.randn({4, 4})
        #Tensor<
        [[ 1.4814,  0.1511, -1.9243,  0.6649],
        [ 0.1308,  0.5038, -0.0844, -0.8609],
        [-0.9535,  0.1651, -0.5081,  0.7449],
        [-1.5848,  0.1389, -0.5299, -0.0702]]
        [
          size: {4, 4},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

        # Get the maximum values alongside the last dimension
        iex> ExTorch.amax(a, -1)
        #Tensor<
        [1.4814, 0.5038, 0.7449, 0.1389]
        [
          size: {4},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

        # Get the maximum value on all dimensions
        iex> ExTorch.amax(a, {0, 1})
        #Tensor<
        1.4814
        [size: {}, dtype: :float, device: :cpu, requires_grad: false]>

    """
    @spec amax(ExTorch.Tensor.t(), nil | integer() | tuple(), boolean(), ExTorch.Tensor.t() | nil) ::
            ExTorch.Tensor.t()
    defbinding(amax(input, dim, keepdim \\ false, out \\ nil))

    @doc """
    Returns the minimum value of each slice of the `input` tensor in the given dimension(s) dim.

    If `keepdim` is `true`, the output tensor is of the same size as `input` except in the dimension(s)
    `dim` where it is of size 1. Otherwise, `dim` is squeezed (see `Extorch.squeeze`), resulting in
    the output tensor having 1 (or `length(dim)`) fewer dimension(s).

    ## Arguments
    - `input` (`ExTorch.Tensor`) - the input tensor.
    - `dim` (`nil | integer() | tuple()`) - the dimension(s) to reduce. If `nil`, then it reduces all the dimensions.

    ## Optional arguments
    - `keepdim` (`boolean()`) - whether the output tensor has dim retained or not, this . Default: `false`
    - `out` (`ExTorch.Tensor | nil`) - the optional output pre-allocated tensor. Default: `nil`

    ## Notes
    The difference between `ExTorch.max`/`ExTorch.min` and `ExTorch.amax`/`ExTorch.amin` is:
    * `ExTorch.amax`/`ExTorch.amin` supports reducing on multiple dimensions
    * `ExTorch.amax`/`ExTorch.amin` does not return indices
    * `ExTorch.amax`/`ExTorch.amin` evenly distributes gradient between equal values,
    while `max(dim)`/`min(dim)` propagates gradient only to a single index in the source tensor.

    ## Examples
        iex> a = ExTorch.randn({4, 4})
        #Tensor<
        [[ 0.4138,  0.9993,  0.1177, -0.0021],
         [ 0.0340, -0.7703,  1.5916, -0.2477],
         [ 0.4927,  0.7762, -0.9214, -0.3303],
         [-0.4098, -0.1762, -1.4085, -1.4918]]
        [
          size: {4, 4},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

        # Get the minimum values alongside the last dimension
        iex> ExTorch.amin(a, -1)
        #Tensor<
        [-0.0021, -0.7703, -0.9214, -1.4918]
        [
          size: {4},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

        # Get the minimum value on all dimensions
        iex> ExTorch.amin(a, {0, 1})
        #Tensor<
        -1.4918
        [size: {}, dtype: :float, device: :cpu, requires_grad: false]>

    """
    @spec amin(ExTorch.Tensor.t(), nil | integer() | tuple(), boolean(), ExTorch.Tensor.t() | nil) ::
            ExTorch.Tensor.t()
    defbinding(amin(input, dim, keepdim \\ false, out \\ nil))

    @doc """
    Computes the minimum and maximum values of the `input` tensor.

    It will return a tuple `{min, max}` containing the maximum and minimum values, respectively.

    ## Arguments
    - `input` (`ExTorch.Tensor`) - the input tensor.

    ## Optional arguments
    - `dim` (`nil | integer()`) - the dimension to reduce. If `nil`, then it computes the
    values over the entire `input` tensor. Default: `nil`
    - `keepdim` (`boolean()`) - whether the output tensors have dim retained or not. Default: `false`
    - `out` (`{ExTorch.Tensor, ExTorch.Tensor} | nil`) - the optional output pre-allocated tensors in a tuple. Default: `nil`

    ## Examples
        iex> a = ExTorch.randn({4, 4})
        #Tensor<
        [[-0.7684,  0.8360,  0.1960, -0.7748],
         [ 0.9795, -0.3725,  0.1304, -0.3627],
         [-0.6206,  0.1624,  0.8514, -1.2361],
         [-1.5297, -0.6418, -0.8179,  1.7531]]
        [
          size: {4, 4},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

        # Find the minimum and maximum values on the entire input
        iex> {min, max} = ExTorch.aminmax(a)
        iex> min
        #Tensor<
        1.7531
        [size: {}, dtype: :float, device: :cpu, requires_grad: false]>
        iex> max
        #Tensor<
        -1.5297
        [size: {}, dtype: :float, device: :cpu, requires_grad: false]>

        # Find the minimum and maximum values alongside the last dimension
        iex> {min, max} = ExTorch.aminmax(a, -1, keepdim: true)
        iex> min
        #Tensor<
        [[0.8360],
         [0.9795],
         [0.8514],
         [1.7531]]
        [
          size: {4, 1},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>
        iex> max
        #Tensor<
        [[-0.7748],
         [-0.3725],
         [-1.2361],
         [-1.5297]]
        [
          size: {4, 1},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

    """
    @spec aminmax(
            ExTorch.Tensor.t(),
            integer() | nil,
            boolean(),
            {ExTorch.Tensor.t(), ExTorch.Tensor.t()} | nil
          ) :: {ExTorch.Tensor.t(), ExTorch.Tensor.t()}
    defbinding(aminmax(input, dim \\ nil, keepdim \\ false, out \\ nil))

    @doc """
    Returns the p-norm of (`input` - `other`)

    The shapes of input and other must be broadcastable.

    ## Arguments
    - `input` (`ExTorch.Tensor`) - the input tensor.
    - `other` (`ExTorch.Tensor`) - the right-hand side input tensor.

    ## Optional arguments
    - `p` (`number`) - the norm to be computed. Default: 2.0

    ## Examples
        iex> a = ExTorch.randn(3)
        #Tensor<
        [ 0.3934,  0.6799, -0.1292]
        [
          size: {3},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>
        iex> b = ExTorch.randn(3)
        #Tensor<
        [-0.3785, -1.5249,  0.2093]
        [
          size: {3},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

        # Compute the Euclidean norm
        iex> ExTorch.dist(a, b)
        #Tensor<
        2.3605\

        [size: {}, dtype: :float, device: :cpu, requires_grad: false]>

        # Compute the L1 distance
        iex> ExTorch.dist(a, b, 1)
        #Tensor<
        3.3152
        [size: {}, dtype: :float, device: :cpu, requires_grad: false]>
    """
    @spec dist(ExTorch.Tensor.t(), ExTorch.Tensor.t(), number()) :: ExTorch.Tensor.t()
    defbinding(dist(input, other, p \\ 2.0))

    @doc ~S"""
    Returns the log of summed exponentials of each row of the `input` tensor in the given dimension `dim`.
    The computation is numerically stabilized.

    For summation index $j$ given by dim and other indices $i$, the result is:

    $$\text{logsumexp}(x)\_i = \text{log}\sum\_j \text{exp}\left(x\_{ij}\right)$$

    If `keepdim` is `true`, the output tensor is of the same size as `input` except in the dimension(s) `dim`
    where it is of size 1. Otherwise, `dim` is squeezed (see `ExTorch.squeeze`), resulting in the
    output tensor having 1 (or `length(dim)`) fewer dimension(s).

    ## Arguments
    - `input` (`ExTorch.Tensor`) - the input tensor.
    - `dim` (`nil | integer() | tuple()`) - the dimension or dimensions to reduce. If `nil`, all dimensions are reduced.

    ## Optional arguments
    - `keepdim` (`boolean`) - whether the output tensor has `dim` retained or not.
    - `out` (`ExTorch.Tensor | nil`) - the optional output pre-allocated tensor. Default: `nil`

    ## Examples
        iex> a = ExTorch.randn({3, 3})
        #Tensor<
        [[ 0.2292, -1.0899,  0.0889],
         [-2.0117,  0.4716, -0.3893],
         [-0.9382,  1.0590, -0.0838]]
        [
          size: {3, 3},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

        # Compute logsumexp in all dimensions
        iex> ExTorch.logsumexp(a)
        #Tensor<
        2.2295
        [size: {}, dtype: :float, device: :cpu, requires_grad: false]>

        # Compute logsumexp in the last dimension, preserve dimensions
        iex> ExTorch.logsumexp(a, -1, keepdim: true)
        #Tensor<
        [[0.9883],
         [0.8812],
         [1.4338]]
        [
          size: {3, 1},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

    """
    @spec logsumexp(
            ExTorch.Tensor.t(),
            integer() | tuple() | nil,
            boolean(),
            ExTorch.Tensor.t() | nil
          ) :: ExTorch.Tensor.t()
    defbinding(logsumexp(input, dim \\ nil, keepdim \\ false, out \\ nil),
      dim:
        if dim do
          dim
        else
          max_dim = ExTorch.Tensor.dim(input) - 1
          Enum.map(0..max_dim, fn x -> x end) |> List.to_tuple()
        end
    )

    @doc """
    Returns the sum of all elements (or alongside an axis) in the input tensor.

    ## Arguments
    - `input` (`ExTorch.Tensor`) - the input tensor.

    ## Optional arguments
    - `dim` (`integer | tuple | nil`) - the dimension or dimensions to reduce. If `nil`,
    all dimensions are reduced. Default: `nil`
    - `keepdim` (`boolean`) - whether the output tensor has `dim` retained or not. Default: `false`
    - `dtype` (`ExTorch.DType` or `nil`) - the desired data type of returned tensor.
    If specified, the `input` tensor is casted to `dtype` before the operation
    is performed. This is useful for preventing data type overflows. Default: `nil`.

    ## Examples
        iex> a = ExTorch.rand({3, 3})
        #Tensor<
        [[0.7281, 0.9280, 0.5829],
        [0.4569, 0.4785, 0.1352],
        [0.9905, 0.0698, 0.1905]]
        [
          size: {3, 3},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

        # Sum all elements in a tensor.
        iex> ExTorch.sum(a)
        #Tensor<
        4.5604
        [size: {}, dtype: :float, device: :cpu, requires_grad: false]>

        # Sum all elements in the last dimension, keeping dims and casting to double
        iex> ExTorch.sum(a, 1, keepdim: true, dtype: :double)
        #Tensor<
        [[2.2390],
         [1.0707],
         [1.2507]]
        [
          size: {3, 1},
          dtype: :double,
          device: :cpu,
          requires_grad: false
        ]>
    """
    @spec sum(ExTorch.Tensor.t(), integer() | tuple() | nil, boolean(), ExTorch.DType.dtype()) ::
            ExTorch.Tensor.t()
    defbinding(sum(input, dim \\ nil, keepdim \\ false, dtype \\ nil),
      input:
        if dtype do
          ExTorch.Tensor.to(input, dtype: dtype)
        else
          input
        end,
      dim: dim || {},
      omitted_args: [:dtype]
    )

    @doc """
    Returns the mean value of all elements (or alongside an axis) in the input tensor.

    ## Arguments
      - `input` (`ExTorch.Tensor`) - the input tensor.

    ## Optional arguments
    - `dim` (`integer | tuple | nil`) - the dimension or dimensions to reduce. If `nil`,
    all dimensions are reduced. Default: `nil`
    - `keepdim` (`boolean`) - whether the output tensor has `dim` retained or not. Default: `false`
    - `dtype` (`ExTorch.DType` or `nil`) - the desired data type of returned tensor.
    If specified, the `input` tensor is casted to `dtype` before the operation
    is performed. This is useful for preventing data type overflows. Default: `nil`.
    - `out` (`ExTorch.Tensor | nil`) - the optional output pre-allocated tensor. Default: `nil`

    ## Examples
        iex> a = ExTorch.rand({3, 3})
        #Tensor<
        [[0.0945, 0.3992, 0.5090],
         [0.0142, 0.1471, 0.4568],
         [0.1428, 0.2121, 0.6163]]
        [
          size: {3, 3},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

        # Mean of all elements in a tensor.
        iex> ExTorch.mean(a)
        #Tensor<
        0.2880
        [size: {}, dtype: :float, device: :cpu, requires_grad: false]>

        # Mean of elements in the last dimension, keeping dims and casting to double
        iex> ExTorch.mean(a, -1, keepdim: true, dtype: :double)
        #Tensor<
        [[0.3343],
         [0.2060],
         [0.3237]]
        [
          size: {3, 1},
          dtype: :double,
          device: :cpu,
          requires_grad: false
        ]>
    """
    @spec mean(
            ExTorch.Tensor.t(),
            integer() | tuple() | nil,
            boolean(),
            ExTorch.DType.dtype(),
            ExTorch.Tensor.t()
          ) ::
            ExTorch.Tensor.t()
    defbinding(mean(input, dim \\ nil, keepdim \\ false, dtype \\ nil, out \\ nil),
      input:
        if dtype do
          ExTorch.Tensor.to(input, dtype: dtype)
        else
          input
        end,
      dim: dim || {},
      omitted_args: [:dtype]
    )

    @doc """
    Computes the mean of all non-NaN elements along the specified dimensions.

    This function is identical to `ExTorch.mean/5` when there are no `:nan` values in the `input` tensor.
    In the presence of `:nan`, `ExTorch.mean` will propagate the `:nan` to the output whereas `ExTorch.nanmean`
    will ignore the NaN values.

    If `keepdim` is `true`, the output tensor is of the same size as `input` except in the dimension(s) `dim`
    where it is of size 1. Otherwise, `dim` is squeezed (see `ExTorch.squeeze`), resulting in the output
    tensor having 1 (or `length(dim)`) fewer dimension(s).

    ## Arguments
    - `input` (`ExTorch.Tensor`) - the input tensor.

    ## Optional arguments
    - `dim` (`integer | tuple | nil`) - the dimension or dimensions to reduce. If `nil`,
    all dimensions are reduced. Default: `nil`
    - `keepdim` (`boolean`) - whether the output tensor has `dim` retained or not. Default: `false`
    - `dtype` (`ExTorch.DType` or `nil`) - the desired data type of returned tensor.
    If specified, the `input` tensor is casted to `dtype` before the operation
    is performed. This is useful for preventing data type overflows. Default: `nil`.
    - `out` (`ExTorch.Tensor | nil`) - the optional output pre-allocated tensor. Default: `nil`

    ## Examples
        iex> a = ExTorch.tensor([[:nan, 1.0, 2.0], [-1.0, :nan, 2.0], [1.0, -1.0, :nan]])
        #Tensor<
        [[    nan,  1.0000,  2.0000],
        [-1.0000,     nan,  2.0000],
        [ 1.0000, -1.0000,     nan]]
        [
          size: {3, 3},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

        # Compute mean of all array elements without :nan
        iex> ExTorch.nanmean(a)
        #Tensor<
        0.6667
        [size: {}, dtype: :float, device: :cpu, requires_grad: false]>

        # Compute mean of all array elements on the last dimension, keep all dims
        iex> ExTorch.nanmean(a, -1, keepdim: true)
        #Tensor<
        [[1.5000],
         [0.5000],
         [0.0000]]
        [
          size: {3, 1},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

    """
    @spec nanmean(
            ExTorch.Tensor.t(),
            integer() | tuple() | nil,
            boolean(),
            ExTorch.DType.dtype(),
            ExTorch.Tensor.t()
          ) ::
            ExTorch.Tensor.t()
    defbinding(nanmean(input, dim \\ nil, keepdim \\ false, dtype \\ nil, out \\ nil),
      input:
        if dtype do
          ExTorch.Tensor.to(input, dtype: dtype)
        else
          input
        end,
      dim: dim || {},
      omitted_args: [:dtype]
    )

    @doc """
    Returns the median of the values in `input`.

    If `dim` is `nil`, it returns the median of all values in the tensor. Else, it
    returns a tuple {`values`, `indices`} where `values` contains the median
    of each row of `input` in the dimension `dim`, and `indices`
    contains the index of the median values found in the dimension `dim`.

    If `keepdim` is `true`, the output tensors are of the same size as `input`
    except in the dimension `dim` where they are of size 1.
    Otherwise, `dim` is squeezed (see `ExTorch.squeeze`), resulting in the outputs
    tensor having 1 fewer dimension than `input`.

    ## Arguments
    - `input` (`ExTorch.Tensor`) - the input tensor.

    ## Optional arguments
    - `dim` (`integer | nil`) - the dimension or dimensions to reduce. If `nil`,
    all dimensions are reduced. Default: `nil`
    - `keepdim` (`boolean`) - whether the output tensor has `dim` retained or not. Default: `false`
    - `out` (`{ExTorch.Tensor, ExTorch.Tensor} | nil`) - the optional output pre-allocated tuple tensor. Default: `nil`

    ## Notes
    * The median is not unique for `input` tensors with an even number of elements.
    In this case the lower of the two medians is returned. To compute the mean of both medians,
    use `ExTorch.quantile` with q=0.5 instead.

    * `indices` does not necessarily contain the first occurrence of each median value found,
    unless it is unique. The exact implementation details are device-specific. Do not expect
    the same result when run on CPU and GPU in general. For the same reason do not expect
    the gradients to be deterministic.

    * `keepdim` and `out` will not take effect when `dim = nil`.

    ## Examples
        iex> a = ExTorch.randn({3, 3})
        #Tensor<
        [[-0.7721, -2.0910, -0.4622],
         [ 0.1119,  2.4266,  1.3471],
         [-0.1450, -0.2876, -2.3025]]
        [
          size: {3, 3},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

        # Compute overall median of the input tensor.
        iex> ExTorch.median(a)
        #Tensor<
        -0.2876
        [size: {}, dtype: :float, device: :cpu, requires_grad: false]>

        # Compute the median of the last dimension, keeping dimensions.
        iex> {values, indices} = ExTorch.median(a, -1, keepdim: true)
        iex> values
        #Tensor<
        [[-0.7721],
         [ 1.3471],
         [-0.2876]]
        [
          size: {3, 1},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>
        iex> indices
        #Tensor<
        [[0],
         [2],
         [1]]
        [
          size: {3, 1},
          dtype: :long,
          device: :cpu,
          requires_grad: false
        ]>
    """
    @spec median(
            ExTorch.Tensor.t(),
            integer() | nil,
            boolean(),
            {ExTorch.Tensor.t(), ExTorch.Tensor.t()} | nil
          ) ::
            ExTorch.Tensor.t() | {ExTorch.Tensor.t(), ExTorch.Tensor.t()}
    defbinding(median(input, dim \\ nil, keepdim \\ false, out \\ nil))

    @doc """
    Returns the median of the values in input, ignoring NaN values.

    This function is identical to `ExTorch.median/4` when there are no `:nan` values in input.
    When input has one or more `:nan` values, `ExTorch.median` will always return `:nan`,
    while this function will return the median of the non-NaN elements in input.
    If all the elements in input are NaN it will also return `:nan`.

    If `dim` is `nil`, it returns the median of all values in the tensor. Else, it
    returns a tuple {`values`, `indices`} where `values` contains the median
    of each row of `input` in the dimension `dim`, and `indices`
    contains the index of the median values found in the dimension `dim`.

    ## Arguments
    - `input` (`ExTorch.Tensor`) - the input tensor.

    ## Optional arguments
    - `dim` (`integer | nil`) - the dimension or dimensions to reduce. If `nil`,
    all dimensions are reduced. Default: `nil`
    - `keepdim` (`boolean`) - whether the output tensor has `dim` retained or not. Default: `false`
    - `out` (`{ExTorch.Tensor, ExTorch.Tensor} | nil`) - the optional output pre-allocated tuple tensor. Default: `nil`

    ## Examples
        iex> input =
        ...>   ExTorch.tensor([
        ...>     [:nan, 1.0, 2.0],
        ...>     [-1.0, :nan, 2.0],
        ...>     [1.0, -1.0, :nan]
        ...>   ])

        # Compute median of the tensor without :nan
        iex> ExTorch.nanmedian(input)
        #Tensor<
        1.
        [size: {}, dtype: :float, device: :cpu, requires_grad: false]>

        # Compute median of the tensor in the last dimension, keeping all dimensions
        iex> {values, indices} = ExTorch.nanmedian(input, -1, keepdim: true)
        iex> values
        #Tensor<
        [[ 1.],
         [-1.],
         [-1.]]
        [
          size: {3, 1},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>
        iex> indices
        #Tensor<
        [[1],
         [0],
         [1]]
        [
          size: {3, 1},
          dtype: :long,
          device: :cpu,
          requires_grad: false
        ]>

    """
    @spec nanmedian(
            ExTorch.Tensor.t(),
            integer() | nil,
            boolean(),
            {ExTorch.Tensor.t(), ExTorch.Tensor.t()} | nil
          ) ::
            ExTorch.Tensor.t() | {ExTorch.Tensor.t(), ExTorch.Tensor.t()}
    defbinding(nanmedian(input, dim \\ nil, keepdim \\ false, out \\ nil))

    @doc """
    Returns the mode of the values in `input` across dimension `dim`.

    It returns a tuple {`values`, `indices`} where `values` contains the median
    of each row of `input` in the dimension `dim`, and `indices`
    contains the index of the median values found in the dimension `dim`.

    If `keepdim` is `true`, the output tensors are of the same size as `input`
    except in the dimension `dim` where they are of size 1.
    Otherwise, `dim` is squeezed (see `ExTorch.squeeze`), resulting in the outputs
    tensor having 1 fewer dimension than `input`.

    ## Arguments
    - `input` (`ExTorch.Tensor`) - the input tensor.

    ## Optional arguments
    - `dim` (`integer | nil`) - the dimension or dimensions to reduce. Default: -1
    - `keepdim` (`boolean`) - whether the output tensor has `dim` retained or not. Default: `false`
    - `out` (`{ExTorch.Tensor, ExTorch.Tensor} | nil`) - the optional output pre-allocated tuple tensor. Default: `nil`

    ## Notes
    This function is not defined for CUDA tensors yet.

    ## Examples
        iex> a = ExTorch.randint(5, {3, 4}, dtype: :int32)
        #Tensor<
        [[4, 4, 4, 4],
         [3, 4, 1, 1],
         [3, 2, 2, 0]]
        [
          size: {3, 4},
          dtype: :int,
          device: :cpu,
          requires_grad: false
        ]>

        # Compute the mode in the last dimension.
        iex> {values, indices} = ExTorch.mode(a)
        iex> values
        #Tensor<
        [4, 1, 2]
        [size: {3}, dtype: :int, device: :cpu, requires_grad: false]>
        iex> indices
        #Tensor<
        [3, 3, 2]
        [size: {3}, dtype: :long, device: :cpu, requires_grad: false]>

        # Compute the mode in the first dimension, keeping output dimensions.
        iex> {values, indices} = ExTorch.mode(a, 0, keepdim: true)
        iex> values
        #Tensor<
        [[3, 4, 1, 0]]
        [
          size: {1, 4},
          dtype: :int,
          device: :cpu,
          requires_grad: false
        ]>
        iex> indices
        #Tensor<
        [[2, 1, 1, 2]]
        [
          size: {1, 4},
          dtype: :long,
          device: :cpu,
          requires_grad: false
        ]>
    """
    @spec mode(
            ExTorch.Tensor.t(),
            integer() | nil,
            boolean(),
            {ExTorch.Tensor.t(), ExTorch.Tensor.t()} | nil
          ) ::
            {ExTorch.Tensor.t(), ExTorch.Tensor.t()}
    defbinding(mode(input, dim \\ -1, keepdim \\ false, out \\ nil))

    @doc """
    Returns the sum of all elements (or alongside an axis) in the input tensor, treating NaNs as zeros.

    ## Arguments
    - `input` (`ExTorch.Tensor`) - the input tensor.

    ## Optional arguments
    - `dim` (`integer | tuple | nil`) - the dimension or dimensions to reduce. If `nil`,
    all dimensions are reduced. Default: `nil`
    - `keepdim` (`boolean`) - whether the output tensor has `dim` retained or not. Default: `false`
    - `dtype` (`ExTorch.DType` or `nil`) - the desired data type of returned tensor.
    If specified, the `input` tensor is casted to `dtype` before the operation
    is performed. This is useful for preventing data type overflows. Default: `nil`.

    ## Examples
        iex> input =
        ...>   ExTorch.tensor(
        ...>     [
        ...>       [4, 4, 4, :nan],
        ...>       [3, :nan, 1, 1],
        ...>       [3, 2, :nan, 0]
        ...>     ]
        ...>   )

        # Sum all elements in the tensor, ignoring NaNs
        iex> ExTorch.nansum(input)
        #Tensor<
        22.
        [size: {}, dtype: :float, device: :cpu, requires_grad: false]>

        # Sum all elements in the last dimension, keeping dims and casting to double.
        iex> ExTorch.nansum(input, -1, keepdim: true, dtype: :double)
        #Tensor<
        [[12.],
         [ 5.],
         [ 5.]]
        [
          size: {3, 1},
          dtype: :double,
          device: :cpu,
          requires_grad: false
        ]>

    """
    @spec nansum(ExTorch.Tensor.t(), integer() | tuple() | nil, boolean(), ExTorch.DType.dtype()) ::
            ExTorch.Tensor.t()
    defbinding(nansum(input, dim \\ nil, keepdim \\ false, dtype \\ nil),
      input:
        if dtype do
          ExTorch.Tensor.to(input, dtype: dtype)
        else
          input
        end,
      dim: dim || {},
      omitted_args: [:dtype]
    )

    @doc """
    Returns the product of all elements (or alongside an axis) in the input tensor.

    ## Arguments
    - `input` (`ExTorch.Tensor`) - the input tensor.

    ## Optional arguments
    - `dim` (`integer | tuple | nil`) - the dimension or dimensions to reduce. If `nil`,
    all dimensions are reduced. Default: `nil`
    - `keepdim` (`boolean`) - whether the output tensor has `dim` retained or not. Default: `false`
    - `dtype` (`ExTorch.DType` or `nil`) - the desired data type of returned tensor.
    If specified, the `input` tensor is casted to `dtype` before the operation
    is performed. This is useful for preventing data type overflows. Default: `nil`.

    ## Notes
    * `keepdim` does not apply when `dim = nil`.

    ## Examples
        iex> a = ExTorch.randint(1, 3, {3, 4})
        #Tensor<
        [[1., 1., 1., 2.],
         [1., 2., 1., 1.],
         [1., 2., 1., 2.]]
        [
          size: {3, 4},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

        # Multiply all elements in the tensor
        iex> ExTorch.prod(a)
        #Tensor<
        16.
        [size: {}, dtype: :float, device: :cpu, requires_grad: false]>

        # Multiply all elements in the last dimension, keep all dimensions
        iex> ExTorch.prod(a, -1, keepdim: true)
        #Tensor<
        [[2.],
         [2.],
         [4.]]
        [
          size: {3, 1},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>
    """
    @spec prod(ExTorch.Tensor.t(), integer() | tuple() | nil, boolean(), ExTorch.DType.dtype()) ::
            ExTorch.Tensor.t()
    defbinding(prod(input, dim \\ nil, keepdim \\ false, dtype \\ nil),
      input:
        if dtype do
          ExTorch.Tensor.to(input, dtype: dtype)
        else
          input
        end,
      omitted_args: [:dtype]
    )

    @doc """
    Computes the q-th quantiles of each row of the `input` tensor along the dimension `dim`.

    To compute the quantile, we map q in [0, 1] to the range of indices [0, n] to find the
    location of the quantile in the sorted input. If the quantile lies between two data points
    `a < b` with indices `i` and `j` in the sorted order, result is computed according to the
    given `interpolation` method as follows:

    * `:linear`: `a + (b - a) * fraction`, where `fraction` is the fractional part of the computed quantile index.
    * `lower`: `a`.
    * `higher`: `b`.
    * `nearest`: `a` or `b`, whichever’s index is closer to the computed quantile index (rounding down for .5 fractions).
    * `midpoint`: `(a + b) / 2`.

    If `q` is a 1D tensor, the first dimension of the output represents the quantiles and has size equal
    to the size of `q`, the remaining dimensions are what remains from the reduction.

    ## Arguments
    - `input` (`ExTorch.Tensor`) - the input tensor.
    - `q` (`ExTorch.Tensor` | `floating`) - a scalar or 1D tensor of values in the range [0, 1].

    ## Optional arguments
    - `dim` (`integer` | `nil`) - the dimension to reduce. If `nil`, `input` will be flattened before
    computation. Default: `nil`
    - `keepdim` (`boolean`) - whether the output has `dim` retained or not. Default: `false`
    - `interpolation` (`atom`) interpolation method to use when the desired quantile lies between two data points.
    Can be `:linear`, `:lower`, `:higher`, `:midpoint` and `:nearest`. Default: `:linear`.
    - `out` (`ExTorch.Tensor | nil`) - the optional output pre-allocated tensor. Default: `nil`

    ## Examples
        iex> a = ExTorch.randn({2, 3})
        #Tensor<
        [[ 2.1818, -1.5810,  0.6152],
         [ 0.2525, -0.7425,  0.3769]]
        [
          size: {2, 3},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>
        iex> q = torch.tensor([0.25, 0.5, 0.75])

        # Quantiles in the last dimension, keep output dimensions
        iex> ExTorch.quantile(a, q, dim: 1, keepdim: true)
        #Tensor<
        [[[-0.4829],
          [-0.2450]],

         [[ 0.6152],
          [ 0.2525]],

         [[ 1.3985],
          [ 0.3147]]]
        [
          size: {3, 2, 1},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

        iex> a = ExTorch.arange(4)
        #Tensor<
        [0.0000, 1.0000, 2.0000, 3.0000]
        [
          size: {4},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

        # Interpolation modes
        iex> ExTorch.quantile(a, 0.6, interpolation: :linear)
        #Tensor<
        [1.8000]
        [size: {1}, dtype: :float, device: :cpu, requires_grad: false]>
        iex> ExTorch.quantile(a, 0.6, interpolation: :lower)
        #Tensor<
        [1.]
        [size: {1}, dtype: :float, device: :cpu, requires_grad: false]>
        iex> ExTorch.quantile(a, 0.6, interpolation: :higher)
        #Tensor<
        [2.]
        [size: {1}, dtype: :float, device: :cpu, requires_grad: false]>
        iex> ExTorch.quantile(a, 0.6, interpolation: :midpoint)
        #Tensor<
        [1.5000]
        [size: {1}, dtype: :float, device: :cpu, requires_grad: false]>
        iex> ExTorch.quantile(a, 0.6, interpolation: :nearest)
        #Tensor<
        [2.]
        [size: {1}, dtype: :float, device: :cpu, requires_grad: false]>

    """
    @spec quantile(
            ExTorch.Tensor.t(),
            float() | ExTorch.Tensor.t(),
            integer() | nil,
            boolean(),
            :linear | :lower | :higher | :midpoint | :nearest,
            ExTorch.Tensor.t() | nil
          ) :: ExTorch.Tensor.t()
    defbinding(
      quantile(input, q, dim \\ nil, keepdim \\ false, interpolation \\ :linear, out \\ nil),
      q:
        case q do
          %ExTorch.Tensor{} -> q
          _ -> ExTorch.tensor(q, dtype: :float)
        end
    )

    @doc """
    This is a variant of `ExTorch.quantile/6` that “ignores” NaN values, computing the quantiles `q` as
    if NaN values in `input` did not exist. If all values in a reduced row are NaN then the quantiles
    for that reduction will be NaN. See the documentation for `ExTorch.quantile/6`.

    ## Arguments
    - `input` (`ExTorch.Tensor`) - the input tensor.
    - `q` (`ExTorch.Tensor` | `floating`) - a scalar or 1D tensor of values in the range [0, 1].

    ## Optional arguments
    - `dim` (`integer` | `nil`) - the dimension to reduce. If `nil`, `input` will be flattened before
    computation. Default: `nil`
    - `keepdim` (`boolean`) - whether the output has `dim` retained or not. Default: `false`
    - `interpolation` (`atom`) interpolation method to use when the desired quantile lies between two data points.
    Can be `:linear`, `:lower`, `:higher`, `:midpoint` and `:nearest`. Default: `:linear`.
    - `out` (`ExTorch.Tensor | nil`) - the optional output pre-allocated tensor. Default: `nil`

    ## Examples
        # Compute quantiles throughout all tensor elements, ignoring :nan values
        iex> a = ExTorch.tensor([:nan, 1, 2])
        iex> ExTorch.nanquantile(a, 0.5)
        #Tensor<
        [1.5000]
        [size: {1}, dtype: :float, device: :cpu, requires_grad: false]>

        # Compute quantiles across specific dimensions, ignoring :nan values
        iex> a = ExTorch.tensor([[:nan, :nan], [1, 2]])
        iex> ExTorch.nanquantile(a, 0.5, dim: 0)
        #Tensor<
        [[1., 2.]]
        [
          size: {1, 2},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>
        iex> ExTorch.nanquantile(a, 0.5, dim: 1)
        #Tensor<
        [[   nan, 1.5000]]
        [
          size: {1, 2},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

    """
    @spec nanquantile(
            ExTorch.Tensor.t(),
            float() | ExTorch.Tensor.t(),
            integer() | nil,
            boolean(),
            :linear | :lower | :higher | :midpoint | :nearest,
            ExTorch.Tensor.t() | nil
          ) :: ExTorch.Tensor.t()
    defbinding(
      nanquantile(input, q, dim \\ nil, keepdim \\ false, interpolation \\ :linear, out \\ nil),
      q:
        case q do
          %ExTorch.Tensor{} -> q
          _ -> ExTorch.tensor(q, dtype: :float)
        end
    )

    @doc ~S"""
    Calculates the standard deviation over the dimensions specified by `dim`.

    `dim` can be a single dimension, list of dimensions, or `nil` to reduce over all dimensions.

    The standard deviation ($\sigma$) is calculated as

    $$\sigma = \sqrt{\frac{1}{N - \delta N} \sum\_{i=0}^{N - 1} (x\_i - \bar{x})^2}$$

    where $x$ is the sample set of elements, $\bar{x}$ is the sample mean, $N$ is the number of samples
    and $\delta N$ is the `correction`.

    If `keepdim` is `true`, the output tensors are of the same size as `input`
    except in the dimension `dim` where they are of size 1.
    Otherwise, `dim` is squeezed (see `ExTorch.squeeze`), resulting in the outputs
    tensor having 1 fewer dimension than `input`.

    ## Arguments
    - `input` (`ExTorch.Tensor`) - the input tensor.

    ## Optional arguments
    - `dim` (`integer | tuple | nil`) - the dimension or dimensions to reduce. If `nil`,
    all dimensions are reduced. Default: `nil`
    - `correction` (`integer`) - difference between the sample size and sample degrees of freedom.
    Defaults to [Bessel's correction](https://en.wikipedia.org/wiki/Bessel%27s_correction). Default: 1
    - `keepdim` (`boolean`) - whether the output tensor has `dim` retained or not. Default: `false`
    - `out` (`ExTorch.Tensor | nil`) - the optional output pre-allocated tensor. Default: `nil`

    ## Examples
        iex> a = ExTorch.randn({4, 4})
        #Tensor<
        [[ 0.0686,  0.7169,  0.2143,  1.5755],
         [-1.6080,  0.9169, -0.0937,  1.2906],
         [ 0.5432,  2.4151, -0.3814,  0.2830],
         [-0.0724,  0.7037, -0.1951, -0.1191]]
        [
          size: {4, 4},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

        # Compute the standard deviation of all tensor elements
        iex> ExTorch.std(a)
        #Tensor<
        0.9167
        [size: {}, dtype: :float, device: :cpu, requires_grad: false]>

        # Compute the standard deviation of elements in the last dimension, keeping total dimensions
        iex> ExTorch.std(a, -1, keepdim: true)
        #Tensor<
        [[0.6804],
         [1.2957],
         [1.1984],
         [0.4194]]
        [
          size: {4, 1},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>
    """
    @spec std_dev(
            ExTorch.Tensor.t(),
            integer() | tuple() | nil,
            integer(),
            boolean(),
            ExTorch.Tensor.t() | nil
          ) :: ExTorch.Tensor.t()
    defbinding(std_dev(input, dim \\ nil, correction \\ 1, keepdim \\ false, out \\ nil),
      fn_aliases: [:std]
    )

    @doc """
    Calculates the standard deviation and mean over the dimensions specified by `dim`.

    `dim` can be a single dimension, list of dimensions, or `nil` to reduce over all dimensions.
    It returns a tuple `{std, mean}` containing the standard deviation and mean, respectively.

    The standard deviation ($\sigma$) is calculated as

    $$\sigma = \sqrt{\frac{1}{N - \delta N} \sum\_{i=0}^{N - 1} (x\_i - \bar{x})^2}$$

    where $x$ is the sample set of elements, $\bar{x}$ is the sample mean, $N$ is the number of samples
    and $\delta N$ is the `correction`.

    If `keepdim` is `true`, the output tensors are of the same size as `input`
    except in the dimension `dim` where they are of size 1.
    Otherwise, `dim` is squeezed (see `ExTorch.squeeze`), resulting in the outputs
    tensor having 1 fewer dimension than `input`.

    ## Arguments
    - `input` (`ExTorch.Tensor`) - the input tensor.

    ## Optional arguments
    - `dim` (`integer | tuple | nil`) - the dimension or dimensions to reduce. If `nil`,
    all dimensions are reduced. Default: `nil`
    - `correction` (`integer`) - difference between the sample size and sample degrees of freedom.
    Defaults to [Bessel's correction](https://en.wikipedia.org/wiki/Bessel%27s_correction). Default: 1
    - `keepdim` (`boolean`) - whether the output tensor has `dim` retained or not. Default: `false`
    - `out` (`{ExTorch.Tensor, ExTorch.Tensor} | nil`) - a tuple containing the optional output pre-allocated tensors. Default: `nil`

    ## Examples
        iex> a = ExTorch.randn({4, 4})
        #Tensor<
        [[-0.8204,  0.0761, -0.5242,  0.7905],
         [ 0.4202,  0.5431, -0.9726,  0.7407],
         [-1.5224,  1.1669, -1.4509,  0.0034],
         [-0.8064,  1.2111,  1.3384, -1.2709]]
        [
          size: {4, 4},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

        # Compute standard deviation and mean of all tensor elements
        iex> {std, mean} = ExTorch.std_mean(a)
        iex> std
        #Tensor<
        0.9926
        [size: {}, dtype: :float, device: :cpu, requires_grad: false]>
        iex> mean
        #Tensor<
        -0.0673
        [size: {}, dtype: :float, device: :cpu, requires_grad: false]>

        # Compute standard deviation and mean of all tensor elements in the last dimension
        iex> {std, mean} = ExTorch.std_mean(a, -1, keepdim: true)
        iex> std
        #Tensor<
        [[0.7121],
        [0.7815],
        [1.2874],
        [1.3500]]
        [
          size: {4, 1},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>
        iex> mean
        #Tensor<
        [[-0.1195],
        [ 0.1829],
        [-0.4507],
        [ 0.1180]]
        [
          size: {4, 1},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

    """
    @spec std_mean(
            ExTorch.Tensor.t(),
            integer() | tuple() | nil,
            integer(),
            boolean(),
            {ExTorch.Tensor.t(), ExTorch.Tensor.t()} | nil
          ) :: {ExTorch.Tensor.t(), ExTorch.Tensor.t()}
    defbinding(std_mean(input, dim \\ nil, correction \\ 1, keepdim \\ false, out \\ nil))

    @doc ~S"""
    Returns the unique elements of the `input` tensor.

    Depending on the value of `return_inverse` and `return_counts`, this function
    can return either a single tensor, a tuple of two tensors, or a tuple of three tensors.

    ## Arguments
    - `input` (`ExTorch.Tensor`) - the input tensor.

    ## Optional arguments
    - `sorted` (`boolean`) -  whether to sort the unique elements in ascending order before returning. Default: `true`
    - `return_inverse` (`boolean`) - whether to also return the indices for where elements in the original input
    ended up in the returned unique list. Default: `false`
    - `return_counts` (`boolean`) - whether to also return the counts for each unique element. Default: `false`
    - `dim` (`integer | nil`) - the dimension to operate upon. If `nil`, the unique of the flattened input
    is returned. Otherwise, each of the tensors indexed by the given dimension is treated as one of the
    elements to apply the unique operation upon. See examples for more details. Default: `nil`

    ## Examples
        iex> a = ExTorch.randint(-1, 4, {4, 4}, dtype: :int64)
        #Tensor<
        [[ 2,  2, -1,  0],
         [ 1,  3,  1, -1],
         [-1,  2,  0,  1],
         [ 3,  2, -1,  3]]
        [
          size: {4, 4},
          dtype: :long,
          device: :cpu,
          requires_grad: false
        ]>

        # Compute a tensor's unique values
        iex> ExTorch.unique(a)
        #Tensor<
        [-1,  0,  1,  2,  3]
        [
          size: {5},
          dtype: :long,
          device: :cpu,
          requires_grad: false
        ]>

        # Compute unique values and inverse tensor
        iex> {unique, inverse} = ExTorch.unique(a, return_inverse: true)
        iex> inverse
        #Tensor<
        [[3, 3, 0, 1],
         [2, 4, 2, 0],
         [0, 3, 1, 2],
         [4, 3, 0, 4]]
        [
          size: {4, 4},
          dtype: :long,
          device: :cpu,
          requires_grad: false
        ]>

        # Compute unique values and count tensor
        iex> {unique, count} = ExTorch.unique(a, return_counts: true)
        iex> count
        #Tensor<
        [4, 2, 3, 4, 3]
        [
          size: {5},
          dtype: :long,
          device: :cpu,
          requires_grad: false
        ]>

        # Compute unique values, inverse and count tensors
        iex> {unique, inverse, count} = ExTorch.unique(a, return_inverse: true, return_counts: true)

        # Compute unique values across a dimension
        iex> a = ExTorch.tensor([[0, 1, 1], [-1, -1, -1], [0, 1, 1]], dtype: :int64)
        iex> ExTorch.unique(a, dim: 0)
        #Tensor<
        [[-1, -1, -1],
         [ 0,  1,  1]]
        [
          size: {2, 3},
          dtype: :long,
          device: :cpu,
          requires_grad: false
        ]>

    """
    @spec unique(ExTorch.Tensor.t(), boolean(), boolean(), boolean(), integer() | nil) ::
            ExTorch.Tensor.t()
            | {ExTorch.Tensor.t(), ExTorch.Tensor.t()}
            | {ExTorch.Tensor.t(), ExTorch.Tensor.t(), ExTorch.Tensor.t()}
    defbinding(
      unique(input, sorted \\ true, return_inverse \\ false, return_counts \\ false, dim \\ nil)
    )

    @doc """
    Eliminates all but the first element from every consecutive group of equivalent elements.

    Depending on the value of `return_inverse` and `return_counts`, this function
    can return either a single tensor, a tuple of two tensors, or a tuple of three tensors.

    This function is different from `ExTorch.unique/5` in the sense that this function only
    eliminates consecutive duplicate values. This semantics is similar to `std::unique` in C++.

    ## Arguments
    - `input` (`ExTorch.Tensor`) - the input tensor.

    ## Optional arguments
    - `return_inverse` (`boolean`) - whether to also return the indices for where elements in the original input
    ended up in the returned unique list. Default: `false`
    - `return_counts` (`boolean`) - whether to also return the counts for each unique element. Default: `false`
    - `dim` (`integer | nil`) - the dimension to operate upon. If `nil`, the unique of the flattened input
    is returned. Otherwise, each of the tensors indexed by the given dimension is treated as one of the
    elements to apply the unique operation upon. See examples for more details. Default: `nil`

    ## Examples
        iex> a = ExTorch.tensor([1, 1, 2, 2, 3, 1, 1, 2], dtype: :int32)

        # Find unique consecutive elements in a tensor
        iex> ExTorch.unique_consecutive(a)
        #Tensor<
        [1, 2, 3, 1, 2]
        [
          size: {5},
          dtype: :int,
          device: :cpu,
          requires_grad: false
        ]>

        # Compute unique values and inverse tensor
        iex> {unique, inverse} = ExTorch.unique_consecutive(a, return_inverse: true)
        iex> inverse
        #Tensor<
        [0, 0, 1, 1, 2, 3, 3, 4]
        [
          size: {8},
          dtype: :long,
          device: :cpu,
          requires_grad: false
        ]>

        # Compute unique values and count tensor
        iex> {unique, count} = ExTorch.unique_consecutive(a, return_counts: true)
        iex> count
        #Tensor<
        [2, 2, 1, 2, 1]
        [
          size: {5},
          dtype: :long,
          device: :cpu,
          requires_grad: false
        ]>


        # Compute unique values, inverse and count tensors
        iex> {unique, inverse, count} = ExTorch.unique(a, return_inverse: true, return_counts: true)

        # Compute unique consecutive values across a dimension
        iex> a = ExTorch.tensor([[-1, -1, -1], [0, 1, 1], [0, 1, 1], [-1, -1, -1]], dtype: :int64)
        iex> ExTorch.unique(a, dim: 0)
        #Tensor<
        [[-1, -1, -1],
         [ 0,  1,  1]]
        [
          size: {2, 3},
          dtype: :long,
          device: :cpu,
          requires_grad: false
        ]>

    """
    @spec unique_consecutive(ExTorch.Tensor.t(), boolean(), boolean(), integer() | nil) ::
            ExTorch.Tensor.t()
            | {ExTorch.Tensor.t(), ExTorch.Tensor.t()}
            | {ExTorch.Tensor.t(), ExTorch.Tensor.t(), ExTorch.Tensor.t()}
    defbinding(
      unique_consecutive(
        input,
        return_inverse \\ false,
        return_counts \\ false,
        dim \\ nil
      )
    )

    @doc ~S"""
    Calculates the variance over the dimensions specified by `dim`.

    `dim` can be a single dimension, list of dimensions, or `nil` to reduce over all dimensions.

    The variance ($\sigma^2$) is calculated as

    $$\sigma^2 = \frac{1}{N - \delta N} \sum\_{i=0}^{N - 1} (x\_i - \bar{x})^2$$

    where $x$ is the sample set of elements, $\bar{x}$ is the sample mean, $N$ is the number of samples
    and $\delta N$ is the `correction`.

    If `keepdim` is `true`, the output tensors are of the same size as `input`
    except in the dimension `dim` where they are of size 1.
    Otherwise, `dim` is squeezed (see `ExTorch.squeeze`), resulting in the outputs
    tensor having 1 fewer dimension than `input`.

    ## Arguments
    - `input` (`ExTorch.Tensor`) - the input tensor.

    ## Optional arguments
    - `dim` (`integer | tuple | nil`) - the dimension or dimensions to reduce. If `nil`,
    all dimensions are reduced. Default: `nil`
    - `correction` (`integer`) - difference between the sample size and sample degrees of freedom.
    Defaults to [Bessel's correction](https://en.wikipedia.org/wiki/Bessel%27s_correction). Default: 1
    - `keepdim` (`boolean`) - whether the output tensor has `dim` retained or not. Default: `false`
    - `out` (`ExTorch.Tensor | nil`) - the optional output pre-allocated tensor. Default: `nil`

    ## Examples
        iex> a = ExTorch.randn({4, 4})
        #Tensor<
        [[-0.9319,  0.1259,  0.0744,  0.3516],
         [-0.1965,  0.8596, -1.2986, -0.6350],
         [-0.0211,  0.2856, -1.3375, -1.4459],
         [-0.0489,  0.4821, -0.5326, -2.3099]]
        [
          size: {4, 4},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

        # Compute the variance of all tensor elements
        iex> ExTorch.var(a)
        #Tensor<
        0.7327
        [size: {}, dtype: :float, device: :cpu, requires_grad: false]>

        # Compute the variance of elements in the last dimension, keeping total dimensions
        iex> ExTorch.var(a, -1, keepdim: true)
        #Tensor<
        [[0.3258],
         [0.8211],
         [0.7917],
         [1.4677]]
        [
          size: {4, 1},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>
    """
    @spec var(
            ExTorch.Tensor.t(),
            integer() | tuple() | nil,
            integer(),
            boolean(),
            ExTorch.Tensor.t() | nil
          ) :: ExTorch.Tensor.t()
    defbinding(var(input, dim \\ nil, correction \\ 1, keepdim \\ false, out \\ nil))

    @doc ~S"""
    Calculates the variance and mean over the dimensions specified by `dim`.

    `dim` can be a single dimension, list of dimensions, or `nil` to reduce over all dimensions.
    It returns a tuple `{var, mean}` containing the variance and mean, respectively.

    The variance ($\sigma^2$) is calculated as

    $$\sigma^2 = \frac{1}{N - \delta N} \sum\_{i=0}^{N - 1} (x\_i - \bar{x})^2$$

    where $x$ is the sample set of elements, $\bar{x}$ is the sample mean, $N$ is the number of samples
    and $\delta N$ is the `correction`.

    If `keepdim` is `true`, the output tensors are of the same size as `input`
    except in the dimension `dim` where they are of size 1.
    Otherwise, `dim` is squeezed (see `ExTorch.squeeze`), resulting in the outputs
    tensor having 1 fewer dimension than `input`.

    ## Arguments
    - `input` (`ExTorch.Tensor`) - the input tensor.

    ## Optional arguments
    - `dim` (`integer | tuple | nil`) - the dimension or dimensions to reduce. If `nil`,
    all dimensions are reduced. Default: `nil`
    - `correction` (`integer`) - difference between the sample size and sample degrees of freedom.
    Defaults to [Bessel's correction](https://en.wikipedia.org/wiki/Bessel%27s_correction). Default: 1
    - `keepdim` (`boolean`) - whether the output tensor has `dim` retained or not. Default: `false`
    - `out` (`{ExTorch.Tensor, ExTorch.Tensor} | nil`) - a tuple containing the optional output pre-allocated tensors. Default: `nil`

    ## Examples
        iex> a = ExTorch.randn({4, 4})
        #Tensor<
        [[-0.9319,  0.1259,  0.0744,  0.3516],
         [-0.1965,  0.8596, -1.2986, -0.6350],
         [-0.0211,  0.2856, -1.3375, -1.4459],
         [-0.0489,  0.4821, -0.5326, -2.3099]]
        [
          size: {4, 4},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

        # Compute variance and mean of all tensor elements
        iex> {var, mean} = ExTorch.var_mean(a)
        iex> var
        #Tensor<
        0.7327
        [size: {}, dtype: :float, device: :cpu, requires_grad: false]>
        iex> mean
        #Tensor<
        -0.4112
        [size: {}, dtype: :float, device: :cpu, requires_grad: false]>

        # Compute variance and mean of all tensor elements in the last dimension
        iex> {var, mean} = ExTorch.var_mean(a, -1, keepdim: true)
        iex> var
        #Tensor<
        [[0.3258],
         [0.8211],
         [0.7917],
         [1.4677]]
        [
          size: {4, 1},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>
        iex> mean
        #Tensor<
        [[-0.0950],
         [-0.3176],
         [-0.6297],
         [-0.6023]]
        [
          size: {4, 1},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

    """
    @spec var_mean(
            ExTorch.Tensor.t(),
            integer() | tuple() | nil,
            integer(),
            boolean(),
            {ExTorch.Tensor.t(), ExTorch.Tensor.t()} | nil
          ) :: {ExTorch.Tensor.t(), ExTorch.Tensor.t()}
    defbinding(var_mean(input, dim \\ nil, correction \\ 1, keepdim \\ false, out \\ nil))

    @doc """
    Counts the number of non-zero values in the tensor `input` along the given `dim`.
    If no dim is specified then all non-zeros in the tensor are counted.

    ## Arguments
    - `input` (`ExTorch.Tensor`) - the input tensor.

    ## Optional arguments
    - `dim` (`integer | tuple | nil`) - the dimension or dimensions to reduce. If `nil`,
    all dimensions are reduced. Default: `nil`

    ## Examples
        iex> x = ExTorch.zeros({3, 3})
        iex> x = ExTorch.index_put(x, ExTorch.gt(ExTorch.randn({3, 3}), 0.5), 1)
        #Tensor<
        [[0.0000, 0.0000, 1.0000],
         [0.0000, 0.0000, 0.0000],
         [1.0000, 0.0000, 0.0000]]
        [
          size: {3, 3},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

        # Count overall nonzero elements
        iex> ExTorch.count_nonzero(x)
        #Tensor<
        2
        [size: {}, dtype: :long, device: :cpu, requires_grad: false]>

        # Count nonzero elements in the first dimension
        iex> ExTorch.count_nonzero(x, 0)
        #Tensor<
        [1, 0, 1]
        [size: {3}, dtype: :long, device: :cpu, requires_grad: false]>
    """
    @spec count_nonzero(ExTorch.Tensor.t(), integer() | tuple() | nil) :: ExTorch.Tensor.t()
    defbinding(count_nonzero(input, dim \\ nil))
  end
end
