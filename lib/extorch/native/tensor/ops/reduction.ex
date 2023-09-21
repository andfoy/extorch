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
    - `dim` (`nil | integer() | tuple()`) - the dimension(s) to reduce.

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
    @spec amax(ExTorch.Tensor.t(), integer() | tuple(), boolean(), ExTorch.Tensor.t() | nil) ::
            ExTorch.Tensor.t()
    defbinding(amax(input, dim, keepdim \\ false, out \\ nil))

    @doc """
    Returns the minimum value of each slice of the `input` tensor in the given dimension(s) dim.

    If `keepdim` is `true`, the output tensor is of the same size as `input` except in the dimension(s)
    `dim` where it is of size 1. Otherwise, `dim` is squeezed (see `Extorch.squeeze`), resulting in
    the output tensor having 1 (or `length(dim)`) fewer dimension(s).

    ## Arguments
    - `input` (`ExTorch.Tensor`) - the input tensor.
    - `dim` (`nil | integer() | tuple()`) - the dimension(s) to reduce.

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
    @spec amin(ExTorch.Tensor.t(), integer() | tuple(), boolean(), ExTorch.Tensor.t() | nil) ::
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
  end
end
