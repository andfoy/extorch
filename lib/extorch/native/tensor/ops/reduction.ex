defmodule ExTorch.Native.Tensor.Ops.Reduction do
  @moduledoc false
  use ExTorch.Native.BindingDeclaration

  # @spec max(ExTorch.Tensor.t(), ExTorch.Tensor.t()) :: ExTorch.Tensor.t()
  # def max(%ExTorch.Tensor{} = input, %ExTorch.Tensor{} = other)
  #     when is_struct(input, ExTorch.Tensor) and is_struct(other, ExTorch.Tensor) do
  #   ExTorch.maximum(input, other)
  # end

  # @spec max(ExTorch.Tensor.t(), ExTorch.Tensor.t(), ExTorch.Tensor.t()) :: ExTorch.Tensor.t()
  # def max(%ExTorch.Tensor{} = input, %ExTorch.Tensor{} = other, %ExTorch.Tensor{} = out)
  #     when is_struct(input, ExTorch.Tensor) and is_struct(other, ExTorch.Tensor) do
  #   ExTorch.maximum(input, other, out)
  # end

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
  end
end
