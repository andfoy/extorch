defmodule ExTorch.Native.Tensor.Ops.Comparison do
  @moduledoc false
  use ExTorch.Native.BindingDeclaration

  defbindings(:tensor_comparison) do
    @doc ~S"""
    This function checks if `input` and `other` satisfy the condition:
    $$|\text{input} - \text{other}| \leq \texttt{atol} + \texttt{rtol} \times |\text{other}|$$
    elementwise, for all elements of `input` and `other`.

    ## Arguments
    - `input` - First tensor to compare (`ExTorch.Tensor`)
    - `other` - Second tensor to compare (`ExTorch.Tensor`)

    ## Optional arguments
    - `rtol` - Relative tolerance (`float`). Default: 1.0e-5
    - `atol` - Absolute tolerance (`float`). Default: 1.0e-8
    - `equal_nan` - If `true`, then two `NaN`s will be considered equal. Default: `false`.

    ## Examples
        iex> ExTorch.allclose(ExTorch.tensor([10000.0, 1.0e-07]), ExTorch.tensor([10000.1, 1.0e-08]))
        false
        iex> ExTorch.allclose(ExTorch.tensor([10000.0, 1.0e-08]), ExTorch.tensor([10000.1, 1.0e-09]))
        true
        iex> ExTorch.allclose(ExTorch.tensor([1.0, :nan]), ExTorch.tensor([1.0, :nan]))
        false
        iex> ExTorch.allclose(ExTorch.tensor([1.0, :nan]), ExTorch.tensor([1.0, :nan]), equal_nan: true)
        true
    """
    @spec allclose(ExTorch.Tensor.t(), ExTorch.Tensor.t(), float(), float(), boolean()) ::
            boolean()
    defbinding(allclose(input, other, rtol \\ 1.0e-5, atol \\ 1.0e-8, equal_nan \\ false))

    @doc """
    Returns the indices that sort a tensor along a given dimension in ascending order by value.

    This is the second value returned by `ExTorch.sort/5`. See its documentation for the exact
    semantics of this method. If `stable` is `true` then the sorting routine becomes stable,
    preserving the order of equivalent elements. If `false`, the relative order of values
    which compare equal is not guaranteed. `true` is slower.

    ## Arguments
    - `input` - Input tensor. (`ExTorch.Tensor`)

    ## Optional arguments
    - `dim` - the dimension to sort along ('integer()'). Default: -1
    - `descending` - controls the sorting order (ascending or descending) (`boolean()`). Default: `false`
    - `stable` - controls the relative order of equivalent elements (`boolean()`). Default: `false`

    ## Examples
        iex> a = ExTorch.randn({4, 4})
        #Tensor<
        [[-1.2732,  0.8419, -0.0140,  0.4717],
         [-1.1627, -0.2813, -0.5655, -0.1348],
         [ 1.5269, -0.2712,  0.5134, -1.5580],
         [ 0.6169, -1.0332,  0.4478, -0.9864]]
        [
          size: {4, 4},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

        # Sort alongside an specific dimension
        iex> ExTorch.argsort(a, dim: 1)
        #Tensor<
        [[0, 2, 3, 1],
         [0, 2, 1, 3],
         [3, 1, 2, 0],
         [1, 3, 2, 0]]
        [
          size: {4, 4},
          dtype: :long,
          device: :cpu,
          requires_grad: false
        ]>
    """
    @spec argsort(ExTorch.Tensor.t(), integer(), boolean(), boolean()) :: ExTorch.Tensor.t()
    defbinding(argsort(input, dim \\ -1, descending \\ false, stable \\ false))

    @doc """
    Sorts the elements of the input tensor along a given dimension in ascending order by value.

    * If `dim` is not given, the last dimension of the `input` is chosen.
    * If `descending` is `true` then the elements are sorted in descending order by value.
    * If `stable` is `true` then the sorting routine becomes stable, preserving
    the order of equivalent elements.

    A tuple of `{values, indices}` is returned, where the `values` are the sorted values
    and `indices` are the indices of the elements in the original input tensor.

    ## Arguments
    - `input` - Input tensor. (`ExTorch.Tensor`)

    ## Optional arguments
    - `dim` - the dimension to sort along. ('integer()'). Default: -1
    - `descending` - controls the sorting order. (ascending or descending) (`boolean()`). Default: `false`
    - `stable` - controls the relative order of equivalent elements. (`boolean()`). Default: `false`
    - `out` - the output tuple of `{values, indices}` that can be optionally given as
    output buffers. (`{ExTorch.Tensor, ExTorch.Tensor}`). Default: `nil`

    ## Examples
        iex> a = ExTorch.randn({3, 4})
        #Tensor<
        [[ 0.7517,  0.5590, -0.1417, -0.1662],
         [-0.1247,  0.5669,  0.0484,  0.4289],
         [ 0.0876, -0.5951, -1.0296,  0.0093]]
        [
          size: {3, 4},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

        # Sort tensor on the last dimension
        iex> {values, indices} = ExTorch.sort(a)
        iex> values
        #Tensor<
        [[-0.1662, -0.1417,  0.5590,  0.7517],
         [-0.1247,  0.0484,  0.4289,  0.5669],
         [-1.0296, -0.5951,  0.0093,  0.0876]]
        [
          size: {3, 4},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>
        iex> indices
        #Tensor<
        [[3, 2, 1, 0],
         [0, 2, 3, 1],
         [2, 1, 3, 0]]
        [
          size: {3, 4},
          dtype: :long,
          device: :cpu,
          requires_grad: false
        ]>

        # Sort tensor on the first dimension while reusing values and indices
        iex> ExTorch.sort(a, 0, out: {values, indices})
        iex> values
        #Tensor<
        [[-0.1247, -0.5951, -1.0296, -0.1662],
         [ 0.0876,  0.5590, -0.1417,  0.0093],
         [ 0.7517,  0.5669,  0.0484,  0.4289]]
        [
          size: {3, 4},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>
        iex> indices
        #Tensor<
        [[1, 2, 2, 0],
         [2, 0, 0, 2],
         [0, 1, 1, 1]]
        [
          size: {3, 4},
          dtype: :long,
          device: :cpu,
          requires_grad: false
        ]>

    """
    @spec sort(
            ExTorch.Tensor.t(),
            integer(),
            boolean(),
            boolean(),
            {ExTorch.Tensor.t(), ExTorch.Tensor.t()} | nil
          ) :: {ExTorch.Tensor.t(), ExTorch.Tensor.t()}
    defbinding(sort(input, dim \\ -1, descending \\ false, stable \\ false, out \\ nil))

    @doc """
    Computes element-wise equality.

    The second argument can be a number or a tensor whose shape is broadcastable with the first argument.
    It will return a boolean tensor of the same shape as `input`, where a `true` entry
    represents a value that is equal on both `input` and `other`, and `false` otherwise.

    ## Arguments
    - `input` - the tensor to compare (`ExTorch.Tensor`).
    - `other` - the tensor or value to compare (`Extorch.Tensor` or value)

    ## Optional arguments
    - `out` - an optional pre-allocated tensor used to store the comparison result. (`ExTorch.Tensor`)

    ## Examples
        # Compare against an scalar value.
        iex> a = ExTorch.tensor([[1, 2], [3, 4]])
        iex> ExTorch.eq(a, 1)
        #Tensor<
        [[ true, false],
         [false, false]]
        [
          size: {2, 2},
          dtype: :bool,
          device: :cpu,
          requires_grad: false
        ]>

        # Compare against a broadcastable value.
        iex> ExTorch.eq(a, [1, 2])
        #Tensor<
        [[ true,  true],
         [false, false]]
        [
          size: {2, 2},
          dtype: :bool,
          device: :cpu,
          requires_grad: false
        ]>

        # Compare against another tensor.
        iex> ExTorch.eq(a, ExTorch.tensor([[1, 1], [4, 4]]))
        #Tensor<
        [[ true, false],
         [false,  true]]
        [
          size: {2, 2},
          dtype: :bool,
          device: :cpu,
          requires_grad: false
        ]>

    """
    @spec eq(
            ExTorch.Tensor.t(),
            ExTorch.Tensor.t()
            | list()
            | number()
            | boolean()
            | ExTorch.Complex.t()
            | :nan
            | :inf
            | :ninf,
            ExTorch.Tensor.t() | nil
          ) :: ExTorch.Tensor.t()
    defbinding(eq(input, other, out \\ nil),
      other:
        case other do
          %ExTorch.Tensor{} ->
            other

          _ ->
            ExTorch.tensor(other,
              device: ExTorch.Tensor.device(input),
              requires_grad: false
            )
        end
    )

    @doc """
    Strict element-wise equality for two tensors.

    This function will return `true` if both inputs have the same size and elements.
    `false`, otherwise.

    ## Arguments
      - `input` - tensor to compare (`ExTorch.Tensor`)
      - `other` - tensor to compare (`ExTorch.Tensor`)

    ## Examples
        iex> ExTorch.equal(ExTorch.tensor([1, 2]), ExTorch.tensor([1, 2]))
        true

        iex> ExTorch.equal(ExTorch.tensor([1, 2]), ExTorch.tensor([1]))
        false
    """
    @spec equal(ExTorch.Tensor.t(), ExTorch.Tensor.t()) :: boolean()
    defbinding(equal(input, other))
  end
end
