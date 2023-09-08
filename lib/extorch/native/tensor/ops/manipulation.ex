defmodule ExTorch.Native.Tensor.Ops.Manipulation do
  @moduledoc false

  use ExTorch.Native.BindingDeclaration

  defbindings(:tensor_manipulation) do
    @doc """
    Append an empty dimension to a tensor on a given dimension.

    ## Arguments
      - `tensor`: Input tensor (`ExTorch.Tensor`)
      - `dim`: Dimension (`integer()`)

    ## Examples
        iex> x = ExTorch.full({2, 2}, -2)
        #Tensor<
        [[-2., -2.],
         [-2., -2.]]
        [
          size: {2, 2},
          dtype: :double,
          device: :cpu,
          requires_grad: false
        ]>

        iex> ExTorch.unsqueeze(x, -1)
        #Tensor<
        [[[-2.],
          [-2.]],

         [[-2.],
          [-2.]]]
        [
          size: {2, 2, 1},
          dtype: :double,
          device: :cpu,
          requires_grad: false
        ]>

        iex> ExTorch.unsqueeze(x, 1)
        #Tensor<
        [[[-2., -2.]],

         [[-2., -2.]]]
        [
          size: {2, 1, 2},
          dtype: :double,
          device: :cpu,
          requires_grad: false
        ]>

        iex> ExTorch.unsqueeze(x, 0)
        #Tensor<
        [[[-2., -2.],
          [-2., -2.]]]
        [
          size: {1, 2, 2},
          dtype: :double,
          device: :cpu,
          requires_grad: false
        ]>

    """
    @spec unsqueeze(
            ExTorch.Tensor.t(),
            integer()
          ) :: ExTorch.Tensor.t()
    defbinding(unsqueeze(tensor, dim))

    @doc """
    Returns a tensor with the same data and number of elements as `input`, but
    with the specified shape.

    When possible, the returned tensor will be a view of `input`. Otherwise, it
    will be a copy. Contiguous inputs and inputs with compatible strides can be
    reshaped without copying, but you should not depend on the copying vs.
    viewing behavior.

    A single dimension may be -1, in which case it’s inferred from the\
    remaining dimensions and the number of elements in input.

    ## Arguments
      - `tensor`: input tensor (`ExTorch.Tensor`)
      - `shape`: the new shape (`[integer()] | tuple()`)

    ## Examples
        iex> a = ExTorch.arange(0, 20) |> ExTorch.reshape({5, 4})
        #Tensor<
        [[ 0.,  1.,  2.,  3.],
         [ 4.,  5.,  6.,  7.],
         [ 8.,  9., 10., 11.],
         [12., 13., 14., 15.],
         [16., 17., 18., 19.]]
        [
          size: {5, 4},
          dtype: :double,
          device: :cpu,
          requires_grad: false
        ]>

        iex> b = ExTorch.tensor([[0, 1], [2, 3]]) |> ExTorch.reshape({-1})
        #Tensor<
        [0, 1, 2, 3]
        [
          size: {4},
          dtype: :byte,
          device: :cpu,
          requires_grad: false
        ]>
    """
    @spec reshape(ExTorch.Tensor.t(), tuple() | [integer()]) :: ExTorch.Tensor.t()
    defbinding(reshape(tensor, shape))

    @doc """
    Returns a view of `input` with a flipped conjugate bit.
    If `input` has a non-complex dtype, this function just returns `input`.

    ## Arguments
      - `tensor`: Input tensor (`ExTorch.Tensor`)

    ## Notes
    `ExTorch.conj/1` performs a lazy conjugation, but the actual conjugated
    tensor can be materialized at any time using `ExTorch.resolve_conj/1`.

    ## Examples
        iex> a = ExTorch.rand({2, 2}, dtype: :complex64)
        #Tensor<
        [[0.5885+0.0263j, 0.8141+0.0605j],
         [0.9169+0.3126j, 0.6344+0.2768j]]
        [
          size: {2, 2},
          dtype: :complex_float,
          device: :cpu,
          requires_grad: false
        ]>

        # Conjugate the input
        iex> b = ExTorch.conj(a)
        #Tensor<
        [[0.5885-0.0263j, 0.8141-0.0605j],
         [0.9169-0.3126j, 0.6344-0.2768j]]
        [
          size: {2, 2},
          dtype: :complex_float,
          device: :cpu,
          requires_grad: false
        ]>

        # Check that conj bit is set to true
        iex> ExTorch.Tensor.is_conj(b)
        true

    """
    @spec conj(ExTorch.Tensor.t()) :: ExTorch.Tensor.t()
    defbinding(conj(tensor))
  end
end
