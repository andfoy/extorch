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

    @doc """
    Returns a view of the tensor conjugated and with the last two dimensions transposed.

    `ExTorch.adjoint(x)` is equivalent to `ExTorch.conj(ExTorch.transpose(x, -2, -1))` and to
    `ExTorch.transpose(x, -2, -1)` for real tensors.

    ## Arguments
    - `input` (`ExTorch.Tensor`) - the input tensor.

    ## Examples
        iex> x = ExTorch.arange(4)
        iex> a = ExTorch.complex(x, x) |> ExTorch.reshape({2, 2})
        #Tensor<
        [[0.0000+0.0000j, 1.0000+1.0000j],
         [2.0000+2.0000j, 3.0000+3.0000j]]
        [
          size: {2, 2},
          dtype: :complex_float,
          device: :cpu,
          requires_grad: false
        ]>

        iex> ExTorch.adjoint(a)
        #Tensor<
        [[0.0000-0.0000j, 2.0000-2.0000j],
         [1.0000-1.0000j, 3.0000-3.0000j]]
        [
          size: {2, 2},
          dtype: :complex_float,
          device: :cpu,
          requires_grad: false
        ]>
    """
    @spec adjoint(ExTorch.Tensor.t()) :: ExTorch.Tensor.t()
    defbinding(adjoint(input))

    @doc """
    Returns a tensor that is a transposed version of input. The given dimensions dim0 and dim1 are swapped.

    ## Arguments
    - `input` (`ExTorch.Tensor`) - the input tensor.
    - `dim0` (`integer()`) - the first dimension to transpose.
    - `dim1` (`integer()`) - the second dimension to transpose.

    ## Notes
    * If `input` is a strided tensor then the resulting out tensor shares its underlying storage with the
    `input` tensor, so changing the content of one would change the content of the other.
    * If `input` is a sparse tensor then the resulting out tensor does not share the underlying storage with the input tensor.
    * If input is a sparse tensor with compressed layout (SparseCSR, SparseBSR, SparseCSC or SparseBSC) the arguments
    `dim0` and `dim1` must be both batch dimensions, or must both be sparse dimensions.
    The batch dimensions of a sparse tensor are the dimensions preceding the sparse dimensions.

    ## Examples
        iex> a = ExTorch.arange(6) |> ExTorch.reshape({2, 3})
        #Tensor<
        [[0.0000, 1.0000, 2.0000],
         [3.0000, 4.0000, 5.0000]]
        [
          size: {2, 3},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

        iex> ExTorch.transpose(a, 0, 1)
        #Tensor<
        [[0.0000, 3.0000],
         [1.0000, 4.0000],
         [2.0000, 5.0000]]
        [
          size: {3, 2},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>
    """
    @spec transpose(ExTorch.Tensor.t(), integer(), integer()) :: ExTorch.Tensor.t()
    defbinding(transpose(input, dim0, dim1))

    @doc """
    Concatenates the given sequence of `seq` tensors in the given dimension.
    All tensors must either have the same shape (except in the concatenating dimension) or be empty.

    ## Arguments
    - `tensors` (`[ExTorch.Tensor] | tuple()`) - A sequence of tensors of the same type. Non-empty
    tensors provided must have the same shape, except in the cat dimension.

    ## Optional arguments
    - `dim` (`integer()`) - the dimension over which the tensors are concatenated. Default: 0
    - `out` (`ExTorch.Tensor | nil`) - an optional pre-allocated tensor used to store the
    concatenation output. Default: nil

    ## Examples
        iex> x = ExTorch.arange(5) |> ExTorch.unsqueeze(-1)
        #Tensor<
        [[0.0000],
         [1.0000],
         [2.0000],
         [3.0000],
         [4.0000]]
        [
          size: {5, 1},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

        iex> ExTorch.cat([x, x], -1)
        #Tensor<
        [[0.0000, 0.0000],
         [1.0000, 1.0000],
         [2.0000, 2.0000],
         [3.0000, 3.0000],
         [4.0000, 4.0000]]
        [
          size: {5, 2},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>
    """
    @spec cat(ExTorch.Tensor.t(), integer(), ExTorch.Tensor.t() | nil) :: ExTorch.Tensor.t()
    defbinding(cat(input, dim \\ 0, out \\ nil), fn_aliases: [:concatenate, :concat])
  end
end
