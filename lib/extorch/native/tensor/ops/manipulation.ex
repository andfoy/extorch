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
    @spec cat([ExTorch.Tensor.t()] | tuple(), integer(), ExTorch.Tensor.t() | nil) ::
            ExTorch.Tensor.t()
    defbinding(cat(input, dim \\ 0, out \\ nil), fn_aliases: [:concatenate, :concat])

    @doc """
    Attempts to split a tensor into the specified number of chunks. Each chunk is a view of the input tensor.

    If the tensor size along the given dimension `dim` is divisible by `chunks`, all returned chunks
    will be the same size. If the tensor size along the given dimension `dim` is not divisible by `chunks`,
    all returned chunks will be the same size, except the last one. If such division is not possible,
    this function may return fewer than the specified number of chunks.

    ## Arguments
    - `input` (`ExTorch.Tensor`) - the tensor to split
    - `chunks` (`integer`) - number of chunks to return

    ## Optional arguments
    - `dim` (`integer`) - dimension along which to split the tensor. Default: 0

    ## Notes
    * This function may return fewer than the specified number of chunks!
    * Use `ExTorch.tensor_split/3` to ensure that the result will have the exact number of chunks.

    ## Examples
        iex> ExTorch.arange(11) |> ExTorch.chunk(6)
        [
          #Tensor<
        [0.0000, 1.0000]
        [
            size: {2},
            dtype: :float,
            device: :cpu,
            requires_grad: false
          ]>,
          #Tensor<
        [2., 3.]
        [
            size: {2},
            dtype: :float,
            device: :cpu,
            requires_grad: false
          ]>,
          #Tensor<
        [4., 5.]
        [
            size: {2},
            dtype: :float,
            device: :cpu,
            requires_grad: false
          ]>,
          #Tensor<
        [6., 7.]
        [
            size: {2},
            dtype: :float,
            device: :cpu,
            requires_grad: false
          ]>,
          #Tensor<
        [8., 9.]
        [
            size: {2},
            dtype: :float,
            device: :cpu,
            requires_grad: false
          ]>,
          #Tensor<
        [10.]
        [size: {1}, dtype: :float, device: :cpu, requires_grad: false]>
        ]
    """
    @spec chunk(ExTorch.Tensor.t(), integer(), integer()) :: [ExTorch.Tensor.t()]
    defbinding(chunk(input, chunks, dim \\ 0))

    @doc """
    Splits a tensor into multiple sub-tensors, all of which are views of `input`,
    along dimension `dim` according to the indices or number of sections specified
    by `indices_or_sections`.

    ## Arguments
    - `input` (`ExTorch.Tensor`) - the tensor to split.
    - `indices_or_sections` (`integer | ExTorch.Tensor | [integer()] | tuple`) -
      * If `indices_or_sections` is an integer `n` or a zero dimensional long tensor with value `n`,
      `input` is split into `n` sections along dimension `dim`. If `input` is divisible by `n`
      along dimension `dim`, each section will be of equal size, `input.size[dim] / n`.
      * If `input` is not divisible by n, the sizes of the first `input.size[dim] % n` sections will have size
      `input.size[dim] / n + 1`, and the rest will have size `input.size[dim] / n`.
      * If `indices_or_sections` is a list or tuple of ints, or a one-dimensional long tensor,
      then `input` is split along dimension `dim` at each of the indices in the list, tuple or tensor.
      For instance, `indices_or_sections = [2, 3]` and `dim = 0` would result in the tensors
      `input[:2]`, `input[2:3]`, and `input[3:]`.
      * If `indices_or_sections` is a tensor, it must be a zero-dimensional or one-dimensional long tensor on the CPU.

    ## Optional arguments
    - `dim` (`integer`) - dimension along which to split the tensor. Default: 0

    ## Examples
        # Split a tensor in a given number of chunks
        iex> a = ExTorch.arange(10)
        iex> ExTorch.tensor_split(a, 2)
        [
          #Tensor<
        [0.0000, 1.0000, 2.0000, 3.0000, 4.0000]
        [
            size: {5},
            dtype: :float,
            device: :cpu,
            requires_grad: false
          ]>,
          #Tensor<
        [5., 6., 7., 8., 9.]
        [
            size: {5},
            dtype: :float,
            device: :cpu,
            requires_grad: false
          ]>
        ]

        # Split a tensor into the given sections
        iex> ExTorch.tensor_split(a, [2, 5])
        [
          #Tensor<
          [0.0000, 1.0000]
          [size: {2}, dtype: :float, device: :cpu, requires_grad: false]>,
          #Tensor<
          [2., 3., 4.]
          [size: {3}, dtype: :float, device: :cpu, requires_grad: false]>,
          #Tensor<
          [5., 6., 7., 8., 9.]
          [size: {5}, dtype: :float, device: :cpu, requires_grad: false]>
        ]

    """
    @spec tensor_split(
            ExTorch.Tensor.t(),
            integer() | [integer()] | tuple() | ExTorch.Tensor.t(),
            integer()
          ) :: [ExTorch.Tensor.t()]
    defbinding(tensor_split(input, indices_or_sections, dim \\ 0),
      indices_or_sections:
        case indices_or_sections do
          x when is_integer(x) -> x
          %ExTorch.Tensor{} = x -> x
          x -> ExTorch.tensor(x, device: :cpu, dtype: :int64)
        end
    )

    @doc """
    Splits `input`, a tensor with three or more dimensions, into multiple tensors
    depthwise according to `indices_or_sections`. Each split is a view of `input`.

    This is equivalent to calling `ExTorch.tensor_split(input, indices_or_sections, dim: 2)`
    (the split dimension is 2), except that if `indices_or_sections` is an integer it must
    evenly divide the split dimension or a runtime error will be thrown.

    ## Arguments
    - `input` (`ExTorch.Tensor`) - tensor to split.
    - `indices_or_sections` (`integer() | [integer()] | tuple()`) - See argument in `ExTorch.tensor_split/3`

    ## Examples
        iex> t = ExTorch.arange(16) |> ExTorch.reshape({2, 2, 4})
        #Tensor<
        [[[ 0.0000,  1.0000,  2.0000,  3.0000],
          [ 4.0000,  5.0000,  6.0000,  7.0000]],

         [[ 8.0000,  9.0000, 10.0000, 11.0000],
          [12.0000, 13.0000, 14.0000, 15.0000]]]
        [size: {2, 2, 4}, dtype: :float, device: :cpu, requires_grad: false]>

        iex> ExTorch.dsplit(t, 2)
        [
          #Tensor<
          [[[ 0.0000,  1.0000],
            [ 4.0000,  5.0000]],

           [[ 8.0000,  9.0000],
            [12.0000, 13.0000]]]
          [size: {2, 2, 2}, dtype: :float, device: :cpu, requires_grad: false]>,
          #Tensor<
          [[[ 2.,  3.],
            [ 6.,  7.]],

           [[10., 11.],
            [14., 15.]]]
          [size: {2, 2, 2}, dtype: :float, device: :cpu, requires_grad: false]>
        ]

    """
    @spec dsplit(ExTorch.Tensor.t(), integer() | [integer()] | tuple()) :: [
            ExTorch.Tensor.t()
          ]
    defbinding(dsplit(input, indices_or_sections))

    @doc """
    Creates a new tensor by horizontally stacking the tensors in `tensors`.

    Equivalent to `ExTorch.hstack(tensors)`, except each zero or one dimensional
    tensor `t` in `tensors` is first reshaped into a `(ExTorch.Tensor.numel(t), 1)`
    column before being stacked horizontally.

    ## Arguments
    - tensors (`[ExTorch.Tensor] | tuple()`) - sequence of tensors to concatenate.

    ## Optional arguments
    - out (`ExTorch.Tensor | nil`) - an optional pre-allocated tensor used to
    store the output result. Default: `nil`

    ## Examples
        # Stack two 1D tensors
        iex> a = ExTorch.tensor([1, 2, 3])
        iex> b = ExTorch.tensor([4, 5, 6])
        iex> ExTorch.column_stack([a, b])
        #Tensor<
        [[1, 4],
         [2, 5],
         [3, 6]]
        [size: {3, 2}, dtype: :byte, device: :cpu, requires_grad: false]>

        # Stack 2D tensors
        iex> a = ExTorch.arange(5)
        iex> b = ExTorch.arange(10) |> ExTorch.reshape({5, 2})
        iex> ExTorch.column_stack({a, b, b})
        #Tensor<
        [[0.0000, 0.0000, 1.0000, 0.0000, 1.0000],
         [1.0000, 2.0000, 3.0000, 2.0000, 3.0000],
         [2.0000, 4.0000, 5.0000, 4.0000, 5.0000],
         [3.0000, 6.0000, 7.0000, 6.0000, 7.0000],
         [4.0000, 8.0000, 9.0000, 8.0000, 9.0000]]
        [size: {5, 5}, dtype: :float, device: :cpu, requires_grad: false]>
    """
    @spec column_stack([ExTorch.Tensor.t()] | tuple(), ExTorch.Tensor.t() | nil) ::
            ExTorch.Tensor.t()
    defbinding(column_stack(tensors, out \\ nil))

    @doc """
    Stack tensors in sequence depthwise (along third axis).

    This is equivalent to concatenation along the third axis after 1-D and 2-D tensors have been reshaped
    by `ExTorch.atleast_3d/1`.

    ## Arguments
    - `tensors` (`[ExTorch.Tensor.t()] | tuple()`) - sequence of tensors to concatenate.

    ## Optional arguments
     - out (`ExTorch.Tensor | nil`) - an optional pre-allocated tensor used to
    store the output result. Default: `nil`

    ## Examples
        iex> a = ExTorch.tensor([1, 2, 3])
        iex> b = ExTorch.tensor([4, 5, 6])
        iex> ExTorch.dstack({a, b})
        #Tensor<
        [[[1, 4],
          [2, 5],
          [3, 6]]]
        [size: {1, 3, 2}, dtype: :byte, device: :cpu, requires_grad: false]>

        iex> iex> a = ExTorch.tensor([[1],[2],[3]])
        #Tensor<
        [[1],
         [2],
         [3]]
        [size: {3, 1}, dtype: :byte, device: :cpu, requires_grad: false]>
        iex> b = ExTorch.tensor([[4],[5],[6]])
        #Tensor<
        [[4],
         [5],
         [6]]
        [size: {3, 1}, dtype: :byte, device: :cpu, requires_grad: false]>
        iex> ExTorch.dstack([a, b])
        #Tensor<
        [[[1, 4]],

         [[2, 5]],

         [[3, 6]]]
        [size: {3, 1, 2}, dtype: :byte, device: :cpu, requires_grad: false]>
    """
    @spec dstack([ExTorch.Tensor.t()] | tuple(), ExTorch.Tensor.t() | nil) :: ExTorch.Tensor.t()
    defbinding(dstack(tensors, out \\ nil))

    @doc """
    Gathers values along an axis specified by dim.

    For a 3-D tensor the output is specified by:

    ```
    out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
    out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
    out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2
    ```

    `input` and `index` must have the same number of dimensions. It is also required
    that `ExTorch.Tensor.size(index, d) <= ExTorch.Tensor.size(input, d)` for all dimensions `d != dim`.
    `out` will have the same shape as index. Note that `input` and `index` do not broadcast
    against each other.

    ## Arguments
    - `input` (`ExTorch.Tensor`) - the source tensor.
    - `dim` (`integer()`) - the axis along which to index.
    - `index` (`ExTorch.Tensor`) - the indices of elements to gather. Its dtype must be `:int64` or `:long`

    ## Optional arguments
    - `sparse_grad` (`boolean()`) - if `true`, then the gradient w.r.t. `input` will be a sparse tensor. Default: `false`
    - `out` (`ExTorch.Tensor | nil`) - an optional pre-allocated tensor used to
    store the output result. Default: `nil`

    ## Examples
        iex> t = ExTorch.tensor([[1, 2], [3, 4]])
        iex> ExTorch.gather(t, 1, ExTorch.tensor([[0, 0], [1, 0]], dtype: :int64))
        #Tensor<
        [[1, 1],
         [4, 3]]
        [size: {2, 2}, dtype: :byte, device: :cpu, requires_grad: false]>
    """
    @spec gather(
            ExTorch.Tensor.t(),
            integer(),
            ExTorch.Tensor.t(),
            boolean(),
            ExTorch.Tensor.t() | nil
          ) :: ExTorch.Tensor.t()
    defbinding(gather(input, dim, index, sparse_grad \\ false, out \\ nil))
  end
end
